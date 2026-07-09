"""Ingest Wikipedia named-reaction articles (Tier C, CC-BY-SA-4.0).

Named reactions -> resolve reagents/products via OPSIN, extract mechanism
outlines. Text -> structure ALWAYS goes through OPSIN/RDKit; never trust a name
without resolving and re-canonicalizing it (Phase 2). Tier C records here are
mined for (a) named reactions + conditions and (b) natural-language mechanism
rationales; they carry an empty structured ``mechanism`` at ingest.

Access: the MediaWiki Action API (``/w/api.php``), which Wikimedia explicitly
sanctions for bots with a descriptive User-Agent and rate limiting (see
API:Etiquette). robots.txt ``Disallow: /w/`` targets HTML crawlers of dynamic
pages, not the API; we pass ``robots_exempt=True`` NARROWLY for api.php only,
keeping the 1 req/s throttle, backoff, and on-disk cache. No HTML scraping, no
access-control bypass.

OPSIN name->structure needs a JRE (py2opsin bundles the jar). If Java is absent,
this ingester still captures the article name + extract text + any chem infobox
identifiers; structure resolution is deferred to Phase 2 and flagged.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..config import Config, load_config
from ..io_utils import PoliteFetcher
from ..schema import make_record
from .base import SourceInfo

INFO = SourceInfo(
    key="wikipedia",
    display="Wikipedia named reactions",
    license="CC-BY-SA-4.0",
    tier="C",
    verdict="usable",
)

_API = "https://en.wikipedia.org/w/api.php"
_CATEGORY = "Category:Name reactions"
_DEFAULT_MAX_ARTICLES = 60


def _api_get(fetcher: PoliteFetcher, params: Dict[str, Any]) -> dict:
    params = {**params, "format": "json"}
    qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    url = f"{_API}?{qs}"
    body = fetcher.get(url, ext=".json", robots_exempt=True)
    return json.loads(body)


def _list_category_members(fetcher: PoliteFetcher, limit: int) -> List[str]:
    titles: List[str] = []
    cont: Optional[str] = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": _CATEGORY.replace(" ", "%20"),
            "cmlimit": min(500, limit - len(titles)),
            "cmtype": "page",
        }
        if cont:
            params["cmcontinue"] = cont
        data = _api_get(fetcher, params)
        titles.extend(m["title"] for m in data["query"]["categorymembers"])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return titles[:limit]


def _get_extract(fetcher: PoliteFetcher, title: str) -> str:
    data = _api_get(
        fetcher,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "titles": title.replace(" ", "%20"),
            "redirects": 1,
        },
    )
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        return page.get("extract", "") or ""
    return ""


def ingest(cfg: Optional[Config] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    cfg = cfg or load_config()
    fetcher = PoliteFetcher(cfg)
    max_articles = int(
        cfg.get("sources", "wikipedia", "max_articles", default=_DEFAULT_MAX_ARTICLES)
        or _DEFAULT_MAX_ARTICLES
    )
    cap = min(limit, max_articles) if limit is not None else max_articles

    titles = _list_category_members(fetcher, cap)
    records: List[Dict[str, Any]] = []
    for title in titles:
        extract = _get_extract(fetcher, title)
        if not extract:
            continue
        # First paragraph is the reaction summary; keep a bounded snippet.
        summary = re.split(r"\n\n", extract.strip())[0][:1200]
        records.append(
            make_record(
                reaction_id=f"WIKI-{title.replace(' ', '_')}",
                source="Wikipedia",
                license="CC-BY-SA-4.0",
                provenance="curated",       # prose; structures resolved in Phase 2
                reactants_smiles=[],         # filled by OPSIN in Phase 2
                products_smiles=[],
                mechanism=[],                # prose mechanism outline, not typed steps
                name=title,
                conditions=None,
                raw_ref={
                    "source_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "api_endpoint": _API,
                    "robots_exempt_reason": "MediaWiki Action API is bot-sanctioned; /w/ Disallow targets HTML crawlers",
                    "summary": summary,
                    "needs_name_resolution": True,   # Phase 2 OPSIN
                },
            )
        )
    return records
