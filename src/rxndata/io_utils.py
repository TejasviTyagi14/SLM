"""Shared IO: parquet writing for interim data, and a polite cached HTTP fetcher.

Scraping etiquette (HARD CONSTRAINT): prefer official bulk dumps/APIs; honor
robots.txt; descriptive User-Agent; <=1 req/sec per host with exponential
backoff; cache raw responses to data/raw/ so re-runs don't re-hit servers.
"""

from __future__ import annotations

import hashlib
import time
import urllib.robotparser
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .config import Config, load_config
from .schema import records_to_rows


# ---- parquet ----------------------------------------------------------------

def write_interim(records: List[Dict[str, Any]], source_name: str, cfg: Optional[Config] = None) -> Path:
    """Write normalized records to data/interim/<source>.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = cfg or load_config()
    out_dir = cfg.path("interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_name}.parquet"
    rows = records_to_rows(records)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    return out_path


def read_interim(source_name: str, cfg: Optional[Config] = None) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    from .schema import rows_to_records

    cfg = cfg or load_config()
    path = cfg.path("interim") / f"{source_name}.parquet"
    table = pq.read_table(path)
    return rows_to_records(table.to_pylist())


# ---- polite cached HTTP -----------------------------------------------------

class PoliteFetcher:
    """Rate-limited, robots-respecting, disk-cached HTTP GET.

    One instance per pipeline run. Caches every response body under
    data/raw/<host>/<sha1(url)>.<ext> so re-runs are offline and deterministic.
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or load_config()
        http = self.cfg.raw.get("http", {})
        self.user_agent = http.get("user_agent", "rxndata-bot/0.1")
        self.min_interval = 1.0 / float(http.get("max_requests_per_sec_per_host", 1.0))
        bo = http.get("backoff", {})
        self.bo_initial = float(bo.get("initial_sec", 2.0))
        self.bo_factor = float(bo.get("factor", 2.0))
        self.bo_max = float(bo.get("max_sec", 120.0))
        self.bo_retries = int(bo.get("max_retries", 5))
        self.respect_robots = bool(http.get("respect_robots_txt", True))
        self.cache_raw = bool(http.get("cache_raw", True))
        self.raw_dir = self.cfg.path("raw")
        self._last_hit: Dict[str, float] = {}       # host -> monotonic ts
        self._robots: Dict[str, urllib.robotparser.RobotFileParser] = {}

    # -- robots --
    def _robots_ok(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        host = parsed.netloc
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{parsed.scheme}://{host}/robots.txt")
            try:
                rp.read()
            except Exception:
                # If robots.txt is unreachable, default to allow but note it.
                rp = None  # type: ignore
            self._robots[host] = rp  # type: ignore
        rp = self._robots[host]
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    # -- rate limit --
    def _throttle(self, host: str) -> None:
        now = time.monotonic()
        last = self._last_hit.get(host)
        if last is not None:
            wait = self.min_interval - (now - last)
            if wait > 0:
                time.sleep(wait)
        self._last_hit[host] = time.monotonic()

    def _cache_path(self, url: str, ext: str) -> Path:
        host = urlparse(url).netloc or "nohost"
        digest = hashlib.sha1(url.encode()).hexdigest()[:16]
        d = self.raw_dir / host
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{digest}{ext}"

    def get(self, url: str, ext: str = ".bin", binary: bool = True,
            params: Optional[Dict[str, Any]] = None, force: bool = False,
            robots_exempt: bool = False) -> bytes:
        """GET a URL with caching, throttling, robots + backoff. Returns bytes.

        ``robots_exempt`` narrowly bypasses the robots.txt check for officially
        sanctioned programmatic endpoints (e.g. the MediaWiki Action API, whose
        provider explicitly invites bot use with a proper User-Agent + rate
        limit, while robots.txt ``Disallow: /w/`` targets HTML crawlers). Callers
        must justify each use; rate-limiting, caching, and UA still apply.
        """
        import requests

        cache_key = url + ("?" + "&".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "")
        cache_path = self._cache_path(cache_key, ext)
        if self.cache_raw and cache_path.exists() and not force:
            return cache_path.read_bytes()

        if not robots_exempt and not self._robots_ok(url):
            raise PermissionError(f"robots.txt disallows fetching {url}")

        host = urlparse(url).netloc
        delay = self.bo_initial
        last_exc: Optional[Exception] = None
        for attempt in range(self.bo_retries + 1):
            self._throttle(host)
            try:
                resp = requests.get(
                    url, params=params, headers={"User-Agent": self.user_agent}, timeout=60
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"status {resp.status_code}")
                resp.raise_for_status()
                body = resp.content if binary else resp.text.encode()
                if self.cache_raw:
                    cache_path.write_bytes(body)
                return body
            except Exception as e:  # noqa: BLE001 - retry any transient error
                last_exc = e
                if attempt >= self.bo_retries:
                    break
                time.sleep(min(delay, self.bo_max))
                delay *= self.bo_factor
        raise RuntimeError(f"GET failed after {self.bo_retries} retries: {url}: {last_exc}")
