"""Ingest USPTO reactions (Lowe) from Figshare (Tier B, CC0-1.0 public domain).

Daniel Lowe's text-mined reactions from US patents (1976-Sep2016), ~1.8M grant
reactions. CC0: no obligations (citing Lowe 2017 is courtesy only). Tier B ->
overall transformations only, NEVER a mechanism.

Format: a single tab-delimited ``.rsmi`` inside a 7z archive. Columns:
``ReactionSmiles`` (atom-mapped ``reactants>agents>products``), ``PatentNumber``,
``ParagraphNum``, ``Year``, ``TextMinedYield``, ``CalculatedYield``.

The reaction SMILES is already atom-mapped by Lowe's pipeline; we keep it in
``reaction_smiles_mapped`` as-is (Phase 4 may re-map / verify). Agents (middle
block) are captured as reagents in raw_ref. Bounded by ``limit`` /
``sources.uspto_lowe.max_rows`` so the gate run stays fast; the archive is
downloaded once and cached under data/raw/.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..config import Config, load_config
from ..io_utils import PoliteFetcher
from ..schema import make_record
from .base import SourceInfo

INFO = SourceInfo(
    key="uspto_lowe",
    display="USPTO (Lowe)",
    license="CC0-1.0",
    tier="B",
    verdict="usable",
)

# Grants SMILES archive on Figshare (article 5104873, file 8664379).
_ARCHIVE_URL = "https://ndownloader.figshare.com/files/8664379"
_MEMBER = "1976_Sep2016_USPTOgrants_smiles.rsmi"
_DEFAULT_MAX_ROWS = 20000


def _ensure_extracted(fetcher: PoliteFetcher) -> str:
    """Download (cached) + extract the rsmi once; return the extracted path."""
    import py7zr

    archive_bytes = fetcher.get(_ARCHIVE_URL, ext=".7z")
    archive_path = fetcher._cache_path(_ARCHIVE_URL, ".7z")
    if not archive_path.exists():  # PoliteFetcher writes it, but be defensive
        archive_path.write_bytes(archive_bytes)

    extract_dir = archive_path.with_suffix(".extracted")
    target = extract_dir / _MEMBER
    if not target.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(archive_path, "r") as z:
            z.extractall(path=str(extract_dir))
    return str(target)


def _split_rxn_smiles(rsmi: str):
    """reactants>agents>products -> (reactants[list], agents[list], products[list])."""
    parts = rsmi.split(">")
    if len(parts) != 3:
        return None
    reac, agents, prod = parts
    to_list = lambda s: [x for x in s.split(".") if x]  # noqa: E731
    return to_list(reac), to_list(agents), to_list(prod)


def ingest(cfg: Optional[Config] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    cfg = cfg or load_config()
    fetcher = PoliteFetcher(cfg)
    max_rows = int(
        cfg.get("sources", "uspto_lowe", "max_rows", default=_DEFAULT_MAX_ROWS) or _DEFAULT_MAX_ROWS
    )
    cap = min(limit, max_rows) if limit is not None else max_rows

    path = _ensure_extracted(fetcher)
    records: List[Dict[str, Any]] = []
    rel = os.path.relpath(path, cfg.path("root"))

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line_no, line in enumerate(f):
            if len(records) >= cap:
                break
            fields = line.rstrip("\n").split("\t")
            if len(fields) < len(header):
                continue
            rsmi = fields[col["ReactionSmiles"]]
            split = _split_rxn_smiles(rsmi)
            if split is None:
                continue
            reactants, agents, products = split
            if not reactants or not products:
                continue
            patent = fields[col.get("PatentNumber", 1)] if "PatentNumber" in col else None
            year = fields[col.get("Year", 3)] if "Year" in col else None
            yield_ = fields[col["CalculatedYield"]] if "CalculatedYield" in col else ""
            records.append(
                make_record(
                    reaction_id=f"USPTO-{patent}-{line_no}",
                    source="USPTO-Lowe",
                    license="CC0-1.0",
                    provenance="curated",  # overall reaction, not a mechanism
                    reactants_smiles=reactants,
                    products_smiles=products,
                    mechanism=[],          # Tier B: NO mechanism, ever
                    conditions=("agents:" + ".".join(agents[:6])) if agents else None,
                    reaction_smiles_mapped=rsmi,  # Lowe's atom mapping, kept as-is
                    raw_ref={
                        "file": rel,
                        "line": line_no,
                        "patent": patent,
                        "year": year,
                        "agents": agents,
                        "calculated_yield": yield_ or None,
                    },
                )
            )
    return records
