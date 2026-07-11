"""Ingest RMechDB — Public Database of Elementary RADICAL reaction steps (Tier A).

RMechDB (Baldi group, UC Irvine; JCIM 2023) is the radical counterpart to
PMechDB: manually-curated elementary RADICAL steps with atom mappings and
arrow-pushing mechanisms. Together with PMechDB it is one of the only public
sources of genuine elementary mechanistic steps.

LICENSE = CC-BY-NC-ND-4.0 (verified 2026-07-09; see configs/license_inventory.json).
Same handling as PMechDB: verdict ``quarantine`` (No-Derivatives + click-through),
the gate refuses ingestion into any release until a human clears the license, and
the raw CSV must be user-downloaded to the configured path (never committed here).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import Config, load_config
from .base import SourceInfo
from .mechdb_common import ingest_csv

INFO = SourceInfo(
    key="rmechdb",
    display="RMechDB",
    license="CC-BY-NC-ND-4.0",
    tier="A",
    verdict="quarantine",
)

_DEFAULT_FILE = "rmechdb/manually_curated_train.csv"
# Radical steps: default to a radical-propagation label (source default only;
# every row is flagged needs_remap for Phase 3/5 to remap or reject).
_DEFAULT_TYPE = "radical"
_DEFAULT_SUBTYPE = "radical_propagation"


def _data_path(cfg: Config):
    rel = cfg.get("sources", "rmechdb", "path", default=_DEFAULT_FILE) or _DEFAULT_FILE
    return cfg.path("raw") / rel


def ingest(cfg: Optional[Config] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    cfg = cfg or load_config()
    return ingest_csv(
        _data_path(cfg),
        source="RMechDB",
        license="CC-BY-NC-ND-4.0",
        default_type=_DEFAULT_TYPE,
        default_subtype=_DEFAULT_SUBTYPE,
        id_prefix="RMECHDB",
        root=cfg.path("root"),
        limit=limit,
    )
