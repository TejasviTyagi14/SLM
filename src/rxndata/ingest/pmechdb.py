"""Ingest PMechDB — Public Database of Elementary POLAR reaction steps (Tier A).

PMechDB (Baldi group, UC Irvine; JCIM 2024) is ~100k manually-curated + combinatorial
elementary POLAR steps with atom mappings and arrow-pushing mechanisms -- one of
the very few public corpora with genuine ELEMENTARY mechanistic steps (USPTO/ORD/
Pistachio have none). That makes it high-value for a mechanism specialist.

LICENSE = CC-BY-NC-ND-4.0 (verified 2026-07-09; see configs/license_inventory.json).
The No-Derivatives + click-through terms mean we cannot bundle or redistribute the
data or a derived corpus. So the config verdict is ``quarantine`` and Phase 1's
gate refuses to ingest it into any release until a human clears the license. This
module only defines the (reviewed, tested) schema mapping; the gate keeps it inert
by default. The raw CSV must be downloaded by the user (after accepting the license)
to the configured path -- it is never committed to this repo.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import Config, load_config
from .base import SourceInfo
from .mechdb_common import ingest_csv

INFO = SourceInfo(
    key="pmechdb",
    display="PMechDB",
    license="CC-BY-NC-ND-4.0",
    tier="A",
    verdict="quarantine",
)

# Default filename of the manually-curated training split (see PMechDB docs).
_DEFAULT_FILE = "pmechdb/manually_curated_train.csv"
# Polar steps: default to a heterolytic (2-electron) cleavage label. This is a
# source default only -- every row is flagged needs_remap for Phase 3/5.
_DEFAULT_TYPE = "cleavage"
_DEFAULT_SUBTYPE = "heterolytic_cleavage"


def _data_path(cfg: Config):
    rel = cfg.get("sources", "pmechdb", "path", default=_DEFAULT_FILE) or _DEFAULT_FILE
    return cfg.path("raw") / rel


def ingest(cfg: Optional[Config] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    cfg = cfg or load_config()
    return ingest_csv(
        _data_path(cfg),
        source="PMechDB",
        license="CC-BY-NC-ND-4.0",
        default_type=_DEFAULT_TYPE,
        default_subtype=_DEFAULT_SUBTYPE,
        id_prefix="PMECHDB",
        root=cfg.path("root"),
        limit=limit,
    )
