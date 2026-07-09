"""Ingest oMe-Silver (Tier A, MIT, bundled in this repo's data/).

oMe-Silver is 2,493 reactions / 10,541 typed mechanistic steps already in the
target schema -- the highest-value source. IMPORTANT provenance fact discovered
during ingest design: every silver reaction id has the form
``NR-<tmpl>_1-<smiles>_2-<smiles>`` and derives from one of 151 oMe-Template
base templates, with per-step rationales copied verbatim from the parent
template. It is therefore honestly ``template_expanded``, not curated -- and it
MUST pass Phase 6 decontamination against oMe-Gold before any training use.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import Config, load_config
from ..schema import make_record, make_step
from .base import SourceInfo

INFO = SourceInfo(
    key="ome_silver",
    display="oMe-Silver",
    license="MIT",
    tier="A",
    verdict="usable",
)


def _template_base(reaction_id: str) -> str:
    """The parent oMe-Template id, e.g. 'NR-001' from 'NR-001_1-CC_2-CCl'."""
    return reaction_id.split("_")[0]


def ingest(cfg: Optional[Config] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    cfg = cfg or load_config()
    path = cfg.path("omebench_silver")
    records: List[Dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit is not None and len(records) >= limit:
                break
            r = json.loads(line)
            mech = [
                make_step(
                    step=s.get("step", i + 1),
                    type=s.get("type"),
                    subtype=s.get("subtype"),
                    intermediate_smiles=s.get("intermediate_smiles", ""),
                    rationale=s.get("rationale"),
                )
                for i, s in enumerate(r.get("mechanism", []) or [])
            ]
            rid = r["reaction_id"]
            records.append(
                make_record(
                    reaction_id=f"SILVER-{rid}",
                    source="oMe-Silver",
                    license="MIT",
                    provenance="template_expanded",  # honest: it IS template expansion
                    reactants_smiles=r.get("reactants_smiles", []),
                    products_smiles=r.get("products_smiles", []),
                    mechanism=mech,
                    level=r.get("level"),
                    name=r.get("name"),
                    conditions=r.get("conditions"),
                    raw_ref={
                        "file": str(path.relative_to(cfg.path("root"))),
                        "line": line_no,
                        "orig_id": rid,
                        "template_base": _template_base(rid),
                    },
                )
            )
    return records
