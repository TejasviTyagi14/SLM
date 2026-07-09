"""Ingest oMe-Template (Tier A, MIT, bundled in this repo's data/).

167 generalized named-reaction templates with R-group placeholders (``[*:n]``).
These are expert-written generalized mechanisms -> provenance ``curated``.
They feed Phase 3 (bounded R-group expansion into concrete instances) and are
part of the Phase 6 decontamination blacklist alongside oMe-Gold.

Dirty-label handling: the template file contains a handful of (type, subtype)
pairs that are NOT in the benchmark's allowed ontology (see ontology.json
``template_remap``). We DO NOT remap at ingest -- ingest is faithful parsing.
We only *flag* offending steps here (``_ontology_ok`` per step + a record-level
``needs_remap`` marker in raw_ref) so Phase 3/5 can remap or reject explicitly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import Config, load_config
from ..ontology import load_ontology
from ..schema import make_record, make_step
from .base import SourceInfo

INFO = SourceInfo(
    key="ome_template",
    display="oMe-Template",
    license="MIT",
    tier="A",
    verdict="usable",
)


def ingest(cfg: Optional[Config] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    cfg = cfg or load_config()
    path = cfg.path("omebench_template")
    onto = load_ontology(cfg.path("ontology"))
    allowed = {(p["type"], p["subtype"]) for p in onto["pairs"]}

    with path.open() as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]

    records: List[Dict[str, Any]] = []
    for idx, r in enumerate(data):
        mech: List[Dict[str, Any]] = []
        n_bad = 0
        for i, s in enumerate(r.get("mechanism", []) or []):
            t, st = s.get("type"), s.get("subtype")
            ok = (t, st) in allowed
            if not ok:
                n_bad += 1
            step = make_step(
                step=s.get("step", i + 1),
                type=t,
                subtype=st,
                intermediate_smiles=s.get("intermediate_smiles", ""),
                rationale=s.get("rationale"),
            )
            step["_ontology_ok"] = ok  # Phase 3/5 reads this
            mech.append(step)
        rid = r["reaction_id"]
        records.append(
            make_record(
                reaction_id=f"TMPL-{rid}",
                source="oMe-Template",
                license="MIT",
                provenance="curated",  # expert-written generalized mechanisms
                reactants_smiles=r.get("reactants_smiles", []),
                products_smiles=r.get("products_smiles", []),
                mechanism=mech,
                level=r.get("level"),
                name=r.get("name"),
                conditions=r.get("conditions"),
                raw_ref={
                    "file": str(path.relative_to(cfg.path("root"))),
                    "index": idx,
                    "orig_id": rid,
                    "description": r.get("description"),
                    "has_rgroups": any("[*:" in x for x in r.get("reactants_smiles", [])),
                    "needs_remap": n_bad > 0,
                    "n_nonontology_steps": n_bad,
                },
            )
        )
    return records
