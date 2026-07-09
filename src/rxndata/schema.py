"""The canonical internal record schema.

Every source normalizes into this one record type (a superset of the oMeBench
entry so we can emit the benchmark format losslessly). See the project spec for
the field contract. This module provides:

- ``RECORD_FIELDS`` / ``STEP_FIELDS`` : the authoritative field lists.
- ``Provenance`` / ``Level`` : controlled vocabularies.
- ``make_record`` / ``make_step`` : constructors that enforce presence + types.
- ``validate_record`` : lightweight structural check (NOT chemistry validation;
  that is Phase 5). Returns a list of problems (empty == ok).

Interim parquet stores the raw-but-parsed rows; the ``mechanism`` list and
SMILES lists are stored as JSON strings for a flat, engine-agnostic columnar
layout, then rehydrated by ``rows_to_records`` / ``records_to_rows``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Controlled vocabularies -----------------------------------------------------

PROVENANCE = ("curated", "template_expanded", "inferred")
LEVELS = ("easy", "medium", "hard")

# Field contracts -------------------------------------------------------------

STEP_FIELDS = (
    "step",                 # int, 1-based
    "type",                 # ontology type
    "subtype",              # ontology subtype (the oMeS alignment key)
    "intermediate_smiles",  # canonical, valid, balanced vs previous step
    "rationale",            # nullable free text
)

RECORD_FIELDS = (
    "reaction_id",          # globally unique, source-prefixed
    "source",               # e.g. PMechDB, ORD, oMe-Silver
    "license",              # SPDX-ish label captured at ingest
    "provenance",           # curated | template_expanded | inferred
    "level",                # easy | medium | hard (nullable at ingest)
    "name",                 # named reaction or null
    "reactants_smiles",     # list[str]
    "products_smiles",      # list[str]
    "conditions",           # free text, nullable
    "reaction_smiles_mapped",  # atom-mapped >>, filled in Phase 4 (nullable)
    "mechanism_step_nums",  # int (len of mechanism), nullable for Tier-B-only rows
    "mechanism",            # list[step dict]; may be empty for overall-only rows
    "checks",               # dict of Phase-5 booleans (filled later)
    "raw_ref",              # provenance pointer: {"file":..., "id"/"line":...}
)


def make_step(
    step: int,
    type: str,
    subtype: str,
    intermediate_smiles: str,
    rationale: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "step": int(step),
        "type": type,
        "subtype": subtype,
        "intermediate_smiles": intermediate_smiles,
        "rationale": rationale,
    }


def make_record(
    reaction_id: str,
    source: str,
    license: str,
    provenance: str,
    reactants_smiles: List[str],
    products_smiles: List[str],
    mechanism: Optional[List[Dict[str, Any]]] = None,
    level: Optional[str] = None,
    name: Optional[str] = None,
    conditions: Optional[str] = None,
    reaction_smiles_mapped: Optional[str] = None,
    raw_ref: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mechanism = mechanism or []
    return {
        "reaction_id": reaction_id,
        "source": source,
        "license": license,
        "provenance": provenance,
        "level": level,
        "name": name,
        "reactants_smiles": list(reactants_smiles),
        "products_smiles": list(products_smiles),
        "conditions": conditions,
        "reaction_smiles_mapped": reaction_smiles_mapped,
        "mechanism_step_nums": len(mechanism),
        "mechanism": mechanism,
        "checks": {},
        "raw_ref": raw_ref or {},
    }


def validate_record(rec: Dict[str, Any]) -> List[str]:
    """Structural (not chemical) validation. Returns a list of problems."""
    problems: List[str] = []
    for f in RECORD_FIELDS:
        if f not in rec:
            problems.append(f"missing field: {f}")
    if rec.get("provenance") not in PROVENANCE:
        problems.append(f"bad provenance: {rec.get('provenance')!r}")
    lvl = rec.get("level")
    if lvl is not None and lvl not in LEVELS:
        problems.append(f"bad level: {lvl!r}")
    for key in ("reactants_smiles", "products_smiles"):
        if not isinstance(rec.get(key), list):
            problems.append(f"{key} must be a list")
    if not isinstance(rec.get("mechanism"), list):
        problems.append("mechanism must be a list")
    else:
        for i, s in enumerate(rec["mechanism"]):
            for sf in ("step", "type", "subtype", "intermediate_smiles"):
                if sf not in s:
                    problems.append(f"mechanism[{i}] missing {sf}")
    return problems


# Parquet interchange ---------------------------------------------------------

# Columns stored as JSON strings for a flat columnar layout.
_JSON_COLS = ("reactants_smiles", "products_smiles", "mechanism", "checks", "raw_ref")


def record_to_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a record for parquet: JSON-encode nested columns."""
    row = dict(rec)
    for c in _JSON_COLS:
        row[c] = json.dumps(row.get(c))
    return row


def row_to_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Rehydrate a parquet row back into a record."""
    rec = dict(row)
    for c in _JSON_COLS:
        v = rec.get(c)
        rec[c] = json.loads(v) if isinstance(v, str) else v
    return rec


def records_to_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record_to_row(r) for r in records]


def rows_to_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row_to_record(r) for r in rows]
