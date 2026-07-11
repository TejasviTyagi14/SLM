"""Shared parsing for the Baldi-group elementary-step databases (PMechDB/RMechDB).

Both ship as CSV files of ELEMENTARY reaction steps in SMIRKS form
(``reactants>>products``, atom-mapped) plus arrow codes and a source column.
This module turns one such row into the repo's canonical record schema
(``schema.make_record``) with a SINGLE-step mechanism (the elementary step's
product side is that step's ``intermediate_smiles``).

LICENSE (critical): PMechDB and RMechDB data are BOTH CC-BY-NC-ND-4.0 (see
configs/license_inventory.json). The No-Derivatives + click-through terms mean we
must NOT bundle or redistribute the raw data or a derived corpus. Therefore:
  * the ingesters read from a user-provided cache under data/raw/ that the user
    downloads themselves after accepting the license (never committed here),
  * their config verdict is ``quarantine`` so Phase 1's gate refuses to ingest
    them into any released artifact unless a human explicitly clears the license.
The code exists so the mapping is reviewed + tested; the gate keeps it inert.

Ontology honesty: these sources do not carry the oMeBench (type, subtype)
vocabulary. We map structure faithfully and attach a best-effort source-default
subtype, but ALWAYS flag the record ``needs_remap`` (mirroring ome_template) so
Phase 3 remap / Phase 5 validation is the authority on the final label -- we do
not invent verified mechanism labels at ingest.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..schema import make_record, make_step

# Column-name candidates seen across the released CSVs / upload templates.
_SMIRKS_COLS = ("Reaction SMIRKS", "ReactionSMIRKS", "smirks", "reaction_smirks",
                "Reaction Smiles", "reaction_smiles", "SMIRKS")
_ARROW_COLS = ("Arrow codes", "arrow_codes", "Arrow Codes", "arrows")
_SOURCE_COLS = ("Original source", "source", "Original Source", "origin")


def _first_present(row: Dict[str, str], names) -> Optional[str]:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def split_smirks(smirks: str):
    """``reactants>>products`` (or ``reactants>agents>products``) -> lists.

    Returns (reactants, agents, products) as lists of SMILES fragments, or None
    if the string is not a well-formed reaction SMIRKS.
    """
    if not smirks:
        return None
    core = smirks.strip().split()[0]  # drop any trailing whitespace-delimited notes
    parts = core.split(">")
    if len(parts) == 2:
        reac, prod = parts
        agents = ""
    elif len(parts) == 3:
        reac, agents, prod = parts
    else:
        return None
    to_list = [x for x in reac.split(".") if x], \
              [x for x in agents.split(".") if x], \
              [x for x in prod.split(".") if x]
    reac_l, agents_l, prod_l = to_list
    if not reac_l or not prod_l:
        return None
    return reac_l, agents_l, prod_l


def row_to_record(
    row: Dict[str, str],
    idx: int,
    *,
    source: str,
    license: str,
    default_type: str,
    default_subtype: str,
    id_prefix: str,
    file_rel: str,
    allowed_pairs: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Map one CSV row (an elementary step) into a single-step record, or None."""
    smirks = _first_present(row, _SMIRKS_COLS)
    if not smirks:
        return None
    split = split_smirks(smirks)
    if split is None:
        return None
    reactants, agents, products = split

    ontology_ok = (
        allowed_pairs is None or (default_type, default_subtype) in allowed_pairs
    )
    step = make_step(
        step=1,
        type=default_type,
        subtype=default_subtype,
        intermediate_smiles=".".join(products),  # the step's product is the intermediate
        rationale=None,
    )
    step["_ontology_ok"] = ontology_ok

    orig_source = _first_present(row, _SOURCE_COLS)
    arrows = _first_present(row, _ARROW_COLS)
    return make_record(
        reaction_id=f"{id_prefix}-{idx}",
        source=source,
        license=license,
        provenance="curated",  # manually curated elementary steps
        reactants_smiles=reactants,
        products_smiles=products,
        mechanism=[step],
        conditions=("agents:" + ".".join(agents[:6])) if agents else None,
        reaction_smiles_mapped=smirks.strip(),
        raw_ref={
            "file": file_rel,
            "row": idx,
            "orig_source": orig_source,
            "arrow_codes": arrows,
            "agents": agents,
            # Elementary-step DBs don't carry the oMeBench ontology; the label is
            # a source default that Phase 3/5 must remap or reject.
            "needs_remap": True,
        },
    )


def ingest_csv(
    path: Path,
    *,
    source: str,
    license: str,
    default_type: str,
    default_subtype: str,
    id_prefix: str,
    root: Path,
    limit: Optional[int] = None,
    allowed_pairs: Optional[set] = None,
    row_filter: Optional[Callable[[Dict[str, str]], bool]] = None,
) -> List[Dict[str, Any]]:
    """Parse an elementary-step CSV into records. Raises if the file is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"{source} data not found at {path}. This CC-BY-NC-ND dataset is NOT "
            f"bundled; download it yourself after accepting the license at the "
            f"provider's portal and place the CSV there. See "
            f"configs/license_inventory.json (verdict: quarantine)."
        )
    try:
        file_rel = str(path.relative_to(root))
    except ValueError:
        file_rel = str(path)

    records: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit is not None and len(records) >= limit:
                break
            if row_filter is not None and not row_filter(row):
                continue
            rec = row_to_record(
                row, idx,
                source=source, license=license,
                default_type=default_type, default_subtype=default_subtype,
                id_prefix=id_prefix, file_rel=file_rel,
                allowed_pairs=allowed_pairs,
            )
            if rec is not None:
                records.append(rec)
    return records
