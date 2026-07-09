"""Phase 4: atom-atom mapping via RXNMapper (transformer-based).

We do NOT hand-roll atom mapping. RXNMapper maps a ``reactants>>products``
reaction SMILES and returns a confidence score. We map:

- the OVERALL reaction (reactants >> products) for every record, and
- each ELEMENTARY step as ``intermediate_{n-1} >> intermediate_{n}`` (the first
  step uses the reactants as the left side), which is what makes the step-level
  electron bookkeeping checkable in Phase 5.

Mapping is skipped for records still carrying R-group placeholders (templates)
and for reactions whose SMILES don't sanitize. Maps below
``atommap.min_confidence`` are QUARANTINED (kept but flagged), never silently
trusted.

RXNMapper loads a transformer model on first use; we lazy-init a single shared
instance. If the import/model is unavailable, the module degrades: records pass
through unmapped with reason ``rxnmapper_unavailable``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=1)
def _get_mapper():
    try:
        from rxnmapper import RXNMapper

        return RXNMapper()
    except Exception:  # noqa: BLE001
        return None


def _has_rgroup(smi: str) -> bool:
    return "[*" in (smi or "") or "*" in (smi or "")


def _valid_rxn_side(smis: List[str]) -> bool:
    return bool(smis) and all(s and not _has_rgroup(s) and Chem.MolFromSmiles(s) for s in smis)


@dataclass
class MapResult:
    mapped_rxn: Optional[str]
    confidence: float
    ok: bool
    reason: Optional[str] = None


def map_reaction(reactant_smis: List[str], product_smis: List[str]) -> MapResult:
    """Map one reactants>>products reaction. Returns MapResult (never raises)."""
    if not _valid_rxn_side(reactant_smis) or not _valid_rxn_side(product_smis):
        return MapResult(None, 0.0, False, "invalid_or_rgroup")
    mapper = _get_mapper()
    if mapper is None:
        return MapResult(None, 0.0, False, "rxnmapper_unavailable")
    rxn = f"{'.'.join(reactant_smis)}>>{'.'.join(product_smis)}"
    try:
        out = mapper.get_attention_guided_atom_maps([rxn])[0]
    except Exception as e:  # noqa: BLE001
        return MapResult(None, 0.0, False, f"map_error:{type(e).__name__}")
    return MapResult(out.get("mapped_rxn"), float(out.get("confidence", 0.0)), True)


def map_record(rec: Dict, min_confidence: float, map_steps: bool = True) -> Dict:
    """Map a record's overall reaction (and, optionally, each elementary step).

    Adds:
      reaction_smiles_mapped (overall), _map_confidence, _map_reason
      per-step: _step_mapped_rxn, _step_map_confidence (when map_steps)
    Records below min_confidence are flagged ``_map_quarantine=True``.
    Preserves an existing reaction_smiles_mapped (e.g. Lowe's) as _premapped.
    """
    out = dict(rec)

    # Overall reaction mapping.
    pre = rec.get("reaction_smiles_mapped")
    res = map_reaction(rec.get("reactants_smiles", []), rec.get("products_smiles", []))
    if pre and not res.ok:
        # Keep a source-provided mapping (e.g. USPTO/Lowe) if we couldn't remap.
        out["reaction_smiles_mapped"] = pre
        out["_map_confidence"] = None
        out["_map_reason"] = f"kept_premapped({res.reason})"
        out["_map_quarantine"] = False
    else:
        out["reaction_smiles_mapped"] = res.mapped_rxn if res.ok else pre
        out["_map_confidence"] = round(res.confidence, 4) if res.ok else None
        out["_map_reason"] = res.reason
        out["_map_quarantine"] = bool(res.ok and res.confidence < min_confidence)
        if pre:
            out["_premapped"] = pre

    # Elementary-step mapping: intermediate_{n-1} >> intermediate_{n}.
    if map_steps and rec.get("mechanism"):
        prev_sides = [rec.get("reactants_smiles", [])]
        new_mech = []
        prev = rec.get("reactants_smiles", [])
        step_confs = []
        for st in rec["mechanism"]:
            cur = [st.get("intermediate_smiles", "")]
            sres = map_reaction(prev, cur)
            s2 = dict(st)
            if sres.ok:
                s2["_step_mapped_rxn"] = sres.mapped_rxn
                s2["_step_map_confidence"] = round(sres.confidence, 4)
                step_confs.append(sres.confidence)
            else:
                s2["_step_mapped_rxn"] = None
                s2["_step_map_confidence"] = None
                s2["_step_map_reason"] = sres.reason
            new_mech.append(s2)
            prev = cur
        out["mechanism"] = new_mech
        out["_step_map_min_confidence"] = round(min(step_confs), 4) if step_confs else None
    return out
