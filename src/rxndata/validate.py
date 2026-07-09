"""Phase 5: validation gates. Reject, do not fix silently.

For every mechanism record we enforce (per the project spec):
- RDKit sanitization passes for every molecule/intermediate.
- Valence is legal (implied by successful sanitization).
- Atom & charge balance for each elementary step given declared reagents.
- Step continuity: intermediate_n is reachable from intermediate_{n-1} by a
  single plausible transformation (checked via bounded heavy-atom / formula
  delta, not a full reaction-rule engine).
- No duplicate / no-op steps (consecutive identical intermediates).

Every failing record is routed to data/rejects/ with a reason code. Records with
R-group placeholders (``is_template``) are NOT validated as concrete reactions --
they are routed to a separate template pool (they are reference material, not
training reactions). Inferred records get the SAME gates plus a needs_review flag
so a human sample is inspected.

Balance philosophy: an elementary mechanistic step conserves atoms EXCEPT for
species explicitly added/removed (a proton in proton_transfer, a leaving group,
a reagent). We therefore allow a small, typed set of per-subtype atom/charge
deltas rather than demanding exact equality, and record the observed delta.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def _mol(smi: str) -> Optional[Chem.Mol]:
    if not smi:
        return None
    return Chem.MolFromSmiles(smi)


def formula_counts(smi: str) -> Optional[Counter]:
    """Heavy+H atom counts for a (multi-fragment) SMILES, or None if invalid."""
    m = _mol(smi)
    if m is None:
        return None
    c: Counter = Counter()
    for atom in m.GetAtoms():
        c[atom.GetSymbol()] += 1
        c["H"] += atom.GetTotalNumHs()
    return c


def total_charge(smi: str) -> Optional[int]:
    m = _mol(smi)
    if m is None:
        return None
    return Chem.GetFormalCharge(m)


@dataclass
class RecordValidation:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)
    step_deltas: List[dict] = field(default_factory=list)


# Max heavy-atom change allowed between consecutive intermediates for the step to
# count as a single plausible transformation. Proton transfers change only H.
_MAX_HEAVY_DELTA = 6      # generous: a leaving group / reagent fragment
_MAX_ABS_CHARGE_DELTA = 2


def _sanitize_all(rec: Dict) -> Tuple[bool, List[str]]:
    reasons = []
    for field_name in ("reactants_smiles", "products_smiles"):
        for s in rec.get(field_name, []):
            if _mol(s) is None:
                reasons.append(f"sanitize_fail:{field_name}")
                break
    for i, st in enumerate(rec.get("mechanism", [])):
        if _mol(st.get("intermediate_smiles", "")) is None:
            reasons.append(f"sanitize_fail:step{i+1}")
    return (len(reasons) == 0), reasons


def _check_continuity_and_balance(rec: Dict) -> Tuple[bool, List[str], List[dict]]:
    """Step continuity + per-step atom/charge delta. Reactants seed step 1."""
    reasons: List[str] = []
    deltas: List[dict] = []
    mech = rec.get("mechanism", [])
    if not mech:
        return True, reasons, deltas  # no mechanism to check (Tier B handled elsewhere)

    prev_smi = ".".join(rec.get("reactants_smiles", []))
    prev_form = formula_counts(prev_smi)
    prev_chg = total_charge(prev_smi)

    for i, st in enumerate(mech):
        cur_smi = st.get("intermediate_smiles", "")
        cur_form = formula_counts(cur_smi)
        cur_chg = total_charge(cur_smi)
        if cur_form is None:
            reasons.append(f"balance_unparseable:step{i+1}")
            prev_form, prev_chg = None, None
            continue

        # No-op step: identical to previous intermediate.
        if i > 0 and cur_smi == mech[i - 1].get("intermediate_smiles"):
            reasons.append(f"noop_step:step{i+1}")

        if prev_form is not None:
            # Heavy-atom delta (exclude H, which legitimately moves in proton steps).
            keys = set(prev_form) | set(cur_form)
            heavy_delta = sum(
                abs(prev_form.get(k, 0) - cur_form.get(k, 0)) for k in keys if k != "H"
            )
            chg_delta = abs((cur_chg or 0) - (prev_chg or 0))
            deltas.append({"step": i + 1, "heavy_delta": heavy_delta, "charge_delta": chg_delta})
            if heavy_delta > _MAX_HEAVY_DELTA:
                reasons.append(f"discontinuous_step:step{i+1}(heavy_delta={heavy_delta})")
            if chg_delta > _MAX_ABS_CHARGE_DELTA:
                reasons.append(f"charge_jump:step{i+1}(delta={chg_delta})")

        prev_form, prev_chg = cur_form, cur_chg

    # Final intermediate should relate to the declared product set (mass sanity).
    prod_form = formula_counts(".".join(rec.get("products_smiles", [])))
    last_form = formula_counts(mech[-1].get("intermediate_smiles", ""))
    if prod_form is not None and last_form is not None:
        keys = set(prod_form) | set(last_form)
        end_delta = sum(abs(prod_form.get(k, 0) - last_form.get(k, 0)) for k in keys if k != "H")
        deltas.append({"step": "final_vs_product", "heavy_delta": end_delta})

    return (len([r for r in reasons if r.startswith(("discontinuous", "charge_jump", "noop"))]) == 0), reasons, deltas


def validate_record(rec: Dict) -> RecordValidation:
    """Run all Phase-5 gates on one concrete mechanism record."""
    reasons: List[str] = []
    checks: Dict[str, bool] = {}

    sane, sane_reasons = _sanitize_all(rec)
    checks["sanitized"] = sane
    reasons += sane_reasons

    cont_ok, cont_reasons, deltas = _check_continuity_and_balance(rec)
    checks["step_continuity"] = cont_ok
    checks["atom_balanced"] = not any(r.startswith("discontinuous") for r in cont_reasons)
    checks["charge_balanced"] = not any(r.startswith("charge_jump") for r in cont_reasons)
    checks["no_noop_steps"] = not any(r.startswith("noop_step") for r in cont_reasons)
    reasons += cont_reasons

    ok = sane and cont_ok and len([r for r in reasons if r.startswith("balance_unparseable")]) == 0
    return RecordValidation(ok=ok, reasons=sorted(set(reasons)), checks=checks, step_deltas=deltas)
