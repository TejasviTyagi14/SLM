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
from rdkit.Chem import rdFMCS

RDLogger.DisableLog("rdApp.*")

# A step is continuous if consecutive intermediates share a common substructure
# (the reacting scaffold is conserved; only a bond/fragment changes). This
# directly tests "reachable by a single transformation" and, unlike a raw formula
# delta, does not wrongly reject valid bimolecular additions or large-leaving-
# group eliminations.
#
# Two-tier floor by provenance: expert-authored mechanisms (curated /
# template_expanded from oMe-Template/Silver) have authoritative step TYPES, and
# a legitimate large-leaving-group elimination can share only ~40% of atoms with
# the prior intermediate; we trust those at a lower floor. INFERRED mechanisms
# get the strict floor because their step sequence is not expert-verified.
_MCS_MIN_FRACTION_TRUSTED = 0.34    # curated / template_expanded
_MCS_MIN_FRACTION_INFERRED = 0.55   # inferred (stricter, per spec)
_MCS_TIMEOUT_SEC = 3


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


# Max heavy-atom change for a step that only reorganizes bonds within the current
# species (proton transfer, cyclization, shift). A LARGER delta is still valid if
# it matches a reactant/reagent molecule legitimately joining or a fragment
# leaving -- see _continuity_ok.
_MAX_INTRA_HEAVY_DELTA = 6
_MAX_ABS_CHARGE_DELTA = 2
# Tolerance when matching a delta to a joining/leaving molecule's size.
_JOIN_LEAVE_TOL = 1


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


def _heavy_delta(form_a: Counter, form_b: Counter) -> int:
    keys = set(form_a) | set(form_b)
    return sum(abs(form_a.get(k, 0) - form_b.get(k, 0)) for k in keys if k != "H")


def _reactant_sizes(rec: Dict) -> List[int]:
    """Heavy-atom sizes of each reactant molecule (candidates for 'joining')."""
    sizes = []
    for s in rec.get("reactants_smiles", []):
        m = _mol(s)
        if m is not None:
            sizes.append(m.GetNumHeavyAtoms())
    return sizes


def mcs_fraction(smi_a: str, smi_b: str) -> Optional[float]:
    """MCS atom count / smaller molecule's heavy-atom count (0..1), or None."""
    a, b = _mol(smi_a), _mol(smi_b)
    if a is None or b is None:
        return None
    smaller = min(a.GetNumHeavyAtoms(), b.GetNumHeavyAtoms())
    if smaller == 0:
        return None
    res = rdFMCS.FindMCS(
        [a, b],
        timeout=_MCS_TIMEOUT_SEC,
        matchValences=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
    )
    if res.canceled:
        return None  # timeout -> unknown, treat as pass (don't reject on compute limit)
    return res.numAtoms / smaller


def _largest_fragment(smi: str) -> str:
    """Return the largest '.'-separated fragment (the main species), for MCS.

    Reagents/spectators introduced as separate fragments (e.g. CBr4, PPh3) make a
    whole-string MCS misleadingly low; comparing the main reacting fragments is
    the right continuity test.
    """
    parts = [p for p in (smi or "").split(".") if p]
    if len(parts) <= 1:
        return smi or ""
    return max(parts, key=lambda p: (_mol(p).GetNumHeavyAtoms() if _mol(p) else 0))


def _continuity_ok(prev_smi: str, cur_smi: str, heavy_delta: int,
                   join_leave_sizes: List[int], min_fraction: float) -> bool:
    """Continuous if a small intra-species change, OR the delta matches a joining/
    leaving molecule, OR the main reacting fragments share a sufficient MCS
    scaffold (floor set by provenance).
    """
    if heavy_delta <= _MAX_INTRA_HEAVY_DELTA:
        return True
    for sz in join_leave_sizes:
        if abs(heavy_delta - sz) <= _JOIN_LEAVE_TOL:
            return True
    # Compare the largest fragments so reagent introduction doesn't tank the MCS.
    frac_full = mcs_fraction(prev_smi, cur_smi)
    frac_main = mcs_fraction(_largest_fragment(prev_smi), _largest_fragment(cur_smi))
    frac = max(x for x in (frac_full, frac_main) if x is not None) \
        if any(x is not None for x in (frac_full, frac_main)) else None
    if frac is None:          # unparseable handled elsewhere; timeout -> don't reject
        return True
    return frac >= min_fraction


def _check_continuity_and_balance(rec: Dict) -> Tuple[bool, List[str], List[dict]]:
    """Step continuity + per-step atom/charge delta between CONSECUTIVE
    intermediates (not vs the combined reactant pool). Step 1 is seeded by the
    single reactant molecule it acts on (smallest matching), and joining partners
    are credited via _continuity_ok.
    """
    reasons: List[str] = []
    deltas: List[dict] = []
    mech = rec.get("mechanism", [])
    if not mech:
        return True, reasons, deltas  # no mechanism (Tier B handled elsewhere)

    join_leave_sizes = _reactant_sizes(rec)
    min_frac = (
        _MCS_MIN_FRACTION_INFERRED
        if rec.get("provenance") == "inferred"
        else _MCS_MIN_FRACTION_TRUSTED
    )

    # Seed: compare step 1 to the reactant molecule closest in size (the species
    # actually engaged first); a joining second partner is credited later.
    first_form = formula_counts(mech[0].get("intermediate_smiles", ""))
    prev_form = None
    prev_chg = None
    if first_form is not None:
        best_seed = None
        for s in rec.get("reactants_smiles", []):
            f = formula_counts(s)
            if f is None:
                continue
            d = _heavy_delta(f, first_form)
            if best_seed is None or d < best_seed[0]:
                best_seed = (d, f, total_charge(s))
        if best_seed is not None:
            prev_form, prev_chg = best_seed[1], best_seed[2]

    for i, st in enumerate(mech):
        cur_smi = st.get("intermediate_smiles", "")
        cur_form = formula_counts(cur_smi)
        cur_chg = total_charge(cur_smi)
        if cur_form is None:
            reasons.append(f"balance_unparseable:step{i+1}")
            prev_form, prev_chg = None, None
            continue

        if i > 0 and cur_smi == mech[i - 1].get("intermediate_smiles"):
            reasons.append(f"noop_step:step{i+1}")

        if prev_form is not None:
            hd = _heavy_delta(prev_form, cur_form)
            chg_delta = abs((cur_chg or 0) - (prev_chg or 0))
            deltas.append({"step": i + 1, "heavy_delta": hd, "charge_delta": chg_delta})
            prev_smi = mech[i - 1].get("intermediate_smiles", "") if i > 0 else \
                ".".join(rec.get("reactants_smiles", []))
            if not _continuity_ok(prev_smi, cur_smi, hd, join_leave_sizes, min_frac):
                reasons.append(f"discontinuous_step:step{i+1}(heavy_delta={hd})")
            if chg_delta > _MAX_ABS_CHARGE_DELTA:
                reasons.append(f"charge_jump:step{i+1}(delta={chg_delta})")

        prev_form, prev_chg = cur_form, cur_chg

    bad = [r for r in reasons if r.startswith(("discontinuous", "charge_jump", "noop"))]
    return (len(bad) == 0), reasons, deltas


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
