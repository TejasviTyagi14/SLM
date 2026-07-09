"""Phase 5 validation-gate tests."""

from __future__ import annotations

from rxndata.schema import make_record, make_step
from rxndata.validate import mcs_fraction, validate_record


def _rec(mech_steps, reactants, products, provenance="template_expanded"):
    return make_record(
        "T", "s", "MIT", provenance, reactants, products,
        mechanism=[make_step(i + 1, t, st, smi) for i, (t, st, smi) in enumerate(mech_steps)],
    )


def test_valid_aldol_passes():
    # Bimolecular aldol: step2 nucleophilic addition brings in the 2nd carbonyl.
    rec = _rec(
        [
            ("proton_transfer", "acid_base_proton_transfer", "[CH2-]C=O"),
            ("addition", "nucleophilic_addition", "CC([O-])CC=O"),
            ("proton_transfer", "acid_base_proton_transfer", "CC(O)CC=O"),
        ],
        ["CC=O", "CC=O"],
        ["CC(O)CC=O"],
    )
    v = validate_record(rec)
    assert v.ok, v.reasons
    assert v.checks["sanitized"] and v.checks["step_continuity"]


def test_unsanitizable_intermediate_rejected():
    rec = _rec(
        [("proton_transfer", "acid_base_proton_transfer", "C(C)(C)(C)(C)C")],  # 5-valent C
        ["CC=O"], ["CCO"],
    )
    v = validate_record(rec)
    assert not v.ok
    assert any("sanitize_fail" in r for r in v.reasons)


def test_noop_step_rejected():
    rec = _rec(
        [
            ("addition", "nucleophilic_addition", "CC([O-])CC=O"),
            ("addition", "nucleophilic_addition", "CC([O-])CC=O"),  # identical -> no-op
        ],
        ["CC=O", "CC=O"], ["CC(O)CC=O"],
    )
    v = validate_record(rec)
    assert not v.ok
    assert any("noop_step" in r for r in v.reasons)


def test_discontinuous_step_rejected():
    # Second intermediate is an unrelated molecule -> not reachable in one step.
    rec = _rec(
        [
            ("addition", "nucleophilic_addition", "CC(O)CC=O"),
            ("substitution", "nucleophilic_substitution", "c1ccc(Cl)nc1C(F)(F)F"),
        ],
        ["CC=O", "CC=O"], ["CC(O)CC=O"],
    )
    v = validate_record(rec)
    assert not v.ok
    assert any("discontinuous_step" in r for r in v.reasons)


def test_mcs_fraction_bounds():
    assert mcs_fraction("CCCCCCCCO", "c1ccncc1Cl") < 0.3
    assert mcs_fraction("CC(O)CC=O", "CC([O-])CC=O") >= 0.8
