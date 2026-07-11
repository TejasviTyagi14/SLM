"""Reward-function tests: gating, validity, format, round-trip, weight wiring.

These pin the reward behavior on hand-built (prediction, reference) pairs with
known oMeS scores, so a regression in gating or weighting fails loudly. No GPU /
TRL needed -- rewards are pure functions of text + reference.
"""

from __future__ import annotations

import json
from typing import List, Optional

from omebench_eval.scoring import canonical_smiles
from training.forward_model import ForwardModel, NoOpForwardModel, step_feasible
from training.reward import (
    FORMAT_FAIL,
    W_FORMAT,
    W_VALIDITY,
    build_reward_funcs,
    format_reward,
    make_omes_reward,
    make_roundtrip_reward,
    score_components,
    validity_reward,
)

# A 3-step aldol reference (subtype, canonical_smiles, weight).
REF = [
    ["acid_base_proton_transfer", canonical_smiles("[CH2-]C=O"), 0.333333],
    ["nucleophilic_addition", canonical_smiles("CC([O-])CC=O"), 0.333333],
    ["acid_base_proton_transfer", canonical_smiles("CC(O)CC=O"), 0.333334],
]


def _mech_text(steps: List[dict]) -> str:
    return "[ANSWER]\n" + json.dumps(steps) + "\n[/ANSWER]"


GOOD = _mech_text([
    {"step": 1, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
     "intermediate_smiles": "[CH2-]C=O"},
    {"step": 2, "type": "addition", "subtype": "nucleophilic_addition",
     "intermediate_smiles": "CC([O-])CC=O"},
    {"step": 3, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
     "intermediate_smiles": "CC(O)CC=O"},
])

# Right (valid) structures but wrong subtypes -> S_partial must be 0.
WRONG_TYPE = _mech_text([
    {"step": 1, "type": "radical", "subtype": "radical_initiation",
     "intermediate_smiles": "[CH2-]C=O"},
    {"step": 2, "type": "radical", "subtype": "radical_coupling",
     "intermediate_smiles": "CC([O-])CC=O"},
    {"step": 3, "type": "radical", "subtype": "radical_termination",
     "intermediate_smiles": "CC(O)CC=O"},
])

# Parseable JSON but the single SMILES is invalid -> gate must trip.
INVALID = _mech_text([
    {"step": 1, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
     "intermediate_smiles": "not-a-smiles"},
])

EMPTY = "I cannot determine the mechanism."


def test_score_components_perfect():
    comp = score_components(GOOD, REF)
    assert comp["parseable"] is True
    assert comp["n_steps"] == 3
    assert comp["validity"] == 1.0
    assert comp["S_partial"] == 1.0
    assert comp["S_total"] == 1.0
    assert comp["L"] == 1.0


def test_omes_reward_perfect_includes_bonuses():
    r = make_omes_reward()
    (val,) = r(completions=[GOOD], reference=[REF])
    # 1.0*S_partial + W_VALIDITY*1.0 + W_FORMAT
    assert abs(val - (1.0 + W_VALIDITY + W_FORMAT)) < 1e-9


def test_wrong_subtype_scores_only_validity_and_format():
    r = make_omes_reward()
    (val,) = r(completions=[WRONG_TYPE], reference=[REF])
    comp = score_components(WRONG_TYPE, REF)
    assert comp["S_partial"] == 0.0
    assert comp["validity"] == 1.0
    assert abs(val - (W_VALIDITY + W_FORMAT)) < 1e-9


def test_multiplicative_gate_on_invalid_and_empty():
    r = make_omes_reward()
    assert r(completions=[INVALID], reference=[REF]) == [FORMAT_FAIL]
    assert r(completions=[EMPTY], reference=[REF]) == [FORMAT_FAIL]
    # Gate must NOT hand out validity/format credit for degenerate output.
    assert score_components(INVALID, REF)["S_partial"] == 0.0


def test_validity_and_format_standalone():
    assert validity_reward(completions=[GOOD, INVALID]) == [1.0, 0.0]
    assert format_reward(completions=[GOOD, EMPTY]) == [1.0, 0.0]


def test_missing_reference_raises():
    r = make_omes_reward()
    try:
        r(completions=[GOOD], reference=None)
    except ValueError:
        return
    raise AssertionError("expected ValueError when reference column is missing")


def test_build_reward_funcs_shapes():
    assert [f.__name__ for f in build_reward_funcs("omes")] == [
        "omes_reward", "validity_reward", "format_reward"
    ]
    assert [f.__name__ for f in build_reward_funcs("omes+roundtrip")] == [
        "omes_reward", "validity_reward", "format_reward", "roundtrip_reward"
    ]


def test_roundtrip_noop_is_neutral():
    # No-op forward model abstains -> standalone round-trip reward contributes 0,
    # and the gated reward equals the plain oMeS reward (no penalty for enabling).
    rt = make_roundtrip_reward(NoOpForwardModel())
    assert rt(completions=[GOOD]) == [0.0]
    gated = make_omes_reward(w_roundtrip=0.10, forward_model=NoOpForwardModel())
    plain = make_omes_reward()
    assert gated(completions=[GOOD], reference=[REF]) == plain(completions=[GOOD], reference=[REF])


class _FakeForward(ForwardModel):
    """Forward model that declares every transition feasible (echoes target)."""

    def predict(self, reactant_smiles: str) -> Optional[List[str]]:
        # Claim the whole reference intermediate set is reachable.
        return [row[1] for row in REF]


def test_roundtrip_reward_with_real_model_rewards_feasible_steps():
    rt = make_roundtrip_reward(_FakeForward())
    (val,) = rt(completions=[GOOD])
    # Two consecutive transitions, both feasible under the fake model.
    assert val == 1.0
    # And it lifts the gated reward above the no-roundtrip baseline.
    gated = make_omes_reward(w_roundtrip=0.10, forward_model=_FakeForward())
    plain = make_omes_reward()
    assert gated(completions=[GOOD], reference=[REF])[0] > plain(
        completions=[GOOD], reference=[REF]
    )[0]


def test_step_feasible_abstains_on_noop_and_bad_smiles():
    m = NoOpForwardModel()
    assert step_feasible(m, "CCO", "CC=O") is None       # no-op abstains
    assert step_feasible(_FakeForward(), "CCO", "@@bad@@") is None  # unparseable target
