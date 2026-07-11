"""Monitoring callback tests (CPU-only; no transformers/TRL required)."""

from __future__ import annotations

import json

from omebench_eval.scoring import canonical_smiles
from training.callbacks import MechMonitor, instrument_reward_funcs
from training.reward import make_omes_reward

REF = [
    ["acid_base_proton_transfer", canonical_smiles("[CH2-]C=O"), 0.333333],
    ["nucleophilic_addition", canonical_smiles("CC([O-])CC=O"), 0.333333],
    ["acid_base_proton_transfer", canonical_smiles("CC(O)CC=O"), 0.333334],
]


def _mech_text(steps):
    return "[ANSWER]\n" + json.dumps(steps) + "\n[/ANSWER]"


GOOD = _mech_text([
    {"step": 1, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
     "intermediate_smiles": "[CH2-]C=O"},
    {"step": 2, "type": "addition", "subtype": "nucleophilic_addition",
     "intermediate_smiles": "CC([O-])CC=O"},
    {"step": 3, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
     "intermediate_smiles": "CC(O)CC=O"},
])
INVALID = _mech_text([
    {"step": 1, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
     "intermediate_smiles": "not-a-smiles"},
])


def test_monitor_tracks_means():
    mon = MechMonitor(window=64, validity_floor=0.8)
    mon.record([GOOD, GOOD], [REF, REF], [1.15, 1.15])
    snap = mon.snapshot()
    assert snap["mech/mean_S_partial"] == 1.0
    assert snap["mech/mean_validity"] == 1.0
    assert snap["mech/mean_reward"] == 1.15
    assert snap["mech/mean_completion_chars"] > 0
    assert snap["mech/window_n"] == 2


def test_validity_floor_trips_on_collapse():
    mon = MechMonitor(window=32, validity_floor=0.8)
    # Feed enough invalid completions to exceed the min-window guard.
    mon.record([INVALID] * 16, [REF] * 16, [-0.5] * 16)
    assert mon.validity_below_floor() is True
    snap = mon.snapshot()
    assert snap["mech/mean_validity"] == 0.0


def test_validity_floor_quiet_before_enough_signal():
    mon = MechMonitor(window=256, validity_floor=0.8)
    mon.record([INVALID, INVALID], [REF, REF], [-0.5, -0.5])
    # Only 2 samples; window//4 = 64 not reached -> must not warn yet.
    assert mon.validity_below_floor() is False


def test_instrument_reward_funcs_feeds_monitor_without_changing_rewards():
    mon = MechMonitor(window=64)
    base = make_omes_reward()
    funcs = instrument_reward_funcs([base], mon)
    assert len(funcs) == 1

    plain = base(completions=[GOOD, INVALID], reference=[REF, REF])
    wrapped = funcs[0](completions=[GOOD, INVALID], reference=[REF, REF])
    assert plain == wrapped  # reward values unchanged
    # ...but the monitor now has 2 records.
    assert mon.n_seen == 2
    snap = mon.snapshot()
    assert snap["mech/mean_validity"] == 0.5  # one valid, one invalid
