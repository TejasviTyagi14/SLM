"""CoT rejection-sampling tests: only verifier-passing traces survive.

Uses a FAKE provider (no network) that returns a known-good trace for one
reaction and a known-bad trace for another; asserts only the good one is kept,
and that the kept row is in the SFT chat schema the trainers consume.
"""

from __future__ import annotations

import json

from omebench_eval.scoring import canonical_smiles
from training.distill_cot import distill_rows, score_trace

# Two silver-like reactions with gold mechanisms.
GOOD_RXN = {
    "reaction_id": "NR-good",
    "level": "easy",
    "reactants_smiles": ["CC=O", "CC=O"],
    "products_smiles": ["CC(O)CC=O"],
    "conditions": "Base",
    "mechanism": [
        {"step": 1, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
         "intermediate_smiles": "[CH2-]C=O"},
        {"step": 2, "type": "addition", "subtype": "nucleophilic_addition",
         "intermediate_smiles": "CC([O-])CC=O"},
        {"step": 3, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
         "intermediate_smiles": "CC(O)CC=O"},
    ],
}

BAD_RXN = {
    "reaction_id": "NR-bad",
    "level": "hard",
    "reactants_smiles": ["CCBr"],
    "products_smiles": ["CCO"],
    "conditions": "Aqueous",
    "mechanism": [
        {"step": 1, "type": "substitution", "subtype": "nucleophilic_substitution",
         "intermediate_smiles": "CCO"},
    ],
}


def _answer_block(steps):
    return "Here is my reasoning.\n[ANSWER]\n" + json.dumps(steps) + "\n[/ANSWER]"


# Good trace == the gold mechanism reproduced exactly (S_partial == 1.0).
GOOD_TRACE = _answer_block([
    {"step": 1, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
     "intermediate_smiles": "[CH2-]C=O"},
    {"step": 2, "type": "addition", "subtype": "nucleophilic_addition",
     "intermediate_smiles": "CC([O-])CC=O"},
    {"step": 3, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
     "intermediate_smiles": "CC(O)CC=O"},
])

# Bad trace: wrong subtype AND wrong structure -> S_partial 0, must be rejected.
BAD_TRACE = _answer_block([
    {"step": 1, "type": "radical", "subtype": "radical_initiation",
     "intermediate_smiles": "C1CCCCC1"},
])


def _fake_generate(system, user):
    # Route on which reaction's SMILES is in the prompt.
    if "CCBr" in user:
        return BAD_TRACE
    return GOOD_TRACE


def test_only_verified_trace_survives():
    kept, stats = distill_rows(
        [GOOD_RXN, BAD_RXN],
        generate=_fake_generate,
        keep_threshold=0.9,
        require_validity=1.0,
    )
    assert stats["reactions_in"] == 2
    assert stats["kept"] == 1
    assert stats["dropped"] == 1
    assert len(kept) == 1

    row = kept[0]
    assert row["reaction_id"] == "DISTILL-NR-good"
    # SFT chat schema: system/user/assistant + prompt + reference.
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert row["messages"][2]["content"].strip() == GOOD_TRACE.strip()
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    # reference is [subtype, canonical_smiles, weight] rows aligned to gold.
    assert len(row["reference"]) == 3
    assert row["reference"][0][0] == "acid_base_proton_transfer"
    assert row["reference"][0][1] == canonical_smiles("[CH2-]C=O")


def test_threshold_gates_partial_traces():
    # A trace that matches only 1 of 3 gold steps: S_partial ~0.33, below 0.9.
    partial = _answer_block([
        {"step": 1, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
         "intermediate_smiles": "[CH2-]C=O"},
    ])
    kept, stats = distill_rows(
        [GOOD_RXN],
        generate=lambda s, u: partial,
        keep_threshold=0.9,
        require_validity=1.0,
    )
    assert stats["kept"] == 0

    # Same trace passes if we lower the threshold.
    ref = [[s["subtype"], canonical_smiles(s["intermediate_smiles"]), 1 / 3]
           for s in GOOD_RXN["mechanism"]]
    sc = score_trace(partial, ref)
    kept2, stats2 = distill_rows(
        [GOOD_RXN], generate=lambda s, u: partial,
        keep_threshold=sc["S_partial"], require_validity=1.0,
    )
    assert stats2["kept"] == 1


def test_invalid_smiles_trace_rejected_even_if_types_match():
    invalid = _answer_block([
        {"step": 1, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
         "intermediate_smiles": "not-a-smiles"},
        {"step": 2, "type": "addition", "subtype": "nucleophilic_addition",
         "intermediate_smiles": "also-bad"},
        {"step": 3, "type": "proton_transfer", "subtype": "acid_base_proton_transfer",
         "intermediate_smiles": "still-bad"},
    ])
    kept, stats = distill_rows(
        [GOOD_RXN], generate=lambda s, u: invalid,
        keep_threshold=0.0,  # even with zero score threshold, validity gate rejects
        require_validity=1.0,
    )
    assert stats["kept"] == 0
