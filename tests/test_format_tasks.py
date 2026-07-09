"""Phase 8 task-formatting tests + the oMeS format-check invariant."""

from __future__ import annotations

import json

from omebench_eval.scoring import oMeS

from rxndata.format_tasks import (
    SYSTEM_PROMPT,
    make_mechanism_full,
    make_next_step,
)
from rxndata.ontology import load_ontology
from rxndata.schema import make_record, make_step


def _aldol():
    return make_record(
        "SILVER-X", "oMe-Silver", "MIT", "template_expanded",
        ["CC=O", "CC=O"], ["CC(O)CC=O"],
        mechanism=[
            make_step(1, "proton_transfer", "acid_base_proton_transfer", "[CH2-]C=O"),
            make_step(2, "addition", "nucleophilic_addition", "CC([O-])CC=O"),
            make_step(3, "proton_transfer", "acid_base_proton_transfer", "CC(O)CC=O"),
        ],
        level="easy", name="Aldol", conditions="Base",
    )


def test_mechanism_full_mirrors_eval_and_is_valid_json():
    tmpl = "Reactants: { reactants_smiles } Products: { products_smiles } Cond: { conditions }"
    row = make_mechanism_full(_aldol(), tmpl, style="default")
    assert row["messages"][0]["content"] == SYSTEM_PROMPT
    # user prompt had tokens substituted
    assert "{ reactants_smiles }" not in row["messages"][1]["content"]
    steps = json.loads(row["messages"][2]["content"])
    assert [s["subtype"] for s in steps] == [
        "acid_base_proton_transfer", "nucleophilic_addition", "acid_base_proton_transfer"
    ]
    # reference present for GRPO
    assert len(row["reference"]) == 3 and row["reference"][0][2] > 0


def test_format_check_target_scores_perfect_on_omes():
    """The load-bearing acceptance invariant: our own target maxes out oMeS."""
    tmpl = "R { reactants_smiles } P { products_smiles } C { conditions }"
    row = make_mechanism_full(_aldol(), tmpl, style="cot")
    content = row["messages"][2]["content"].split("[ANSWER]")[1].split("[/ANSWER]")[0]
    steps = json.loads(content)
    gold = [(r[0], r[1], r[2]) for r in row["reference"]]
    pred = [(s["subtype"], s["intermediate_smiles"]) for s in steps]
    res = oMeS(gold, pred)
    assert res.S_partial >= 0.99 and res.S_total >= 0.99


def test_next_step_one_row_per_prefix():
    rows = make_next_step(_aldol())
    assert len(rows) == 2  # 3 steps -> predict step2 given [1], step3 given [1,2]
    for r in rows:
        target = json.loads(r["messages"][-1]["content"])
        assert set(target) >= {"step", "type", "subtype", "intermediate_smiles"}


def test_subtypes_are_ontology_valid():
    onto = load_ontology()
    allowed = set(onto["subtypes"])
    tmpl = "R { reactants_smiles } P { products_smiles } C { conditions }"
    row = make_mechanism_full(_aldol(), tmpl, style="default")
    steps = json.loads(row["messages"][2]["content"])
    assert all(s["subtype"] in allowed for s in steps)
