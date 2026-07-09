"""Phase 6 decontamination tests: positive + negative controls are load-bearing."""

from __future__ import annotations

import json
from pathlib import Path

from rxndata.decontaminate import (
    build_blacklist,
    contamination_reason,
    inchikey_signature,
    mechanism_hash,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    with (ROOT / "data" / name).open() as f:
        return json.load(f)


def test_positive_control_all_gold_caught():
    """Every oMe-Gold reaction, re-fed as a candidate, MUST be flagged.

    This is the load-bearing decontamination guarantee: no held-out test record
    can survive into training.
    """
    gold = _load("oMe_Gold.json")
    template = _load("oMe_Template.json")
    bl = build_blacklist(gold, template)
    caught = sum(
        1 for g in gold
        if contamination_reason(
            {"reactants_smiles": g["reactants_smiles"],
             "products_smiles": g["products_smiles"],
             "mechanism": g["mechanism"]},
            bl, 0.95,
        )
    )
    assert caught == len(gold), f"{len(gold) - caught} gold reactions would leak!"


def test_negative_control_unrelated_not_flagged():
    gold = _load("oMe_Gold.json")
    template = _load("oMe_Template.json")
    bl = build_blacklist(gold, template)
    unrelated = [
        {"reactants_smiles": ["CCCCCCCCCCBr"], "products_smiles": ["CCCCCCCCCCO"], "mechanism": []},
        {"reactants_smiles": ["FC(F)(F)c1ccccc1I"], "products_smiles": ["FC(F)(F)c1ccccc1O"], "mechanism": []},
    ]
    flagged = sum(1 for u in unrelated if contamination_reason(u, bl, 0.95))
    assert flagged == 0


def test_inchikey_signature_order_independent():
    a = inchikey_signature(["CC=O", "CCO"], ["CC(O)CC=O"])
    b = inchikey_signature(["CCO", "CC=O"], ["CC(O)CC=O"])
    assert a == b and a is not None


def test_mechanism_hash_canonical():
    m1 = [{"subtype": "acid_base_proton_transfer", "intermediate_smiles": "CC(O)=O"}]
    m2 = [{"subtype": "acid_base_proton_transfer", "intermediate_smiles": "OC(C)=O"}]
    assert mechanism_hash(m1) == mechanism_hash(m2)  # canonicalized SMILES
