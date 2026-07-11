"""Tests for SFT data building: augmentation (canonical targets) + decontam."""

from __future__ import annotations

import json

from omebench_eval.scoring import canonical_smiles
from rdkit import Chem
from training.build_sft_data import (
    augment_train_rows,
    decontaminate_rows,
    make_row,
    randomize_smiles,
    split_and_write,
    strip_meta,
)


def _same_mol(a: str, b: str) -> bool:
    return Chem.MolToSmiles(Chem.MolFromSmiles(a)) == Chem.MolToSmiles(Chem.MolFromSmiles(b))


def test_randomize_smiles_same_molecule_different_string():
    variants = randomize_smiles("CC(=O)Oc1ccccc1C(=O)O", 5)  # aspirin
    assert variants, "expected at least one randomized variant"
    canon = canonical_smiles("CC(=O)Oc1ccccc1C(=O)O")
    for v in variants:
        assert v != canon           # a different (non-canonical) string...
        assert _same_mol(v, canon)  # ...but the same molecule


def _row(reactants, products, level="easy", rid="R1"):
    return make_row(
        reaction_id=rid,
        level=level,
        user=f"reactants={reactants} products={products}",
        assistant_content="[ANSWER]\n[]\n[/ANSWER]",
        reference=[["nucleophilic_substitution", canonical_smiles(products[0]), 1.0]],
        reactants=reactants,
        products=products,
        conditions="none",
        style="cot",
    )


def test_augment_adds_variants_and_keeps_target_canonical():
    rows = [_row(["CC(=O)Oc1ccccc1C(=O)O"], ["CC(=O)Oc1ccccc1C(=O)O"])]
    out = augment_train_rows(rows, n=3)
    assert len(out) > len(rows)
    # Every row (original + augmented) keeps the identical canonical target.
    target = rows[0]["messages"][2]["content"]
    for r in out:
        assert r["messages"][2]["content"] == target
        # reference (scoring target) is untouched by augmentation.
        assert r["reference"] == rows[0]["reference"]


def test_strip_meta_removes_internal_keys():
    row = _row(["CCO"], ["CC=O"])
    cleaned = strip_meta(row)
    for k in ("_reactants", "_products", "_conditions", "_style"):
        assert k not in cleaned
    # Public schema survives.
    for k in ("reaction_id", "level", "messages", "prompt", "reference"):
        assert k in cleaned
    # Round-trips through JSON (no non-serializable objects).
    json.loads(json.dumps(cleaned))


def test_decontaminate_drops_gold_reactions():
    # A clearly-unrelated reaction should pass; a real gold reaction should be cut.
    from pathlib import Path

    gold = json.loads((Path(__file__).resolve().parents[1] / "data" / "oMe_Gold.json").read_text())
    g0 = gold[0]
    leaked = _row(g0["reactants_smiles"], g0["products_smiles"], rid="LEAK")
    clean = _row(["CCCCCCCCCCBr"], ["CCCCCCCCCCO"], rid="CLEAN")

    kept, removed = decontaminate_rows([leaked, clean])
    kept_ids = {r["reaction_id"] for r in kept}
    assert "LEAK" not in kept_ids, "a gold reaction leaked through decontamination!"
    assert "CLEAN" in kept_ids
    assert removed >= 1


def test_split_and_write_augments_then_decontaminates(tmp_path):
    # Spec-critical ordering: a leaked reaction AND all of its augmented variants
    # must be removed. Feed one gold reaction (augmentable) + one clean reaction.
    import json as _json
    from pathlib import Path

    gold = _json.loads((Path(__file__).resolve().parents[1] / "data" / "oMe_Gold.json").read_text())
    # Pick a gold reaction whose reactants have enough atoms to randomize.
    g = next(x for x in gold if any(len(s) > 6 for s in x["reactants_smiles"]))
    leaked = _row(g["reactants_smiles"], g["products_smiles"], rid="LEAK")
    # A real transformation unrelated to gold (the existing decontam negative
    # control) so its augmented variants survive.
    clean = _row(["CCCCCCCCCCBr"], ["CCCCCCCCCCO"], rid="CLEAN")

    train_path, val_path = split_and_write(
        [leaked, clean],
        out_dir=tmp_path,
        train_name="t.jsonl",
        val_name="v.jsonl",
        val_frac=0.0,
        seed=1,
        augment=3,
        decontaminate=True,
    )
    written = [_json.loads(line) for line in train_path.read_text().splitlines()]
    ids = {r["reaction_id"] for r in written}
    assert "LEAK" not in ids, "leaked reaction (or a variant) survived augment+decontam!"
    assert "CLEAN" in ids  # clean reaction and its variants survive
