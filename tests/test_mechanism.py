"""Phase 3 tests: typing/remap, molecule-level R-group substitution, expansion."""

from __future__ import annotations

from rdkit import Chem

from rxndata.mechanism import (
    derive_level,
    expand_template,
    rgroup_labels,
    substitute_rgroups,
    type_and_remap,
)
from rxndata.ontology import load_ontology
from rxndata.schema import make_record, make_step


def _mol_eq(a: str, b: str) -> bool:
    return Chem.MolToSmiles(Chem.MolFromSmiles(a)) == Chem.MolToSmiles(Chem.MolFromSmiles(b))


def test_substitute_rgroups_no_ring_digit_collision():
    # Regression: string substitution of an aromatic fragment reuses ring-bond
    # digits and corrupts the molecule; molecule-level substitution must not.
    tmpl = "c1(C([*:2])=C([*:3]))c(N(=O)=O)cc([*:1])cc1"
    out = substitute_rgroups(tmpl, {1: "c1ccccc1", 2: "C", 3: "[H]"})
    assert out is not None
    m = Chem.MolFromSmiles(out)
    assert m is not None
    # biphenyl linkage present, molecule sane
    assert "-c" in out or "c1ccccc1" in out


def test_substitute_all_methyl_matches_hand_smiles():
    tmpl = "c1(C([*:2])=C([*:3]))c(N(=O)=O)cc([*:1])cc1"
    out = substitute_rgroups(tmpl, {1: "C", 2: "C", 3: "C"})
    assert _mol_eq(out, "CC=C(C)c1ccc(C)cc1[N+](=O)[O-]")


def test_rgroup_labels():
    assert rgroup_labels(["C([*:1])[*:2]", "N[*:1]"]) == [1, 2]


def test_type_and_remap_passthrough_and_reject():
    onto = load_ontology()
    good = make_record(
        "G", "s", "MIT", "curated", ["CC=O"], ["CCO"],
        mechanism=[make_step(1, "proton_transfer", "acid_base_proton_transfer", "CC=O")],
    )
    typed, reason = type_and_remap(good, onto)
    assert typed is not None and reason is None
    assert typed["level"] == "easy"

    bad = make_record(
        "B", "s", "MIT", "curated", ["CC=O"], ["CCO"],
        mechanism=[make_step(1, "rearrangement", "rearrangement", "CC=O")],  # explicit reject
    )
    typed2, reason2 = type_and_remap(bad, onto)
    assert typed2 is None
    assert "unmapped_step_type" in reason2


def test_type_and_remap_applies_remap_table():
    onto = load_ontology()
    # sigmatropic_rearrangement is filed under 'rearrangement' in dirty templates;
    # remap should move it to 'pericyclic'.
    rec = make_record(
        "R", "s", "MIT", "curated", ["C"], ["C"],
        mechanism=[make_step(1, "rearrangement", "sigmatropic_rearrangement", "C")],
    )
    typed, reason = type_and_remap(rec, onto)
    assert typed is not None
    assert typed["mechanism"][0]["type"] == "pericyclic"
    assert typed["mechanism"][0]["subtype"] == "sigmatropic_rearrangement"


def test_expand_template_produces_concrete_valid_instances():
    onto = load_ontology()
    tmpl = make_record(
        "T", "oMe-Template", "MIT", "curated",
        ["C([*:1])=O"], ["C([*:1])O"],
        mechanism=[make_step(1, "addition", "nucleophilic_addition", "C([*:1])[O-]")],
    )
    typed, _ = type_and_remap(tmpl, onto)
    insts = expand_template(typed, max_instances=4, ontology=onto)
    assert insts
    for inst in insts:
        assert inst["provenance"] == "template_expanded"
        for s in inst["reactants_smiles"] + inst["products_smiles"]:
            assert "[*" not in s  # fully concrete
            assert Chem.MolFromSmiles(s) is not None


def test_derive_level():
    assert derive_level(2) == "easy"
    assert derive_level(5) == "medium"
    assert derive_level(9) == "hard"
