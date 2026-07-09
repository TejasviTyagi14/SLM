"""Phase 2 normalization tests: canonicalization conventions + robustness."""

from __future__ import annotations

from rxndata.normalize import canonicalize, has_rgroup, strip_cxsmiles
from rxndata.schema import _as_smiles_list, make_record


def test_charges_and_radicals_preserved():
    # Mechanisms carry real charges/radicals; canonicalization must keep them.
    r = canonicalize("[CH2-]C=O")
    assert r.ok and r.had_charge
    assert "-" in r.canonical
    rad = canonicalize("[CH3]")  # methyl radical
    assert rad.ok and rad.had_radical


def test_atom_maps_preserved_when_requested():
    r = canonicalize("[Br:1][CH2:2][CH2:3][OH:4]", keep_maps=True)
    assert r.ok and r.had_map
    assert ":1" in r.canonical or ":4" in r.canonical
    r2 = canonicalize("[Br:1][CH2:2][CH2:3][OH:4]", keep_maps=False)
    assert r2.ok and ":" not in r2.canonical  # maps stripped


def test_cxsmiles_suffix_stripped():
    s = "[C:23]([OH:25])=[O:24] |f:3"
    assert strip_cxsmiles(s) == "[C:23]([OH:25])=[O:24]"
    r = canonicalize(s, from_source="USPTO-Lowe")
    assert r.ok


def test_rgroup_fragment_flagged_not_crashed():
    r = canonicalize("C([*:1])([*:2])C=O", from_source="oMe-Template")
    # This particular R-group SMILES DOES parse in RDKit; the point is no crash
    # and R-group detection works.
    assert has_rgroup("C([*:1])([*:2])C=O")
    assert r.ok or r.reason == "rgroup_template"


def test_bare_string_smiles_not_exploded():
    # Regression: some template records store SMILES as a bare string, not list.
    # list("CCO") would explode to ['C','C','O']; _as_smiles_list must not.
    assert _as_smiles_list("OC([*:1])([*:2])") == ["OC([*:1])([*:2])"]
    assert _as_smiles_list("A.B") == ["A", "B"]
    rec = make_record("X", "s", "MIT", "curated", "CCO", "CC=O")
    assert rec["reactants_smiles"] == ["CCO"]
    assert rec["products_smiles"] == ["CC=O"]


def test_inchikey_dedup_key_stable():
    a = canonicalize("OC(=O)C")   # acetic acid
    b = canonicalize("CC(O)=O")   # same molecule, different SMILES
    assert a.ok and b.ok
    assert a.inchikey == b.inchikey
