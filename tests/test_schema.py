"""Schema + ontology structural tests (Phase 0/1)."""

from __future__ import annotations

from pathlib import Path

from rxndata.config import load_config
from rxndata.ontology import build_ontology, load_ontology
from rxndata.schema import (
    make_record,
    make_step,
    record_to_row,
    row_to_record,
    validate_record,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ontology_gold_is_clean():
    onto = build_ontology()
    # Every subtype used by the graded gold set must be in the allowed vocabulary.
    assert onto["crosscheck"]["gold_is_clean"] is True
    # No subtype maps to two different types (subtype is the oMeS alignment key).
    assert onto["crosscheck"]["subtype_type_conflicts"] == {}


def test_ontology_artifact_matches_source():
    """The committed configs/ontology.json equals a fresh parse (no drift)."""
    onto_file = load_ontology()
    onto_fresh = build_ontology()
    assert onto_file["subtypes"] == onto_fresh["subtypes"]
    assert onto_file["types"] == onto_fresh["types"]


def test_record_roundtrip_parquet_encoding():
    step = make_step(1, "proton_transfer", "acid_base_proton_transfer", "CC=O", "note")
    rec = make_record(
        reaction_id="X-1",
        source="test",
        license="MIT",
        provenance="curated",
        reactants_smiles=["CC=O"],
        products_smiles=["CCO"],
        mechanism=[step],
    )
    assert validate_record(rec) == []
    row = record_to_row(rec)
    # JSON columns must be strings in the flat row.
    assert isinstance(row["mechanism"], str)
    back = row_to_record(row)
    assert back["mechanism"] == rec["mechanism"]
    assert back["reactants_smiles"] == rec["reactants_smiles"]


def test_validate_record_catches_bad_provenance():
    rec = make_record("X-2", "s", "MIT", "curated", ["C"], ["C"])
    rec["provenance"] = "made_up"
    problems = validate_record(rec)
    assert any("provenance" in p for p in problems)


def test_config_source_verdicts_block_ncnd():
    cfg = load_config()
    # The NC-ND / no-AI sources must be disabled + quarantined (spec HARD constraint).
    for key in ("pmechdb", "rmechdb", "openstax"):
        s = cfg.source(key)
        assert s["enabled"] is False
        assert s["verdict"] == "quarantine"
