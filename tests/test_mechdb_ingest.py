"""Smoke tests for the PMechDB / RMechDB elementary-step ingesters.

CRITICAL: the real PMechDB/RMechDB data is CC-BY-NC-ND-4.0 and is NOT bundled in
this repo. These tests use a small SYNTHETIC CSV fixture (hand-written, not from
the licensed datasets) to exercise the schema mapping, and assert the Phase 1
gate keeps the sources inert by default. They also confirm a mapped record flows
through Phase 5 validation + Phase 6 decontamination unchanged in shape.
"""

from __future__ import annotations

from rxndata.config import load_config
from rxndata.decontaminate import build_blacklist, contamination_reason
from rxndata.ingest import get_module
from rxndata.ingest.mechdb_common import ingest_csv, split_smirks
from rxndata.schema import validate_record

# Synthetic elementary steps (NOT from the licensed datasets). Simple, valid
# atom-mapped-ish SMIRKS in reactants>>products form.
_SYNTHETIC_CSV = """Reaction SMIRKS,Arrow codes,Original source,Auxiliary information
[CH3:1][Br:2].[OH-:3]>>[CH3:1][OH:3].[Br-:2],2-3=1-2,synthetic-1,none
[CH2:1]=[CH2:2].[H+:3]>>[CH3:1][CH2+:2],1-2=2-3,synthetic-2,none
not-a-reaction,,,skip-me
"""


def _write_fixture(tmp_path):
    csv_path = tmp_path / "steps.csv"
    csv_path.write_text(_SYNTHETIC_CSV)
    return csv_path


def test_split_smirks_forms():
    assert split_smirks("CCBr.[OH-]>>CCO.[Br-]") == (["CCBr", "[OH-]"], [], ["CCO", "[Br-]"])
    assert split_smirks("CCBr>reagent>CCO") == (["CCBr"], ["reagent"], ["CCO"])
    assert split_smirks("garbage") is None
    assert split_smirks("") is None


def test_ingest_csv_maps_elementary_steps(tmp_path):
    csv_path = _write_fixture(tmp_path)
    recs = ingest_csv(
        csv_path,
        source="PMechDB",
        license="CC-BY-NC-ND-4.0",
        default_type="cleavage",
        default_subtype="heterolytic_cleavage",
        id_prefix="PMECHDB",
        root=tmp_path,
        limit=None,
    )
    # 2 well-formed rows; the "not-a-reaction" row is skipped.
    assert len(recs) == 2
    for r in recs:
        assert validate_record(r) == []               # structurally valid record
        assert r["source"] == "PMechDB"
        assert r["license"] == "CC-BY-NC-ND-4.0"
        assert r["provenance"] == "curated"
        assert len(r["mechanism"]) == 1               # single elementary step
        assert r["mechanism"][0]["intermediate_smiles"]
        assert r["raw_ref"]["needs_remap"] is True     # ontology honesty flag
        assert r["reaction_smiles_mapped"]             # SMIRKS preserved


def test_missing_data_raises_informative_error(tmp_path):
    missing = tmp_path / "nope.csv"
    try:
        ingest_csv(
            missing, source="RMechDB", license="CC-BY-NC-ND-4.0",
            default_type="radical", default_subtype="radical_propagation",
            id_prefix="RMECHDB", root=tmp_path,
        )
    except FileNotFoundError as e:
        assert "not bundled" in str(e).lower() or "download" in str(e).lower()
        return
    raise AssertionError("expected FileNotFoundError for absent CC-BY-NC-ND data")


def test_gate_blocks_quarantined_sources_by_default():
    cfg = load_config()
    for key in ("pmechdb", "rmechdb"):
        mod = get_module(key)
        assert mod.INFO.verdict == "quarantine"
        # Config ships disabled -> gate must not permit ingestion.
        assert mod.INFO.gate_ok(cfg) is False
        src = cfg.raw["sources"][key]
        assert src["enabled"] is False


def test_mapped_record_passes_decontamination_flow(tmp_path):
    # A synthetic step unrelated to gold must survive decontamination (proves the
    # mapped record is shaped correctly for Phase 6, not that it leaks).
    csv_path = _write_fixture(tmp_path)
    recs = ingest_csv(
        csv_path, source="PMechDB", license="CC-BY-NC-ND-4.0",
        default_type="cleavage", default_subtype="heterolytic_cleavage",
        id_prefix="PMECHDB", root=tmp_path,
    )
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    gold = json.loads((root / "data" / "oMe_Gold.json").read_text())
    template = json.loads((root / "data" / "oMe_Template.json").read_text())
    bl = build_blacklist(gold, template)
    # Contamination check runs without error and (for these synthetic steps)
    # returns None => not contaminated.
    for r in recs:
        assert contamination_reason(r, bl, 0.95) is None
