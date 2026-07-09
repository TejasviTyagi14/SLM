"""Golden mini-run smoke tests for the local (no-network) ingesters."""

from __future__ import annotations

from rxndata.config import load_config
from rxndata.ingest import get_module
from rxndata.schema import validate_record


def test_ome_silver_ingest_small():
    cfg = load_config()
    recs = get_module("ome_silver").ingest(cfg, limit=10)
    assert len(recs) == 10
    for r in recs:
        assert validate_record(r) == []
        assert r["source"] == "oMe-Silver"
        assert r["license"] == "MIT"
        assert r["provenance"] == "template_expanded"  # honest provenance
        assert r["mechanism"], "silver records must carry a mechanism"
        assert r["raw_ref"]["template_base"].startswith("NR") or r["raw_ref"]["template_base"].startswith("NG")


def test_ome_template_ingest_flags_bad_ontology():
    cfg = load_config()
    recs = get_module("ome_template").ingest(cfg)
    assert recs
    # At least one template step is flagged as outside the allowed ontology
    # (the known dirty template labels); ingest must FLAG, not drop.
    flagged = [r for r in recs if r["raw_ref"].get("needs_remap")]
    assert flagged, "expected some template records flagged needs_remap"
    for r in recs:
        assert r["provenance"] == "curated"
