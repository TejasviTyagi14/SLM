"""Phase 7 dedup tests."""

from __future__ import annotations

from rxndata.dedup import deduplicate
from rxndata.schema import make_record, make_step


def _rec(rid, prov, smi_prod="CC(O)CC=O"):
    return make_record(
        rid, "s", "MIT", prov, ["CC=O", "CC=O"], [smi_prod],
        mechanism=[make_step(1, "addition", "nucleophilic_addition", "CC([O-])CC=O")],
    )


def test_dedup_keeps_highest_provenance():
    recs = [
        _rec("A-inferred", "inferred"),
        _rec("B-curated", "curated"),
        _rec("C-expanded", "template_expanded"),
    ]
    kept, stats = deduplicate(recs)
    assert stats["n_kept"] == 1
    assert kept[0]["provenance"] == "curated"  # highest rank wins


def test_dedup_distinct_reactions_survive():
    recs = [
        _rec("A", "curated", "CC(O)CC=O"),
        make_record("B", "s", "MIT", "curated", ["CCBr"], ["CCO"],
                    mechanism=[make_step(1, "substitution", "nucleophilic_substitution", "CCO")]),
    ]
    kept, stats = deduplicate(recs)
    assert stats["n_kept"] == 2
