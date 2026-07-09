"""Phase 9 split tests: determinism + stratification + no test leakage."""

from __future__ import annotations

from rxndata.schema import make_record, make_step
from rxndata.splits import stratified_split


def _recs(n):
    out = []
    subtypes = ["acid_base_proton_transfer", "nucleophilic_addition", "1,2-shift"]
    levels = ["easy", "medium", "hard"]
    for i in range(n):
        st = subtypes[i % len(subtypes)]
        out.append(make_record(
            f"R-{i}", "s", "MIT", "template_expanded", ["CC=O"], ["CCO"],
            mechanism=[make_step(1, "addition", st, "CCO")],
            level=levels[i % len(levels)],
        ))
    return out


def test_split_is_deterministic():
    recs = _recs(120)
    t1, v1, _ = stratified_split(recs, 0.1, seed=42)
    t2, v2, _ = stratified_split(recs, 0.1, seed=42)
    assert [r["reaction_id"] for r in t1] == [r["reaction_id"] for r in t2]
    assert [r["reaction_id"] for r in v1] == [r["reaction_id"] for r in v2]


def test_split_no_overlap_covers_all():
    recs = _recs(120)
    train, val, stats = stratified_split(recs, 0.1, seed=1)
    ids_t = {r["reaction_id"] for r in train}
    ids_v = {r["reaction_id"] for r in val}
    assert ids_t.isdisjoint(ids_v)
    assert len(ids_t | ids_v) == 120
    assert stats["n_val"] > 0


def test_val_contains_multiple_strata():
    recs = _recs(120)
    _, val, _ = stratified_split(recs, 0.2, seed=7)
    strata = {(s.get("subtype"), r.get("level"))
              for r in val for s in r["mechanism"]}
    assert len(strata) >= 3  # rare types represented in val
