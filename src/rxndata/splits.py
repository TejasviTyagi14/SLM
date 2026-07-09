"""Phase 9: stratified train/val split.

Test = the official oMe-Gold (held out; NEVER trained on). We split the clean,
deduped, decontaminated mechanism records into train/val, stratified by
(dominant-subtype, level) so rare mechanism types and hard reactions appear in
both splits. Deterministic given the global seed.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def _dominant_subtype(rec: dict) -> str:
    subs = [s.get("subtype") for s in rec.get("mechanism", []) if s.get("subtype")]
    if not subs:
        return "none"
    return Counter(subs).most_common(1)[0][0]


def _stratum(rec: dict) -> str:
    return f"{_dominant_subtype(rec)}|{rec.get('level') or 'na'}"


def stratified_split(
    records: List[dict], val_fraction: float, seed: int
) -> Tuple[List[dict], List[dict], Dict]:
    """Return (train, val, stats). Stratified by (dominant subtype, level)."""
    rng = random.Random(seed)
    by_stratum: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        by_stratum[_stratum(rec)].append(rec)

    train: List[dict] = []
    val: List[dict] = []
    for stratum, recs in sorted(by_stratum.items()):
        recs = sorted(recs, key=lambda r: r["reaction_id"])
        rng.shuffle(recs)
        n_val = max(1, round(len(recs) * val_fraction)) if len(recs) > 1 else 0
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    stats = {
        "n_train": len(train),
        "n_val": len(val),
        "n_strata": len(by_stratum),
        "val_fraction_actual": round(len(val) / max(1, len(records)), 4),
    }
    return train, val, stats
