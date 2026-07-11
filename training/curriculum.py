"""Difficulty curriculum ordering (easy -> medium -> hard).

Evidence: hierarchical difficulty curricula help reaction models (RxnNano
arXiv:2603.02215; ether0's advantage-based curriculum; "Rethinking
Retrosynthesis" JCIM 2025). oMe rows carry a ``level`` in {easy, medium, hard};
this module simply orders the dataset by that label so training sees easy
reactions first and hard ones last.

Kept deliberately simple and framework-agnostic (operates on a list of row
dicts) so it unit-tests on CPU and both trainers can reuse it. When curriculum
is OFF, callers shuffle as before; when ON, callers must also disable shuffling
so the ordering is preserved.
"""

from __future__ import annotations

from typing import Dict, List, Optional

LEVEL_ORDER: Dict[Optional[str], int] = {"easy": 0, "medium": 1, "hard": 2}
# Rows with no/unknown level sort AFTER hard (train on the clean, labeled
# easy->hard progression first, then the unlabeled remainder).
_UNKNOWN_RANK = 99


def level_rank(level: Optional[str]) -> int:
    return LEVEL_ORDER.get(level, _UNKNOWN_RANK)


def order_by_difficulty(rows: List[dict], level_key: str = "level") -> List[dict]:
    """Stable-sort rows easy -> medium -> hard -> unknown.

    Stable so within a level the caller's prior (shuffled) order is preserved.
    """
    return sorted(rows, key=lambda r: level_rank(r.get(level_key)))


def level_histogram(rows: List[dict], level_key: str = "level") -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for r in rows:
        lvl = r.get(level_key) or "unknown"
        hist[lvl] = hist.get(lvl, 0) + 1
    return hist
