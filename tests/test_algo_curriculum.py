"""Tests for DAPO algo config building and difficulty curriculum ordering.

Both are framework-agnostic (no TRL import), so they run on CPU. The
filter_supported_kwargs test uses a tiny stand-in dataclass to prove unsupported
knobs are dropped rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
from training.algo import (
    DEFAULT_EPSILON_HIGH,
    build_algo_config_kwargs,
    filter_supported_kwargs,
)
from training.curriculum import level_histogram, level_rank, order_by_difficulty


def test_grpo_leaves_defaults_untouched():
    assert build_algo_config_kwargs("grpo") == {}


def test_dapo_sets_token_loss_and_clip_higher():
    kw = build_algo_config_kwargs("dapo", epsilon_high=0.28)
    assert kw["loss_type"] == "dapo"
    assert kw["epsilon_high"] == 0.28
    assert kw["scale_rewards"] == "none"
    assert kw["mask_truncated_completions"] is True


def test_dapo_default_epsilon_high():
    kw = build_algo_config_kwargs("dapo")
    assert kw["epsilon_high"] == DEFAULT_EPSILON_HIGH


def test_dapo_dynamic_sampling_toggle():
    kw = build_algo_config_kwargs("dapo", dynamic_sampling=False, mask_truncated=False)
    assert "mask_truncated_completions" not in kw
    # Core DAPO loss knobs still present.
    assert kw["loss_type"] == "dapo"


def test_unknown_algo_raises():
    with pytest.raises(ValueError):
        build_algo_config_kwargs("ppo")


@dataclass
class _MiniConfig:
    loss_type: str = "grpo"
    epsilon_high: Optional[float] = None
    # deliberately no scale_rewards / mask_truncated_completions


def test_filter_drops_unsupported_kwargs():
    kw = build_algo_config_kwargs("dapo")
    supported = filter_supported_kwargs(kw, _MiniConfig)
    assert set(supported) == {"loss_type", "epsilon_high"}
    # And the survivors actually construct the config.
    cfg = _MiniConfig(**supported)
    assert cfg.loss_type == "dapo"


# --- Curriculum -------------------------------------------------------------

def test_level_rank_order():
    assert level_rank("easy") < level_rank("medium") < level_rank("hard")
    assert level_rank(None) > level_rank("hard")       # unknown sorts last
    assert level_rank("bogus") > level_rank("hard")


def test_order_by_difficulty_is_easy_to_hard_stable():
    rows = [
        {"id": "h1", "level": "hard"},
        {"id": "e1", "level": "easy"},
        {"id": "m1", "level": "medium"},
        {"id": "e2", "level": "easy"},
        {"id": "u1", "level": None},
    ]
    out = order_by_difficulty(rows)
    assert [r["id"] for r in out] == ["e1", "e2", "m1", "h1", "u1"]


def test_level_histogram():
    rows = [{"level": "easy"}, {"level": "easy"}, {"level": "hard"}, {"level": None}]
    hist = level_histogram(rows)
    assert hist == {"easy": 2, "hard": 1, "unknown": 1}
