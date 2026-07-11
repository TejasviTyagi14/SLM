"""Validity + reward monitoring for RL/SFT (guards against validity collapse).

Evidence: SMILES validity can be >99% after SFT yet collapse during RL
exploration (PSV-PPO, arXiv:2505.00530). This module logs, per logging step,
the mean reward, mean oMeS S_partial, mean validity, and mean completion length
over a rolling window -- and warns (or fails loud) if validity drops below a
floor.

Two pieces, split so the aggregation logic is testable without transformers/TRL:

* ``MechMonitor``          -- a plain rolling-window aggregator. Pure Python.
* ``make_monitor_callback``-- a thin ``transformers.TrainerCallback`` adapter
                              (lazy-imported) that emits the snapshot into the
                              trainer logs each logging step and enforces the
                              validity floor.
* ``instrument_reward_funcs`` -- wraps the main reward func so every reward
                              computation also feeds the monitor (no double
                              scoring of oMeS beyond what the reward already did).
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional

from training.reward import _completion_to_text, score_components


class MechMonitor:
    """Rolling-window aggregator of reward/validity/S_partial/length.

    Frame-agnostic (no transformers dependency) so it unit-tests on CPU. Feed it
    with ``record(completions, reference, rewards)`` from a reward function; read
    aggregates with ``snapshot()``.
    """

    def __init__(self, window: int = 256, validity_floor: float = 0.8):
        self.window = window
        self.validity_floor = validity_floor
        self._reward: deque = deque(maxlen=window)
        self._spartial: deque = deque(maxlen=window)
        self._validity: deque = deque(maxlen=window)
        self._length: deque = deque(maxlen=window)
        self.n_seen = 0

    def record(self, completions, reference=None, rewards: Optional[List[float]] = None) -> None:
        if completions is None:
            return
        refs = reference if reference is not None else [None] * len(completions)
        for i, comp in enumerate(completions):
            text = _completion_to_text(comp)
            self._length.append(len(text))
            ref = refs[i] if i < len(refs) else None
            if ref is not None:
                comp_metrics = score_components(text, ref)
                self._spartial.append(comp_metrics["S_partial"])
                self._validity.append(comp_metrics["validity"])
            if rewards is not None and i < len(rewards):
                self._reward.append(float(rewards[i]))
            self.n_seen += 1

    @staticmethod
    def _mean(dq: deque) -> Optional[float]:
        return round(sum(dq) / len(dq), 4) if dq else None

    def snapshot(self) -> dict:
        return {
            "mech/mean_reward": self._mean(self._reward),
            "mech/mean_S_partial": self._mean(self._spartial),
            "mech/mean_validity": self._mean(self._validity),
            "mech/mean_completion_chars": self._mean(self._length),
            "mech/window_n": len(self._validity) or len(self._length),
        }

    def validity_below_floor(self) -> bool:
        """True once we have a full-ish window and its mean validity < floor."""
        if len(self._validity) < max(8, self.window // 4):
            return False  # not enough signal yet; don't cry wolf early
        mv = self._mean(self._validity)
        return mv is not None and mv < self.validity_floor


def instrument_reward_funcs(reward_funcs: List[Callable], monitor: MechMonitor) -> List[Callable]:
    """Wrap the FIRST reward func so it feeds ``monitor`` while it scores.

    The main reward already parses/scores each completion, so recording here adds
    only the rolling-window bookkeeping, not a second oMeS pass beyond validity.
    """
    if not reward_funcs:
        return reward_funcs
    main = reward_funcs[0]

    def monitored(prompts=None, completions=None, reference=None, **kwargs):
        rewards = main(prompts=prompts, completions=completions, reference=reference, **kwargs)
        try:
            monitor.record(completions, reference, rewards)
        except Exception:
            pass  # monitoring must never break training
        return rewards

    monitored.__name__ = getattr(main, "__name__", "reward")
    return [monitored] + list(reward_funcs[1:])


def make_monitor_callback(
    monitor: MechMonitor,
    raise_on_floor: bool = False,
):
    """Return a transformers.TrainerCallback that logs the monitor snapshot.

    Lazy-imports transformers so this file imports on a CPU-only base install.
    On each ``on_log`` it merges the snapshot into the trainer's ``logs`` dict and
    enforces the validity floor (warn by default; raise if ``raise_on_floor``).
    """
    from transformers import TrainerCallback

    class _MechMonitorCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            snap = monitor.snapshot()
            if logs is not None:
                for k, v in snap.items():
                    if v is not None:
                        logs[k] = v
            mv = snap.get("mech/mean_validity")
            if monitor.validity_below_floor():
                msg = (
                    f"[monitor] mean SMILES validity {mv} < floor "
                    f"{monitor.validity_floor} over last {snap.get('mech/window_n')} "
                    f"completions -- possible validity collapse."
                )
                if raise_on_floor:
                    raise RuntimeError(msg)
                print("WARNING: " + msg)
            return control

    return _MechMonitorCallback()
