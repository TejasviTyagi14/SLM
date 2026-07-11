"""RL algorithm configuration: GRPO (default) vs DAPO.

Evidence: RetroDFM-R (arXiv:2507.17448) chose DAPO over GRPO because GRPO's
sample-level (sequence-mean) loss biases toward SHORT completions -- bad for long
mechanism CoT. DAPO (arXiv:2503.14476) fixes this with:

  * token-level loss aggregation (long, correct traces aren't down-weighted),
  * asymmetric "clip-higher" (epsilon_high > epsilon) to preserve exploration,
  * dynamic sampling (drop prompt groups whose rollouts all get identical reward,
    i.e. zero advantage, so every optimizer step sees a useful gradient),
  * no std-normalization of advantages (avoids a difficulty-dependent bias).

These map onto TRL ``GRPOConfig`` knobs (TRL exposing ``loss_type`` /
``epsilon_high`` / ``mask_truncated_completions`` / ``scale_rewards`` -- present
in recent TRL, roughly >= 0.21 for ``loss_type`` and the DAPO loss). We build the
knob dict here as plain data so it is unit-testable WITHOUT importing TRL, then
filter it against the installed ``GRPOConfig`` fields at wiring time so an older
TRL simply ignores knobs it doesn't have (with a warning).
"""

from __future__ import annotations

from typing import Any, Dict, List

ALGOS = ("grpo", "dapo")

# DAPO recommended clip-higher epsilon (arXiv:2503.14476).
DEFAULT_EPSILON_HIGH = 0.28


def build_algo_config_kwargs(
    algo: str = "grpo",
    epsilon_high: float = DEFAULT_EPSILON_HIGH,
    dynamic_sampling: bool = True,
    mask_truncated: bool = True,
) -> Dict[str, Any]:
    """Return the GRPOConfig kwargs that switch the loss between GRPO and DAPO.

    grpo -> {} (leave TRL defaults untouched: sequence-level loss, symmetric clip).
    dapo -> token-level loss + clip-higher + no-std advantages + (optionally)
            drop degenerate/truncated completions.
    """
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo '{algo}'. Choose from {ALGOS}.")
    if algo == "grpo":
        return {}

    kwargs: Dict[str, Any] = {
        # Token-level aggregation is the core DAPO change vs GRPO.
        "loss_type": "dapo",
        # Asymmetric clipping: raise only the upper bound.
        "epsilon_high": float(epsilon_high),
        # DAPO drops the per-group std normalization of advantages.
        "scale_rewards": "none",
    }
    if dynamic_sampling or mask_truncated:
        # Closest stock-TRL lever to DAPO "dynamic sampling": exclude truncated /
        # degenerate completions from the loss so zero-signal groups don't dilute
        # the gradient. (Groups whose rollouts all score identically already yield
        # zero advantage under GRPO/DAPO normalization and contribute nothing.)
        kwargs["mask_truncated_completions"] = True
    return kwargs


def filter_supported_kwargs(kwargs: Dict[str, Any], config_cls) -> Dict[str, Any]:
    """Keep only kwargs that ``config_cls`` (a dataclass) actually accepts.

    Lets DAPO knobs degrade gracefully on an older TRL: unsupported keys are
    dropped and reported rather than crashing ``GRPOConfig(**kwargs)``.
    """
    import dataclasses

    if dataclasses.is_dataclass(config_cls):
        valid = {f.name for f in dataclasses.fields(config_cls)}
    else:  # fall back to the __init__ signature
        import inspect

        valid = set(inspect.signature(config_cls).parameters)

    supported = {k: v for k, v in kwargs.items() if k in valid}
    dropped: List[str] = [k for k in kwargs if k not in valid]
    if dropped:
        print(
            f"[algo] WARNING: installed {getattr(config_cls, '__name__', config_cls)} "
            f"does not support {dropped}; upgrade TRL for full DAPO behavior."
        )
    return supported
