"""Verifiable oMeS rewards for GRPO/DAPO RL fine-tuning.

The headline reward is the oMeS partial score (S_partial in [0, 1]) of the
generated mechanism against the gold reference. Because oMeS is cheap and fully
programmatic, this is a clean verifiable reward -- no reward model needed.

Design (evidence-linked)
------------------------
* Multiplicative format gate (ether0, arXiv:2506.17238): if the output is
  unparseable OR no predicted intermediate is a valid SMILES, we return
  ``FORMAT_FAIL`` and never call oMeS. This stops "invalid-but-lucky" outputs
  (right subtypes, garbage structures) from collecting format/validity bonuses,
  and pushes the policy to first satisfy format+validity before chasing score.
* Explicit validity signal (PSV-PPO, arXiv:2505.00530): SMILES validity can
  collapse during RL exploration, so we keep validity both inside the gated
  oMeS reward and as a standalone ``validity_reward`` reward_func that always
  contributes to the loss.
* Round-trip feasibility (RTRL arXiv:2510.01527; RetroDFM-R arXiv:2507.17448):
  an optional term that checks each elementary step is feasible under a
  pluggable forward model (``training/forward_model.py``). Off by default and a
  no-op unless a real forward model is injected, so it never needs extra weights.

The oMeS scorer itself (``omebench_eval.scoring.oMeS``) is imported unchanged so
every score stays byte-for-byte comparable to the published leaderboard; the new
signals are separate reward terms layered on top, never edits to oMeS.

TRL calls a reward function as
    reward_func(prompts, completions, **kwargs) -> list[float]
where every other dataset column (here ``reference``) arrives via kwargs as a
list aligned with the batch. Completions may be plain strings or chat-message
lists depending on the dataset format; both are handled.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from omebench_eval.parsing import extract_mechanism
from omebench_eval.scoring import canonical_smiles, oMeS

# --- Reward shaping weights (module constants; overridable via grpo_train CLI) --
W_SPARTIAL = 1.0     # main signal: oMeS partial score
W_VALIDITY = 0.10    # fraction of predicted intermediates that are valid SMILES
W_FORMAT = 0.05      # bonus for producing a non-empty, parseable, valid mechanism
W_ROUNDTRIP = 0.10   # optional forward-model round-trip feasibility term
FORMAT_FAIL = -0.5   # empty / unparseable / all-invalid output (hard gate)


def _completion_to_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # chat format: list of {role, content}
        return " ".join(m.get("content", "") for m in c if isinstance(m, dict))
    if isinstance(c, dict):
        return c.get("content", "")
    return str(c)


def parse_pred_steps(text: str) -> Optional[List[dict]]:
    """Extract the list of step dicts from model text, or None if unparseable."""
    mech = extract_mechanism(text)
    if not isinstance(mech, list) or not mech:
        return None
    steps = [s for s in mech if isinstance(s, dict)]
    return steps or None


def _validity(steps: List[dict]) -> float:
    if not steps:
        return 0.0
    n_valid = sum(1 for s in steps if canonical_smiles(s.get("intermediate_smiles", "")))
    return n_valid / len(steps)


def score_components(text: str, reference: List) -> dict:
    """Single source of truth for reward + monitoring.

    Returns a dict with:
      parseable : bool  -- a non-empty mechanism list was recovered
      n_steps   : int
      validity  : float -- fraction of predicted intermediates that parse
      S_partial : float -- oMeS partial score (0.0 if not scored)
      S_total   : float
      L         : float -- oMeS logical fidelity
    ``S_*``/``L`` are 0.0 when the output is unparseable or the gate trips.
    """
    steps = parse_pred_steps(text)
    if steps is None:
        return {"parseable": False, "n_steps": 0, "validity": 0.0,
                "S_partial": 0.0, "S_total": 0.0, "L": 0.0}

    validity = _validity(steps)
    out = {"parseable": True, "n_steps": len(steps), "validity": validity,
           "S_partial": 0.0, "S_total": 0.0, "L": 0.0}
    # Gate: don't score structure when nothing parses (oMeS would be ~0 anyway,
    # and we must not hand out validity/format credit for degenerate output).
    if validity == 0.0:
        return out

    pred = [(s.get("subtype"), s.get("intermediate_smiles", "")) for s in steps]
    gold = [(row[0], row[1], row[2]) for row in reference]
    try:
        res = oMeS(gold, pred)
    except Exception:
        return out
    out["S_partial"] = float(res.S_partial)
    out["S_total"] = float(res.S_total)
    out["L"] = float(res.L)
    return out


def _roundtrip_frac(steps: List[dict], forward_model) -> Optional[float]:
    """Fraction of consecutive-step transitions the forward model deems feasible.

    Returns None when the model abstains on every transition (no signal), so the
    caller can contribute 0 rather than penalize.
    """
    from training.forward_model import step_feasible

    assessed = 0
    feasible = 0
    prev = None
    for s in steps:
        cur = s.get("intermediate_smiles", "")
        if prev is not None:
            verdict = step_feasible(forward_model, prev, cur)
            if verdict is not None:
                assessed += 1
                feasible += 1 if verdict else 0
        prev = cur
    if assessed == 0:
        return None
    return feasible / assessed


def make_omes_reward(
    w_spartial: float = W_SPARTIAL,
    w_validity: float = W_VALIDITY,
    w_format: float = W_FORMAT,
    w_roundtrip: float = 0.0,
    forward_model=None,
) -> Callable:
    """Build a GRPO/DAPO reward_func over oMeS with configurable weights.

    Multiplicative gate: unparseable or all-invalid -> FORMAT_FAIL (oMeS skipped).
    Otherwise reward = w_spartial*S_partial + w_validity*V + w_format
                       + w_roundtrip*roundtrip_frac (if a forward model assesses).
    """
    if w_roundtrip and forward_model is None:
        from training.forward_model import default_forward_model

        forward_model = default_forward_model()

    def _reward_one(text: str, reference: List) -> float:
        steps = parse_pred_steps(text)
        if steps is None:
            return FORMAT_FAIL
        comp = score_components(text, reference)
        if comp["validity"] == 0.0:
            return FORMAT_FAIL
        reward = (
            w_spartial * comp["S_partial"]
            + w_validity * comp["validity"]
            + w_format
        )
        if w_roundtrip:
            rt = _roundtrip_frac(steps, forward_model)
            if rt is not None:
                reward += w_roundtrip * rt
        return float(reward)

    def omes_reward(prompts=None, completions=None, reference=None, **kwargs) -> List[float]:
        if completions is None:
            return []
        if reference is None:
            raise ValueError(
                "omes_reward needs a 'reference' column in the dataset "
                "(list of [subtype, canonical_smiles, weight] per example)."
            )
        return [
            _reward_one(_completion_to_text(comp), ref)
            for comp, ref in zip(completions, reference)
        ]

    omes_reward.__name__ = "omes_reward"
    return omes_reward


# Default-configured reward function (backward-compatible import surface).
omes_reward = make_omes_reward()


def validity_reward(prompts=None, completions=None, **kwargs) -> List[float]:
    """Standalone validity signal: fraction of parseable intermediate SMILES.

    Kept in the reward_funcs list so the RL loss always carries an explicit
    validity term (guards against validity collapse during exploration).
    """
    out = []
    for comp in completions or []:
        steps = parse_pred_steps(_completion_to_text(comp))
        out.append(_validity(steps) if steps else 0.0)
    return out


def format_reward(prompts=None, completions=None, **kwargs) -> List[float]:
    """1.0 if a non-empty mechanism list parses at all, else 0.0."""
    out = []
    for comp in completions or []:
        steps = parse_pred_steps(_completion_to_text(comp))
        out.append(1.0 if steps else 0.0)
    return out


def make_roundtrip_reward(forward_model=None) -> Callable:
    """Standalone round-trip feasibility reward_func (forward model pluggable)."""
    if forward_model is None:
        from training.forward_model import default_forward_model

        forward_model = default_forward_model()

    def roundtrip_reward(prompts=None, completions=None, **kwargs) -> List[float]:
        out = []
        for comp in completions or []:
            steps = parse_pred_steps(_completion_to_text(comp))
            if not steps:
                out.append(0.0)
                continue
            rt = _roundtrip_frac(steps, forward_model)
            out.append(rt if rt is not None else 0.0)
        return out

    roundtrip_reward.__name__ = "roundtrip_reward"
    return roundtrip_reward


def build_reward_funcs(
    reward: str = "omes",
    w_spartial: float = W_SPARTIAL,
    w_validity: float = W_VALIDITY,
    w_format: float = W_FORMAT,
    w_roundtrip: float = W_ROUNDTRIP,
    forward_model=None,
) -> List[Callable]:
    """Assemble the reward_funcs list for grpo_train from CLI options.

    reward = "omes"           -> [gated oMeS reward, validity, format]
    reward = "omes+roundtrip" -> add a round-trip feasibility term into the main
                                 reward AND keep a standalone round-trip signal.
    """
    if reward not in ("omes", "omes+roundtrip"):
        raise ValueError(f"Unknown reward '{reward}'. Choose omes | omes+roundtrip.")
    use_rt = reward == "omes+roundtrip"
    main = make_omes_reward(
        w_spartial=w_spartial,
        w_validity=w_validity,
        w_format=w_format,
        w_roundtrip=w_roundtrip if use_rt else 0.0,
        forward_model=forward_model,
    )
    funcs: List[Callable] = [main, validity_reward, format_reward]
    if use_rt:
        funcs.append(make_roundtrip_reward(forward_model))
    return funcs
