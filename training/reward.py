"""Verifiable oMeS reward for GRPO/RL fine-tuning.

The headline reward is the oMeS partial score (S_partial in [0, 1]) of the
generated mechanism against the gold reference, with small shaping bonuses for
producing a parseable, valid-SMILES mechanism. Because oMeS is cheap and fully
programmatic, this is a clean verifiable reward — no reward model needed.

TRL's GRPOTrainer calls a reward function as:
    reward_func(prompts, completions, **kwargs) -> list[float]
where every other dataset column (here `reference`) is passed via kwargs as a
list aligned with the batch. Completions may be plain strings or chat-message
lists depending on the dataset format; both are handled.
"""

from __future__ import annotations

from typing import List

from omebench_eval.parsing import extract_mechanism
from omebench_eval.scoring import canonical_smiles, oMeS

# Reward shaping weights.
W_SPARTIAL = 1.0     # main signal: oMeS partial score
W_VALIDITY = 0.10    # fraction of predicted intermediates that are valid SMILES
W_FORMAT = 0.05      # produced a non-empty, parseable mechanism list at all
FORMAT_FAIL = -0.5   # empty / unparseable output


def _completion_to_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # chat format: list of {role, content}
        return " ".join(
            m.get("content", "") for m in c if isinstance(m, dict)
        )
    if isinstance(c, dict):
        return c.get("content", "")
    return str(c)


def _score_one(text: str, reference: List) -> float:
    mech = extract_mechanism(text)
    if not isinstance(mech, list) or not mech:
        return FORMAT_FAIL

    pred = [
        (s.get("subtype"), s.get("intermediate_smiles", ""))
        for s in mech
        if isinstance(s, dict)
    ]
    if not pred:
        return FORMAT_FAIL

    # reference rows are [subtype, canonical_smiles, weight]
    gold = [(row[0], row[1], row[2]) for row in reference]

    try:
        res = oMeS(gold, pred)
    except Exception:
        return FORMAT_FAIL

    validity = sum(1 for _, smi in pred if canonical_smiles(smi)) / max(1, len(pred))
    reward = (
        W_SPARTIAL * res.S_partial
        + W_VALIDITY * validity
        + W_FORMAT  # bonus for producing a usable mechanism
    )
    return float(reward)


def omes_reward(prompts=None, completions=None, reference=None, **kwargs) -> List[float]:
    """GRPO-compatible reward: oMeS partial score + validity/format shaping."""
    if completions is None:
        return []
    if reference is None:
        raise ValueError(
            "omes_reward needs a 'reference' column in the dataset "
            "(list of [subtype, canonical_smiles, weight] per example)."
        )
    rewards = []
    for comp, ref in zip(completions, reference):
        rewards.append(_score_one(_completion_to_text(comp), ref))
    return rewards


# Convenience: separate format-only reward (useful as a second reward_func).
def format_reward(prompts=None, completions=None, **kwargs) -> List[float]:
    out = []
    for comp in completions or []:
        mech = extract_mechanism(_completion_to_text(comp))
        out.append(1.0 if isinstance(mech, list) and mech else 0.0)
    return out
