"""Pluggable forward reaction model for round-trip reward verification.

Evidence: round-trip / feasibility rewards (RTRL arXiv:2510.01527; RetroDFM-R
arXiv:2507.17448) verify a proposed step by running it *forward* and checking the
predicted product matches the claimed one. This module defines a tiny interface
so the round-trip reward in ``training/reward.py`` can call *some* forward model,
while defaulting to a NO-OP so the RL loop runs with zero extra weights or deps.

Contract
--------
A forward model maps a single-step reactant SMILES to a list of candidate product
SMILES (``predict``). The default implementation returns ``None`` ("cannot
assess"), which the round-trip reward treats as *no signal* (weight contributes
nothing) rather than a penalty -- so turning the reward on without real weights
never hurts training. Drop in a real model (a small forward-prediction
transformer, a template applicator, an external API) by subclassing
``ForwardModel`` and passing an instance to ``make_roundtrip_reward``.
"""

from __future__ import annotations

from typing import List, Optional

from omebench_eval.scoring import canonical_smiles


class ForwardModel:
    """Interface: predict product SMILES for a single elementary step.

    ``predict`` returns candidate product SMILES for ``reactant_smiles`` (a single
    combined SMILES for that step's starting material), or ``None`` if this model
    cannot make a prediction. Returning ``None`` means "no signal" to the reward.
    """

    def predict(self, reactant_smiles: str) -> Optional[List[str]]:  # pragma: no cover - interface
        raise NotImplementedError


class NoOpForwardModel(ForwardModel):
    """Default: makes no prediction. Round-trip reward contributes nothing.

    This lets ``--reward omes+roundtrip`` run without any extra model weights;
    the round-trip term stays a no-op until a real ForwardModel is supplied.
    """

    def predict(self, reactant_smiles: str) -> Optional[List[str]]:
        return None


def default_forward_model() -> ForwardModel:
    """The no-op forward model used unless a real one is injected."""
    return NoOpForwardModel()


def step_feasible(
    model: ForwardModel, reactant_smiles: str, product_smiles: str
) -> Optional[bool]:
    """Is ``reactant -> product`` feasible according to ``model``?

    Returns:
      True/False if the model produced predictions to compare against,
      None      if the model abstains (no-op) or either SMILES is unparseable
                (so the round-trip reward can skip this step rather than punish it).
    """
    can_p = canonical_smiles(product_smiles)
    if not can_p:
        return None
    preds = model.predict(reactant_smiles)
    if not preds:
        return None
    return any(canonical_smiles(p) == can_p for p in preds)
