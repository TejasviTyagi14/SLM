"""Phase 8: task formatting. Emit SFT chat views + a pretrain corpus.

Primary product = the SFT instruction set (chat format), designed to fine-tune a
Qwen model: messages=[system,user,assistant], where the user prompt MIRRORS the
oMeBench eval template (prompts/default.txt / cot.txt) so the model trains on the
exact distribution it is graded on, and the assistant target is EXACTLY
reconstructable + machine-checkable (valid JSON, ontology-valid subtypes, RDKit-
valid SMILES) -- i.e. the training targets themselves re-pass the oMeS validators.

Task types generated from a mechanism record:
- mechanism_full      : reactants+conditions -> numbered typed steps (oMeS-critical)
- mechanism_next_step : reactants + first k steps -> step k+1
- forward_prediction  : reactants+reagents -> product(s)
- retrosynthesis      : product -> reactant(s)
- reagent_prediction  : reactants+product -> reagents/conditions
- named_reaction_qa   : (Tier C prose) name/conditions/mechanism Q&A

Every SFT row carries meta (reaction_id, source, license, provenance, level,
task, n_steps) and a ``reference`` list ([subtype, canonical_smiles, weight]) so
the same file can drive GRPO reward via training/reward.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = "You are an expert in organic reaction mechanisms."


def _canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else smi


def _fill_prompt(template: str, reactants, products, conditions) -> str:
    """Mirror omebench_eval.dataset.build_prompt token substitution exactly."""
    repl = {
        "{ reactants_smiles }": json.dumps(reactants),
        "{ products_smiles }": json.dumps(products),
        "{ conditions }": json.dumps(conditions),
    }
    for tok, val in repl.items():
        template = template.replace(tok, val)
    return template


def _steps_payload(mechanism: List[dict]) -> List[dict]:
    """The exact benchmark step objects (step/type/subtype/intermediate_smiles)."""
    return [
        {
            "step": s.get("step", i + 1),
            "type": s["type"],
            "subtype": s["subtype"],
            "intermediate_smiles": s["intermediate_smiles"],
        }
        for i, s in enumerate(mechanism)
    ]


def _reference(mechanism: List[dict]) -> List[list]:
    """[subtype, canonical_smiles, weight] per step for GRPO/oMeS reward."""
    n = len(mechanism)
    w = round(1.0 / n, 6) if n else 0.0
    return [[s["subtype"], _canon(s["intermediate_smiles"]), w] for s in mechanism]


def _assistant_mechanism(mechanism: List[dict], style: str) -> str:
    body = json.dumps(_steps_payload(mechanism))
    if style == "cot":
        lines = []
        for s in mechanism:
            rat = s.get("rationale") or f"Perform a {s['subtype']} step."
            lines.append(f"Step {s.get('step')}: {rat}")
        return "\n".join(lines) + f"\n[ANSWER]\n{body}\n[/ANSWER]"
    return body


def _row(task: str, messages: List[dict], rec: dict, reference=None) -> dict:
    return {
        "task": task,
        "messages": messages,
        "reference": reference,
        "meta": {
            "reaction_id": rec.get("reaction_id"),
            "source": rec.get("source"),
            "license": rec.get("license"),
            "provenance": rec.get("provenance"),
            "level": rec.get("level"),
            "name": rec.get("name"),
            "n_steps": len(rec.get("mechanism", [])),
        },
    }


# ---- per-task builders ------------------------------------------------------

def make_mechanism_full(rec: dict, template: str, style: str) -> Optional[dict]:
    mech = rec.get("mechanism", [])
    if not mech:
        return None
    user = _fill_prompt(template, rec.get("reactants_smiles"),
                        rec.get("products_smiles"), rec.get("conditions"))
    assistant = _assistant_mechanism(mech, style)
    return _row("mechanism_full",
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user},
                 {"role": "assistant", "content": assistant}],
                rec, reference=_reference(mech))


def make_next_step(rec: dict) -> List[dict]:
    """One row per prefix length k -> predict step k+1."""
    mech = rec.get("mechanism", [])
    rows = []
    for k in range(0, len(mech) - 1):
        given = _steps_payload(mech[: k + 1])
        target = _steps_payload([mech[k + 1]])[0]
        user = (
            "You are extending an organic mechanism.\n"
            f"Reactants (SMILES): {json.dumps(rec.get('reactants_smiles'))}\n"
            f"Conditions: {json.dumps(rec.get('conditions'))}\n"
            f"Steps so far: {json.dumps(given)}\n"
            "Return ONLY the next step as a JSON object with keys "
            "step, type, subtype, intermediate_smiles."
        )
        rows.append(_row("mechanism_next_step",
                         [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user},
                          {"role": "assistant", "content": json.dumps(target)}],
                         rec))
    return rows


def make_forward(rec: dict) -> Optional[dict]:
    reac = rec.get("reactants_smiles", [])
    prod = rec.get("products_smiles", [])
    if not reac or not prod:
        return None
    reagents = rec.get("raw_ref", {}).get("agents") or rec.get("raw_ref", {}).get("reagents") or []
    user = (
        "Predict the product(s) of this reaction.\n"
        f"Reactants (SMILES): {json.dumps(reac)}\n"
        f"Reagents/conditions: {json.dumps(reagents or rec.get('conditions'))}\n"
        "Return a JSON list of product SMILES."
    )
    return _row("forward_prediction",
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user},
                 {"role": "assistant", "content": json.dumps([_canon(p) for p in prod])}],
                rec)


def make_retro(rec: dict) -> Optional[dict]:
    reac = rec.get("reactants_smiles", [])
    prod = rec.get("products_smiles", [])
    if not reac or not prod:
        return None
    user = (
        "Propose reactants for this target product (single-step retrosynthesis).\n"
        f"Product (SMILES): {json.dumps(prod)}\n"
        "Return a JSON list of reactant SMILES."
    )
    return _row("retrosynthesis",
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user},
                 {"role": "assistant", "content": json.dumps([_canon(r) for r in reac])}],
                rec)


def make_reagent(rec: dict) -> Optional[dict]:
    reac = rec.get("reactants_smiles", [])
    prod = rec.get("products_smiles", [])
    reagents = rec.get("raw_ref", {}).get("agents") or []
    cond = rec.get("conditions")
    if not reac or not prod or not (reagents or cond):
        return None
    target = json.dumps(reagents) if reagents else json.dumps(cond)
    user = (
        "Given reactants and product, predict the reagents/conditions.\n"
        f"Reactants (SMILES): {json.dumps(reac)}\n"
        f"Product (SMILES): {json.dumps(prod)}\n"
        "Return the reagents as a JSON list, or conditions as a JSON string."
    )
    return _row("reagent_prediction",
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user},
                 {"role": "assistant", "content": target}],
                rec)


def make_named_qa(rec: dict) -> Optional[dict]:
    """Tier-C prose Q&A: only for records with a name + summary and no structures."""
    name = rec.get("name")
    summary = rec.get("raw_ref", {}).get("summary")
    if not name or not summary:
        return None
    user = f"Describe the {name} in organic chemistry: what it does and its conditions."
    return _row("named_reaction_qa",
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user},
                 {"role": "assistant", "content": summary}],
                rec)
