"""CoT rejection-sampling data builder (the single biggest quality lever).

Evidence: ether0 (arXiv:2506.17238) and RetroDFM-R (arXiv:2507.17448) both
warm-start RL on long chain-of-thought traces distilled from a frontier reasoner,
KEPT ONLY IF the final answer passes the verifier. This module implements that:

1. For each silver training reaction, prompt a frontier model
   (``omebench_eval.providers``) with the reaction AND the KNOWN gold mechanism
   (answer-conditioned prompting) to write a step-by-step rationale that ends in
   the ``[ANSWER]...[/ANSWER]`` JSON block that ``extract_mechanism`` parses.
2. Rejection filter: score each generated trace with oMeS; keep only traces with
   ``S_partial >= --keep-threshold`` (default 0.9) AND ``validity == 1.0``.
3. Emit rows in the SAME chat schema as ``build_sft_data.py`` (messages + prompt
   + reference), so ``sft_train.py`` / ``grpo_train.py`` consume them unchanged.
   The distilled assistant message is the model's own reasoning followed by the
   verified answer block -- long CoT the model can imitate.
4. Decontaminate the distilled set against oMe-Gold before writing.

The frontier model is any provider alias in ``omebench_eval.providers`` (opus,
gpt-5.5, ...). A fake provider can be injected for tests (see tests/).

Usage
-----
  python -m training.distill_cot --dataset silver --model opus \
      --limit 500 --keep-threshold 0.9 --out-dir training/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List

from omebench_eval.dataset import build_prompt, load_dataset, load_prompt_template
from omebench_eval.parsing import extract_mechanism
from omebench_eval.scoring import canonical_smiles, oMeS

from training.build_sft_data import decontaminate_rows, make_row, split_and_write

ROOT = Path(__file__).resolve().parent.parent

# Answer-conditioned distillation prompt. The model is shown the gold mechanism
# and asked to *explain* how to derive it, then re-emit it in the graded format.
# This yields high-quality reasoning traces whose final answer is (nearly) the
# gold answer -- exactly what the rejection filter then verifies.
DISTILL_SYSTEM = (
    "You are an expert organic chemist writing a training rationale. You are "
    "given a reaction and its EXPERT-VERIFIED stepwise mechanism. Explain the "
    "mechanism's logic step by step as a chemist would reason it out from the "
    "reactants forward, WITHOUT revealing that you were given the answer. Then "
    "reproduce the mechanism exactly in the required JSON format."
)

DISTILL_INSTRUCTION = """\
Reaction:
- Reactants (SMILES): {reactants}
- Products (SMILES): {products}
- Conditions: {conditions}

Expert-verified mechanism (for your reference only; do not mention that it was provided):
{gold_json}

Write a clear, step-by-step chemical rationale that arrives at this mechanism,
then output the final mechanism as JSON between [ANSWER] and [/ANSWER]. Each step
must have "step", "type", "subtype", and "intermediate_smiles" fields, using the
same types/subtypes and intermediate SMILES as the verified mechanism above.
"""


def _gold_reference(mech: List[dict]):
    """(subtype, canonical_smiles, weight) reference rows for oMeS scoring."""
    steps = [s for s in mech if isinstance(s, dict)]
    if not steps:
        return None
    weight = round(1.0 / len(steps), 6)
    ref = []
    for s in steps:
        can = canonical_smiles(s.get("intermediate_smiles", ""))
        ref.append([s.get("subtype"), can, weight])
    return ref


def _display_gold(mech: List[dict]) -> str:
    payload = [
        {
            "step": s.get("step", i + 1),
            "type": s.get("type"),
            "subtype": s.get("subtype"),
            "intermediate_smiles": s.get("intermediate_smiles", ""),
        }
        for i, s in enumerate(mech)
        if isinstance(s, dict)
    ]
    return json.dumps(payload, indent=2)


def score_trace(trace_text: str, reference: List) -> dict:
    """oMeS S_partial + validity of a distilled trace's final answer block."""
    mech = extract_mechanism(trace_text)
    if not isinstance(mech, list) or not mech:
        return {"ok": False, "S_partial": 0.0, "validity": 0.0, "n_steps": 0}
    pred = [
        (s.get("subtype"), s.get("intermediate_smiles", ""))
        for s in mech
        if isinstance(s, dict)
    ]
    if not pred:
        return {"ok": False, "S_partial": 0.0, "validity": 0.0, "n_steps": 0}
    validity = sum(1 for _, smi in pred if canonical_smiles(smi)) / len(pred)
    gold = [(row[0], row[1], row[2]) for row in reference]
    try:
        res = oMeS(gold, pred)
    except Exception:
        return {"ok": False, "S_partial": 0.0, "validity": validity, "n_steps": len(pred)}
    return {
        "ok": True,
        "S_partial": float(res.S_partial),
        "validity": float(validity),
        "n_steps": len(pred),
    }


def distill_rows(
    reactions: List[dict],
    generate: Callable[[str, str], str],
    style: str = "cot",
    keep_threshold: float = 0.9,
    require_validity: float = 1.0,
    max_attempts: int = 1,
) -> tuple:
    """Generate + rejection-filter distilled CoT rows.

    ``generate(system, user) -> str`` is the frontier-model call (a provider's
    ``.generate``, or a fake in tests). Returns (kept_rows, stats).

    A trace is KEPT only if S_partial >= keep_threshold AND validity >=
    require_validity. ``max_attempts`` re-samples a reaction until one trace
    passes (useful with a stochastic provider); default 1 = single shot.
    """
    template = load_prompt_template(style)
    kept: List[dict] = []
    n_kept = n_dropped = 0

    for r in reactions:
        mech = r.get("mechanism", []) or []
        reference = _gold_reference(mech)
        if reference is None:
            n_dropped += 1
            continue

        reactants = r.get("reactants_smiles")
        products = r.get("products_smiles")
        conditions = r.get("conditions")
        user_graded = build_prompt(template, reactants, products, conditions)
        distill_user = DISTILL_INSTRUCTION.format(
            reactants=json.dumps(reactants),
            products=json.dumps(products),
            conditions=json.dumps(conditions),
            gold_json=_display_gold(mech),
        )

        passed = None
        for _ in range(max(1, max_attempts)):
            try:
                trace = generate(DISTILL_SYSTEM, distill_user)
            except Exception:
                continue
            s = score_trace(trace, reference)
            if s["ok"] and s["S_partial"] >= keep_threshold and s["validity"] >= require_validity:
                passed = trace
                break

        if passed is None:
            n_dropped += 1
            continue

        # The assistant target is the frontier model's own verified trace, kept
        # verbatim so the student imitates the long reasoning (not just the JSON).
        kept.append(
            make_row(
                reaction_id=f"DISTILL-{r.get('reaction_id')}",
                level=r.get("level"),
                user=user_graded,
                assistant_content=passed.strip(),
                reference=reference,
                reactants=reactants,
                products=products,
                conditions=conditions,
                style=style,
            )
        )
        n_kept += 1

    stats = {
        "reactions_in": len(reactions),
        "kept": n_kept,
        "dropped": n_dropped,
        "keep_threshold": keep_threshold,
        "require_validity": require_validity,
    }
    print(f"[distill] kept={n_kept} dropped={n_dropped} "
          f"(threshold S_partial>={keep_threshold}, validity>={require_validity})")
    return kept, stats


def _make_generate(model: str, max_tokens: int, temperature: float) -> Callable[[str, str], str]:
    from omebench_eval.providers import make_provider, resolve_spec

    spec = resolve_spec(model)
    provider = make_provider(spec, max_tokens=max_tokens, temperature=temperature)
    return provider.generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="silver", choices=["silver", "gold", "template"],
                    help="Source reactions to distill. NOTE: never use gold for "
                         "training data -- distilled rows are decontaminated regardless.")
    ap.add_argument("--model", default="opus",
                    help="Frontier provider alias (omebench_eval.providers).")
    ap.add_argument("--style", default="cot", choices=["default", "cot"])
    ap.add_argument("--out-dir", default="training/data")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--keep-threshold", type=float, default=0.9,
                    help="Min oMeS S_partial to keep a distilled trace.")
    ap.add_argument("--require-validity", type=float, default=1.0,
                    help="Min SMILES validity fraction to keep a trace.")
    ap.add_argument("--max-attempts", type=int, default=1,
                    help="Re-sample a reaction up to this many times to pass the filter.")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reactions = load_dataset(args.dataset, limit=args.limit)
    generate = _make_generate(args.model, args.max_tokens, args.temperature)

    rows, stats = distill_rows(
        reactions,
        generate=generate,
        style=args.style,
        keep_threshold=args.keep_threshold,
        require_validity=args.require_validity,
        max_attempts=args.max_attempts,
    )
    if not rows:
        raise SystemExit("No distilled traces passed the rejection filter.")

    rows, removed = decontaminate_rows(rows)
    stats["decontam_removed"] = removed
    if not rows:
        raise SystemExit("All distilled rows removed by decontamination?!")

    out_dir = ROOT / args.out_dir
    split_and_write(
        rows,
        out_dir=out_dir,
        train_name=f"sft_distilled_{args.style}_train.jsonl",
        val_name=f"sft_distilled_{args.style}_val.jsonl",
        val_frac=args.val_frac,
        seed=args.seed,
        augment=0,  # distilled traces are already the target; do not augment inputs
    )
    print(f"[distill] stats: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
