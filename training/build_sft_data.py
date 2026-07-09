"""Build chat-format SFT data from oMeBench splits, validated with RDKit.

The output rows use the *exact same* prompt the eval harness uses, so the model
trains on the distribution it will be graded on. Two target styles:

- default : assistant returns the JSON mechanism list directly.
- cot     : assistant reasons step-by-step (from the silver rationales), then
            returns the JSON between [ANSWER]...[/ANSWER].

Each row also carries a `reference` field (list of [subtype, canonical_smiles,
weight]) so the same file can drive GRPO reward computation later.

Usage
-----
  python -m training.build_sft_data --dataset silver --style cot \
      --out-dir training/data --val-frac 0.03
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

from omebench_eval.dataset import build_prompt, load_dataset, load_prompt_template
from omebench_eval.runner import SYSTEM_PROMPT
from omebench_eval.scoring import canonical_smiles

ROOT = Path(__file__).resolve().parent.parent


def _clean_mechanism(mech: List[dict], min_valid_frac: float, canonicalize: bool):
    """Return (steps, reference, valid_frac) or None if below the validity bar.

    steps     : minimal step dicts for the assistant target
    reference : (subtype, canonical_smiles, weight) tuples for scoring/reward
    """
    steps = []
    n_valid = 0
    for s in mech:
        smi = s.get("intermediate_smiles", "")
        can = canonical_smiles(smi)
        if can:
            n_valid += 1
        out_smi = can if (canonicalize and can) else smi
        steps.append(
            {
                "step": s.get("step"),
                "type": s.get("type"),
                "subtype": s.get("subtype"),
                "intermediate_smiles": out_smi,
                "_rationale": s.get("rationale", ""),
                "_canon": can,
            }
        )
    if not steps:
        return None
    valid_frac = n_valid / len(steps)
    if valid_frac < min_valid_frac:
        return None

    weight = round(1.0 / len(steps), 6)
    reference = [
        [s["subtype"], s["_canon"], weight] for s in steps
    ]
    return steps, reference, valid_frac


def _assistant_target(steps: List[dict], style: str) -> str:
    payload = [
        {
            "step": s["step"],
            "type": s["type"],
            "subtype": s["subtype"],
            "intermediate_smiles": s["intermediate_smiles"],
        }
        for s in steps
    ]
    body = json.dumps(payload)
    if style == "cot":
        reasoning_lines = []
        for s in steps:
            rat = s.get("_rationale") or f"Perform a {s['subtype']} step."
            reasoning_lines.append(f"Step {s['step']}: {rat}")
        reasoning = "\n".join(reasoning_lines)
        return f"{reasoning}\n[ANSWER]\n{body}\n[/ANSWER]"
    return body


def build_rows(
    dataset: str,
    style: str,
    min_valid_frac: float,
    canonicalize: bool,
    limit: Optional[int],
) -> List[dict]:
    data = load_dataset(dataset, limit=limit)
    template = load_prompt_template(style)
    rows = []
    dropped = 0
    for r in data:
        cleaned = _clean_mechanism(r.get("mechanism", []), min_valid_frac, canonicalize)
        if cleaned is None:
            dropped += 1
            continue
        steps, reference, _ = cleaned
        user = build_prompt(
            template,
            r.get("reactants_smiles"),
            r.get("products_smiles"),
            r.get("conditions"),
        )
        rows.append(
            {
                "reaction_id": r.get("reaction_id"),
                "level": r.get("level"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": _assistant_target(steps, style)},
                ],
                # Kept for GRPO: prompt-only messages + gold reference.
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "reference": reference,
            }
        )
    print(f"[build] dataset={dataset} kept={len(rows)} dropped_invalid={dropped}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="silver", choices=["silver", "gold", "template"])
    ap.add_argument("--style", default="cot", choices=["default", "cot"])
    ap.add_argument("--out-dir", default="training/data")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--min-valid-frac", type=float, default=1.0,
                    help="Drop reactions whose fraction of RDKit-valid intermediates "
                         "is below this (1.0 = every intermediate must parse).")
    ap.add_argument("--no-canonicalize", action="store_true",
                    help="Keep original SMILES instead of RDKit-canonical form.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = build_rows(
        dataset=args.dataset,
        style=args.style,
        min_valid_frac=args.min_valid_frac,
        canonicalize=not args.no_canonicalize,
        limit=args.limit,
    )
    if not rows:
        raise SystemExit("No rows produced; loosen --min-valid-frac?")

    random.seed(args.seed)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac)) if args.val_frac > 0 else 0
    val, train = rows[:n_val], rows[n_val:]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"sft_{args.dataset}_{args.style}_train.jsonl"
    val_path = out_dir / f"sft_{args.dataset}_{args.style}_val.jsonl"

    for path, subset in [(train_path, train), (val_path, val)]:
        with path.open("w") as f:
            for row in subset:
                f.write(json.dumps(row) + "\n")
    print(f"[build] wrote {len(train)} train -> {train_path}")
    print(f"[build] wrote {len(val)} val   -> {val_path}")


if __name__ == "__main__":
    main()
