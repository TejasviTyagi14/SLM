"""Build chat-format SFT data from oMeBench splits, validated with RDKit.

The output rows use the *exact same* prompt the eval harness uses, so the model
trains on the distribution it will be graded on. Two target styles:

- default : assistant returns the JSON mechanism list directly.
- cot     : assistant reasons step-by-step (from the silver rationales), then
            returns the JSON between [ANSWER]...[/ANSWER].

Each row also carries a `reference` field (list of [subtype, canonical_smiles,
weight]) so the same file can drive GRPO reward computation later.

Two SOTA levers are wired in here (see docs/sota_features.md):

* --augment N : emit N randomized-SMILES variants of the reactant/product INPUTS
  per train reaction (RDKit doRandom), keeping targets canonical (Bjerrum
  enumeration, arXiv:1703.07076; LlaSMol). Train-only; runs BEFORE decontam/dedup
  so leaks are still caught.
* --decontaminate (default on): drop any row whose reaction matches the held-out
  oMe-Gold/Template blacklist (reuses rxndata.decontaminate). Guards the zero-leak
  invariant even though build reads the raw silver file directly.

Usage
-----
  python -m training.build_sft_data --dataset silver --style cot \
      --out-dir training/data --val-frac 0.03 --augment 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

from omebench_eval.dataset import build_prompt, load_dataset, load_prompt_template
from omebench_eval.runner import SYSTEM_PROMPT
from omebench_eval.scoring import canonical_smiles

ROOT = Path(__file__).resolve().parent.parent

# Internal metadata keys carried on rows for augmentation/decontam and stripped
# before the row is written (they are not part of the SFT schema).
_META_KEYS = ("_reactants", "_products", "_conditions", "_style")


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


def make_row(
    reaction_id,
    level,
    user: str,
    assistant_content: str,
    reference: List,
    reactants=None,
    products=None,
    conditions=None,
    style: str = "cot",
) -> dict:
    """Assemble one SFT chat row (messages + prompt + reference) with metadata.

    The `_reactants/_products/_conditions/_style` keys are internal (used for
    augmentation) and are removed by `strip_meta` before writing.
    """
    return {
        "reaction_id": reaction_id,
        "level": level,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_content},
        ],
        # Kept for GRPO: prompt-only messages + gold reference.
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "reference": reference,
        "_reactants": reactants,
        "_products": products,
        "_conditions": conditions,
        "_style": style,
    }


def strip_meta(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _META_KEYS}


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
        reactants = r.get("reactants_smiles")
        products = r.get("products_smiles")
        conditions = r.get("conditions")
        user = build_prompt(template, reactants, products, conditions)
        rows.append(
            make_row(
                reaction_id=r.get("reaction_id"),
                level=r.get("level"),
                user=user,
                assistant_content=_assistant_target(steps, style),
                reference=reference,
                reactants=reactants,
                products=products,
                conditions=conditions,
                style=style,
            )
        )
    print(f"[build] dataset={dataset} kept={len(rows)} dropped_invalid={dropped}")
    return rows


# --- SMILES augmentation (Feature 6) ----------------------------------------

def randomize_smiles(smi: str, n: int) -> List[str]:
    """Up to n distinct randomized (non-canonical) SMILES for the same molecule.

    Uses RDKit doRandom enumeration (Bjerrum). Returns variants that differ from
    the canonical string; may return fewer than n if the molecule is small.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    canon = Chem.MolToSmiles(mol)
    seen = set()
    out: List[str] = []
    # Oversample: random SMILES collide for small molecules.
    for _ in range(n * 8):
        if len(out) >= n:
            break
        try:
            s = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        except Exception:
            break
        if s and s != canon and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _augment_input_list(smis, n: int) -> List[List[str]]:
    """Produce up to n randomized variants of a list of input SMILES."""
    if not smis:
        return []
    per = [randomize_smiles(s, n) for s in smis]
    variants: List[List[str]] = []
    for k in range(n):
        variant = []
        ok = True
        for orig, alts in zip(smis, per):
            if k < len(alts):
                variant.append(alts[k])
            else:
                variant.append(orig)  # fall back to original if we ran dry
        if ok:
            variants.append(variant)
    return variants


def augment_train_rows(rows: List[dict], n: int) -> List[dict]:
    """Return `rows` plus n randomized-SMILES input variants per row (train only).

    Targets (assistant content + reference) stay canonical; only the user-prompt
    input SMILES are randomized, matching LlaSMol's "canonical target, augmented
    input" recipe.
    """
    if n <= 0:
        return rows
    template_cache = {}

    def _template(style):
        if style not in template_cache:
            template_cache[style] = load_prompt_template(style)
        return template_cache[style]

    augmented: List[dict] = list(rows)
    added = 0
    for row in rows:
        reac = row.get("_reactants")
        prod = row.get("_products")
        if not reac and not prod:
            continue
        style = row.get("_style", "cot")
        tmpl = _template(style)
        reac_variants = _augment_input_list(reac or [], n)
        prod_variants = _augment_input_list(prod or [], n)
        for k in range(n):
            rv = reac_variants[k] if k < len(reac_variants) else reac
            pv = prod_variants[k] if k < len(prod_variants) else prod
            if rv == reac and pv == prod:
                continue  # no new variant available
            user = build_prompt(tmpl, rv, pv, row.get("_conditions"))
            new_row = dict(row)
            new_row["messages"] = [
                row["messages"][0],
                {"role": "user", "content": user},
                row["messages"][2],
            ]
            new_row["prompt"] = [row["prompt"][0], {"role": "user", "content": user}]
            new_row["_reactants"] = rv
            new_row["_products"] = pv
            augmented.append(new_row)
            added += 1
    print(f"[augment] added {added} randomized-SMILES variants (n={n})")
    return augmented


# --- Decontamination against oMe-Gold/Template (invariant guard) ------------

def decontaminate_rows(rows: List[dict], tanimoto_threshold: float = 0.95) -> Tuple[List[dict], int]:
    """Drop rows whose reaction matches the oMe-Gold/Template blacklist.

    Reuses rxndata.decontaminate so the check is identical to Phase 6. Uses each
    row's carried reactant/product SMILES; rows without them are kept (they carry
    no reaction signature to match on).
    """
    import sys

    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from rxndata.decontaminate import build_blacklist, contamination_reason

    gold_path = ROOT / "data" / "oMe_Gold.json"
    tmpl_path = ROOT / "data" / "oMe_Template.json"
    gold = json.loads(gold_path.read_text()) if gold_path.exists() else []
    template = json.loads(tmpl_path.read_text()) if tmpl_path.exists() else []
    bl = build_blacklist(gold, template)

    kept: List[dict] = []
    removed = 0
    for row in rows:
        rec = {
            "reactants_smiles": row.get("_reactants") or [],
            "products_smiles": row.get("_products") or [],
            "mechanism": [],
        }
        if not rec["reactants_smiles"] and not rec["products_smiles"]:
            kept.append(row)
            continue
        if contamination_reason(rec, bl, tanimoto_threshold) is None:
            kept.append(row)
        else:
            removed += 1
    print(f"[decontaminate] kept={len(kept)} removed_leaks={removed} "
          f"(blacklist={bl.n_entries})")
    return kept, removed


# --- Split + write ----------------------------------------------------------

def split_and_write(
    rows: List[dict],
    out_dir: Path,
    train_name: str,
    val_name: str,
    val_frac: float,
    seed: int,
    augment: int = 0,
    decontaminate: bool = False,
    tanimoto_threshold: float = 0.95,
) -> Tuple[Path, Path]:
    """Shuffle, split off a val set, augment TRAIN ONLY, decontaminate, and write.

    Ordering (spec-critical): augmentation runs BEFORE decontamination so every
    emitted row -- including each randomized-SMILES variant -- is individually
    screened against the oMe-Gold/Template blacklist. Decontamination is applied
    to train AND val.
    """
    random.seed(seed)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac)) if val_frac > 0 else 0
    val, train = rows[:n_val], rows[n_val:]

    if augment > 0:
        train = augment_train_rows(train, augment)
        random.shuffle(train)

    if decontaminate:
        train, _ = decontaminate_rows(train, tanimoto_threshold)
        if val:
            val, _ = decontaminate_rows(val, tanimoto_threshold)
        if not train:
            raise SystemExit("All train rows removed by decontamination?!")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / train_name
    val_path = out_dir / val_name
    for path, subset in [(train_path, train), (val_path, val)]:
        with path.open("w") as f:
            for row in subset:
                f.write(json.dumps(strip_meta(row)) + "\n")
    print(f"[build] wrote {len(train)} train -> {train_path}")
    print(f"[build] wrote {len(val)} val   -> {val_path}")
    return train_path, val_path


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
    ap.add_argument("--augment", type=int, default=0,
                    help="Emit N randomized-SMILES input variants per TRAIN reaction "
                         "(0 = off). Targets stay canonical.")
    ap.add_argument("--decontaminate", dest="decontaminate", action="store_true",
                    default=True,
                    help="Drop rows matching the oMe-Gold/Template blacklist (default on).")
    ap.add_argument("--no-decontaminate", dest="decontaminate", action="store_false",
                    help="Skip decontamination (NOT recommended; can leak test data).")
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

    # Augmentation happens inside split_and_write (train-only), and
    # decontamination runs AFTER it there, so every emitted row -- including each
    # randomized-SMILES variant -- is individually screened against oMe-Gold.
    out_dir = ROOT / args.out_dir
    split_and_write(
        rows,
        out_dir=out_dir,
        train_name=f"sft_{args.dataset}_{args.style}_train.jsonl",
        val_name=f"sft_{args.dataset}_{args.style}_val.jsonl",
        val_frac=args.val_frac,
        seed=args.seed,
        augment=args.augment,
        decontaminate=args.decontaminate,
    )


if __name__ == "__main__":
    main()
