"""End-to-end runner: generate predictions, score with oMeS, report."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from .dataset import build_prompt, load_dataset, load_prompt_template
from .parsing import extract_mechanism
from .providers import BaseProvider, ModelSpec, make_provider
from .scoring import oMeS

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

SYSTEM_PROMPT = "You are an expert in organic reaction mechanisms."


def _read_done_ids(path: Path) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done[obj["reaction_id"]] = obj
                except Exception:
                    continue
    return done


def generate_predictions(
    provider: BaseProvider,
    dataset: str,
    prompt_style: str,
    out_path: Path,
    limit: Optional[int] = None,
    concurrency: int = 4,
    resume: bool = True,
) -> None:
    """Query the model for every reaction and append predictions to a .jsonl."""
    data = load_dataset(dataset, limit=limit)
    template = load_prompt_template(prompt_style)

    done = _read_done_ids(out_path) if resume else {}
    todo = [
        (idx, r)
        for idx, r in enumerate(data)
        if r.get("reaction_id", f"rxn_{idx}") not in done
    ]
    print(
        f"[generate] dataset={dataset} total={len(data)} "
        f"already_done={len(done)} to_run={len(todo)} concurrency={concurrency}"
    )
    if not todo:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    def _work(item):
        idx, reaction = item
        rxn_id = reaction.get("reaction_id", f"rxn_{idx}")
        prompt = build_prompt(
            template,
            reaction.get("reactants_smiles"),
            reaction.get("products_smiles"),
            reaction.get("conditions"),
        )
        try:
            raw = provider.generate(SYSTEM_PROMPT, prompt)
            parsed = extract_mechanism(raw)
            return {
                "reaction_id": rxn_id,
                "input": {
                    "reactants_smiles": reaction.get("reactants_smiles"),
                    "products_smiles": reaction.get("products_smiles"),
                    "conditions": reaction.get("conditions"),
                },
                "raw_output": raw,
                "output": parsed if parsed is not None else raw,
            }
        except Exception as e:
            return {
                "reaction_id": rxn_id,
                "input": reaction,
                "output": f"[Error] generation failed: {type(e).__name__}: {e}",
                "error": True,
            }

    completed = 0
    with out_path.open("a") as fout:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_work, item): item for item in todo}
            for fut in as_completed(futures):
                result = fut.result()
                with write_lock:
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
                completed += 1
                if completed % 10 == 0 or completed == len(todo):
                    print(f"[generate] {completed}/{len(todo)} done")


def evaluate_predictions(
    dataset: str,
    pred_path: Path,
    eval_path: Path,
    model_label: str,
    limit: Optional[int] = None,
) -> dict:
    """Score predictions against gold with oMeS; write per-reaction + summary."""
    gold_data = load_dataset(dataset, limit=limit)
    preds = _read_done_ids(pred_path)

    per_reaction: List[dict] = []
    successful = 0
    for gold_item in gold_data:
        rxn_id = gold_item["reaction_id"]
        pred_obj = preds.get(rxn_id, {})
        pred_output = pred_obj.get("output")

        mech = extract_mechanism(pred_output)
        if not isinstance(mech, list):
            per_reaction.append(
                {"reaction_id": rxn_id, "S_total": 0.0, "S_partial": 0.0, "V": 0, "L": 0}
            )
            continue

        try:
            pred = [
                (s.get("subtype"), s.get("intermediate_smiles", ""))
                for s in mech
                if isinstance(s, dict)
            ]
            gold = [
                (s["subtype"], s["intermediate_smiles"], s["step_weight"])
                for s in gold_item["mechanism"]
            ]
            res = oMeS(gold, pred)
            per_reaction.append(
                {
                    "reaction_id": rxn_id,
                    "level": gold_item.get("level"),
                    "S_total": res.S_total,
                    "S_partial": res.S_partial,
                    "V": res.V,
                    "L": res.L,
                }
            )
            successful += 1
        except Exception as e:
            per_reaction.append(
                {
                    "reaction_id": rxn_id,
                    "S_total": 0.0,
                    "S_partial": 0.0,
                    "V": 0,
                    "L": 0,
                    "error": str(e),
                }
            )

    def _avg(key):
        vals = [r[key] for r in per_reaction if key in r]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary = {
        "model": model_label,
        "dataset": dataset,
        "evaluated_reactions": len(per_reaction),
        "successful_evaluations": successful,
        "avg_S_total": _avg("S_total"),
        "avg_S_partial": _avg("S_partial"),
        "avg_V": _avg("V"),
        "avg_L": _avg("L"),
    }

    # Per-difficulty breakdown (levels: easy/medium/hard when present).
    by_level: Dict[str, dict] = {}
    for r in per_reaction:
        lvl = r.get("level")
        if not lvl:
            continue
        by_level.setdefault(lvl, {"n": 0, "S_partial": 0.0, "S_total": 0.0})
        by_level[lvl]["n"] += 1
        by_level[lvl]["S_partial"] += r["S_partial"]
        by_level[lvl]["S_total"] += r["S_total"]
    for lvl, agg in by_level.items():
        n = max(1, agg["n"])
        agg["avg_S_partial"] = round(agg["S_partial"] / n, 3)
        agg["avg_S_total"] = round(agg["S_total"] / n, 3)
    summary["by_level"] = by_level

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w") as f:
        json.dump({"summary": summary, "per_reaction": per_reaction}, f, indent=2)

    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 56)
    print(f"  Model:   {summary['model']}")
    print(f"  Dataset: {summary['dataset']}")
    print("-" * 56)
    print(f"  S_partial : {summary['avg_S_partial']}")
    print(f"  S_total   : {summary['avg_S_total']}")
    print(f"  V (valid) : {summary['avg_V']}")
    print(f"  L (logic) : {summary['avg_L']}")
    print(
        f"  reactions : {summary['successful_evaluations']}/"
        f"{summary['evaluated_reactions']} scored"
    )
    if summary.get("by_level"):
        print("-" * 56)
        print("  By difficulty (avg S_partial):")
        for lvl, agg in summary["by_level"].items():
            print(f"    {lvl:<8} n={agg['n']:<4} S_partial={agg['avg_S_partial']}")
    print("=" * 56 + "\n")


def run_model(
    label: str,
    spec: ModelSpec,
    dataset: str,
    prompt_style: str,
    limit: Optional[int],
    concurrency: int,
    max_tokens: int,
    temperature: float,
    resume: bool = True,
) -> dict:
    provider = make_provider(spec, max_tokens=max_tokens, temperature=temperature)
    pred_path = RESULTS_DIR / dataset / f"{label}.predictions.jsonl"
    eval_path = RESULTS_DIR / dataset / f"{label}.eval.json"

    generate_predictions(
        provider,
        dataset=dataset,
        prompt_style=prompt_style,
        out_path=pred_path,
        limit=limit,
        concurrency=concurrency,
        resume=resume,
    )
    summary = evaluate_predictions(
        dataset=dataset,
        pred_path=pred_path,
        eval_path=eval_path,
        model_label=label,
        limit=limit,
    )
    print_summary(summary)
    return summary
