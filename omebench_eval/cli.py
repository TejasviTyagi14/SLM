"""Command-line entrypoint for the oMeBench API evaluation harness.

Examples
--------
  # Evaluate Opus and GPT-5.5 on the 196-reaction gold set (chain-of-thought):
  python -m omebench_eval.cli run --models opus gpt-5.5 --dataset gold --prompt cot

  # Quick smoke test on 5 reactions:
  python -m omebench_eval.cli run --models opus --dataset gold --limit 5

  # Score an existing predictions file without re-querying:
  python -m omebench_eval.cli score --dataset gold --pred results/gold/opus.predictions.jsonl --label opus

  # Print a leaderboard from all eval JSONs:
  python -m omebench_eval.cli report --dataset gold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import DATASETS
from .providers import MODEL_REGISTRY, resolve_spec
from .runner import (
    RESULTS_DIR,
    evaluate_predictions,
    print_summary,
    run_model,
)


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=f"Model aliases (or raw ids with --provider/--model-id). "
        f"Known aliases: {list(MODEL_REGISTRY)}",
    )
    p.add_argument("--dataset", default="gold", choices=list(DATASETS))
    p.add_argument(
        "--prompt", default="cot", choices=["default", "cot"],
        help="Prompt style: 'default' (direct JSON) or 'cot' (reason then answer).",
    )
    p.add_argument("--limit", type=int, default=None, help="Only run first N reactions.")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16000,
                   help="Output token budget. Keep high for reasoning models: "
                        "reasoning tokens count against this and can starve the answer.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--provider", default=None, help="Override provider for a raw model id.")
    p.add_argument("--model-id", default=None, help="Override the concrete model id.")
    p.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    p.add_argument("--thinking-budget", type=int, default=None,
                   help="Anthropic extended-thinking token budget.")
    p.add_argument("--no-resume", action="store_true", help="Ignore existing predictions.")


def cmd_run(args: argparse.Namespace) -> None:
    summaries = []
    for name in args.models:
        spec = resolve_spec(
            name,
            model_id=args.model_id if len(args.models) == 1 else None,
            provider=args.provider if len(args.models) == 1 else None,
            reasoning_effort=args.reasoning_effort,
            thinking_budget=args.thinking_budget,
        )
        label = name.replace("/", "_")
        print(f"\n>>> Running '{label}' ({spec.provider}:{spec.model_id})")
        summary = run_model(
            label=label,
            spec=spec,
            dataset=args.dataset,
            prompt_style=args.prompt,
            limit=args.limit,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            resume=not args.no_resume,
        )
        summaries.append(summary)
    if len(summaries) > 1:
        _print_leaderboard(summaries)


def cmd_score(args: argparse.Namespace) -> None:
    pred_path = Path(args.pred)
    eval_path = pred_path.with_suffix("").with_suffix(".eval.json")
    summary = evaluate_predictions(
        dataset=args.dataset,
        pred_path=pred_path,
        eval_path=eval_path,
        model_label=args.label,
        limit=args.limit,
    )
    print_summary(summary)


def cmd_report(args: argparse.Namespace) -> None:
    eval_dir = RESULTS_DIR / args.dataset
    summaries = []
    if eval_dir.exists():
        for f in sorted(eval_dir.glob("*.eval.json")):
            data = json.loads(f.read_text())
            summaries.append(data["summary"])
    if not summaries:
        print(f"No eval results found in {eval_dir}")
        return
    _print_leaderboard(summaries)


def _print_leaderboard(summaries: list[dict]) -> None:
    summaries = sorted(summaries, key=lambda s: s["avg_S_partial"], reverse=True)
    print("\n" + "=" * 72)
    print(f"  oMeBench leaderboard  (dataset: {summaries[0]['dataset']})")
    print("=" * 72)
    print(f"  {'Model':<24}{'S_partial':>11}{'S_total':>10}{'V':>7}{'L':>7}")
    print("-" * 72)
    for s in summaries:
        print(
            f"  {s['model']:<24}{s['avg_S_partial']:>11}{s['avg_S_total']:>10}"
            f"{s['avg_V']:>7}{s['avg_L']:>7}"
        )
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="omebench_eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Generate predictions and score them.")
    _add_run_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_score = sub.add_parser("score", help="Score an existing predictions file.")
    p_score.add_argument("--dataset", default="gold", choices=list(DATASETS))
    p_score.add_argument("--pred", required=True)
    p_score.add_argument("--label", required=True)
    p_score.add_argument("--limit", type=int, default=None)
    p_score.set_defaults(func=cmd_score)

    p_report = sub.add_parser("report", help="Print a leaderboard from eval JSONs.")
    p_report.add_argument("--dataset", default="gold", choices=list(DATASETS))
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
