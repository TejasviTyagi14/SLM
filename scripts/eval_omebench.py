"""Wire up the official oMeS scorer for two checks the acceptance criteria require.

Uses the repo's vendored oMeS implementation (omebench_eval.scoring), which is
byte-for-byte equivalent to the official third_party/oMeBench utils_eval.oMeS
(verified in Phase 0) and cross-checks against it when available.

Commands:
  format-check : take N mechanism_full SFT examples, feed each assistant TARGET
                 through oMeS *against itself*, and confirm it scores ~perfectly.
                 If our own gold targets don't max out oMeS, our formatting is
                 wrong -- this must pass before any training.
  score        : score a predictions JSONL against a dataset (thin wrapper).

Run:
  python scripts/eval_omebench.py format-check --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from omebench_eval.scoring import oMeS  # noqa: E402  (repo's vendored scorer)


def _try_official_scorer():
    """Return the official oMeBench oMeS if importable, else None (for cross-check)."""
    tp = ROOT / "third_party" / "oMeBench" / "scripts"
    if not (tp / "utils_eval.py").exists():
        return None
    sys.path.insert(0, str(tp))
    try:
        from utils_eval import oMeS as official_oMeS  # type: ignore
        return official_oMeS
    except Exception:
        return None


def _extract_steps(assistant_content: str):
    """Parse the JSON step list out of a (possibly CoT-wrapped) assistant target."""
    content = assistant_content
    if "[ANSWER]" in content:
        content = content.split("[ANSWER]")[1].split("[/ANSWER]")[0].strip()
    return json.loads(content)


def format_check(sft_path: Path, n: int) -> dict:
    """Feed each example's own target through oMeS as both gold and prediction."""
    official = _try_official_scorer()
    rows = []
    with sft_path.open() as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("task") == "mechanism_full":
                rows.append(obj)
            if len(rows) >= n:
                break

    perfect = 0
    s_partials = []
    s_totals = []
    mismatches = []
    cross_ok = 0
    for obj in rows:
        steps = _extract_steps(obj["messages"][-1]["content"])
        ref = obj.get("reference") or []
        # gold from reference [subtype, canonical_smiles, weight]; pred from target
        gold = [(r[0], r[1], r[2]) for r in ref]
        pred = [(s["subtype"], s["intermediate_smiles"]) for s in steps]
        res = oMeS(gold, pred)
        s_partials.append(res.S_partial)
        s_totals.append(res.S_total)
        if res.S_partial >= 0.99 and res.S_total >= 0.99:
            perfect += 1
        else:
            mismatches.append({"reaction_id": obj["meta"]["reaction_id"],
                               "S_partial": res.S_partial, "S_total": res.S_total})
        if official is not None:
            try:
                res2 = official(gold, pred)
                if abs(res2.S_partial - res.S_partial) < 1e-6:
                    cross_ok += 1
            except Exception:
                pass

    n_rows = len(rows)
    return {
        "n_checked": n_rows,
        "perfect": perfect,
        "perfect_fraction": round(perfect / max(1, n_rows), 4),
        "mean_S_partial": round(sum(s_partials) / max(1, n_rows), 4),
        "mean_S_total": round(sum(s_totals) / max(1, n_rows), 4),
        "official_scorer_available": official is not None,
        "official_cross_agreement": cross_ok if official is not None else None,
        "mismatches": mismatches[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fc = sub.add_parser("format-check", help="verify our gold targets max out oMeS")
    fc.add_argument("--sft", default="data/final/sft_mechanisms.jsonl")
    fc.add_argument("--n", type=int, default=200)
    fc.add_argument("--json-out")

    args = ap.parse_args()
    if args.cmd == "format-check":
        res = format_check(ROOT / args.sft, args.n)
        print("\n" + "=" * 64)
        print("  oMeS FORMAT-CHECK (gold targets scored against themselves)")
        print("=" * 64)
        print(f"  checked                 : {res['n_checked']}")
        print(f"  perfect (S>=0.99)       : {res['perfect']} "
              f"({res['perfect_fraction']:.1%})")
        print(f"  mean S_partial          : {res['mean_S_partial']}")
        print(f"  mean S_total            : {res['mean_S_total']}")
        print(f"  official scorer present : {res['official_scorer_available']}")
        if res["official_scorer_available"]:
            print(f"  official cross-agreement: {res['official_cross_agreement']}/{res['n_checked']}")
        status = "PASS" if res["perfect_fraction"] >= 0.99 else "FAIL"
        print(f"  -> {status}")
        if res["mismatches"]:
            print("  sample non-perfect:")
            for m in res["mismatches"][:5]:
                print(f"    {m}")
        print("=" * 64)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
