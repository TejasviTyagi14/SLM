"""Phase 5 driver: run validation gates over concrete mechanism records.

Splits the Phase-3 output into:
  - PASS  -> data/interim/validated.parquet   (100% must clear every gate)
  - REJECT -> data/rejects/phase5_invalid.jsonl (with reason codes)
  - TEMPLATE POOL -> data/interim/templates.parquet (is_template records; kept as
    reference/decon material, not validated as concrete reactions)

Gate report: pass/reject counts + a reason-code breakdown + the checks summary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List

from .config import load_config
from .schema import records_to_rows, rows_to_records
from .validate import validate_record


def _read(name: str, cfg):
    import pyarrow.parquet as pq

    path = cfg.path("interim") / f"{name}.parquet"
    return rows_to_records(pq.read_table(path).to_pylist())


def _write(records: List[dict], name: str, cfg) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = cfg.path("interim") / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(records_to_rows(records)), path)
    return str(path.relative_to(cfg.path("root")))


def run(source_parquet: str = "mechanisms") -> Dict:
    cfg = load_config()
    records = _read(source_parquet, cfg)

    templates = [r for r in records if r.get("is_template")]
    concrete = [r for r in records if not r.get("is_template")]

    passed: List[dict] = []
    rejects: List[dict] = []
    reason_counter: Counter = Counter()
    check_counter: Counter = Counter()

    for rec in concrete:
        v = validate_record(rec)
        for k, ok in v.checks.items():
            if ok:
                check_counter[k] += 1
        if v.ok:
            out = dict(rec)
            out["checks"] = {
                "sanitized": v.checks.get("sanitized", False),
                "atom_balanced": v.checks.get("atom_balanced", False),
                "charge_balanced": v.checks.get("charge_balanced", False),
                "step_continuity": v.checks.get("step_continuity", False),
            }
            passed.append(out)
        else:
            for r in v.reasons:
                reason_counter[r.split("(")[0]] += 1
            rejects.append({
                "reaction_id": rec["reaction_id"],
                "source": rec.get("source"),
                "provenance": rec.get("provenance"),
                "reasons": v.reasons,
            })

    passed_parquet = _write(passed, "validated", cfg)
    tmpl_parquet = _write(templates, "templates", cfg)
    rej_path = cfg.path("rejects") / "phase5_invalid.jsonl"
    with rej_path.open("w") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    prov_pass = Counter(r["provenance"] for r in passed)
    return {
        "n_concrete": len(concrete),
        "n_passed": len(passed),
        "n_rejected": len(rejects),
        "n_template_pool": len(templates),
        "pass_rate": round(len(passed) / max(1, len(concrete)), 4),
        "passed_by_provenance": dict(prov_pass),
        "reject_reason_breakdown": dict(reason_counter.most_common()),
        "check_pass_counts": dict(check_counter),
        "validated_parquet": passed_parquet,
        "templates_parquet": tmpl_parquet,
        "rejects_path": str(rej_path.relative_to(cfg.path("root"))),
    }


def print_gate(res: Dict) -> None:
    print("\n" + "=" * 74)
    print("  PHASE 5 GATE — validation (reject, don't fix silently)")
    print("=" * 74)
    print(f"  concrete records in        : {res['n_concrete']}")
    print(f"  PASSED (-> validated)      : {res['n_passed']}  ({res['pass_rate']:.1%})")
    print(f"  REJECTED                   : {res['n_rejected']}  -> {res['rejects_path']}")
    print(f"  template pool (R-groups)   : {res['n_template_pool']}  -> {res['templates_parquet']}")
    print("  " + "-" * 70)
    print(f"  passed by provenance       : {res['passed_by_provenance']}")
    print("  " + "-" * 70)
    print("  reject reason breakdown:")
    for reason, n in res["reject_reason_breakdown"].items():
        print(f"    {n:>5}  {reason}")
    if not res["reject_reason_breakdown"]:
        print("    (none)")
    print("  " + "-" * 70)
    print("  per-check pass counts (of concrete):")
    for k, n in res["check_pass_counts"].items():
        print(f"    {k:<18} {n:>6}")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="mechanisms", help="interim parquet stem to validate")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    res = run(source_parquet=args.source)
    print_gate(res)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
