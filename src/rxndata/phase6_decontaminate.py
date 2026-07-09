"""Phase 6 driver: remove any candidate overlapping the oMe-Gold/Template blacklist.

Blacklist source = the RAW oMe-Gold + oMe-Template files (the actual held-out
test material), loaded directly so decontamination doesn't depend on our own
pipeline transforms. Candidates = the Phase-5 ``validated`` records.

Writes:
  data/interim/decontaminated.parquet          (clean training candidates)
  data/final/decontamination_report.json        (removed ids + reasons + counts)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List

from .config import load_config
from .decontaminate import build_blacklist, contamination_reason
from .schema import records_to_rows, rows_to_records


def _read_parquet(name: str, cfg):
    import pyarrow.parquet as pq

    path = cfg.path("interim") / f"{name}.parquet"
    return rows_to_records(pq.read_table(path).to_pylist())


def _load_raw_json(path):
    with open(path) as f:
        return json.load(f)


def _write_parquet(records: List[dict], name: str, cfg) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = cfg.path("interim") / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(records_to_rows(records)), path)
    return str(path.relative_to(cfg.path("root")))


def run() -> Dict:
    cfg = load_config()
    tau = float(cfg.get("decontamination", "tanimoto_threshold", default=0.95))

    gold = _load_raw_json(cfg.path("omebench_gold"))
    template = _load_raw_json(cfg.path("omebench_template"))
    blacklist = build_blacklist(gold, template)

    candidates = _read_parquet("validated", cfg)

    kept: List[dict] = []
    removed: List[dict] = []
    reason_counter: Counter = Counter()
    by_source_removed: Counter = Counter()

    for rec in candidates:
        reason = contamination_reason(rec, blacklist, tau)
        if reason is None:
            kept.append(rec)
        else:
            reason_counter[reason] += 1
            by_source_removed[rec.get("source")] += 1
            removed.append({
                "reaction_id": rec["reaction_id"],
                "source": rec.get("source"),
                "provenance": rec.get("provenance"),
                "reason": reason,
            })

    parquet = _write_parquet(kept, "decontaminated", cfg)

    report = {
        "blacklist_source": ["oMe-Gold", "oMe-Template"],
        "blacklist_entries": blacklist.n_entries,
        "tanimoto_threshold": tau,
        "candidates_in": len(candidates),
        "kept": len(kept),
        "removed": len(removed),
        "removed_by_reason": dict(reason_counter),
        "removed_by_source": dict(by_source_removed),
        "removed_ids": [r["reaction_id"] for r in removed],
    }
    report_path = cfg.path("final") / "decontamination_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    # Independent verification: confirm ZERO surviving records match gold by the
    # exact InChIKey signature (the strongest leak test).
    from .decontaminate import inchikey_signature
    gold_sigs = {s for s in (inchikey_signature(g.get("reactants_smiles", []),
                                                g.get("products_smiles", []))
                             for g in gold) if s}
    leaks = sum(1 for r in kept
                if inchikey_signature(r.get("reactants_smiles", []),
                                      r.get("products_smiles", [])) in gold_sigs)
    report["post_check_gold_inchikey_leaks"] = leaks

    return {
        **report,
        "parquet": parquet,
        "report_path": str(report_path.relative_to(cfg.path("root"))),
    }


def print_gate(res: Dict) -> None:
    print("\n" + "=" * 74)
    print("  PHASE 6 GATE — benchmark decontamination (oMe-Gold + Template)")
    print("=" * 74)
    print(f"  blacklist entries          : {res['blacklist_entries']}")
    print(f"  tanimoto threshold         : {res['tanimoto_threshold']}")
    print(f"  candidates in              : {res['candidates_in']}")
    print(f"  kept (clean)               : {res['kept']}")
    print(f"  REMOVED (contaminated)     : {res['removed']}")
    print("  " + "-" * 70)
    print("  removed by reason:")
    for reason, n in res["removed_by_reason"].items():
        print(f"    {n:>5}  {reason}")
    if not res["removed_by_reason"]:
        print("    (none)")
    print("  removed by source:")
    for src, n in res["removed_by_source"].items():
        print(f"    {n:>5}  {src}")
    print("  " + "-" * 70)
    leaks = res["post_check_gold_inchikey_leaks"]
    status = "PASS (zero gold overlap)" if leaks == 0 else f"FAIL ({leaks} leaks!)"
    print(f"  post-check gold InChIKey leaks : {leaks}  -> {status}")
    print(f"  report -> {res['report_path']}")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    res = run()
    print_gate(res)
    if args.json_out:
        summary = {k: v for k, v in res.items() if k != "removed_ids"}
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
