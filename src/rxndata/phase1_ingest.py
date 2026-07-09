"""Phase 1 driver: run enabled + license-cleared ingesters -> interim parquet.

Gate output: a per-source row-count table (with tier / license / provenance),
a 5-row sample from each source, and a note for any source skipped because its
license verdict does not permit ingestion.

Usage:
    python -m rxndata.phase1_ingest [--only ome_silver ome_template] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

from .config import load_config
from .ingest import all_keys, get_module
from .ingest.base import sample_rows
from .io_utils import write_interim


def run(only: Optional[List[str]] = None, limit: Optional[int] = None) -> Dict[str, dict]:
    cfg = load_config()
    results: Dict[str, dict] = {}
    keys = only or list(all_keys())

    for key in keys:
        mod = get_module(key)
        info = mod.INFO
        src = cfg.raw["sources"].get(key, {})
        enabled = src.get("enabled", False)
        verdict = src.get("verdict", "pending")

        if not enabled:
            results[key] = {"status": "disabled", "verdict": verdict, "tier": info.tier}
            print(f"[skip] {info.display}: disabled in config (verdict={verdict})")
            continue
        if not info.gate_ok(cfg):
            results[key] = {"status": "blocked", "verdict": verdict, "tier": info.tier}
            print(f"[BLOCK] {info.display}: verdict={verdict} does not permit ingestion")
            continue

        print(f"[ingest] {info.display} (tier {info.tier}, {info.license}) ...", flush=True)
        try:
            records = mod.ingest(cfg, limit=limit)
        except Exception as e:  # noqa: BLE001
            results[key] = {"status": "error", "error": f"{type(e).__name__}: {e}", "tier": info.tier}
            print(f"[ERROR] {info.display}: {type(e).__name__}: {e}")
            continue

        out_path = write_interim(records, key, cfg)
        n_mech = sum(1 for r in records if r.get("mechanism"))
        results[key] = {
            "status": "ok",
            "tier": info.tier,
            "license": info.license,
            "verdict": verdict,
            "n_records": len(records),
            "n_with_mechanism": n_mech,
            "parquet": str(out_path.relative_to(cfg.path("root"))),
            "sample": sample_rows(records, 5),
        }
        print(f"[ingest] {info.display}: {len(records)} records "
              f"({n_mech} with mechanism) -> {results[key]['parquet']}")

    return results


def print_gate(results: Dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("  PHASE 1 GATE — ingestion row counts")
    print("=" * 78)
    print(f"  {'source':<16}{'tier':<6}{'license':<16}{'records':>9}{'w/mech':>9}  status")
    print("  " + "-" * 74)
    tier_tot: Dict[str, int] = {}
    for key, r in results.items():
        if r["status"] == "ok":
            tier_tot[r["tier"]] = tier_tot.get(r["tier"], 0) + r["n_records"]
            print(f"  {key:<16}{r['tier']:<6}{r['license']:<16}"
                  f"{r['n_records']:>9}{r['n_with_mechanism']:>9}  ok")
        else:
            print(f"  {key:<16}{r.get('tier',''):<6}{'':<16}{'':>9}{'':>9}  {r['status']}"
                  + (f" ({r.get('error') or r.get('verdict')})" if r['status'] != 'ok' else ""))
    print("  " + "-" * 74)
    for tier in sorted(tier_tot):
        print(f"  Tier {tier} total records: {tier_tot[tier]}")
    print("=" * 78)

    for key, r in results.items():
        if r["status"] != "ok":
            continue
        print(f"\n  ---- {key}: 5-row sample ----")
        for s in r["sample"]:
            print("   " + json.dumps(s, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="run only these source keys")
    ap.add_argument("--limit", type=int, default=None, help="cap records per source (for quick gates)")
    ap.add_argument("--json-out", help="write full results json here")
    args = ap.parse_args()

    results = run(only=args.only, limit=args.limit)
    print_gate(results)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
