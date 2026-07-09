"""Phase 7 driver: near-duplicate removal over the decontaminated records.

Reads data/interim/decontaminated.parquet, dedups (keeping the highest-provenance
representative per cluster), writes data/interim/deduped.parquet, and reports
before/after counts + the duplicate-cluster size distribution.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List

from .config import load_config
from .dedup import deduplicate
from .schema import records_to_rows, rows_to_records


def _read(name, cfg):
    import pyarrow.parquet as pq

    return rows_to_records(pq.read_table(cfg.path("interim") / f"{name}.parquet").to_pylist())


def _write(records: List[dict], name, cfg) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = cfg.path("interim") / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(records_to_rows(records)), path)
    return str(path.relative_to(cfg.path("root")))


def run() -> Dict:
    cfg = load_config()
    tau = float(cfg.get("dedup", "reaction_fp_tanimoto_threshold", default=0.97))
    records = _read("decontaminated", cfg)
    prov_before = Counter(r["provenance"] for r in records)
    kept, stats = deduplicate(records, tanimoto_threshold=tau)
    prov_after = Counter(r["provenance"] for r in kept)
    parquet = _write(kept, "deduped", cfg)
    return {
        **stats,
        "tanimoto_threshold": tau,
        "provenance_before": dict(prov_before),
        "provenance_after": dict(prov_after),
        "parquet": parquet,
    }


def print_gate(res: Dict) -> None:
    print("\n" + "=" * 70)
    print("  PHASE 7 GATE — near-duplicate deduplication")
    print("=" * 70)
    print(f"  records in                 : {res['n_in']}")
    print(f"  kept (unique)              : {res['n_kept']}")
    print(f"  removed (duplicates)       : {res['n_removed']}")
    print(f"  clusters                   : {res['n_clusters']}")
    print(f"  largest cluster            : {res['largest_cluster']}")
    print(f"  fp comparisons             : {res['fp_comparisons']}")
    print("  " + "-" * 66)
    print(f"  provenance before          : {res['provenance_before']}")
    print(f"  provenance after           : {res['provenance_after']}")
    print("  " + "-" * 66)
    print("  cluster-size distribution (size: #clusters):")
    for size, n in res["cluster_size_distribution"].items():
        print(f"    size {size:>3}: {n}")
    print(f"  -> {res['parquet']}")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    res = run()
    print_gate(res)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
