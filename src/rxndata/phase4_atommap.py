"""Phase 4 driver: atom-map mechanism records; emit confidence histogram.

RXNMapper runs a transformer per reaction, so full mapping of every step is slow.
The driver supports a bounded ``--sample`` for the gate (confidence histogram on a
representative subset) and a full run for the actual dataset build
(``--sample 0``). Overall-reaction mapping is always attempted; per-step mapping
is controlled by config ``atommap.map_steps``.

Output: data/interim/mechanisms.mapped.parquet (+ the mapped sample when sampling).
Low-confidence maps are flagged ``_map_quarantine`` (kept, not dropped).
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from .atommap import map_record
from .config import load_config
from .schema import records_to_rows, rows_to_records


def _read(name: str, cfg):
    import pyarrow.parquet as pq

    path = cfg.path("interim") / f"{name}.parquet"
    return rows_to_records(pq.read_table(path).to_pylist())


def _write(records: List[dict], name: str, cfg) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = cfg.path("interim") / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(records_to_pylist(records)), path)
    return str(path.relative_to(cfg.path("root")))


def records_to_pylist(records):
    return records_to_rows(records)


def _histogram(confidences: List[float]) -> Dict[str, int]:
    bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for c in confidences:
        if c < 0.2:
            bins["0.0-0.2"] += 1
        elif c < 0.4:
            bins["0.2-0.4"] += 1
        elif c < 0.6:
            bins["0.4-0.6"] += 1
        elif c < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    return bins


def run(sample: int = 200) -> Dict:
    cfg = load_config()
    min_conf = float(cfg.get("atommap", "min_confidence", default=0.30))
    map_steps = bool(cfg.get("atommap", "map_steps", default=True))

    records = _read("mechanisms", cfg)
    # Skip template records with R-groups (can't be mapped); map concrete ones.
    concrete = [r for r in records if not r.get("is_template")]
    to_map = concrete if sample in (0, None) else concrete[:sample]

    mapped = []
    overall_conf: List[float] = []
    step_conf: List[float] = []
    n_quar = 0
    for rec in to_map:
        m = map_record(rec, min_confidence=min_conf, map_steps=map_steps)
        mapped.append(m)
        if m.get("_map_confidence") is not None:
            overall_conf.append(m["_map_confidence"])
        if m.get("_map_quarantine"):
            n_quar += 1
        for st in m.get("mechanism", []):
            if st.get("_step_map_confidence") is not None:
                step_conf.append(st["_step_map_confidence"])

    out_name = "mechanisms.mapped" if sample in (0, None) else "mechanisms.mapped_sample"
    parquet = _write(mapped, out_name, cfg)

    return {
        "sampled": sample,
        "n_mapped_records": len(mapped),
        "n_low_confidence_quarantined": n_quar,
        "min_confidence": min_conf,
        "overall_confidence_histogram": _histogram(overall_conf),
        "overall_mean_confidence": round(sum(overall_conf) / len(overall_conf), 4) if overall_conf else None,
        "step_confidence_histogram": _histogram(step_conf),
        "step_mean_confidence": round(sum(step_conf) / len(step_conf), 4) if step_conf else None,
        "n_steps_mapped": len(step_conf),
        "parquet": parquet,
    }


def print_gate(res: Dict) -> None:
    print("\n" + "=" * 72)
    print("  PHASE 4 GATE — atom mapping (RXNMapper)")
    print("=" * 72)
    print(f"  sampled                     : {res['sampled']} (0 = full)")
    print(f"  records mapped              : {res['n_mapped_records']}")
    print(f"  low-confidence quarantined  : {res['n_low_confidence_quarantined']} "
          f"(< {res['min_confidence']})")
    print(f"  overall map mean confidence : {res['overall_mean_confidence']}")
    print("  overall confidence histogram:")
    for b, n in res["overall_confidence_histogram"].items():
        bar = "#" * min(50, n)
        print(f"    {b}  {n:>5}  {bar}")
    print(f"  step map mean confidence    : {res['step_mean_confidence']} "
          f"(over {res['n_steps_mapped']} steps)")
    print("  step confidence histogram:")
    for b, n in res["step_confidence_histogram"].items():
        bar = "#" * min(50, n // 5)
        print(f"    {b}  {n:>5}  {bar}")
    print(f"  -> {res['parquet']}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=200, help="records to map (0 = all)")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    res = run(sample=args.sample)
    print_gate(res)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
