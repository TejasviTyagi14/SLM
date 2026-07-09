"""Phase 2 driver: canonicalize every source -> data/interim/<source>.norm.parquet,
resolve Tier-C names via OPSIN, build the InChIKey molecule vocabulary, and emit
the gate report (% molecules that sanitized cleanly + top failure reasons).

Usage:
    python -m rxndata.phase2_normalize [--only ...] [--json-out ...]
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from .config import load_config
from .io_utils import read_interim
from .normalize import SourceNormStats, has_rgroup, normalize_record
from .opsin_resolve import java_available, resolve_name
from .schema import records_to_rows

_ALL_SOURCES = ["ome_silver", "ome_template", "ord", "uspto_lowe", "wikipedia"]


def _write_norm(records: List[dict], source: str, cfg) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = cfg.path("interim")
    out_path = out_dir / f"{source}.norm.parquet"
    pq.write_table(pa.Table.from_pylist(records_to_rows(records)), out_path)
    return str(out_path.relative_to(cfg.path("root")))


def _resolve_wikipedia_names(records: List[dict], stats: dict) -> None:
    """Best-effort OPSIN resolution of Tier-C named reactions (in place)."""
    have_java = java_available()
    stats["java_available"] = have_java
    resolved = 0
    for r in records:
        name = r.get("name")
        if not name or name == "Name reaction":
            r["raw_ref"]["skip_reason"] = "meta_or_unnamed"
            continue
        nr = resolve_name(name)
        r["raw_ref"]["opsin"] = {"ok": nr.ok, "reason": nr.reason, "canonical": nr.canonical}
        # A named *reaction* rarely resolves to a single molecule (it's a process,
        # not a compound); OPSIN success here mostly happens for compound-like
        # names. We record the attempt honestly rather than fabricating products.
        if nr.ok:
            resolved += 1
    stats["names_resolved"] = resolved


def run(only: Optional[List[str]] = None) -> Dict[str, dict]:
    cfg = load_config()
    keep_maps = bool(cfg.get("normalization", "keep_atom_maps_through_canon", default=True))
    sources = only or _ALL_SOURCES

    results: Dict[str, dict] = {}
    # Global molecule vocabulary: inchikey -> {canonical, count, sources}
    vocab: Dict[str, dict] = {}

    for src in sources:
        try:
            records = read_interim(src)
        except Exception as e:  # noqa: BLE001
            results[src] = {"status": "missing", "error": str(e)}
            continue

        stats = SourceNormStats(source=src)
        norm_records: List[dict] = []
        for rec in records:
            nrec, res_list = normalize_record(rec, keep_maps=keep_maps)
            for res in res_list:
                stats.add(res)
                if res.ok and res.inchikey:
                    v = vocab.setdefault(
                        res.inchikey,
                        {"canonical": res.canonical, "count": 0, "sources": set()},
                    )
                    v["count"] += 1
                    v["sources"].add(src)
            norm_records.append(nrec)

        if src == "wikipedia":
            wiki_stats: Dict[str, object] = {}
            _resolve_wikipedia_names(norm_records, wiki_stats)
            extra = wiki_stats
        else:
            extra = {}

        parquet = _write_norm(norm_records, src, cfg)
        results[src] = {
            "status": "ok",
            "n_records": len(norm_records),
            "n_molecules": stats.n_mols,
            "n_sanitized": stats.n_ok,
            "pct_sanitized": round(stats.pct_ok, 2),
            "top_failure_reasons": dict(
                sorted(stats.reasons.items(), key=lambda x: -x[1])
            ),
            "parquet": parquet,
            **extra,
        }

    # Molecule vocabulary summary + artifact.
    unique_mols = len(vocab)
    rgroup_keys = sum(1 for k in vocab if k.startswith("SMI:") and has_rgroup(vocab[k]["canonical"]))
    vocab_out = cfg.path("interim") / "molecule_vocab.json"
    with vocab_out.open("w") as f:
        json.dump(
            {k: {"canonical": v["canonical"], "count": v["count"],
                 "sources": sorted(v["sources"])}
             for k, v in list(vocab.items())},
            f,
        )
    results["_vocab"] = {
        "unique_molecules": unique_mols,
        "rgroup_fragment_keys": rgroup_keys,
        "artifact": str(vocab_out.relative_to(cfg.path("root"))),
    }
    return results


def print_gate(results: Dict[str, dict]) -> None:
    print("\n" + "=" * 82)
    print("  PHASE 2 GATE — normalization (RDKit canonicalization + molecule vocab)")
    print("=" * 82)
    print(f"  {'source':<14}{'records':>9}{'molecules':>11}{'sanitized':>11}{'%ok':>8}  top failures")
    print("  " + "-" * 78)
    for src in _ALL_SOURCES:
        r = results.get(src)
        if not r or r["status"] != "ok":
            print(f"  {src:<14}{'-':>9}  {r.get('status','?') if r else 'missing'}")
            continue
        fails = ", ".join(f"{k}={v}" for k, v in list(r["top_failure_reasons"].items())[:3]) or "none"
        print(f"  {src:<14}{r['n_records']:>9}{r['n_molecules']:>11}"
              f"{r['n_sanitized']:>11}{r['pct_sanitized']:>7.1f}%  {fails}")
    print("  " + "-" * 78)
    v = results.get("_vocab", {})
    print(f"  molecule vocabulary: {v.get('unique_molecules')} unique keys "
          f"({v.get('rgroup_fragment_keys')} R-group fragments) -> {v.get('artifact')}")
    wiki = results.get("wikipedia", {})
    if wiki.get("status") == "ok":
        print(f"  OPSIN: java_available={wiki.get('java_available')} "
              f"names_resolved={wiki.get('names_resolved')}")
    print("=" * 82)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    results = run(only=args.only)
    print_gate(results)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
