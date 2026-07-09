"""Phase 3 driver: decomposition & typing.

Pipeline:
1. Load normalized records for every source.
2. Tier-A mechanism sources (silver, template): type_and_remap each record onto
   the parsed ontology; records with an unmappable step go to rejects.
3. Template expansion: expand the uncovered-by-silver R-group templates into
   bounded, deduplicated concrete instances (provenance=template_expanded),
   dedup against silver's molecule set.
4. Inference: for Tier-B/C overall reactions that match a curated template by
   name, attach an inferred typed step pattern (provenance=inferred), capped at
   ``inferred_fraction_cap`` of the total mechanism records and with a held-back
   review sample.
5. Write data/interim/mechanisms.parquet + rejects; emit the gate report:
   ontology distribution, under-covered subtypes, 10 fully-decomposed examples.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List

from .config import load_config
from .infer import infer_mechanisms
from .mechanism import expand_template, type_and_remap
from .ontology import load_ontology
from .schema import records_to_rows


def _read_norm(source: str, cfg) -> List[dict]:
    import pyarrow.parquet as pq

    from .schema import rows_to_records

    path = cfg.path("interim") / f"{source}.norm.parquet"
    if not path.exists():
        return []
    return rows_to_records(pq.read_table(path).to_pylist())


def _write_parquet(records: List[dict], name: str, cfg) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = cfg.path("interim") / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(records_to_rows(records)), path)
    return str(path.relative_to(cfg.path("root")))


def _write_rejects(rejects: List[dict], cfg) -> str:
    path = cfg.path("rejects") / "phase3_unmapped.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path.relative_to(cfg.path("root")))


def run() -> Dict:
    cfg = load_config()
    onto = load_ontology(cfg.path("ontology"))
    allowed_subtypes = list(onto["subtypes"])

    kept: List[dict] = []
    rejects: List[dict] = []
    counts: Dict[str, int] = {}

    # --- Tier A: type + remap curated/expanded mechanism records -------------
    silver = _read_norm("ome_silver", cfg)
    template = _read_norm("ome_template", cfg)

    for src_name, recs in [("ome_silver", silver), ("ome_template", template)]:
        n_ok = n_rej = 0
        for rec in recs:
            typed, reason = type_and_remap(rec, onto)
            if typed is None:
                rejects.append({"reaction_id": rec["reaction_id"], "source": src_name,
                                "reason": reason})
                n_rej += 1
            else:
                kept.append(typed)
                n_ok += 1
        counts[f"{src_name}_typed_ok"] = n_ok
        counts[f"{src_name}_rejected"] = n_rej

    # Silver's molecule set (to avoid re-minting silver during expansion).
    silver_mols = set()
    for r in silver:
        silver_mols.update(r.get("products_smiles", []))

    # --- Template expansion (uncovered-by-silver templates only) -------------
    exp_cfg = cfg.get("mechanism", "template_expansion", default={}) or {}
    expanded: List[dict] = []
    if exp_cfg.get("enabled", True):
        max_inst = int(exp_cfg.get("max_instances_per_template", 40))
        only_uncovered = bool(exp_cfg.get("only_uncovered_by_silver", True))
        silver_bases = {r["raw_ref"].get("template_base") for r in silver}
        for rec in template:
            base = rec["raw_ref"].get("orig_id")
            if only_uncovered and base in silver_bases:
                continue
            # Type+remap the template first so expansions inherit clean labels.
            typed, reason = type_and_remap(rec, onto)
            if typed is None:
                continue
            insts = expand_template(typed, max_instances=max_inst,
                                    exclude_canonical=silver_mols, ontology=onto)
            expanded.extend(insts)
    counts["template_expanded"] = len(expanded)
    kept.extend(expanded)

    # --- Inference (capped) for Tier-B/C overall reactions -------------------
    inf_cfg = cfg.get("mechanism", "inference", default={}) or {}
    inferred: List[dict] = []
    review_sample: List[dict] = []
    if inf_cfg.get("enabled", True):
        cap_frac = float(cfg.get("mechanism", "inferred_fraction_cap", default=0.20))
        # cap is a fraction of TOTAL mechanism records (kept so far are all mechanism records)
        n_mech_so_far = len(kept)
        max_inferred = int(cap_frac / (1 - cap_frac) * n_mech_so_far)
        candidates = _read_norm("wikipedia", cfg) + _read_norm("ord", cfg) + _read_norm("uspto_lowe", cfg)
        inferred = infer_mechanisms(candidates, template, max_records=max_inferred, cfg=cfg)
        # type+remap inferred too (they use template subtypes, should be clean)
        typed_inferred = []
        for rec in inferred:
            typed, reason = type_and_remap(rec, onto)
            if typed is not None:
                typed_inferred.append(typed)
        inferred = typed_inferred
        kept.extend(inferred)
        rs = int(inf_cfg.get("review_sample_size", 25))
        review_sample = inferred[:rs]
    counts["inferred"] = len(inferred)

    # --- Provenance + ontology distribution ---------------------------------
    prov = Counter(r["provenance"] for r in kept)
    subtype_dist = Counter()
    type_dist = Counter()
    for r in kept:
        for s in r["mechanism"]:
            subtype_dist[s["subtype"]] += 1
            type_dist[s["type"]] += 1
    under_covered = [st for st in allowed_subtypes if subtype_dist.get(st, 0) < 5]

    # --- Write outputs -------------------------------------------------------
    parquet = _write_parquet(kept, "mechanisms", cfg)
    rejects_path = _write_rejects(rejects, cfg)
    review_path = cfg.path("interim") / "phase3_inferred_review_sample.jsonl"
    with review_path.open("w") as f:
        for r in review_sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    inferred_frac = (prov["inferred"] / max(1, len(kept)))
    return {
        "n_kept": len(kept),
        "n_rejects": len(rejects),
        "provenance": dict(prov),
        "inferred_fraction": round(inferred_frac, 4),
        "inferred_cap": float(cfg.get("mechanism", "inferred_fraction_cap", default=0.20)),
        "counts": counts,
        "type_distribution": dict(type_dist.most_common()),
        "subtype_distribution": dict(subtype_dist.most_common()),
        "under_covered_subtypes": under_covered,
        "parquet": parquet,
        "rejects_path": rejects_path,
        "review_sample_path": str(review_path.relative_to(cfg.path("root"))),
        "examples": kept[:10],
    }


def print_gate(res: Dict) -> None:
    print("\n" + "=" * 82)
    print("  PHASE 3 GATE — mechanism decomposition & typing")
    print("=" * 82)
    print(f"  kept mechanism records : {res['n_kept']}")
    print(f"  rejected (unmapped)    : {res['n_rejects']}  -> {res['rejects_path']}")
    print(f"  provenance             : {res['provenance']}")
    print(f"  inferred fraction      : {res['inferred_fraction']:.1%} "
          f"(cap {res['inferred_cap']:.0%})  "
          f"{'OK' if res['inferred_fraction'] <= res['inferred_cap'] else 'OVER CAP!'}")
    print("  " + "-" * 78)
    print("  step TYPE distribution:")
    for t, n in res["type_distribution"].items():
        print(f"    {t:<18} {n:>6}")
    print("  " + "-" * 78)
    print("  step SUBTYPE distribution (all 31; the oMeS alignment key):")
    for st, n in res["subtype_distribution"].items():
        print(f"    {st:<30} {n:>6}")
    if res["under_covered_subtypes"]:
        print("  " + "-" * 78)
        print(f"  UNDER-COVERED subtypes (<5 steps): {res['under_covered_subtypes']}")
    print("=" * 82)
    print("\n  ---- 10 fully-decomposed examples ----")
    for ex in res["examples"]:
        steps = " | ".join(f"{s['step']}.{s['subtype']}" for s in ex["mechanism"])
        print(f"   [{ex['provenance']:<16}] {ex['reaction_id']}  ({ex.get('name')})")
        print(f"      R: {ex['reactants_smiles']} -> P: {ex['products_smiles']}")
        print(f"      steps: {steps}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    res = run()
    print_gate(res)
    if args.json_out:
        # examples can be large; write a trimmed summary
        summary = {k: v for k, v in res.items() if k != "examples"}
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
