"""Phase 9 driver: stratified splits + data card.

- Reads data/interim/deduped.parquet (clean, validated, decontaminated).
- Splits into train/val (stratified); test is the official oMe-Gold (referenced,
  never materialized here).
- Writes split id manifests to data/final/splits/.
- Assembles data/final/DATA_CARD.md from every phase's gate JSON: counts per
  source/tier/provenance/subtype/level, license inventory, decontamination
  summary, known gaps.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .config import load_config
from .schema import rows_to_records
from .splits import stratified_split


def _read(name, cfg):
    import pyarrow.parquet as pq

    path = cfg.path("interim") / f"{name}.parquet"
    if not path.exists():
        return []
    return rows_to_records(pq.read_table(path).to_pylist())


def _load_gate(name, cfg):
    p = cfg.path("interim") / f"_{name}_gate.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def run() -> Dict:
    cfg = load_config()
    seed = cfg.seed
    val_frac = float(cfg.get("splits", "val_fraction", default=0.05))

    records = _read("deduped", cfg)
    train, val, split_stats = stratified_split(records, val_frac, seed)

    splits_dir = cfg.path("final") / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train_ids.json").write_text(json.dumps([r["reaction_id"] for r in train]))
    (splits_dir / "val_ids.json").write_text(json.dumps([r["reaction_id"] for r in val]))

    # --- aggregate distributions for the data card ---
    prov = Counter(r["provenance"] for r in records)
    src = Counter(r["source"] for r in records)
    lvl = Counter(r.get("level") for r in records)
    subtype = Counter()
    for r in records:
        for s in r["mechanism"]:
            subtype[s["subtype"]] += 1

    license_inv = json.loads((cfg.path("configs") / "license_inventory.json").read_text())
    decon = json.loads((cfg.path("final") / "decontamination_report.json").read_text())
    onto = json.loads((cfg.path("ontology")).read_text())

    gates = {name: _load_gate(name, cfg) for name in
             ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "phase8"]}
    fc = _load_gate("format_check", cfg)
    if not fc:
        p = cfg.path("interim") / "_format_check.json"
        fc = json.loads(p.read_text()) if p.exists() else {}

    card = {
        "records_clean": len(records),
        "splits": split_stats,
        "provenance": dict(prov),
        "sources": dict(src),
        "levels": dict(lvl),
        "subtype_distribution": dict(subtype.most_common()),
        "ontology": {"n_types": onto["_meta"]["n_types"], "n_subtypes": onto["_meta"]["n_subtypes"]},
        "decontamination": {
            "removed": decon["removed"],
            "post_check_gold_inchikey_leaks": decon.get("post_check_gold_inchikey_leaks", "n/a"),
        },
        "format_check": fc,
        "gates": gates,
        "license_verdicts": {
            v["source"][:40]: v["verdict"] for v in license_inv.get("verdicts", [])
        },
    }
    card_path = cfg.path("final") / "data_card.json"
    card_path.write_text(json.dumps(card, indent=2))

    md = _render_data_card_md(card, cfg)
    md_path = cfg.path("final") / "DATA_CARD.md"
    md_path.write_text(md)

    return {
        "n_clean": len(records),
        "split_stats": split_stats,
        "data_card_json": str(card_path.relative_to(cfg.path("root"))),
        "data_card_md": str(md_path.relative_to(cfg.path("root"))),
        "train_ids": str((splits_dir / "train_ids.json").relative_to(cfg.path("root"))),
        "val_ids": str((splits_dir / "val_ids.json").relative_to(cfg.path("root"))),
        "provenance": dict(prov),
        "top_subtypes": dict(subtype.most_common(5)),
        "under_covered": [k for k in onto["subtypes"] if subtype.get(k, 0) < 5],
    }


def _render_data_card_md(card: Dict, cfg) -> str:
    lines: List[str] = []
    A = lines.append
    A("# Data Card — rxndata mechanism-step dataset\n")
    A("A training-ready dataset for a reaction-mechanism LM, optimized for "
      "oMeBench/oMeS. The core unit is a typed, balanced elementary mechanistic "
      "step with a valid intermediate SMILES.\n")

    A("## Headline\n")
    A(f"- Clean mechanism records (post validate+decon+dedup): **{card['records_clean']}**")
    A(f"- Train / Val: **{card['splits']['n_train']} / {card['splits']['n_val']}** "
      f"(stratified by dominant subtype + level; {card['splits']['n_strata']} strata)")
    A("- Test: the official **oMe-Gold** (196 rxns), held out — never trained on.")
    fc = card.get("format_check", {})
    if fc:
        A(f"- oMeS format-check: {fc.get('perfect')}/{fc.get('n_checked')} gold "
          f"targets score S_partial={fc.get('mean_S_partial')}, "
          f"S_total={fc.get('mean_S_total')} (ceiling).")
    A(f"- Ontology: {card['ontology']['n_types']} types / "
      f"{card['ontology']['n_subtypes']} subtypes (parsed from the benchmark).\n")

    A("## Provenance\n")
    for k, v in card["provenance"].items():
        A(f"- {k}: {v}")
    A("")
    A("## Sources (clean records)\n")
    for k, v in card["sources"].items():
        A(f"- {k}: {v}")
    A("")
    A("## Difficulty levels\n")
    for k, v in card["levels"].items():
        A(f"- {k}: {v}")
    A("")

    A("## License inventory\n")
    A("| Source | Verdict |")
    A("| --- | --- |")
    for s, verdict in card["license_verdicts"].items():
        A(f"| {s} | {verdict} |")
    A("\nNo paywalled/copyrighted sources are ingested. NC-ND sources "
      "(PMechDB, RMechDB) and the no-AI-clause source (OpenStax) are quarantined; "
      "LibreTexts (NC) is research-only and kept separable.\n")

    A("## Decontamination\n")
    d = card["decontamination"]
    A(f"- Removed as contaminated vs oMe-Gold+Template: **{d['removed']}**")
    A(f"- Post-check gold InChIKey leaks: **{d['post_check_gold_inchikey_leaks']}** "
      "(0 required).\n")

    A("## Ontology coverage (step subtype counts)\n")
    A("| subtype | steps |")
    A("| --- | ---: |")
    for st, n in card["subtype_distribution"].items():
        A(f"| {st} | {n} |")
    A("")

    A("## Task views (Phase 8)\n")
    p8 = card["gates"].get("phase8", {})
    if p8.get("sft_task_counts"):
        for t, n in p8["sft_task_counts"].items():
            A(f"- {t}: {n}")
        tc = p8.get("tokenizer_check", {})
        if tc.get("status") == "ok":
            A(f"\nQwen tokenizer ({tc['model_id']}): p50={tc['p50_tokens']} "
              f"p95={tc['p95_tokens']} max={tc['max_tokens']} tokens, "
              f"{tc['over_budget']}/{tc['checked']} over the {tc['max_len_budget']} budget.")
    A("")

    A("## Known gaps\n")
    A("- Tier-A elementary-step supply is limited to oMe-Silver + template "
      "expansion because PMechDB/RMechDB (the spec's primary sources) are "
      "CC-BY-NC-ND and cannot be redistributed as a derived corpus.")
    A("- Under-covered subtypes (<5 steps) may score poorly on oMeS; consider "
      "targeted expansion or licensed sourcing.")
    A("- A small set of curated mechanisms whose step 1 introduces reagents "
      "disconnected from the substrate (e.g. Appel/NG-007) are rejected by the "
      "continuity gate; recoverable with a reagent-introduction step type.")
    A("- Atom mapping (RXNMapper) confidence is modest on exotic charged "
      "intermediates; low-confidence maps are flagged, not trusted.\n")

    A("## Reproduce\n")
    A("```\nmake setup   # venv + pinned deps + clone oMeBench\nmake all     "
      "# phases 1-9\nmake eval-format-check\nmake test\n```\n")
    return "\n".join(lines)


def print_gate(res: Dict) -> None:
    print("\n" + "=" * 74)
    print("  PHASE 9 GATE — splits & data card")
    print("=" * 74)
    print(f"  clean records              : {res['n_clean']}")
    print(f"  train / val                : {res['split_stats']['n_train']} / "
          f"{res['split_stats']['n_val']}  "
          f"({res['split_stats']['n_strata']} strata, "
          f"val≈{res['split_stats']['val_fraction_actual']:.1%})")
    print("  test                       : official oMe-Gold (held out)")
    print(f"  provenance                 : {res['provenance']}")
    print(f"  top subtypes               : {res['top_subtypes']}")
    print(f"  under-covered (<5 steps)   : {res['under_covered']}")
    print("  " + "-" * 70)
    print(f"  data card (md)  -> {res['data_card_md']}")
    print(f"  data card (json)-> {res['data_card_json']}")
    print(f"  train ids       -> {res['train_ids']}")
    print(f"  val ids         -> {res['val_ids']}")
    print("=" * 74)


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
