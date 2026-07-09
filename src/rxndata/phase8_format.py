"""Phase 8 driver: emit SFT JSONL corpora + a continued-pretraining corpus.

Inputs:
  - data/interim/deduped.parquet        (clean, validated mechanism records)
  - data/interim/uspto_lowe.norm.parquet, ord.norm.parquet  (Tier B overall rxns)
  - data/interim/wikipedia.norm.parquet (Tier C prose)

Outputs (data/final/):
  - sft_mechanisms.jsonl   (mechanism_full + mechanism_next_step; the core product)
  - sft_reactions.jsonl    (forward/retro/reagent from Tier A + B)
  - sft_named_qa.jsonl     (Tier C named-reaction Q&A)
  - pretrain_corpus.txt    (dedup plain text: reaction SMILES lines + step seqs + prose)

Self-check: every mechanism_full assistant target is parsed back and must yield
valid JSON with ontology-valid subtypes and RDKit-valid SMILES (the training
targets must re-pass the validators the benchmark applies). A Qwen tokenizer
length check flags any example exceeding the context budget.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from rdkit import Chem, RDLogger

from .config import load_config
from .format_tasks import (
    make_forward,
    make_mechanism_full,
    make_named_qa,
    make_next_step,
    make_reagent,
    make_retro,
)
from .ontology import load_ontology
from .schema import rows_to_records

RDLogger.DisableLog("rdApp.*")


def _read(name, cfg):
    import pyarrow.parquet as pq

    path = cfg.path("interim") / f"{name}.parquet"
    if not path.exists():
        return []
    return rows_to_records(pq.read_table(path).to_pylist())


def _load_template(cfg, style: str) -> str:
    name = "cot" if style == "cot" else "default"
    return (cfg.path("root") / "prompts" / f"{name}.txt").read_text()


def _write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _selfcheck_mechanism_full(rows: List[dict], allowed_subtypes: set) -> Dict:
    """Every mechanism_full target must be valid JSON + ontology + SMILES."""
    bad_json = bad_subtype = bad_smiles = 0
    for r in rows:
        content = r["messages"][-1]["content"]
        if "[ANSWER]" in content:
            content = content.split("[ANSWER]")[1].split("[/ANSWER]")[0].strip()
        try:
            steps = json.loads(content)
        except Exception:
            bad_json += 1
            continue
        for s in steps:
            if s.get("subtype") not in allowed_subtypes:
                bad_subtype += 1
            if Chem.MolFromSmiles(s.get("intermediate_smiles", "")) is None:
                bad_smiles += 1
    return {"n": len(rows), "bad_json": bad_json,
            "bad_subtype": bad_subtype, "bad_smiles": bad_smiles}


def _tokenizer_check(rows: List[dict], model_id: str, max_len: int) -> Dict:
    """Optional Qwen tokenizer length check; degrades if transformers absent."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "reason": f"{type(e).__name__}: {e}"}
    lengths = []
    over = 0
    for r in rows[:2000]:
        try:
            text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
            n = len(tok(text)["input_ids"])
        except Exception:
            text = "\n".join(m["content"] for m in r["messages"])
            n = len(tok(text)["input_ids"])
        lengths.append(n)
        if n > max_len:
            over += 1
    lengths.sort()
    return {
        "status": "ok",
        "model_id": model_id,
        "checked": len(lengths),
        "max_len_budget": max_len,
        "over_budget": over,
        "p50_tokens": lengths[len(lengths) // 2] if lengths else 0,
        "p95_tokens": lengths[int(len(lengths) * 0.95)] if lengths else 0,
        "max_tokens": lengths[-1] if lengths else 0,
    }


def run(style: str = "cot", tokenizer_model: Optional[str] = None,
        max_len: int = 8192) -> Dict:
    cfg = load_config()
    onto = load_ontology(cfg.path("ontology"))
    allowed_subtypes = set(onto["subtypes"])
    template = _load_template(cfg, style)

    mechanisms = _read("deduped", cfg)
    tier_b = _read("uspto_lowe.norm", cfg) + _read("ord.norm", cfg)
    tier_c = _read("wikipedia.norm", cfg)

    # --- core mechanism SFT ---
    mech_rows: List[dict] = []
    for rec in mechanisms:
        row = make_mechanism_full(rec, template, style)
        if row:
            mech_rows.append(row)
        mech_rows.extend(make_next_step(rec))

    # --- reaction tasks (forward/retro/reagent) from Tier A mechanisms + Tier B ---
    rxn_rows: List[dict] = []
    for rec in mechanisms + tier_b:
        for fn in (make_forward, make_retro, make_reagent):
            row = fn(rec)
            if row:
                rxn_rows.append(row)

    # --- named-reaction QA (Tier C prose) ---
    qa_rows = [r for r in (make_named_qa(rec) for rec in tier_c) if r]

    # --- write SFT corpora ---
    final = cfg.path("final")
    _write_jsonl(mech_rows, final / "sft_mechanisms.jsonl")
    _write_jsonl(rxn_rows, final / "sft_reactions.jsonl")
    _write_jsonl(qa_rows, final / "sft_named_qa.jsonl")

    # --- continued-pretraining corpus (dedup plain text) ---
    pre_lines: set = set()
    for rec in mechanisms:
        if rec.get("reaction_smiles_mapped"):
            pre_lines.add(rec["reaction_smiles_mapped"])
        seq = " . ".join(s["intermediate_smiles"] for s in rec.get("mechanism", []))
        if seq:
            pre_lines.add(seq)
    for rec in tier_b:
        if rec.get("reaction_smiles_mapped"):
            pre_lines.add(rec["reaction_smiles_mapped"])
    for rec in tier_c:
        summ = rec.get("raw_ref", {}).get("summary")
        if summ:
            pre_lines.add(summ)
    pretrain_path = final / "pretrain_corpus.txt"
    with pretrain_path.open("w") as f:
        for line in sorted(pre_lines):
            f.write(line + "\n")

    # --- self-check + token stats ---
    mf_rows = [r for r in mech_rows if r["task"] == "mechanism_full"]
    selfcheck = _selfcheck_mechanism_full(mf_rows, allowed_subtypes)
    tok_model = tokenizer_model or cfg.get("tasks", "tokenizer_model", default=None)
    token_check = (
        _tokenizer_check(mf_rows, tok_model, max_len)
        if tok_model else {"status": "not_requested"}
    )

    # rough token count of the pretrain corpus (whitespace proxy)
    pretrain_tokens = sum(len(line.split()) for line in pre_lines)

    task_counts = Counter(r["task"] for r in (mech_rows + rxn_rows + qa_rows))
    return {
        "style": style,
        "sft_task_counts": dict(task_counts),
        "n_mechanism_sft": len(mech_rows),
        "n_reaction_sft": len(rxn_rows),
        "n_named_qa_sft": len(qa_rows),
        "pretrain_lines": len(pre_lines),
        "pretrain_approx_tokens": pretrain_tokens,
        "mechanism_full_selfcheck": selfcheck,
        "tokenizer_check": token_check,
        "outputs": {
            "sft_mechanisms": str((final / "sft_mechanisms.jsonl").relative_to(cfg.path("root"))),
            "sft_reactions": str((final / "sft_reactions.jsonl").relative_to(cfg.path("root"))),
            "sft_named_qa": str((final / "sft_named_qa.jsonl").relative_to(cfg.path("root"))),
            "pretrain_corpus": str(pretrain_path.relative_to(cfg.path("root"))),
        },
    }


def print_gate(res: Dict) -> None:
    print("\n" + "=" * 74)
    print("  PHASE 8 GATE — task formatting (SFT + pretrain)")
    print("=" * 74)
    print(f"  style                      : {res['style']}")
    print("  SFT task counts:")
    for t, n in res["sft_task_counts"].items():
        print(f"    {t:<22} {n:>7}")
    print("  " + "-" * 70)
    sc = res["mechanism_full_selfcheck"]
    ok = sc["bad_json"] == 0 and sc["bad_subtype"] == 0 and sc["bad_smiles"] == 0
    print(f"  mechanism_full self-check  : n={sc['n']} bad_json={sc['bad_json']} "
          f"bad_subtype={sc['bad_subtype']} bad_smiles={sc['bad_smiles']}  "
          f"{'PASS' if ok else 'FAIL'}")
    tc = res["tokenizer_check"]
    if tc.get("status") == "ok":
        print(f"  tokenizer ({tc['model_id']}): p50={tc['p50_tokens']} p95={tc['p95_tokens']} "
              f"max={tc['max_tokens']} over_budget={tc['over_budget']}/{tc['checked']}")
    else:
        print(f"  tokenizer check            : {tc.get('status')} ({tc.get('reason','')})")
    print(f"  pretrain corpus            : {res['pretrain_lines']} lines, "
          f"~{res['pretrain_approx_tokens']} tokens")
    print("  " + "-" * 70)
    for k, v in res["outputs"].items():
        print(f"    {k:<16} -> {v}")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", default="cot", choices=["default", "cot"])
    ap.add_argument("--tokenizer-model", default=None,
                    help="HF id for a length check, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    res = run(style=args.style, tokenizer_model=args.tokenizer_model, max_len=args.max_len)
    print_gate(res)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[wrote] {args.json_out}")


if __name__ == "__main__":
    main()
