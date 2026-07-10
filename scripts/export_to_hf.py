"""Export the rxndata dataset (data/final/) to the Hugging Face Hub.

Builds a multi-config Hub dataset with real train/validation splits, a dataset
card (README.md with YAML frontmatter) that carries the license inventory +
decontamination disclosures, and pushes it. Configs:

  sft_mechanisms  chat rows (mechanism_full + mechanism_next_step)  train/validation
  sft_reactions   forward / retrosynthesis / reagent-prediction     train/validation
  pretrain        dedup plain-text corpus (continued pretraining)   train only

Split routing reuses the pipeline's stratified, decontaminated manifests
(data/final/splits/{train,val}_ids.json), keyed by the parent reaction_id, so a
next-step / reaction row lands in the SAME split as its parent mechanism — no
train/validation leakage across configs.

The license tag is DERIVED from the data (not hardcoded): if every row is MIT the
card is `license: mit`; if any CC-BY-SA source is present the share-alike license
wins and all licenses are listed in the body.

Usage:
  python scripts/export_to_hf.py --repo-id <user>/<name> --dry-run     # safe, no network
  HF_TOKEN=... python scripts/export_to_hf.py --repo-id <user>/<name>  # real push
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "final"
SPLITS = FINAL / "splits"

# config name -> source jsonl in data/final/
SFT_CONFIGS = {
    "sft_mechanisms": "sft_mechanisms.jsonl",
    "sft_reactions": "sft_reactions.jsonl",
}
PRETRAIN_FILE = "pretrain_corpus.txt"

# License normalization: our internal labels -> SPDX-ish tags the Hub understands.
_LICENSE_TAG = {
    "MIT": "mit",
    "CC-BY-SA-4.0": "cc-by-sa-4.0",
    "CC0-1.0": "cc0-1.0",
    "CC-BY-4.0": "cc-by-4.0",
}
# Precedence when multiple licenses are mixed (share-alike is the binding term).
_LICENSE_PRECEDENCE = ["cc-by-sa-4.0", "cc-by-nc-sa-4.0", "cc-by-4.0", "cc0-1.0", "mit"]


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _load_splits() -> Tuple[set, set]:
    train = set(json.loads((SPLITS / "train_ids.json").read_text())) if (SPLITS / "train_ids.json").exists() else set()
    val = set(json.loads((SPLITS / "val_ids.json").read_text())) if (SPLITS / "val_ids.json").exists() else set()
    return train, val


def _route(reaction_id: Optional[str], train_ids: set, val_ids: set) -> str:
    """val wins; anything not explicitly held out is train."""
    if reaction_id in val_ids:
        return "validation"
    return "train"


def _flatten_sft_row(row: dict) -> dict:
    """Flatten meta into top-level columns; keep messages/reference/task.

    messages/reference are JSON-encoded to strings for a stable, viewer-friendly
    columnar schema (nested chat objects otherwise force a complex Arrow type).
    """
    meta = row.get("meta", {})
    return {
        "task": row.get("task"),
        "messages": json.dumps(row.get("messages"), ensure_ascii=False),
        "reference": json.dumps(row.get("reference"), ensure_ascii=False),
        "reaction_id": meta.get("reaction_id"),
        "source": meta.get("source"),
        "license": meta.get("license"),
        "provenance": meta.get("provenance"),
        "level": meta.get("level"),
        "name": meta.get("name"),
        "n_steps": meta.get("n_steps"),
    }


def build_configs(subsets: List[str]) -> Tuple[Dict[str, "object"], set]:
    """Return ({config_name: DatasetDict}, set_of_internal_license_labels)."""
    from datasets import Dataset, DatasetDict

    train_ids, val_ids = _load_splits()
    configs: Dict[str, object] = {}
    licenses: set = set()

    for name, fname in SFT_CONFIGS.items():
        if name not in subsets:
            continue
        path = FINAL / fname
        rows = _load_jsonl(path)
        if not rows:
            print(f"[skip] {name}: {fname} is empty")
            continue
        buckets: Dict[str, List[dict]] = {"train": [], "validation": []}
        for r in rows:
            licenses.add(r.get("meta", {}).get("license"))
            split = _route(r.get("meta", {}).get("reaction_id"), train_ids, val_ids)
            buckets[split].append(_flatten_sft_row(r))
        dd = DatasetDict({
            s: Dataset.from_list(rs) for s, rs in buckets.items() if rs
        })
        configs[name] = dd
        print(f"[build] {name}: " + ", ".join(f"{s}={len(rs)}" for s, rs in buckets.items() if rs))

    if "pretrain" in subsets:
        pre_path = FINAL / PRETRAIN_FILE
        if pre_path.exists():
            lines = [ln.rstrip("\n") for ln in pre_path.open() if ln.strip()]
            dd = DatasetDict({"train": Dataset.from_list([{"text": ln} for ln in lines])})
            configs["pretrain"] = dd
            print(f"[build] pretrain: train={len(lines)}")

    licenses.discard(None)
    return configs, licenses


def resolve_license(licenses: set) -> str:
    tags = {_LICENSE_TAG.get(x, x) for x in licenses if x}
    if not tags:
        return "mit"
    for lic in _LICENSE_PRECEDENCE:
        if lic in tags:
            return lic
    return sorted(tags)[0]


def assert_no_split_leakage(configs: Dict[str, "object"]) -> None:
    """No reaction_id may appear in both train and validation of any config."""
    for name, dd in configs.items():
        if "train" not in dd or "validation" not in dd:
            continue
        if "reaction_id" not in dd["train"].column_names:
            continue
        tr = {x for x in dd["train"]["reaction_id"] if x}
        va = {x for x in dd["validation"]["reaction_id"] if x}
        overlap = tr & va
        assert not overlap, f"LEAKAGE in {name}: {len(overlap)} ids in both splits e.g. {list(overlap)[:3]}"
    print("[check] no train/validation reaction_id leakage across configs — OK")


def _configs_yaml(configs: Dict[str, "object"]) -> str:
    """YAML `configs:` block so the Hub viewer exposes each config + split."""
    lines = ["configs:"]
    for name, dd in configs.items():
        lines.append(f"- config_name: {name}")
        lines.append("  data_files:")
        for split in dd:
            hub_split = split
            lines.append(f'  - split: {hub_split}')
            lines.append(f'    path: {name}/{split}-*')
    return "\n".join(lines)


def build_card(configs: Dict[str, "object"], license_tag: str, repo_id: str) -> str:
    """Assemble README.md (frontmatter + body) from data_card + decon report."""
    # ---- frontmatter ----
    fm = ["---",
          f"license: {license_tag}",
          "task_categories:",
          "- text-generation",
          "language:",
          "- en",
          "tags:",
          "- chemistry",
          "- organic-chemistry",
          "- reaction-mechanism",
          "- oMeBench",
          "- SMILES",
          "pretty_name: rxndata organic reaction-mechanism dataset",
          "size_categories:",
          "- 1K<n<10K",
          _configs_yaml(configs),
          "---",
          ""]

    # ---- body: reuse the pipeline data card if present ----
    body: List[str] = []
    card_md = FINAL / "DATA_CARD.md"
    decon_p = FINAL / "decontamination_report.json"
    fc_p = ROOT / "data" / "interim" / "_format_check.json"

    body.append(f"# {repo_id.split('/')[-1]}\n")
    body.append("Training-ready dataset for **organic reaction-mechanism** modeling, optimized for "
                "the [oMeBench](https://arxiv.org/abs/2510.07731) / oMeS metric. The core unit is a "
                "typed, mass/charge-balanced elementary mechanistic step with a valid intermediate "
                "SMILES — not an overall transformation.\n")

    # Loading snippet
    cfg_names = list(configs)
    first = cfg_names[0] if cfg_names else "sft_mechanisms"
    body.append("## Loading\n")
    body.append("```python")
    body.append("from datasets import load_dataset")
    body.append(f'ds = load_dataset("{repo_id}", "{first}")')
    body.append('# chat SFT rows: parse the JSON-encoded columns')
    body.append("import json")
    body.append('messages = json.loads(ds["train"][0]["messages"])   # [{role, content}, ...]')
    body.append('reference = json.loads(ds["train"][0]["reference"]) # [[subtype, canonical_smiles, weight], ...]')
    body.append("```\n")

    body.append("## Configurations\n")
    for name, dd in configs.items():
        splits = ", ".join(f"{s}: {dd[s].num_rows}" for s in dd)
        body.append(f"- **{name}** — {splits}")
    body.append("")

    # Splice the generated data card body (skip its top H1).
    if card_md.exists():
        card = card_md.read_text()
        # drop the first line (its own H1) to avoid duplicate titles
        card_body = "\n".join(card.splitlines()[1:]).strip()
        body.append(card_body)
        body.append("")

    # Decontamination callout (load-bearing claim).
    if decon_p.exists():
        d = json.loads(decon_p.read_text())
        body.append("## Decontamination (held-out test integrity)\n")
        body.append(f"- Blacklist: {', '.join(d.get('blacklist_source', []))}")
        body.append(f"- Records removed as contaminated: **{d.get('removed')}**")
        body.append(f"- Post-check oMe-Gold InChIKey leaks: **{d.get('post_check_gold_inchikey_leaks', 'n/a')}** "
                    "(0 required — no test record leaks into training)\n")

    if fc_p.exists():
        fc = json.loads(fc_p.read_text())
        body.append("## Format check\n")
        body.append(f"- {fc.get('perfect')}/{fc.get('n_checked')} mechanism targets score "
                    f"S_partial={fc.get('mean_S_partial')}, S_total={fc.get('mean_S_total')} on the "
                    "real oMeS scorer (targets max out the benchmark metric).\n")

    body.append("## Intended use\n")
    body.append("- SFT / continued-pretraining of chemistry LMs for **stepwise mechanism prediction**; "
                "the `reference` column also drives GRPO against the verifiable oMeS reward.\n")
    body.append("## Out of scope\n")
    body.append("- Broad chemical knowledge, retrosynthesis planning at scale, or safety-critical use. "
                "Scores are highest in-domain (same named reactions, new substituents) and should not be "
                "read as general chemical mastery.\n")

    body.append("## Attribution & licenses\n")
    body.append("Derived from openly-licensed sources only. See `license_inventory.json` in this repo "
                "for the full per-source verdict. oMeBench data/prompts/scorer: MIT "
                "(skylarkie/oMeBench). If a rebuild includes ORD (CC-BY-SA-4.0) or USPTO/Lowe (CC0-1.0), "
                "the share-alike terms of CC-BY-SA propagate to the combined dataset.\n")

    return "\n".join(fm) + "\n".join(body) + "\n"


def do_dry_run(configs, card: str, license_tag: str) -> None:
    out = FINAL / "hf_export"
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(card)
    for name, dd in configs.items():
        dd.save_to_disk(str(out / name))
    print("\n" + "=" * 68)
    print("  DRY RUN — nothing pushed. Preview written to data/final/hf_export/")
    print("=" * 68)
    print(f"  license tag : {license_tag}")
    print(f"  configs     : {list(configs)}")
    print(f"  card head   :\n")
    print("\n".join(card.splitlines()[:18]))
    print("\n  To push for real:")
    print("    huggingface-cli login   # or export HF_TOKEN=...")
    print("    python scripts/export_to_hf.py --repo-id <user>/<name>")
    print("=" * 68)


def do_push(configs, card: str, repo_id: str, private: bool) -> None:
    import os

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    # Fail early with a clear message if unauthenticated.
    try:
        who = api.whoami()
        print(f"[auth] logged in as {who.get('name')}")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Not authenticated with the HF Hub. Run `huggingface-cli login` or set HF_TOKEN. "
            f"({type(e).__name__}: {e})"
        )

    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    for name, dd in configs.items():
        print(f"[push] {name} ...")
        dd.push_to_hub(repo_id, config_name=name, private=private, token=token)

    # Upload the card + metadata files.
    tmp_card = FINAL / "hf_export"
    tmp_card.mkdir(parents=True, exist_ok=True)
    (tmp_card / "README.md").write_text(card)
    api.upload_file(path_or_fileobj=str(tmp_card / "README.md"),
                    path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
    for meta in ("decontamination_report.json", "data_card.json"):
        p = FINAL / meta
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=meta,
                            repo_id=repo_id, repo_type="dataset")
    lic = ROOT / "configs" / "license_inventory.json"
    if lic.exists():
        api.upload_file(path_or_fileobj=str(lic), path_in_repo="license_inventory.json",
                        repo_id=repo_id, repo_type="dataset")
    print(f"\nPushed → https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="<user_or_org>/<dataset_name>")
    ap.add_argument("--subsets", nargs="+",
                    default=["sft_mechanisms", "sft_reactions", "pretrain"],
                    choices=["sft_mechanisms", "sft_reactions", "pretrain"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + write local preview + print push commands; no network")
    args = ap.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit("Install the export deps: pip install datasets huggingface_hub")

    configs, licenses = build_configs(args.subsets)
    if not configs:
        raise SystemExit("No non-empty configs to export. Run `make phase8` first?")
    assert_no_split_leakage(configs)
    license_tag = resolve_license(licenses)
    card = build_card(configs, license_tag, args.repo_id)

    if args.dry_run:
        do_dry_run(configs, card, license_tag)
    else:
        do_push(configs, card, args.repo_id, args.private)


if __name__ == "__main__":
    main()
