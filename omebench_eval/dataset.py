"""Loading oMeBench datasets and building prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROMPT_DIR = ROOT / "prompts"

DATASETS: Dict[str, str] = {
    "gold": "oMe_Gold.json",
    "template": "oMe_Template.json",
    "silver": "oMe_Silver.jsonl",
}


def load_dataset(name: str, limit: int | None = None) -> List[dict]:
    """Load a named oMeBench split. Supports .json and .jsonl."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS)}.")
    path = DATA_DIR / DATASETS[name]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file missing: {path}")

    if path.suffix == ".jsonl":
        records = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with path.open() as f:
            records = json.load(f)

    if limit is not None:
        records = records[:limit]
    return records


def load_prompt_template(style: str) -> str:
    """Load a prompt template: 'default' (direct) or 'cot' (reason then answer)."""
    path = PROMPT_DIR / f"{style}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template missing: {path}")
    return path.read_text()


def build_prompt(template: str, reactants, products, conditions) -> str:
    """Fill the oMeBench prompt template placeholders."""
    replacements = {
        "{ reactants_smiles }": json.dumps(reactants),
        "{ products_smiles }": json.dumps(products),
        "{ conditions }": json.dumps(conditions),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template
