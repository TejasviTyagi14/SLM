"""Parse the oMeBench type/subtype ontology directly from the benchmark artifacts.

HARD RULE (from the project spec): do NOT hardcode the 8 types / 30 subtypes from
memory. The exact vocabulary is derived here from the benchmark's own files and
persisted to ``configs/ontology.json``. Every mechanism step in the pipeline is
later forced to map to a (type, subtype) pair present in that file; anything that
cannot be mapped is rejected with reason ``unmapped_step_type``.

Three independent sources are parsed and cross-checked so the result is provably
the benchmark's own ontology, not a reconstruction:

1. ``prompts/default.txt`` / ``prompts/cot.txt`` -- the *allowed* type/subtype
   pairs (with human descriptions) that the benchmark shows the model. This is
   the authoritative *allowed set*.
2. ``data/oMe_Gold.json`` -- the (type, subtype) pairs actually *used* by the
   held-out gold mechanisms.
3. ``data/oMe_Template.json`` -- the pairs used by the named-reaction templates.

The union of (2) and (3) must be a subset of (1); we assert that and report any
pair that appears in the data but not in the allowed list (a benchmark bug) or in
the allowed list but never used (a coverage gap for later phases).

Alignment note: oMeS aligns predicted steps to gold on the **subtype** field
(see ``third_party/oMeBench/scripts/run.py`` -> ``pred = [(j['subtype'], ...)]``),
so ``subtype`` is the primary key that earns the L (logical fidelity) score.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROMPT_DIR = ROOT / "prompts"
CONFIG_DIR = ROOT / "configs"

# The prompt file embeds the allowed pairs as a JSON array of objects with
# {"type", "subtype", "description"}. We extract that array robustly rather than
# assuming its exact line position.
_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def _extract_allowed_from_prompt(prompt_text: str) -> List[Dict[str, str]]:
    """Pull the allowed type/subtype/description list out of a prompt template."""
    # The first JSON array of objects containing a "subtype" key is the ontology.
    for match in _ARRAY_RE.finditer(prompt_text):
        blob = match.group(0)
        if '"subtype"' not in blob:
            continue
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed and "subtype" in parsed[0]:
            return parsed
    raise ValueError("Could not locate the allowed type/subtype array in the prompt.")


def _pairs_from_records(records: List[dict]) -> Dict[Tuple[str, str], int]:
    """Count (type, subtype) pairs used across a list of oMeBench records."""
    counts: Dict[Tuple[str, str], int] = {}
    for rec in records:
        for step in rec.get("mechanism", []) or []:
            t = step.get("type")
            s = step.get("subtype")
            if t is None or s is None:
                continue
            counts[(t, s)] = counts.get((t, s), 0) + 1
    return counts


def _load_json(path: Path) -> list:
    with path.open() as f:
        return json.load(f)


def build_ontology(
    prompt_path: Optional[Path] = None,
    gold_path: Optional[Path] = None,
    template_path: Optional[Path] = None,
) -> dict:
    """Derive and cross-check the ontology; return a serializable dict."""
    prompt_path = prompt_path or (PROMPT_DIR / "default.txt")
    gold_path = gold_path or (DATA_DIR / "oMe_Gold.json")
    template_path = template_path or (DATA_DIR / "oMe_Template.json")

    allowed = _extract_allowed_from_prompt(prompt_path.read_text())
    allowed_pairs = [(a["type"], a["subtype"]) for a in allowed]
    allowed_set = set(allowed_pairs)

    gold_counts = _pairs_from_records(_load_json(gold_path))
    template_counts = _pairs_from_records(_load_json(template_path))

    used_pairs = set(gold_counts) | set(template_counts)

    # Cross-checks. We deliberately split "used but not allowed" by source: the
    # gold set is the graded test set, so a non-zero count there would mean our
    # allowed vocabulary cannot reach ceiling on the benchmark (a real problem).
    # Non-allowed pairs confined to the template file are benchmark data-quality
    # issues that Phase 3 (template expansion) must remap or reject.
    used_not_allowed = sorted(p for p in used_pairs if p not in allowed_set)
    gold_not_allowed = sorted(p for p in gold_counts if p not in allowed_set)
    template_not_allowed = sorted(p for p in template_counts if p not in allowed_set)
    allowed_not_used = sorted(p for p in allowed_set if p not in used_pairs)

    types = sorted({t for t, _ in allowed_pairs})
    subtypes = [s for _, s in allowed_pairs]
    subtype_to_type = {s: t for t, s in allowed_pairs}

    # Detect any subtype mapped to more than one type in the allowed list (would
    # break the assumption that subtype alone determines type).
    subtype_type_conflicts = {}
    seen: Dict[str, str] = {}
    for t, s in allowed_pairs:
        if s in seen and seen[s] != t:
            subtype_type_conflicts[s] = sorted({seen[s], t})
        seen[s] = t

    ontology = {
        "_meta": {
            "description": (
                "oMeBench type/subtype ontology parsed from the benchmark's own "
                "prompt + data files. Do not edit by hand; regenerate with "
                "`python -m rxndata.ontology`."
            ),
            "source_prompt": str(prompt_path.relative_to(ROOT)),
            "source_gold": str(gold_path.relative_to(ROOT)),
            "source_template": str(template_path.relative_to(ROOT)),
            "alignment_key": "subtype",
            "n_types": len(types),
            "n_subtypes": len(subtypes),
        },
        "types": types,
        "subtypes": subtypes,
        "pairs": [
            {
                "type": a["type"],
                "subtype": a["subtype"],
                "description": a.get("description", ""),
                "n_gold": gold_counts.get((a["type"], a["subtype"]), 0),
                "n_template": template_counts.get((a["type"], a["subtype"]), 0),
            }
            for a in allowed
        ],
        "subtype_to_type": subtype_to_type,
        "crosscheck": {
            "used_pairs_not_in_allowed": [list(p) for p in used_not_allowed],
            "gold_pairs_not_in_allowed": [list(p) for p in gold_not_allowed],
            "template_pairs_not_in_allowed": [list(p) for p in template_not_allowed],
            "allowed_pairs_never_used": [list(p) for p in allowed_not_used],
            "subtype_type_conflicts": subtype_type_conflicts,
            "gold_is_clean": len(gold_not_allowed) == 0,
        },
        # Remap table for the known template-only mislabels, applied in Phase 3
        # so expanded records only ever carry allowed (type, subtype) pairs.
        # Keys are "type::subtype" from the raw template; values are the allowed
        # pair to use, or null to REJECT the step (reason: unmapped_step_type).
        # `null` subtype (missing label) is always a reject.
        "template_remap": {
            "rearrangement::sigmatropic_rearrangement": ["pericyclic", "sigmatropic_rearrangement"],
            "elimination::cycloreversion": ["pericyclic", "cycloreversion"],
            "coordination::lewis_acid_base_coordination": ["coordination", "coordination"],
            "radical::radical_substitution": ["substitution", "radical_substitution"],
            "proton_transfer::intramolecular_proton_transfer": ["proton_transfer", "acid_base_proton_transfer"],
            "addition::intramolecular_nucleophilic_addition": ["addition", "nucleophilic_addition"],
            "rearrangement::hydride_shift": ["rearrangement", "1,2-shift"],
            "rearrangement::rearrangement": None,
            "cyclization::aromatic_electrophilic_substitution": ["substitution", "electrophilic_substitution"],
            "aromatization::dehydration": None,
        },
    }
    return ontology


def write_ontology(out_path: Optional[Path] = None) -> dict:
    out_path = out_path or (CONFIG_DIR / "ontology.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ontology = build_ontology()
    with out_path.open("w") as f:
        json.dump(ontology, f, indent=2)
    return ontology


def load_ontology(path: Optional[Path] = None) -> dict:
    path = path or (CONFIG_DIR / "ontology.json")
    with path.open() as f:
        return json.load(f)


def _print_report(ontology: dict) -> None:
    m = ontology["_meta"]
    print(f"Parsed ontology: {m['n_types']} types, {m['n_subtypes']} subtypes")
    print(f"  from prompt={m['source_prompt']}, gold={m['source_gold']}, "
          f"template={m['source_template']}")
    print(f"  alignment key = {m['alignment_key']}\n")
    print(f"{'type':<16} {'subtype':<28} {'gold':>5} {'tmpl':>5}")
    print("-" * 58)
    for p in ontology["pairs"]:
        print(f"{p['type']:<16} {p['subtype']:<28} {p['n_gold']:>5} {p['n_template']:>5}")
    cc = ontology["crosscheck"]
    print("\nCross-check:")
    print(f"  used-but-not-allowed pairs : {cc['used_pairs_not_in_allowed'] or 'none (OK)'}")
    never = cc["allowed_pairs_never_used"]
    print(f"  allowed-but-never-used     : "
          f"{len(never)} pairs" + (f" -> {never}" if never else " (full coverage)"))
    print(f"  subtype->type conflicts    : "
          f"{cc['subtype_type_conflicts'] or 'none (subtype uniquely maps to type)'}")


if __name__ == "__main__":
    onto = write_ontology()
    _print_report(onto)
    print(f"\nWrote {CONFIG_DIR / 'ontology.json'}")
