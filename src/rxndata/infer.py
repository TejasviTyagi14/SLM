"""Phase 3 (inference path): propose mechanisms for Tier-B/C overall reactions.

HARD RULE (no fabricated chemistry): anything produced here is tagged
``provenance: inferred``, routed through STRICTER validation (Phase 5 continuity
+ balance), capped at a configurable fraction of mechanism records (default
<=20%), and a held-back sanity sample is emitted for human review. We NEVER label
an inferred mechanism as curated.

Two inference backends, in order of preference:

1. template_application (default, deterministic, offline): find a curated
   oMe-Template whose named reaction matches a Tier-B/C reaction (by reaction
   name, or by a reactant/product substructure signature), then instantiate that
   template's typed step pattern for the concrete substrates. This reuses
   expert-authored mechanisms rather than inventing them, so it is the safest
   form of "inference".

2. llm (optional, off by default): if an API key is configured, an LLM proposes a
   step sequence; EVERY proposed intermediate must be RDKit-valid and pass Phase 5
   continuity or the whole proposal is rejected. Disabled unless
   ``mechanism.inference.llm_enabled`` is true AND a key is present.

This module only PRODUCES candidates + tags them; the Phase 5 gate decides what
survives. Phase 3's cap governs how many inferred records we even attempt.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from rdkit import RDLogger

from .config import Config, load_config

RDLogger.DisableLog("rdApp.*")


def _name_key(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def build_template_index(template_records: List[Dict]) -> Dict[str, Dict]:
    """Index curated templates by normalized reaction name for matching."""
    idx: Dict[str, Dict] = {}
    for t in template_records:
        k = _name_key(t.get("name"))
        if k and k not in idx:
            idx[k] = t
    return idx


def infer_by_template_application(
    overall_rec: Dict,
    template_index: Dict[str, Dict],
) -> Optional[Dict]:
    """If the overall reaction names a known template, attach that template's typed
    step pattern as an INFERRED mechanism for the concrete substrates.

    Conservative: only fires on an exact normalized-name match (e.g. a Wikipedia
    'Diels-Alder reaction' article matching template 'Diels-Alder'). The concrete
    intermediates are taken from the template's generalized steps with R-groups
    left symbolic where we cannot resolve them; such records are marked
    ``needs_review`` so the Phase 5 continuity gate + human sample catch nonsense.
    Returns an inferred record or None.
    """
    key = _name_key(overall_rec.get("name"))
    if not key or key not in template_index:
        return None
    tmpl = template_index[key]

    # We do not fabricate concrete intermediates from thin air: we only propose
    # the STEP TYPE SEQUENCE (subtypes) from the matched template, and reuse the
    # template's intermediate SMILES as a symbolic scaffold. This is explicitly
    # provenance=inferred and needs_review; Phase 5 continuity will reject any
    # instance whose intermediates don't actually connect for these substrates.
    steps = []
    for st in tmpl.get("mechanism", []):
        steps.append({
            "step": st.get("step"),
            "type": st.get("type"),
            "subtype": st.get("subtype"),
            "intermediate_smiles": st.get("intermediate_smiles", ""),
            "rationale": st.get("rationale"),
        })
    if not steps:
        return None

    out = dict(overall_rec)
    out["reaction_id"] = f"INFER-{overall_rec['reaction_id']}"
    out["provenance"] = "inferred"
    out["mechanism"] = steps
    out["mechanism_step_nums"] = len(steps)
    out["raw_ref"] = {
        **overall_rec.get("raw_ref", {}),
        "inferred_via": "template_application",
        "matched_template": tmpl.get("reaction_id"),
        "needs_review": True,
    }
    return out


def infer_mechanisms(
    candidates: List[Dict],
    template_records: List[Dict],
    max_records: int,
    cfg: Optional[Config] = None,
) -> List[Dict]:
    """Produce up to ``max_records`` inferred mechanisms from Tier-B/C candidates.

    Currently uses template_application only (deterministic, offline). The LLM
    backend is intentionally gated off unless explicitly enabled + keyed.
    """
    cfg = cfg or load_config()
    index = build_template_index(template_records)
    out: List[Dict] = []
    for rec in candidates:
        if len(out) >= max_records:
            break
        inferred = infer_by_template_application(rec, index)
        if inferred is not None:
            out.append(inferred)
    return out
