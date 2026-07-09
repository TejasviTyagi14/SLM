"""Phase 3: mechanism decomposition & typing.

Three jobs, in order of trust (curated > template_expanded > inferred):

1. type_and_remap(rec): map every mechanism step to the parsed ontology. Steps
   whose (type, subtype) is already allowed pass through. Steps flagged in Phase 1
   as outside the ontology are remapped via ontology.json's ``template_remap``
   table, or rejected (reason ``unmapped_step_type``) when the table maps them to
   null / they have a null subtype. Also derives ``level`` from step count when
   absent. Returns (typed_record | None, reject_reason | None).

2. expand_template(rec, ...): bounded, deduplicated R-group expansion of a
   generalized template into concrete instances via MOLECULE-LEVEL substitution
   (RDKit RWMol -- never string substitution, which corrupts ring-closure digits).
   Tags provenance ``template_expanded``.

3. (inference lives in infer.py to keep the LLM/template-application path isolated
   and its stricter validation obvious.)

Level heuristic (documented): easy <=3 steps, medium 4-6, hard >=7 -- matched to
the oMe-Gold distribution; only applied when ``level`` is missing.
"""

from __future__ import annotations

import itertools
import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger

from .config import Config, load_config
from .ontology import load_ontology

RDLogger.DisableLog("rdApp.*")

_RGROUP_RE = re.compile(r"\[\*:(\d+)\]")

# Small, valence-safe substituent library for our own expansion. Deliberately
# conservative: common organic caps that keep mechanisms chemically sensible.
DEFAULT_SUBSTITUENTS: Dict[str, str] = {
    "H": "[H]",
    "Me": "C",
    "Et": "CC",
    "Ph": "c1ccccc1",
    "Cl": "Cl",
    "OMe": "OC",
}


def derive_level(n_steps: int) -> str:
    if n_steps <= 3:
        return "easy"
    if n_steps <= 6:
        return "medium"
    return "hard"


# ---- 1. typing / remap ------------------------------------------------------

def type_and_remap(
    rec: Dict, ontology: Optional[dict] = None
) -> Tuple[Optional[Dict], Optional[str]]:
    """Ensure every step maps to an allowed (type, subtype). Returns (rec|None, reason)."""
    ontology = ontology or load_ontology()
    allowed = {(p["type"], p["subtype"]) for p in ontology["pairs"]}
    remap = ontology.get("template_remap", {})

    out = dict(rec)
    new_mech: List[Dict] = []
    for step in rec.get("mechanism", []):
        t, st = step.get("type"), step.get("subtype")
        pair = (t, st)
        if pair in allowed:
            new_mech.append(_clean_step(step))
            continue
        # Try the remap table (keyed "type::subtype").
        key = f"{t}::{st}"
        mapped = remap.get(key, "__MISSING__")
        if mapped == "__MISSING__":
            # subtype not in allowed and not in remap -> unmappable.
            return None, f"unmapped_step_type:{key}"
        if mapped is None:
            return None, f"unmapped_step_type:{key}(explicit_reject)"
        new_t, new_st = mapped
        if (new_t, new_st) not in allowed:
            return None, f"remap_target_not_allowed:{new_t}/{new_st}"
        s = _clean_step(step)
        s["type"], s["subtype"] = new_t, new_st
        s["_remapped_from"] = key
        new_mech.append(s)

    out["mechanism"] = new_mech
    out["mechanism_step_nums"] = len(new_mech)
    if out.get("level") is None and new_mech:
        out["level"] = derive_level(len(new_mech))

    # Mark records that still contain generalized R-group placeholders. These are
    # valid *reference/template* material (decon blacklist, inference source) but
    # are NOT concrete training reactions -- Phase 5 routes them to a template
    # pool rather than rejecting them as invalid chemistry.
    all_smis = (
        out.get("reactants_smiles", [])
        + out.get("products_smiles", [])
        + [s.get("intermediate_smiles", "") for s in new_mech]
    )
    out["is_template"] = any(_RGROUP_RE.search(s or "") for s in all_smis)
    return out, None


def _clean_step(step: Dict) -> Dict:
    """Drop Phase-1/2 internal annotations, keep the canonical step fields."""
    return {
        "step": step.get("step"),
        "type": step.get("type"),
        "subtype": step.get("subtype"),
        "intermediate_smiles": step.get("intermediate_smiles", ""),
        "rationale": step.get("rationale"),
    }


# ---- 2. molecule-level R-group substitution --------------------------------

def rgroup_labels(smiles_list: List[str]) -> List[int]:
    labels = set()
    for s in smiles_list:
        labels.update(int(x) for x in _RGROUP_RE.findall(s or ""))
    return sorted(labels)


def substitute_rgroups(template_smiles: str, sub_map: Dict[int, str],
                       default: str = "[H]") -> Optional[str]:
    """Replace [*:n] dummy atoms with fragments at the MOLECULE level (RDKit).

    String substitution corrupts ring-closure digits (e.g. inserting c1ccccc1
    into a ring reuses bond number 1); doing it on the molecule graph avoids that.
    ``sub_map`` maps an R-group label to a fragment SMILES whose FIRST atom is the
    attachment point. Returns canonical SMILES, or None if any step fails to
    sanitize.
    """
    mol = Chem.MolFromSmiles(template_smiles)
    if mol is None:
        return None
    guard = 0
    while True:
        guard += 1
        if guard > 64:
            return None
        dummy = next((a for a in mol.GetAtoms() if a.GetAtomicNum() == 0), None)
        if dummy is None:
            break
        mapnum = dummy.GetAtomMapNum()
        frag_smi = sub_map.get(mapnum, default)
        frag = Chem.MolFromSmiles(frag_smi)
        if frag is None:
            return None
        nbrs = dummy.GetNeighbors()
        if len(nbrs) != 1:
            return None
        attach_idx = nbrs[0].GetIdx()
        dummy_idx = dummy.GetIdx()
        n_before = mol.GetNumAtoms()
        combined = Chem.CombineMols(mol, frag)
        rw = Chem.RWMol(combined)
        rw.AddBond(attach_idx, n_before, Chem.BondType.SINGLE)  # frag root = n_before
        rw.RemoveAtom(dummy_idx)
        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _substitution_combos(labels: List[int], subs: Dict[str, str],
                         max_instances: int) -> List[Dict[int, str]]:
    """Enumerate bounded (label -> fragment) assignments.

    To keep counts sane, we vary at most 2 labels across the full substituent
    set and cap remaining labels to a single sensible default; then truncate to
    max_instances. Deterministic order (sorted) for reproducibility.
    """
    frags = list(subs.values())
    if not labels:
        return []
    combos: List[Dict[int, str]] = []
    # Vary the first up-to-2 labels fully; fix the rest to methyl (C) as a neutral cap.
    vary = labels[:2]
    fixed = labels[2:]
    for choice in itertools.product(frags, repeat=len(vary)):
        m = {lab: choice[i] for i, lab in enumerate(vary)}
        for lab in fixed:
            m[lab] = "C"
        combos.append(m)
        if len(combos) >= max_instances:
            break
    return combos


def expand_template(
    rec: Dict,
    max_instances: int,
    substituents: Optional[Dict[str, str]] = None,
    exclude_canonical: Optional[set] = None,
    ontology: Optional[dict] = None,
) -> List[Dict]:
    """Expand one R-group template record into concrete instances.

    Only emits an instance if ALL of its molecules (reactants, products, every
    intermediate) substitute + sanitize cleanly. Deduplicates by the tuple of
    canonical product SMILES and skips instances whose molecules already appear
    in ``exclude_canonical`` (e.g. silver's molecule set) to avoid re-minting
    silver. Tags provenance ``template_expanded``.
    """
    substituents = substituents or DEFAULT_SUBSTITUENTS
    exclude_canonical = exclude_canonical or set()

    all_smis = (
        rec.get("reactants_smiles", [])
        + rec.get("products_smiles", [])
        + [s.get("intermediate_smiles", "") for s in rec.get("mechanism", [])]
    )
    labels = rgroup_labels(all_smis)
    if not labels:
        return []  # nothing to expand

    combos = _substitution_combos(labels, substituents, max_instances)
    out: List[Dict] = []
    seen_products: set = set()

    for ci, sub_map in enumerate(combos):
        def sub(s: str) -> Optional[str]:
            if not _RGROUP_RE.search(s or ""):
                # already concrete; canonicalize
                m = Chem.MolFromSmiles(s)
                return Chem.MolToSmiles(m) if m else None
            return substitute_rgroups(s, sub_map)

        reac = [sub(s) for s in rec.get("reactants_smiles", [])]
        prod = [sub(s) for s in rec.get("products_smiles", [])]
        if any(x is None for x in reac + prod):
            continue
        mech = []
        ok = True
        for st in rec.get("mechanism", []):
            inter = sub(st.get("intermediate_smiles", ""))
            if inter is None:
                ok = False
                break
            mech.append({**_clean_step(st), "intermediate_smiles": inter})
        if not ok:
            continue

        prod_key = tuple(prod)
        if prod_key in seen_products:
            continue
        seen_products.add(prod_key)
        # Skip if the product set is already present in the exclude set (silver).
        if all(p in exclude_canonical for p in prod):
            continue

        inst = dict(rec)
        inst["reaction_id"] = f"{rec['reaction_id']}__x{ci}"
        inst["provenance"] = "template_expanded"
        inst["reactants_smiles"] = reac
        inst["products_smiles"] = prod
        inst["mechanism"] = mech
        inst["mechanism_step_nums"] = len(mech)
        if inst.get("level") is None:
            inst["level"] = derive_level(len(mech))
        inst["raw_ref"] = {**rec.get("raw_ref", {}),
                           "expanded_from": rec["reaction_id"],
                           "sub_map": {str(k): v for k, v in sub_map.items()}}
        out.append(inst)
    return out
