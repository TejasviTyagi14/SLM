"""Phase 2 normalization: RDKit canonicalization + molecule vocabulary.

Rules (documented conventions, applied everywhere):
- Canonical SMILES via RDKit ``MolToSmiles`` after sanitization.
- Mechanisms carry REAL charges and radicals (e.g. ``[CH2-]``, ``[CH3]`` radical);
  we DO NOT neutralize or strip them -- doing so would destroy the chemistry the
  benchmark grades. ``normalization.neutralize_charges: false`` in config.
- Atom-map numbers are preserved through canonicalization when present
  (``keep_atom_maps_through_canon: true``) so Phase 4 mapping survives.
- Stereo is kept (oMeS canonicalization is stereo-aware).
- Molecule dedup key = InChIKey (falls back to canonical SMILES if InChI fails,
  e.g. for fragments containing R-group ``[*:n]`` placeholders).

Source-specific pre-clean:
- USPTO (Lowe) reaction SMILES carry a trailing CXSMILES fragment-grouping
  suffix `` |f:...`` that RDKit will not parse; it is stripped before parsing.
- R-group template SMILES (``[*:n]``) are left as-is here; malformed template
  syntax is handled in Phase 3 expansion, not silently repaired.

This module NEVER mutates chemistry to force a parse. If a molecule will not
sanitize, it is reported as a failure with a reason code; the record is routed
to rejects only in Phase 5, not here (Phase 2 annotates, Phase 5 gates).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# Trailing CXSMILES extension (e.g. " |f:1.2|", " |f:3", " |c:...|"). Lowe emits
# a bare "|f:N" without the closing bar for fragment grouping.
_CX_SUFFIX = re.compile(r"\s+\|[^|]*\|?\s*$")

# Presence of an R-group / attachment-point placeholder.
_HAS_RGROUP = re.compile(r"\[\*(?::\d+)?\]|\*")


@dataclass
class NormResult:
    canonical: Optional[str]
    inchikey: Optional[str]
    ok: bool
    reason: Optional[str] = None
    had_map: bool = False
    had_charge: bool = False
    had_radical: bool = False


def strip_cxsmiles(smi: str) -> str:
    """Remove a trailing CXSMILES extension block that RDKit can't parse."""
    return _CX_SUFFIX.sub("", smi).strip()


def has_rgroup(smi: str) -> bool:
    return bool(_HAS_RGROUP.search(smi or ""))


def _remove_atom_maps(mol: Chem.Mol) -> None:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)


def canonicalize(
    smi: str,
    keep_maps: bool = True,
    from_source: Optional[str] = None,
) -> NormResult:
    """Canonicalize one SMILES. Returns a NormResult (never raises)."""
    if not smi or not isinstance(smi, str):
        return NormResult(None, None, False, "empty")

    cleaned = smi
    # Source-aware pre-clean (only strip syntactic wrappers, never chemistry).
    if from_source in ("USPTO-Lowe",) or " |" in smi:
        cleaned = strip_cxsmiles(cleaned)

    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        # Retry with CXSMILES allowed (handles well-formed |...| blocks).
        ps = Chem.SmilesParserParams()
        ps.allowCXSMILES = True
        mol = Chem.MolFromSmiles(cleaned, ps)
    if mol is None:
        reason = "rgroup_template" if has_rgroup(cleaned) else "unparseable"
        return NormResult(None, None, False, reason)

    had_map = any(a.GetAtomMapNum() for a in mol.GetAtoms())
    had_charge = any(a.GetFormalCharge() for a in mol.GetAtoms())
    had_radical = any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())

    if not keep_maps and had_map:
        _remove_atom_maps(mol)

    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:  # noqa: BLE001
        return NormResult(None, None, False, f"canon_error:{type(e).__name__}")

    # InChIKey for molecule dedup; skip for R-group fragments (InChI rejects *).
    inchikey: Optional[str] = None
    if not has_rgroup(canonical):
        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = None
    if not inchikey:
        inchikey = f"SMI:{canonical}"  # fallback key so dedup still works

    return NormResult(
        canonical=canonical,
        inchikey=inchikey,
        ok=True,
        had_map=had_map,
        had_charge=had_charge,
        had_radical=had_radical,
    )


@dataclass
class SourceNormStats:
    source: str
    n_mols: int = 0
    n_ok: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)

    def add(self, res: NormResult) -> None:
        self.n_mols += 1
        if res.ok:
            self.n_ok += 1
        else:
            self.reasons[res.reason or "unknown"] = self.reasons.get(res.reason or "unknown", 0) + 1

    @property
    def pct_ok(self) -> float:
        return 100.0 * self.n_ok / self.n_mols if self.n_mols else 0.0


def normalize_record(rec: Dict, keep_maps: bool = True) -> Tuple[Dict, List[NormResult]]:
    """Canonicalize all SMILES fields in a record in place-ish (returns a copy).

    Returns (normalized_record, list_of_NormResults_for_every_molecule).
    Records with R-group placeholders (templates) keep original SMILES for the
    unparseable fragments so Phase 3 can expand them.
    """
    out = dict(rec)
    src = rec.get("source")
    results: List[NormResult] = []

    def _norm_list(smis: List[str]) -> List[str]:
        new = []
        for s in smis:
            res = canonicalize(s, keep_maps=keep_maps, from_source=src)
            results.append(res)
            new.append(res.canonical if res.ok else s)
        return new

    out["reactants_smiles"] = _norm_list(rec.get("reactants_smiles", []))
    out["products_smiles"] = _norm_list(rec.get("products_smiles", []))

    new_mech = []
    for step in rec.get("mechanism", []):
        st = dict(step)
        res = canonicalize(step.get("intermediate_smiles", ""), keep_maps=keep_maps, from_source=src)
        results.append(res)
        if res.ok:
            st["intermediate_smiles"] = res.canonical
        st["_norm_ok"] = res.ok
        st["_norm_reason"] = res.reason
        new_mech.append(st)
    out["mechanism"] = new_mech
    return out, results
