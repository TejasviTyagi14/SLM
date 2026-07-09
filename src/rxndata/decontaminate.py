"""Phase 6: benchmark decontamination (do NOT skip).

No training record may overlap with the held-out oMe-Gold test set. Because
oMe-Silver ships in the same repo as oMe-Gold, we treat oMe-Gold + oMe-Template
as a contamination blacklist and remove any training candidate that matches by:

  (a) exact reactant-set + product-set InChIKey signature,
  (b) reaction-fingerprint Tanimoto >= threshold (default 0.95), or
  (c) normalized mechanism-string hash.

A leaked test set invalidates every reported number, so this errs toward removal.
Applies to oMe-Silver-derived records too. Removed ids are written to
data/final/decontamination_report.json.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Set

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def _inchikey(smi: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return None


def inchikey_signature(reactants: List[str], products: List[str]) -> Optional[str]:
    """Order-independent signature of reactant-set + product-set InChIKeys."""
    r_keys = sorted(k for k in (_inchikey(s) for s in reactants) if k)
    p_keys = sorted(k for k in (_inchikey(s) for s in products) if k)
    if not r_keys or not p_keys:
        return None
    return "R:" + ".".join(r_keys) + ">>P:" + ".".join(p_keys)


def _skeleton_inchikey_signature(reactants: List[str], products: List[str]) -> Optional[str]:
    """Signature using only the first InChIKey block (connectivity, no stereo/charge).

    Catches contamination where a training candidate differs from a gold entry
    only by protonation state or stereo -- still effectively the same reaction.
    """
    def block1(smi):
        k = _inchikey(smi)
        return k.split("-")[0] if k else None
    r = sorted(x for x in (block1(s) for s in reactants) if x)
    p = sorted(x for x in (block1(s) for s in products) if x)
    if not r or not p:
        return None
    return "r:" + ".".join(r) + ">>p:" + ".".join(p)


def reaction_fingerprint(reactants: List[str], products: List[str],
                         radius: int = 2, nbits: int = 2048):
    """Difference reaction fingerprint: sum(product FPs) - sum(reactant FPs)."""
    def combine(smis):
        acc = None
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
            if acc is None:
                acc = DataStructs.cDataStructs.ExplicitBitVect(nbits)
            acc |= fp
        return acc
    rfp = combine(reactants)
    pfp = combine(products)
    if rfp is None or pfp is None:
        return None
    # XOR captures the symmetric difference (bonds made/broken); good enough as a
    # reaction similarity proxy and cheap with Tanimoto.
    diff = rfp ^ pfp
    return diff


def mechanism_hash(mechanism: List[dict]) -> Optional[str]:
    """Normalized hash of the (subtype, canonical intermediate) step sequence."""
    if not mechanism:
        return None
    parts = []
    for s in mechanism:
        smi = s.get("intermediate_smiles", "")
        m = Chem.MolFromSmiles(smi)
        can = Chem.MolToSmiles(m) if m else smi
        parts.append(f"{s.get('subtype')}|{can}")
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()


@dataclass
class Blacklist:
    inchikey_sigs: Set[str] = field(default_factory=set)
    skeleton_sigs: Set[str] = field(default_factory=set)
    mech_hashes: Set[str] = field(default_factory=set)
    fingerprints: list = field(default_factory=list)      # list[ExplicitBitVect]
    n_entries: int = 0


def build_blacklist(gold: List[dict], template: List[dict]) -> Blacklist:
    bl = Blacklist()
    for rec in list(gold) + list(template):
        reac = rec.get("reactants_smiles", [])
        prod = rec.get("products_smiles", [])
        sig = inchikey_signature(reac, prod)
        if sig:
            bl.inchikey_sigs.add(sig)
        sk = _skeleton_inchikey_signature(reac, prod)
        if sk:
            bl.skeleton_sigs.add(sk)
        mh = mechanism_hash(rec.get("mechanism", []))
        if mh:
            bl.mech_hashes.add(mh)
        fp = reaction_fingerprint(reac, prod)
        if fp is not None:
            bl.fingerprints.append(fp)
        bl.n_entries += 1
    return bl


def contamination_reason(rec: dict, bl: Blacklist, tanimoto_threshold: float) -> Optional[str]:
    """Return a reason string if the record is contaminated, else None."""
    reac = rec.get("reactants_smiles", [])
    prod = rec.get("products_smiles", [])

    sig = inchikey_signature(reac, prod)
    if sig and sig in bl.inchikey_sigs:
        return "inchikey_set_match"
    sk = _skeleton_inchikey_signature(reac, prod)
    if sk and sk in bl.skeleton_sigs:
        return "skeleton_inchikey_match"
    mh = mechanism_hash(rec.get("mechanism", []))
    if mh and mh in bl.mech_hashes:
        return "mechanism_hash_match"
    fp = reaction_fingerprint(reac, prod)
    if fp is not None and bl.fingerprints:
        sims = DataStructs.BulkTanimotoSimilarity(fp, bl.fingerprints)
        if sims and max(sims) >= tanimoto_threshold:
            return f"reaction_fp_tanimoto>={tanimoto_threshold}"
    return None
