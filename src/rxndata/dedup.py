"""Phase 7: near-duplicate removal across surviving records.

Two records are duplicates if they share either
  - the same normalized mechanism hash (subtype + canonical-intermediate seq), or
  - the same reactant/product InChIKey signature.
Within a duplicate cluster we keep ONE representative, preferring the highest
provenance (curated > template_expanded > inferred), then the one with more steps
(richer), then the lexicographically-smallest reaction_id for determinism.

We also union near-duplicates by reaction-fingerprint Tanimoto >= threshold via a
union-find over an LSH-free O(n^2)-within-bucket pass bucketed by product InChIKey
first-block (keeps it tractable and deterministic).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from rdkit import DataStructs

from .decontaminate import (
    inchikey_signature,
    mechanism_hash,
    reaction_fingerprint,
)

PROV_RANK = {"curated": 3, "template_expanded": 2, "inferred": 1}


def _prov_rank(rec: dict) -> int:
    return PROV_RANK.get(rec.get("provenance"), 0)


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _representative(cluster: List[dict]) -> dict:
    """Keep highest provenance, then most steps, then smallest id (deterministic)."""
    return sorted(
        cluster,
        key=lambda r: (-_prov_rank(r), -len(r.get("mechanism", [])), r.get("reaction_id", "")),
    )[0]


def deduplicate(records: List[dict], tanimoto_threshold: float = 0.97) -> Tuple[List[dict], Dict]:
    """Return (kept_records, stats)."""
    n = len(records)
    uf = _UnionFind(n)

    # Exact keys: mechanism hash and InChIKey signature.
    by_mech: Dict[str, int] = {}
    by_sig: Dict[str, int] = {}
    fps: List[Optional[object]] = [None] * n
    bucket: Dict[str, List[int]] = defaultdict(list)

    for i, rec in enumerate(records):
        mh = mechanism_hash(rec.get("mechanism", []))
        if mh:
            if mh in by_mech:
                uf.union(i, by_mech[mh])
            else:
                by_mech[mh] = i
        sig = inchikey_signature(rec.get("reactants_smiles", []), rec.get("products_smiles", []))
        if sig:
            if sig in by_sig:
                uf.union(i, by_sig[sig])
            else:
                by_sig[sig] = i
        fps[i] = reaction_fingerprint(rec.get("reactants_smiles", []), rec.get("products_smiles", []))
        # Bucket by first product InChIKey block to bound the fp comparison.
        prods = rec.get("products_smiles", [])
        bkey = sig.split(">>")[-1][:16] if sig else f"_none_{i}"
        bucket[bkey].append(i)

    # Fingerprint near-duplicate union within buckets (bounded).
    fp_pairs = 0
    for idxs in bucket.values():
        if len(idxs) < 2:
            continue
        for a_pos in range(len(idxs)):
            ia = idxs[a_pos]
            if fps[ia] is None:
                continue
            for b_pos in range(a_pos + 1, len(idxs)):
                ib = idxs[b_pos]
                if fps[ib] is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(fps[ia], fps[ib])
                fp_pairs += 1
                if sim >= tanimoto_threshold:
                    uf.union(ia, ib)

    clusters: Dict[int, List[dict]] = defaultdict(list)
    for i, rec in enumerate(records):
        clusters[uf.find(i)].append(rec)

    kept: List[dict] = []
    cluster_sizes: List[int] = []
    for members in clusters.values():
        cluster_sizes.append(len(members))
        kept.append(_representative(members))

    from collections import Counter
    size_dist = Counter(cluster_sizes)
    stats = {
        "n_in": n,
        "n_kept": len(kept),
        "n_removed": n - len(kept),
        "n_clusters": len(clusters),
        "cluster_size_distribution": dict(sorted(size_dist.items())),
        "largest_cluster": max(cluster_sizes) if cluster_sizes else 0,
        "fp_comparisons": fp_pairs,
    }
    return kept, stats
