"""Ingest the Open Reaction Database (Tier B, CC-BY-SA-4.0).

ORD is ~2M reactions incl. Lowe's USPTO grants: conditions, yields, procedures.
Tier B teaches "what forms from what" -- overall transformations only, NEVER a
mechanism. Records here carry an empty ``mechanism`` and are used in Phase 8 for
forward/retro/reagent tasks and to seed Phase 3 mechanism inference.

Data access: the ord-data GitHub repo stores each dataset as a git-LFS
``.parquet`` (columns: reaction_id, reaction=serialized Reaction protobuf). We
fetch shards via the LFS media endpoint (media.githubusercontent.com), cache
them under data/raw/, and parse the protobuf with the light ``_ord_proto`` shim
(no psycopg2/flask). Bounded by config: ``ord.max_shards`` / ``limit``.

Share-alike duty (CC-BY-SA): derived datasets that include ORD-derived records
must themselves be BY-SA; recorded per-record via the ``license`` field and in
the data card so downstream release honors copyleft.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import Config, load_config
from ..io_utils import PoliteFetcher
from ..schema import make_record
from ._ord_proto import load_ord_protos
from .base import SourceInfo

INFO = SourceInfo(
    key="ord",
    display="Open Reaction Database",
    license="CC-BY-SA-4.0",
    tier="B",
    verdict="usable",
)

# GitHub API to list shard directories, LFS media endpoint to fetch content.
_GH_API = "https://api.github.com/repos/open-reaction-database/ord-data/contents/data"
_MEDIA = "https://media.githubusercontent.com/media/open-reaction-database/ord-data/main"

# How many top-level shard dirs to sample by default (00..ff = 256 dirs, ~2M rxns).
_DEFAULT_MAX_SHARDS = 2


def _list_shard_dirs(fetcher: PoliteFetcher, max_dirs: int) -> List[str]:
    body = fetcher.get(_GH_API, ext=".json")
    entries = json.loads(body)
    dirs = sorted(e["name"] for e in entries if e["type"] == "dir")
    return dirs[:max_dirs]


def _list_parquets(fetcher: PoliteFetcher, shard: str) -> List[str]:
    body = fetcher.get(f"{_GH_API}/{shard}", ext=".json")
    entries = json.loads(body)
    return [e["name"] for e in entries if e["name"].endswith(".parquet")]


def _extract_reaction(reaction_pb2, blob: bytes) -> Dict[str, Any]:
    # Enum constants live on the message CLASSES, not instances, in this
    # protobuf build. Resolve them once.
    SMILES = reaction_pb2.CompoundIdentifier.SMILES
    RXN_SMILES = reaction_pb2.ReactionIdentifier.REACTION_SMILES
    REACTANT = reaction_pb2.ReactionRole.REACTANT

    r = reaction_pb2.Reaction()
    r.ParseFromString(blob)

    reactants: List[str] = []
    reagents: List[str] = []
    for _, inp in r.inputs.items():
        for c in inp.components:
            smis = [i.value for i in c.identifiers if i.type == SMILES]
            if not smis:
                continue
            # role: REACTANT=1; everything else (reagent/solvent/catalyst) is a reagent.
            (reactants if c.reaction_role == REACTANT else reagents).append(smis[0])

    products: List[str] = []
    for oc in r.outcomes:
        for prod in oc.products:
            smis = [i.value for i in prod.identifiers if i.type == SMILES]
            if smis:
                products.append(smis[0])

    rxn_smiles = [i.value for i in r.identifiers if i.type == RXN_SMILES]

    # Conditions: a compact free-text summary if available.
    cond_bits = []
    if r.conditions.temperature.setpoint.value:
        cond_bits.append(f"T={r.conditions.temperature.setpoint.value}")
    if reagents:
        cond_bits.append("reagents:" + ".".join(reagents[:6]))
    return {
        "reactants": reactants,
        "reagents": reagents,
        "products": products,
        "rxn_smiles": rxn_smiles[0] if rxn_smiles else None,
        "conditions": "; ".join(cond_bits) or None,
    }


def ingest(cfg: Optional[Config] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    cfg = cfg or load_config()
    reaction_pb2, _ = load_ord_protos()
    fetcher = PoliteFetcher(cfg)

    max_shards = int(cfg.get("sources", "ord", "max_shards", default=_DEFAULT_MAX_SHARDS) or _DEFAULT_MAX_SHARDS)
    records: List[Dict[str, Any]] = []

    import pyarrow.parquet as pq

    for shard in _list_shard_dirs(fetcher, max_shards):
        for fname in _list_parquets(fetcher, shard):
            if limit is not None and len(records) >= limit:
                return records
            url = f"{_MEDIA}/data/{shard}/{fname}"
            blob = fetcher.get(url, ext=".parquet")
            cache_path = fetcher._cache_path(url, ".parquet")
            table = pq.read_table(cache_path)
            for row in table.to_pylist():
                if limit is not None and len(records) >= limit:
                    return records
                try:
                    ex = _extract_reaction(reaction_pb2, row["reaction"])
                except Exception:
                    continue
                if not ex["reactants"] or not ex["products"]:
                    continue
                rid = row["reaction_id"]
                rec = make_record(
                    reaction_id=f"ORD-{rid}",
                    source="ORD",
                    license="CC-BY-SA-4.0",
                    provenance="curated",   # overall reaction, not a mechanism
                    reactants_smiles=ex["reactants"],
                    products_smiles=ex["products"],
                    mechanism=[],           # Tier B: NO mechanism, ever
                    conditions=ex["conditions"],
                    reaction_smiles_mapped=None,
                    raw_ref={
                        "file": f"data/{shard}/{fname}",
                        "orig_id": rid,
                        "reagents": ex["reagents"],
                        "rxn_smiles_raw": ex["rxn_smiles"],
                    },
                )
                records.append(rec)
    return records
