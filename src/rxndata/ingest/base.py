"""Common helpers for ingest modules.

Each ingester is a callable ``ingest(cfg, limit=None) -> list[record]`` that
reads from an official source (cached under data/raw/) and emits normalized
records (schema.make_record). Phase 1 does NOT canonicalize SMILES or validate
chemistry -- that is Phase 2/5. It only parses and captures provenance+license.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import Config


def sample_rows(records: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    """Compact preview rows for the Phase 1 gate table."""
    out = []
    for r in records[:n]:
        out.append(
            {
                "reaction_id": r["reaction_id"],
                "provenance": r["provenance"],
                "name": r.get("name"),
                "n_reactants": len(r.get("reactants_smiles", [])),
                "n_products": len(r.get("products_smiles", [])),
                "n_steps": r.get("mechanism_step_nums"),
                "first_reactant": (r.get("reactants_smiles") or [None])[0],
            }
        )
    return out


class SourceInfo:
    """Static descriptor a module exposes so the CLI can gate on license verdict."""

    def __init__(self, key: str, display: str, license: str, tier: str, verdict: str):
        self.key = key
        self.display = display
        self.license = license
        self.tier = tier
        self.verdict = verdict

    def gate_ok(self, cfg: Config) -> bool:
        """True if this source is enabled and its verdict permits ingestion."""
        s = cfg.raw["sources"].get(self.key, {})
        if not s.get("enabled"):
            return False
        return s.get("verdict") in ("usable", "usable_for_research_only")
