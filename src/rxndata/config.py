"""Load and expose the pipeline configuration (configs/pipeline.yaml).

Everything stochastic or path-dependent in the pipeline reads from here so that
`make all` is reproducible and behavior changes only via the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline.yaml"


@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        """Resolve a configured path (relative to repo root) to an absolute Path."""
        rel = self.raw["paths"][key]
        p = Path(rel)
        return p if p.is_absolute() else (ROOT / p)

    def source(self, name: str) -> Dict[str, Any]:
        return self.raw["sources"][name]

    def enabled_sources(self) -> Dict[str, Dict[str, Any]]:
        return {k: v for k, v in self.raw["sources"].items() if v.get("enabled")}

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


@lru_cache(maxsize=4)
def load_config(path: str | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
