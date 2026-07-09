"""Ingest modules registry.

Each module exposes ``INFO`` (SourceInfo) and ``ingest(cfg, limit) -> [record]``.
The registry maps the config source key to its module so the Phase 1 driver can
iterate enabled + license-cleared sources.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict

_MODULES = {
    "ome_silver": "rxndata.ingest.ome_silver",
    "ome_template": "rxndata.ingest.ome_template",
    "ord": "rxndata.ingest.ord",
    "uspto_lowe": "rxndata.ingest.uspto_lowe",
    "wikipedia": "rxndata.ingest.wikipedia",
}


def get_module(key: str):
    return import_module(_MODULES[key])


def all_keys() -> Dict[str, str]:
    return dict(_MODULES)
