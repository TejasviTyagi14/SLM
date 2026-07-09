"""Load ORD's generated protobuf modules WITHOUT importing ord_schema/__init__.

The full ``ord_schema`` package pulls heavy deps (psycopg2, werkzeug, flask) at
import time that fail to build on a bare macOS system Python and that we do not
need: reading a dataset only requires the generated ``reaction_pb2`` /
``dataset_pb2`` protobuf classes, which depend on nothing but the ``protobuf``
runtime. This shim loads those modules by file path so ingestion stays
dependency-light and reproducible.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import types
from functools import lru_cache


@lru_cache(maxsize=1)
def load_ord_protos():
    """Return (reaction_pb2, dataset_pb2), importing them in isolation."""
    site_candidates = glob.glob(
        os.path.join(sys.prefix, "lib", "python*", "site-packages", "ord_schema", "proto")
    )
    if not site_candidates:
        # Fall back to a normal import if the package is fully installed.
        from ord_schema.proto import dataset_pb2, reaction_pb2  # type: ignore
        return reaction_pb2, dataset_pb2

    proto_dir = site_candidates[0]
    pkg_dir = os.path.dirname(proto_dir)

    # Register lightweight package shims so relative proto imports resolve.
    if "ord_schema" not in sys.modules:
        pkg = types.ModuleType("ord_schema")
        pkg.__path__ = [pkg_dir]
        sys.modules["ord_schema"] = pkg
    if "ord_schema.proto" not in sys.modules:
        ppkg = types.ModuleType("ord_schema.proto")
        ppkg.__path__ = [proto_dir]
        sys.modules["ord_schema.proto"] = ppkg

    def _load(name: str):
        full = f"ord_schema.proto.{name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, os.path.join(proto_dir, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    reaction_pb2 = _load("reaction_pb2")
    dataset_pb2 = _load("dataset_pb2")
    return reaction_pb2, dataset_pb2
