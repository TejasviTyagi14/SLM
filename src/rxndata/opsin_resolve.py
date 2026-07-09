"""OPSIN name->structure resolution for textbook/encyclopedia-mined reactions.

HARD RULE: text -> structure ALWAYS goes through OPSIN + RDKit re-canonicalization;
never trust a name without resolving and re-canonicalizing it. We do NOT hand-roll
name parsing.

OPSIN runs on the JVM (py2opsin bundles the jar). If no Java runtime is present,
this module degrades gracefully: ``java_available()`` returns False and
``resolve_name`` returns None with reason ``no_java`` so the pipeline can proceed
(Tier C name resolution is deferred/flagged rather than crashing the run).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from .normalize import canonicalize


def _java_works(java_bin: str = "java") -> bool:
    try:
        subprocess.run([java_bin, "-version"], capture_output=True, check=True, timeout=15)
        return True
    except Exception:
        return False


def _ensure_java_on_path() -> None:
    """Ensure a WORKING java is first on PATH.

    macOS ships a /usr/bin/java stub that shutil.which() finds but which exits
    non-zero when no JDK is installed; so we test that java actually runs and,
    if not, prepend a known (keg-only Homebrew) JDK bin dir. Keeps make/CI
    reproducible without the caller exporting PATH.
    """
    if shutil.which("java") is not None and _java_works():
        return
    candidates = [
        "/opt/homebrew/opt/openjdk/bin",       # Apple Silicon Homebrew
        "/usr/local/opt/openjdk/bin",          # Intel Homebrew
        "/opt/homebrew/opt/openjdk@21/bin",
    ]
    for c in candidates:
        cand = os.path.join(c, "java")
        if os.path.exists(cand) and _java_works(cand):
            os.environ["PATH"] = c + os.pathsep + os.environ.get("PATH", "")
            return


@lru_cache(maxsize=1)
def java_available() -> bool:
    _ensure_java_on_path()
    if shutil.which("java") is None:
        return False
    try:
        subprocess.run(
            ["java", "-version"],
            capture_output=True,
            check=True,
            timeout=15,
        )
        return True
    except Exception:
        return False


@dataclass
class NameResolution:
    name: str
    smiles: Optional[str]
    canonical: Optional[str]
    ok: bool
    reason: Optional[str] = None


@lru_cache(maxsize=4096)
def resolve_name(name: str) -> NameResolution:
    """Resolve an IUPAC/systematic name to canonical SMILES via OPSIN."""
    if not name or not name.strip():
        return NameResolution(name, None, None, False, "empty")
    if not java_available():
        return NameResolution(name, None, None, False, "no_java")

    try:
        from py2opsin import py2opsin
    except Exception as e:  # noqa: BLE001
        return NameResolution(name, None, None, False, f"no_py2opsin:{type(e).__name__}")

    try:
        raw = py2opsin(name, output_format="SMILES")
    except Exception as e:  # noqa: BLE001
        return NameResolution(name, None, None, False, f"opsin_error:{type(e).__name__}")

    if not raw or not isinstance(raw, str) or not raw.strip():
        return NameResolution(name, None, None, False, "opsin_no_result")

    smiles = raw.strip().splitlines()[0].strip()
    res = canonicalize(smiles, keep_maps=False)
    if not res.ok:
        return NameResolution(name, smiles, None, False, f"canon_fail:{res.reason}")
    return NameResolution(name, smiles, res.canonical, True)
