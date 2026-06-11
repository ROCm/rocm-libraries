#!/usr/bin/env python3
################################################################################
# Subtile Python <-> C++ boundary guard.
#
# Pure source-text test (no rocisa / no compiled extension needed): it scans
# production Python under Tensile/ for imports of the Subtile package and
# asserts the external surface stays minimal, matching the contract in
# docs/subtile_cpp_boundary.md.
#
# Production code outside Tensile/Components/Subtile/ may import only:
#   * Tensile/KernelWriter.py        -> explicit named import from Subtile.Kernel
#   * Tensile/Components/StreamK.py  -> Kernel.localReadResetOffsetsSubtile
#
# The regex matches the opening line of a multi-line import block; the
# ALLOWED_PRODUCTION_SITES set contains the exact stripped first line.
# A new hit here means the boundary widened and must be triaged before merge.
#
# Usage: pytest test_subtile_boundary.py -v
################################################################################

import re
from pathlib import Path

# tensilelite/Tensile/Tests/unit/<this file> -> Tensile dir is parents[2].
TENSILE_DIR = Path(__file__).resolve().parents[2]
SUBTILE_PKG = TENSILE_DIR / "Components" / "Subtile"
TESTS_DIR = TENSILE_DIR / "Tests"

_IMPORT_RE = re.compile(r"(?:from\s+\S*Subtile|import\s+\S*Subtile)")

# (relative-to-Tensile path) -> set of sanctioned import lines (verbatim).
# KernelWriter.py uses an explicit multi-line import; only the opening line
# matches the regex (grc.150: wildcard replaced with named symbols).
ALLOWED_PRODUCTION_SITES = {
    "KernelWriter.py": {"from .Components.Subtile.Kernel import ("},
    "Components/StreamK.py": {
        "from .Subtile.Kernel import localReadResetOffsetsSubtile"
    },
}


def _production_subtile_imports():
    """Map of relpath -> set(import lines) for non-test, non-Subtile sources."""
    found = {}
    for path in TENSILE_DIR.rglob("*.py"):
        rp = path.resolve()
        if SUBTILE_PKG in rp.parents or rp == SUBTILE_PKG:
            continue
        if TESTS_DIR in rp.parents:
            continue
        lines = {
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if _IMPORT_RE.search(ln)
        }
        if lines:
            found[str(path.relative_to(TENSILE_DIR))] = lines
    return found


def test_production_subtile_import_sites_are_minimal():
    found = _production_subtile_imports()

    unexpected_files = set(found) - set(ALLOWED_PRODUCTION_SITES)
    assert not unexpected_files, (
        "New production importer(s) of the Subtile package: "
        f"{sorted(unexpected_files)}. Update docs/subtile_cpp_boundary.md "
        "and this guard if the boundary intentionally changed."
    )

    for site, allowed in ALLOWED_PRODUCTION_SITES.items():
        assert site in found, (
            f"Expected production Subtile importer '{site}' is gone; "
            "update the boundary contract if this is intentional."
        )
        extra = found[site] - allowed
        assert not extra, (
            f"{site} added unsanctioned Subtile import(s): {sorted(extra)}. "
            "Only the surface in docs/subtile_cpp_boundary.md is allowed."
        )


def test_streamk_uses_only_sanctioned_lr_symbol():
    streamk = (TENSILE_DIR / "Components" / "StreamK.py").read_text(
        encoding="utf-8"
    )
    subtile_imports = [
        ln.strip()
        for ln in streamk.splitlines()
        if _IMPORT_RE.search(ln)
    ]
    assert subtile_imports == [
        "from .Subtile.Kernel import localReadResetOffsetsSubtile"
    ], (
        "StreamK.py may import only localReadResetOffsetsSubtile from the "
        "Subtile package (via Kernel.py); got: " + repr(subtile_imports)
    )
