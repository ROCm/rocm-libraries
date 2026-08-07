# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The import-time staleness guard must scan exactly what _stinkytofu.so is built from.

A source edit under one of the scanned trees (src/, include/, hardware/,
tools/tablegen, python_module/src) must mark the bindings stale so a rebuild is
demanded. src/conversion/ is the one carve-out: it holds the rocisa<->stinkytofu
glue compiled into _rocisa.so only, not into _stinkytofu.so, so editing it must
NOT flag the standalone binding.

These tests create their own probe files rather than touching tracked sources, so
a crash between setup and cleanup leaves a stray file rather than a source tree
whose mtimes are hours in the future -- which would deadlock every later import.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# The guard signals staleness with an ImportError of its own. On some CI runners
# (notably Windows) checkout-vs-build mtime ordering can trip that guard on tracked
# sources -- an environmental false positive, not a real regression. Skip the module
# in that case rather than hard-failing; the guard's real behaviour is still exercised
# by the subprocess probes below.
try:
    from stinkytofu import _build_info as _bi
except ImportError as exc:  # pragma: no cover - depends on the tree's state
    pytest.skip(f"cannot import stinkytofu bindings: {exc}", allow_module_level=True)

_SOURCE_ROOT = Path(_bi.SOURCE_ROOT)
_GFX_DIR = _SOURCE_ROOT / "hardware" / "src" / "gfx"
# The guard compares against the .so, so a probe must be newer than *that*, not
# merely newer than its neighbours. The extension module is private to the
# package, so find it by path rather than importing it.
_SO = next(iter(sorted(Path(_bi.__file__).parent.glob("_stinkytofu*.so"))), None)
if _SO is None:
    pytest.skip(
        "no _stinkytofu extension module beside _build_info", allow_module_level=True
    )


def _write_future_source(path: Path):
    """Create a source file newer than the .so, so the guard flags it if it looks there.

    The margin is seconds rather than hours on purpose: a probe leaked by a SIGKILL
    lands in a directory the guard scans, and a far-future mtime would then block
    every import until wall-clock time caught up, no matter how often you rebuild.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("// staleness guard probe\n")
    future = max(path.stat().st_mtime, _SO.stat().st_mtime) + 5
    os.utime(path, (future, future))


def _import_in_subprocess():
    """Import the same package the parent inspected, in a fresh interpreter.

    Other test modules in this directory put a build tree on sys.path
    unconditionally, so the ambient PYTHONPATH is not necessarily where
    `stinkytofu` was imported from. Leading the child's PYTHONPATH with the
    parent's actual package location keeps the two from testing different builds
    and reporting the disagreement as a guard failure.
    """
    package_root = str(Path(_bi.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (package_root, env.get("PYTHONPATH")) if p
    )
    return subprocess.run(
        [sys.executable, "-c", "import stinkytofu"],
        capture_output=True,
        text=True,
        env=env,
    )


def test_source_edit_under_gfx_marks_bindings_stale():
    """A newer source anywhere under the scanned hardware/src/gfx tree is stale."""
    if not _GFX_DIR.is_dir():
        pytest.skip("no hardware/src/gfx tree to probe")
    probe = _GFX_DIR / "_staleness_probe.def"
    assert (
        not probe.exists()
    ), f"{probe} already exists. If a previous run was killed, delete it and re-run."
    try:
        _write_future_source(probe)
        result = _import_in_subprocess()
    finally:
        probe.unlink(missing_ok=True)

    assert (
        result.returncode != 0
    ), "guard failed to notice a stale source under hardware/src/gfx"
    assert (
        probe.name in result.stderr
    ), f"guard flagged something other than the probe:\n{result.stderr}"


def test_conversion_glue_is_excluded_from_the_scan():
    """src/conversion/ feeds _rocisa.so only; editing it must not flag _stinkytofu.so."""
    conversion = _SOURCE_ROOT / "src" / "conversion"
    if not conversion.is_dir():
        pytest.skip("no src/conversion tree to probe")
    probe = conversion / "_staleness_probe.cpp"
    assert (
        not probe.exists()
    ), f"{probe} already exists. If a previous run was killed, delete it and re-run."
    try:
        _write_future_source(probe)
        result = _import_in_subprocess()
    finally:
        probe.unlink(missing_ok=True)

    assert (
        result.returncode == 0
    ), f"editing src/conversion glue wrongly marked the standalone binding stale:\n{result.stderr}"
