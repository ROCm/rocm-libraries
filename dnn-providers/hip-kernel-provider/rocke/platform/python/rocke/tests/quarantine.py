# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Skip the tests ``rocke.tests.pending_lanes.QUARANTINED`` holds.

Each pytest root that applies the quarantine calls :func:`skip_quarantined_items`
from its ``pytest_collection_modifyitems`` hook, with its own directory and that
directory's path relative to the rocke directory (one of
``pending_lanes.QUARANTINE_TREES``). Items are matched by their path below that
directory, which the install keeps (``platform/tests`` installs as ``tests``,
``library/tests`` as ``tests/library/tests``)::

    def pytest_collection_modifyitems(config, items):
        skip_quarantined_items(items, Path(__file__).parent, "library/tests")

A quarantined item is skipped with the reason ``quarantined: <issue>``, so it
stays visible in every report. An entry that names GPU arches skips its item
only when this process's GPU is one of them (``rocke.tests.gpu``), and adds
``on <arch>`` to the reason; one that names compilers skips it only when the
COMGR this process loads runs one of them (``rocke.tests.comgr``), and adds
``with llvm-project <commit>`` (or ``with LLVM <major>``). Each is probed only
when such an entry covers a collected item, after the test modules are
imported, and the compiler before the GPU: probing the GPU loads HIP, which
maps its own COMGR, and a process with a COMGR mapped loads no different one.
A compiler that cannot be identified (no COMGR loads here) is none of the
named ones, so its tests run. ``ROCKE_QUARANTINE=ignore`` runs every
quarantined test, to debug or fix it.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Iterable
from pathlib import Path

import pytest

from .comgr import CompilerBuild, loaded_compiler_build
from .gpu import probe_device_arch
from .pending_lanes import (
    QUARANTINE_SKIP_PREFIX,
    QUARANTINE_TREES,
    QUARANTINED,
    quarantined_test,
)

__all__ = ["QUARANTINE_ENV", "skip_quarantined_items"]

# Set to "ignore" to run quarantined tests.
QUARANTINE_ENV = "ROCKE_QUARANTINE"


def skip_quarantined_items(
    items: Iterable[pytest.Item], tree_root: Path, tree_path: str
) -> None:
    """Mark each of ``items`` under ``tree_root`` that a quarantine entry covers
    to be skipped; ``tree_path`` is ``tree_root`` relative to the rocke
    directory."""
    if tree_path not in QUARANTINE_TREES:
        raise ValueError(
            f"skip_quarantined_items failed - tree_path={tree_path!r}, expected "
            f"one of pending_lanes.QUARANTINE_TREES {QUARANTINE_TREES}"
        )
    mode = os.environ.get(QUARANTINE_ENV, "")
    if mode not in ("", "ignore"):
        raise ValueError(
            f"skip_quarantined_items failed - {QUARANTINE_ENV}={mode!r}, expected "
            "unset or 'ignore'"
        )
    if mode == "ignore":
        return
    # The quarantined test files of this tree, by their resolved path here.
    files = {
        (tree_root / entry.path.removeprefix(f"{tree_path}/")).resolve(): entry.path
        for entry in QUARANTINED
        if entry.path.startswith(f"{tree_path}/")
    }
    if not files:
        return
    resolved: dict[Path, Path] = {}
    for item in items:
        item_path = Path(item.path)
        if item_path not in resolved:
            resolved[item_path] = item_path.resolve()
        path = files.get(resolved[item_path])
        if path is None:
            continue
        entry = quarantined_test(path, item.nodeid.partition("::")[2])
        if entry is None:
            continue
        # The compiler first: probing the GPU loads HIP (see the docstring).
        build = _loaded_compiler_build() if entry.compilers else None
        if build is not None and not entry.holds_out_with(build):
            continue
        arch = _device_arch() if entry.arches else None
        if not entry.holds_out_on(arch):
            continue
        reason = f"{QUARANTINE_SKIP_PREFIX}{entry.issue}"
        if entry.arches:
            reason += f" on {arch}"
        if build is not None:
            reason += (
                f" with llvm-project {build.llvm_commit}"
                if build.llvm_commit
                else f" with LLVM {build.llvm_major}"
            )
        item.add_marker(pytest.mark.skip(reason=reason))


@functools.cache
def _device_arch() -> str | None:
    """This process's GPU arch, or None without a GPU; probed only when an entry
    that names arches covers a collected item."""
    return probe_device_arch()[0]


@functools.cache
def _loaded_compiler_build() -> CompilerBuild:
    """The compiler this process's COMGR runs; probed only when an entry that
    names compilers covers a collected item."""
    return loaded_compiler_build()
