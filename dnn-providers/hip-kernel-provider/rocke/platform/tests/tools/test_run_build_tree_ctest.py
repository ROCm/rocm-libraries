# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The build-tree CTest runner finds the entries it is asked for, or refuses.

``tools/run_build_tree_ctest.py`` configures the platform alone, without the
sibling library, and runs named CTest entries of that tree. The entries the
pre-commit hook names must be registered there. A name CTest's ``-R`` filter
does not match would just leave nothing to run for it, so a renamed or removed
entry would drop out of the check unnoticed; the runner refuses such a name
before it builds anything.
"""

from __future__ import annotations

import importlib.util
import shutil

import pytest
from rocke.assets import platform_tools_dir
from rocke.tests.lanes import SOURCE_TREE_CASES, skip_until_reason
from rocke.tests.platform_build import configure_platform

RUNNER = platform_tools_dir() / "run_build_tree_ctest.py"

# The installed test layout ships no tools/ tree.
pytestmark = [
    pytest.mark.skipif(
        not RUNNER.exists(),
        reason=skip_until_reason(
            SOURCE_TREE_CASES,
            reason="runs a platform tool, which an installed test layout omits",
        ),
    ),
    pytest.mark.skipif(
        shutil.which("cmake") is None or shutil.which("ctest") is None,
        reason="CMake and CTest required",
    ),
]

# The build-tree entries the rocke-emitter-pair-parity hook runs, which the
# platform registers on every host with Python.
HOOK_ENTRIES = (
    "rocke_optimization_barrier_hip_pair",
    "rocke_coalesced_loader_async_refusals",
)


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("rocke_run_build_tree_ctest", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def configured_tree(runner, tmp_path_factory):
    """A platform build tree configured as the runner configures it, unbuilt."""
    build_root = tmp_path_factory.mktemp("build")
    configure_platform(build_root, *runner.PLATFORM_ONLY, config=runner.CONFIG)
    return build_root


def test_the_platform_alone_registers_the_hook_entries(runner, configured_tree):
    runner.require_registered(configured_tree, HOOK_ENTRIES)


def test_an_unregistered_entry_is_refused_by_name(runner, configured_tree):
    with pytest.raises(
        ValueError,
        match=r"unregistered=\['rocke_no_such_entry'\], expected every named test "
        r"registered in ",
    ):
        runner.require_registered(
            configured_tree, [*HOOK_ENTRIES, "rocke_no_such_entry"]
        )
