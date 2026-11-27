# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU tests run only in GPU entries.

An installed CTest entry whose tests launch GPU work takes the ``rocke_gpu``
resource lock, so a parallel ``ctest -j`` never runs two such entries on the
one GPU, and the GPU per-test policy (``test_engine_linkage`` checks both on the
gpu-labelled entries). A test declares that it launches GPU work with the
``gpu`` mark: on the test, its class, one ``pytest.param`` or its module's
``pytestmark``.

This check configures the installed entries (nothing is built) and refuses a
pytest entry without the ``rocke_gpu`` lock that selects a test module

1. that marks a test ``gpu``, unless the entry's ``-m`` expression deselects
   the mark whatever other marks the module puts on that test; or
2. that uses one of ``LAUNCH_ENTRY_POINTS`` and marks no test ``gpu``, unless
   it lies in one of ``FIXTURE_MARKED_TREES`` or is one of
   ``HOST_ONLY_IMPORTERS``.

What it sees, and what it does not:

- It reads the source. A mark that a conftest adds at collection is not seen.
  The conftest of each of ``FIXTURE_MARKED_TREES`` marks ``gpu`` the tests that
  ask for one of its GPU fixtures, and the tree's entries select by ``-m gpu``
  and ``-m "not gpu"``; rule 2 skips those trees whole, so a test there that
  launches without asking for a GPU fixture is not seen.
- It reads an entry's ``-m`` expression, not its ``-k`` filters or node ids, so
  an entry that leaves gpu tests out another way still deselects them with
  ``-m``.
- Rule 2 reads imports of the entry points named below (from their module, or
  ``<module>.<name>`` after importing the module), not calls. A test that
  launches through a helper of another module, a fixture or a child process,
  and carries no ``gpu`` mark, is not seen, and one ``gpu`` mark in a module
  turns rule 2 off for the module's other tests.
"""

from __future__ import annotations

import ast
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
from rocke.assets import platform_root
from rocke.tests.ctest_listing import (
    GPU_RESOURCE_LOCK,
    PROVIDER_CTEST_SUBDIR,
    configure_installed_ctest,
    list_installed_ctest_entries,
)
from rocke.tests.entry_selection import (
    InstallLayout,
    Layout,
    SourceLayout,
    arguments_after_interpreter,
    keeps_marked_tests,
    mark_names,
    pytest_selection,
    runs_pytest,
    selected_test_files,
)
from rocke.tests.lanes import SOURCE_TREE, skip_until

ROCKE = platform_root().parent.resolve()
GPU_MARK = "gpu"
# The device entry points a test calls to launch a kernel or move data to the
# GPU: the runtime itself (also re-exported by rocke.runtime) and the test
# helpers built on it.
LAUNCH_ENTRY_POINTS: Mapping[str, frozenset[str]] = {
    "rocke.runtime": frozenset({"Runtime"}),
    "rocke.runtime.hip_module": frozenset({"Runtime"}),
    "rocke.tests.gpu": frozenset({"launch_with_buffers", "run_on_device"}),
    "rocke.tests.universal_gemm": frozenset({"launch_universal_gemm"}),
}
# Test trees whose conftest marks gpu the tests that ask for its GPU fixtures.
FIXTURE_MARKED_TREES = (ROCKE / "platform/python/rocke/tests/helpers/tiling",)
# Test modules that use a launch entry point but launch nothing, each with
# why. Read the module before adding one.
HOST_ONLY_IMPORTERS: Mapping[str, str] = {
    "platform/tests/test_rocke.py": "patches the runtime's HIP calls or fakes it",
    "platform/tests/runtime/test_cluster_launch.py": "patches the runtime's HIP calls",
    "library/tests/test_aot_launch_zeroing.py": "inspects Runtime's signature only",
}


def used_launch_entry_points(source: str) -> set[str]:
    """The ``LAUNCH_ENTRY_POINTS`` a test module's source uses, as
    ``module.name``: imported by name from their module, or read as an
    attribute of the module imported under a local name, wherever the import
    statement is."""
    tree = ast.parse(source)
    used = set()
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name in LAUNCH_ENTRY_POINTS.get(node.module, ()):
                    used.add(f"{node.module}.{alias.name}")
                submodule = f"{node.module}.{alias.name}"
                if submodule in LAUNCH_ENTRY_POINTS:
                    module_aliases[alias.asname or alias.name] = submodule
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in LAUNCH_ENTRY_POINTS and alias.asname:
                    module_aliases[alias.asname] = alias.name
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in module_aliases
        ):
            module = module_aliases[node.value.id]
            if node.attr in LAUNCH_ENTRY_POINTS[module]:
                used.add(f"{module}.{node.attr}")
    return used


def holds_gpu_lock(entry: Mapping) -> bool:
    """Whether a listed CTest entry takes the ``rocke_gpu`` resource lock."""
    for prop in entry.get("properties", []):
        if prop["name"] == "RESOURCE_LOCK":
            return GPU_RESOURCE_LOCK in prop["value"]
    return False


def module_breach(entry: str, marker_expression: str | None, test_file: Path) -> str:
    """Why entry ``entry``, which holds no GPU lock and selects ``test_file``
    with ``-m marker_expression``, runs GPU tests of it; "" when it does not."""
    source = test_file.read_text(encoding="utf-8")
    shown = test_file.relative_to(ROCKE) if ROCKE in test_file.parents else test_file
    if GPU_MARK in mark_names(source):
        if not keeps_marked_tests(marker_expression, test_file, GPU_MARK):
            return ""
        return (
            f"{entry} runs the gpu tests of {shown.as_posix()} without the "
            f'{GPU_RESOURCE_LOCK} lock; deselect them with -m "not {GPU_MARK}" '
            f"and run them in a GPU entry"
        )
    if any(tree in test_file.parents for tree in FIXTURE_MARKED_TREES):
        return ""
    if shown.as_posix() in HOST_ONLY_IMPORTERS:
        return ""
    entry_points = used_launch_entry_points(source)
    if not entry_points:
        return ""
    return (
        f"{entry} runs {shown.as_posix()}, which uses "
        f"{', '.join(sorted(entry_points))} and marks no test {GPU_MARK}; mark the "
        f"tests that launch GPU work pytest.mark.{GPU_MARK} and run them in a GPU "
        f"entry"
    )


def _without_marker_expression(arguments: list[str]) -> list[str]:
    """``arguments`` less every ``-m <expression>`` pair."""
    kept: list[str] = []
    skip_next = False
    for argument in arguments:
        if skip_next:
            skip_next = False
        elif argument == "-m":
            skip_next = True
        else:
            kept.append(argument)
    return kept


def placement_breaches(
    entries: Iterable[Mapping], layout: Layout, base: str = ""
) -> list[str]:
    """One line per test module that a pytest entry without the GPU lock runs
    GPU tests of, by the two rules of this file's docstring."""
    breaches = []
    for entry in entries:
        command = entry.get("command", [])
        if not runs_pytest(command) or holds_gpu_lock(entry):
            continue
        arguments = arguments_after_interpreter(command)
        marker_expression = pytest_selection(arguments).marker_expression
        # The files the entry's paths and ignores select, whatever its -m
        # expression: selected_test_files drops a module by its pytestmark
        # alone, which misses a gpu test with marks of its own.
        unfiltered = _without_marker_expression(arguments)
        for test_file in sorted(selected_test_files(unfiltered, layout, base)):
            if breach := module_breach(entry["name"], marker_expression, test_file):
                breaches.append(breach)
    return breaches


def _entry(name: str, arguments: list[str], lock: str | None = None) -> dict:
    """A listed CTest entry as ``list_ctest_entries`` returns it."""
    properties = [] if lock is None else [{"name": "RESOURCE_LOCK", "value": [lock]}]
    return {
        "name": name,
        "command": ["python", "-u", "-m", "pytest", *arguments],
        "properties": properties,
    }


_MODULES = {
    "test_gpu.py": """
import pytest

@pytest.mark.gpu
def test_launches():
    pass

def test_lowers():
    pass
""",
    "test_gpu_exhaustive.py": """
import pytest

@pytest.mark.gpu
@pytest.mark.exhaustive
def test_launches_every_case():
    pass
""",
    "test_unmarked_launch.py": """
from rocke.tests.gpu import run_on_device

def test_launches():
    run_on_device
""",
    "test_reexported_runtime.py": """
from rocke.runtime import Runtime

def test_launches():
    Runtime
""",
    "test_runtime_by_module.py": """
from rocke.runtime import hip_module as hip

def test_launches():
    hip.Runtime
""",
    "test_host.py": """
from rocke.runtime import hip_module

def test_reads_the_arch():
    hip_module.get_device_arch
""",
}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name, source in _MODULES.items():
        (tmp_path / name).write_text(source)
    return tmp_path


def test_a_host_entry_that_keeps_gpu_tests_is_refused(tree):
    path = (tree / "test_gpu.py").as_posix()
    [breach] = placement_breaches([_entry("host", [path])], SourceLayout())
    assert breach.startswith(f"host runs the gpu tests of {path} without the")
    [breach] = placement_breaches(
        [_entry("host", [path, "-m", "not integration"])], SourceLayout()
    )
    assert 'deselect them with -m "not gpu"' in breach


def test_a_gpu_test_with_another_mark_is_refused_where_that_mark_selects_it(tree):
    """The module's pytestmark sets no mark, but its gpu test also carries
    exhaustive, which an entry selecting -m exhaustive keeps."""
    path = (tree / "test_gpu_exhaustive.py").as_posix()
    [breach] = placement_breaches(
        [_entry("host", [path, "-m", "exhaustive"])], SourceLayout()
    )
    assert breach.startswith(f"host runs the gpu tests of {path} without the")
    entries = [_entry("host", [path, "-m", "exhaustive and not gpu"])]
    assert placement_breaches(entries, SourceLayout()) == []


def test_a_host_entry_that_deselects_gpu_tests_passes(tree):
    entries = [
        _entry(
            "host", [str(tree / "test_gpu.py"), "-m", "not integration and not gpu"]
        ),
        _entry("device", [str(tree / "test_gpu.py")], lock=GPU_RESOURCE_LOCK),
    ]
    assert placement_breaches(entries, SourceLayout()) == []


@pytest.mark.parametrize(
    "module, entry_point",
    [
        ("test_unmarked_launch.py", "rocke.tests.gpu.run_on_device"),
        ("test_reexported_runtime.py", "rocke.runtime.Runtime"),
        ("test_runtime_by_module.py", "rocke.runtime.hip_module.Runtime"),
    ],
)
def test_an_unmarked_module_that_uses_a_launch_entry_point_is_refused(
    tree, module, entry_point
):
    path = (tree / module).as_posix()
    [breach] = placement_breaches([_entry("host", [path])], SourceLayout())
    assert breach == (
        f"host runs {path}, which uses {entry_point} and marks no test gpu; mark "
        "the tests that launch GPU work pytest.mark.gpu and run them in a GPU entry"
    )
    entries = [_entry("device", [path], lock=GPU_RESOURCE_LOCK)]
    assert placement_breaches(entries, SourceLayout()) == []


def test_reading_the_device_arch_is_not_a_launch(tree):
    entries = [_entry("host", [str(tree / "test_host.py")])]
    assert placement_breaches(entries, SourceLayout()) == []


def test_a_module_of_a_fixture_marked_tree_needs_no_source_mark():
    """test_leading_axes.py launches through the tiling tree's GPU fixtures,
    which mark its GPU tests at collection."""
    module = ROCKE / "platform/python/rocke/tests/helpers/tiling/test_leading_axes.py"
    if not module.is_file():
        skip_until(SOURCE_TREE, reason="the tiling tests are absent here")
    source = module.read_text(encoding="utf-8")
    assert used_launch_entry_points(source)
    assert GPU_MARK not in mark_names(source)
    assert module_breach("host", None, module) == ""


@pytest.mark.parametrize("path", sorted(HOST_ONLY_IMPORTERS))
def test_a_host_only_importer_uses_an_entry_point_and_marks_no_gpu_test(path):
    """An entry of HOST_ONLY_IMPORTERS that no longer uses an entry point, or
    that now marks a gpu test, is stale: remove it."""
    module = ROCKE / path
    if not module.is_file():
        skip_until(SOURCE_TREE, reason=f"{path} is absent from this tree")
    source = module.read_text(encoding="utf-8")
    assert used_launch_entry_points(source), path
    assert GPU_MARK not in mark_names(source), path


def test_another_resource_lock_is_not_the_gpu_lock(tree):
    entries = [
        _entry("gate", [str(tree / "test_gpu.py")], lock="rocke_counted_sync_gate")
    ]
    assert len(placement_breaches(entries, SourceLayout())) == 1


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("ctest") is None,
    reason="CMake and CTest required",
)
def test_no_installed_host_entry_runs_gpu_tests(tmp_path):
    build, ctest_dir = configure_installed_ctest(
        tmp_path, f"-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR={PROVIDER_CTEST_SUBDIR}"
    )
    entries = list_installed_ctest_entries(ctest_dir, tmp_path)
    assert any(
        runs_pytest(entry.get("command", [])) for entry in entries
    ), "CTest listed no pytest command"
    breaches = placement_breaches(
        entries, InstallLayout.of_build_tree(build), f"bin/{PROVIDER_CTEST_SUBDIR}/"
    )
    assert not breaches, "\n".join(breaches)
