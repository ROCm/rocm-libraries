# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import os
import types
from typing import Any, Tuple
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

import Tensile.BenchmarkProblems as bp
from Tensile import LibraryIO
from Tensile.Common import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, defaultSolution, globalParameters
from Tensile.SolutionStructs.Problem import ProblemType
from Tensile.SolutionStructs.Solution import Solution
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

POOL_FILE = os.path.join(os.path.dirname(__file__), "test_data", "solution_pool_gfx950.yaml")
_POOL_ISA = IsaVersion(9, 5, 0)


@pytest.fixture(scope="module")
def pool_env() -> types.SimpleNamespace:
    """GPU-free assembler + ISA map matching ``solution_pool_gfx950.yaml``."""
    cxxCompiler, _c_compiler, offload_bundler = validateToolchain(
        ToolchainDefaults.CXX_COMPILER,
        ToolchainDefaults.C_COMPILER,
        ToolchainDefaults.OFFLOAD_BUNDLER,
    )
    isa_info_map = makeIsaInfoMap([_POOL_ISA], cxxCompiler)
    assignGlobalParameters({}, isa_info_map)
    assembler = makeAssemblyToolchain(cxxCompiler, offload_bundler, "default").assembler
    return types.SimpleNamespace(
        assembler=assembler,
        isaInfoMap=isa_info_map,
        debugConfig=types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=False,
            printIndexAssignmentInfo=False,
        ),
    )


@pytest.fixture(autouse=True)
def _sequential_cpu_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid oversubscribing workers when pytest-xdist runs the unit suite."""
    monkeypatch.setitem(globalParameters, "CpuThreads", 1)


def _exact_logic_list_to_dict(exact_logic: list[Any] | None) -> dict[tuple[Any, ...], Any]:
    """Convert legacy ``ExactLogic`` list-of-pairs to the dict ``createLibraryLogic`` expects.

    Args:
        exact_logic: ``ExactLogic`` as produced by :func:`LibraryIO.parseLibraryLogicList`
            (list of ``[size_list, index_and_eff]``), or ``None``.

    Returns:
        Mapping from size tuple to the per-size payload (typically ``[sol_idx, eff]``).

    Raises:
        None.
    """
    if not exact_logic:
        return {}
    out: dict[tuple[Any, ...], Any] = {}
    for size_list, payload in exact_logic:
        out[tuple(size_list)] = payload
    return out


def _logic_tuple_from_parsed_pool(
    pdata: dict[str, Any],
    solutions: list[Solution],
) -> tuple[Any, ...]:
    """Build the ``logicTuple`` argument for :func:`LibraryIO.createLibraryLogic`.

    Args:
        pdata: Parsed pool mapping from :func:`Tensile.BenchmarkProblems._parsePoolFile`.
        solutions: Concrete :class:`~Tensile.SolutionStructs.Solution.Solution` instances.

    Returns:
        Tuple ``(problemType, solutions, indexOrder, exactLogic, rangeLogic,
        tileSelectionSolutions, tileSelectionIndices, perfMetric)``.

    Raises:
        KeyError: If *pdata* is missing required keys.
    """
    problem_type = ProblemType(pdata["ProblemType"], False)
    index_order = pdata["IndexOrder"]
    exact_dict = _exact_logic_list_to_dict(pdata.get("ExactLogic"))
    range_logic = pdata.get("RangeLogic")
    perf_metric = pdata.get("PerfMetric", "DeviceEfficiency")
    return (
        problem_type,
        solutions,
        index_order,
        exact_dict,
        range_logic,
        None,
        None,
        perf_metric,
    )


def _parse_pool(pool_env: types.SimpleNamespace) -> Tuple[dict[str, Any], list[Solution]]:
    """Load the gfx950 pool file and build :class:`Solution` objects.

    Args:
        pool_env: Toolchain fixture (assembler + ``isaInfoMap`` + debug flags).

    Returns:
        ``(parsed_data, solutions)`` where *parsed_data* is the dict from
        :func:`LibraryIO.parseLibraryLogicList`.

    Raises:
        AssertionError: If the pool yields zero solutions.
    """
    _pt_str, pool_path, pdata = bp._parsePoolFile(POOL_FILE)
    pool_entries = [(pool_path, pdata)]
    solutions = bp._constructAllPoolSolutions(
        pool_entries,
        pool_env.assembler,
        pool_env.debugConfig,
        pool_env.isaInfoMap,
    )
    assert len(solutions) >= 1
    return pdata, solutions


def test_parse_library_logic_data_list_input(pool_env: types.SimpleNamespace) -> None:
    """List-format YAML is converted then parsed into a ``LibraryLogic`` record."""
    raw = LibraryIO.read(POOL_FILE, customizedLoader=True)
    assert isinstance(raw, list)
    fields = LibraryIO.parseLibraryLogicData(
        copy.deepcopy(raw),
        POOL_FILE,
        pool_env.assembler,
        False,
        False,
        False,
        pool_env.isaInfoMap,
        False,
    )
    assert fields.schedule == "gfx950"
    assert fields.architecture == "gfx950"
    assert len(fields.solutions) == 1
    assert isinstance(fields.solutions[0], Solution)


def test_parse_library_logic_data_dict_input_matches_list(pool_env: types.SimpleNamespace) -> None:
    """Dict input (post-:func:`parseLibraryLogicList`) parses like the equivalent list."""
    raw_list = LibraryIO.read(POOL_FILE, customizedLoader=True)
    assert isinstance(raw_list, list)
    as_dict = LibraryIO.parseLibraryLogicList(copy.deepcopy(raw_list), POOL_FILE)

    from_list = LibraryIO.parseLibraryLogicData(
        copy.deepcopy(raw_list),
        POOL_FILE,
        pool_env.assembler,
        False,
        False,
        False,
        pool_env.isaInfoMap,
        False,
    )
    from_dict = LibraryIO.parseLibraryLogicData(
        copy.deepcopy(as_dict),
        POOL_FILE,
        pool_env.assembler,
        False,
        False,
        False,
        pool_env.isaInfoMap,
        False,
    )
    assert from_list.schedule == from_dict.schedule
    assert from_list.architecture == from_dict.architecture
    assert len(from_list.solutions) == len(from_dict.solutions)


def test_parse_library_logic_data_default_solution_overrides_solution_state(
    pool_env: types.SimpleNamespace,
) -> None:
    """``DefaultSolution`` supplies values for keys omitted from each solution dict."""
    pdata, _solutions = _parse_pool(pool_env)
    data = copy.deepcopy(pdata)
    baseline_stagger = defaultSolution["StaggerU"]
    override = 999 if baseline_stagger != 999 else 1001
    data["DefaultSolution"] = {"StaggerU": override}
    for sol in data["Solutions"]:
        sol.pop("StaggerU", None)

    fields = LibraryIO.parseLibraryLogicData(
        data,
        POOL_FILE,
        pool_env.assembler,
        False,
        False,
        False,
        pool_env.isaInfoMap,
        False,
    )
    assert fields.solutions[0]["StaggerU"] == override


def test_parse_library_logic_data_normalizes_top_level_gridbased(
    pool_env: types.SimpleNamespace,
) -> None:
    """Top-level ``LibraryType: GridBased`` is normalized to ``Matching`` + ``Library``."""
    pdata, _solutions = _parse_pool(pool_env)
    data = copy.deepcopy(pdata)
    data["LibraryType"] = "GridBased"
    data.pop("Library", None)

    LibraryIO.parseLibraryLogicData(
        data,
        POOL_FILE,
        pool_env.assembler,
        False,
        False,
        False,
        pool_env.isaInfoMap,
        False,
    )
    assert data["LibraryType"] == "Matching"
    assert isinstance(data.get("Library"), dict)
    assert data["Library"].get("distance") == "GridBased"


@pytest.mark.parametrize("architecture", ["gfx950", "gfx1250"])
@patch.object(LibraryIO, "getCUCount", return_value=304)
def test_create_library_logic_returns_dict_format(
    _mock_cu: Any,
    architecture: str,
    pool_env: types.SimpleNamespace,
) -> None:
    """``createLibraryLogic`` always returns dict-format YAML-ready data."""
    pdata, solutions = _parse_pool(pool_env)
    logic_tuple = _logic_tuple_from_parsed_pool(pdata, solutions)
    schedule = pdata["ScheduleName"]
    devices = pdata["DeviceNames"]
    out = LibraryIO.createLibraryLogic(
        schedule,
        architecture,
        devices,
        "GridBased",
        logic_tuple,
    )
    assert isinstance(out, dict)
    assert out["ArchitectureName"] == architecture
    assert out["ScheduleName"] == schedule
    assert out["LibraryType"] == "GridBased"
    assert "DefaultSolution" in out
    assert isinstance(out["Solutions"], list)
    assert len(out["Solutions"]) == len(solutions)
    assert out["IndexOrder"] == pdata["IndexOrder"]


def test_reorder_solutions_params_puts_naming_keys_first() -> None:
    """``reorderSolutionsParams`` moves naming keys to the front of each solution."""
    data = {
        "Solutions": [
            {
                "WorkGroupMapping": 8,
                "SolutionIndex": 3,
                "KernelNameMin": "kern_min",
                "SolutionNameMin": "sol_min",
                "NumThreads": 256,
            }
        ]
    }
    LibraryIO.reorderSolutionsParams(data)
    keys = list(data["Solutions"][0].keys())
    assert keys[:3] == ["SolutionIndex", "KernelNameMin", "SolutionNameMin"]
    assert keys[3:] == ["WorkGroupMapping", "NumThreads"]


def test_reorder_solutions_params_no_solutions_is_noop() -> None:
    """``reorderSolutionsParams`` returns immediately when ``Solutions`` is absent."""
    data: dict[str, Any] = {}
    LibraryIO.reorderSolutionsParams(data)
    assert data == {}


def test_prepare_library_logic_dict_freesize() -> None:
    """``prepareLibraryLogicDict`` builds a FreeSize library table."""
    data: dict[str, Any] = {
        "LibraryType": "FreeSize",
        "Solutions": [{"SolutionIndex": 0}, {"SolutionIndex": 1}],
    }
    LibraryIO.prepareLibraryLogicDict(data)
    assert data["Library"]["indexOrder"] is None
    assert data["Library"]["table"] == [0, 2]
    assert data["Library"]["distance"] is None


@patch.object(LibraryIO, "getCUCount", return_value=128)
def test_create_library_logic_dict_adds_cu_count_for_gfx942(
    _mock_cu: Any,
    pool_env: types.SimpleNamespace,
) -> None:
    """``createLibraryLogic`` records non-default ``CUCount`` for gfx942."""
    pdata, solutions = _parse_pool(pool_env)
    logic_tuple = _logic_tuple_from_parsed_pool(pdata, solutions)
    out = LibraryIO.createLibraryLogic(
        pdata["ScheduleName"],
        "gfx942",
        pdata["DeviceNames"],
        "GridBased",
        logic_tuple,
    )
    assert out["CUCount"] == 128


# Canonical top-level key order for dict-format library logic. Both the
# LibraryLogic write path (``createLibraryLogic``) and the merge/convert path
# (``parseLibraryLogicList``/``convertToDict``) must emit exactly this order.
_CANONICAL_LOGIC_KEYS = [
    "MinimumRequiredVersion", "ScheduleName", "ArchitectureName", "CUCount",
    "DeviceNames", "ProblemType", "DefaultSolution", "Solutions", "IndexOrder",
    "ExactLogic", "RangeLogic", "TileSelectionIndices", "PerfMetric", "LibraryType",
]


@pytest.mark.parametrize("architecture", ["gfx950", "gfx1250"])
@patch.object(LibraryIO, "getCUCount", return_value=304)
def test_create_library_logic_canonical_key_order(
    _mock_cu: Any,
    architecture: str,
    pool_env: types.SimpleNamespace,
) -> None:
    """``createLibraryLogic`` emits keys in the canonical order with ``CUCount`` present."""
    pdata, solutions = _parse_pool(pool_env)
    logic_tuple = _logic_tuple_from_parsed_pool(pdata, solutions)
    out = LibraryIO.createLibraryLogic(
        pdata["ScheduleName"], architecture, pdata["DeviceNames"], "GridBased", logic_tuple
    )
    assert list(out.keys()) == _CANONICAL_LOGIC_KEYS
    # Non-gfx942 architectures still carry a (null) CUCount key.
    assert out["CUCount"] is None
    assert out["TileSelectionIndices"] is None


def test_parse_library_logic_list_has_tile_selection_and_canonical_order() -> None:
    """``parseLibraryLogicList`` adds ``TileSelectionIndices`` (null) in canonical position."""
    raw = LibraryIO.read(POOL_FILE, customizedLoader=True)
    assert isinstance(raw, list)
    parsed = LibraryIO.parseLibraryLogicList(copy.deepcopy(raw), POOL_FILE)
    assert "TileSelectionIndices" in parsed
    # ExactLogic precedes TileSelectionIndices which precedes PerfMetric/LibraryType.
    keys = list(parsed.keys())
    assert keys.index("ExactLogic") < keys.index("TileSelectionIndices")
    assert keys.index("TileSelectionIndices") < keys.index("LibraryType")


def test_prepare_library_logic_dict_gridbased() -> None:
    """``prepareLibraryLogicDict`` promotes GridBased to Matching library metadata."""
    data: dict[str, Any] = {
        "LibraryType": "GridBased",
        "IndexOrder": [2, 3, 0, 1],
        "ExactLogic": [[[128, 128, 1, 128], [0, 0.0]]],
        "Solutions": [{"SolutionIndex": 0}],
    }
    LibraryIO.prepareLibraryLogicDict(data)
    assert data["LibraryType"] == "Matching"
    assert data["Library"]["distance"] == "GridBased"
    assert data["Library"]["indexOrder"] == [2, 3, 0, 1]
    assert data["Library"]["table"] == [[[128, 128, 1, 128], [0, 0.0]]]


def test_reorder_solution_dict_for_dict_merge_sorts_internal_support_params() -> None:
    """``reorderSolutionDictForDictMerge`` sorts nested ``InternalSupportParams``."""
    state = {
        "WorkGroup": [16, 16, 1],
        "InternalSupportParams": {"b": 2, "a": 1},
    }
    out = LibraryIO.reorderSolutionDictForDictMerge(state)
    assert list(out["InternalSupportParams"].keys()) == ["a", "b"]
