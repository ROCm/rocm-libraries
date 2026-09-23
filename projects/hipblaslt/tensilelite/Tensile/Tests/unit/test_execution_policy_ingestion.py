# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Legacy library input must keep policy identity through defaults and merging."""

from copy import deepcopy
from pathlib import Path

import pytest

from config_harness import _isolated_globals_with_isa, _toolchain_for
from Tensile import LibraryIO
from Tensile.ExecutionPolicy import (
    ALIASES, SELECTORS, normalize_execution_policy, normalize_execution_policy_with_defaults,
)
from Tensile.ExperimentalLibrary import _apply_overrides
from Tensile.SolutionStructs.Naming import getSolutionNameFull
from Tensile.TensileMergeLibrary import addKernel, reNameSolutions

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).parent / "test_data/solution_pool_gfx950.yaml"
_POLICIES = (
    (0, 0, "None", "StaticGrid"),
    (3, 1, "DataParallel", "StaticGrid"),
    (3, 0, "StreamK", "StaticGrid"),
    (4, 0, "StreamK", "DynamicWorkQueue"),
    (5, 0, "StreamK", "Hybrid"),
)


def _logic(selectors, defaults=None):
    raw = LibraryIO.read(str(_FIXTURE), customizedLoader=True)
    data = LibraryIO.parseLibraryLogicList(raw, str(_FIXTURE))
    solution = data["Solutions"][0]
    for key in SELECTORS | set(ALIASES) | set(ALIASES.values()) | {"_PersistentLoop"}:
        solution.pop(key, None)
    solution["StreamKAtomic"] = 0
    solution.update(selectors)
    data["DefaultSolution"] = dict(defaults or {})
    return data


def _parse(data):
    assembler, isa_info_map = _toolchain_for("gfx950")
    with _isolated_globals_with_isa(isa_info_map):
        return LibraryIO.parseLibraryLogicData(
            data, str(_FIXTURE), assembler, False, False, False, isa_info_map, False
        ).solutions[0]


@pytest.mark.parametrize("mode,force,strategy,assignment", _POLICIES)
@pytest.mark.parametrize("in_defaults", (False, True))
@pytest.mark.parametrize("flag_type", (int, bool), ids=("integer", "boolean"))
def test_library_parser_preserves_legacy_policy_identity(mode, force, strategy, assignment, in_defaults, flag_type):
    old = {"StreamK": mode, "StreamKForceDPOnly": flag_type(force)}
    new = {"TileProcessingStrategy": strategy, "WorkAssignment": assignment}
    legacy = _parse(_logic({}, old) if in_defaults else _logic(old))
    canonical = _parse(_logic({}, new) if in_defaults else _logic(new))
    assert legacy["TileProcessingStrategy"] == strategy
    assert legacy["WorkAssignment"] == assignment
    assert "StreamK" not in legacy and "StreamKForceDPOnly" not in legacy
    assert legacy["InternalSupportParams"]["PersistentLoopArgsVersion"] == int(strategy == "DataParallel")
    if strategy == "DataParallel":
        assert legacy["InternalSupportParams"]["KernArgsVersion"] == 3
    assert getSolutionNameFull(legacy, False) == getSolutionNameFull(canonical, False)


@pytest.mark.parametrize("defaults,override,expected", (
    ({"StreamK": 3}, {"StreamKForceDPOnly": 1}, ("DataParallel", "StaticGrid")),
    ({"StreamK": 3, "StreamKForceDPOnly": 1}, {"StreamKForceDPOnly": 0}, ("StreamK", "StaticGrid")),
    ({"TileProcessingStrategy": "StreamK"}, {"StreamKForceDPOnly": 1}, ("DataParallel", "StaticGrid")),
    ({"TileProcessingStrategy": "DataParallel"}, {"StreamKForceDPOnly": 0}, ("StreamK", "StaticGrid")),
    ({"StreamK": 3}, {"WorkAssignment": "Hybrid"}, ("StreamK", "Hybrid")),
    ({"StreamK": 5}, {"TileProcessingStrategy": "DataParallel", "WorkAssignment": "StaticGrid"}, ("DataParallel", "StaticGrid")),
    ({"TileProcessingStrategy": "DataParallel"}, {"StreamK": 5, "StreamKForceDPOnly": 0}, ("StreamK", "Hybrid")),
    ({"WorkAssignment": "Hybrid"}, {"StreamK": 5}, ("StreamK", "Hybrid")),
    ({"WorkAssignment": "DynamicWorkQueue"}, {"StreamK": 4}, ("StreamK", "DynamicWorkQueue")),
    ({"StreamKForceDPOnly": 1}, {"TileProcessingStrategy": "DataParallel"}, ("DataParallel", "StaticGrid")),
    ({"PrefetchAcrossPersistent": 1}, {"TileProcessingStrategy": "DataParallel"}, ("DataParallel", "StaticGrid")),
))
def test_solution_policy_overrides_file_defaults(defaults, override, expected):
    solution = _parse(_logic(override, defaults))
    assert (solution["TileProcessingStrategy"], solution["WorkAssignment"]) == expected


@pytest.mark.parametrize("old,new", ALIASES.items())
@pytest.mark.parametrize("legacy_override", (False, True))
def test_shared_control_override_replaces_inherited_spelling(old, new, legacy_override):
    inherited, requested = (new, old) if legacy_override else (old, new)
    defaults = {"StreamK": 4, inherited: 1}
    override = {requested: 0}
    before = deepcopy((defaults, override))
    result = normalize_execution_policy_with_defaults(override, defaults)
    assert result[new] == 0 and old not in result
    assert (defaults, override) == before


@pytest.mark.parametrize("override", (
    {"StreamK": 5, "TileProcessingStrategy": "DataParallel"},
    {"StreamK": 4, "WorkAssignment": "Hybrid"},
    {"StreamKXCCMapping": 4, "PersistentXCCMapping": 8},
    {"StreamKWorkStealing": 0, "WorkQueueStealing": 1},
))
def test_explicit_conflicts_remain_errors_after_default_merge(override):
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_execution_policy_with_defaults(override, {"StreamK": 4})


@pytest.mark.parametrize("strategy", ("None", "DataParallel", "StreamK"))
def test_active_inherited_stealing_requires_explicit_override_for_static_assignment(strategy):
    defaults = {"StreamK": 5, "StreamKWorkStealing": 1}
    override = {"TileProcessingStrategy": strategy, "WorkAssignment": "StaticGrid"}
    with pytest.raises(ValueError, match="WorkQueueStealing requires"):
        normalize_execution_policy_with_defaults(override, defaults)
    override["WorkQueueStealing"] = 0
    result = normalize_execution_policy_with_defaults(override, defaults)
    assert result["TileProcessingStrategy"] == strategy
    assert result["WorkQueueStealing"] == 0


def test_merge_names_deduplicate_equivalent_policies_and_keep_distinct_policies():
    pool, by_name = [], {}
    for mode, force, strategy, assignment in _POLICIES:
        names = []
        for selectors in (
            {"StreamK": mode, "StreamKForceDPOnly": force},
            {"TileProcessingStrategy": strategy, "WorkAssignment": assignment},
        ):
            data = _logic(selectors)
            reNameSolutions(data)
            solution = data["Solutions"][0]
            names.append((solution["SolutionNameMin"], solution["KernelNameMin"]))
            pool, by_name, _ = addKernel(pool, by_name, solution)
        assert names[0] == names[1]
    assert len(pool) == len(by_name) == len(_POLICIES)


def test_merge_naming_uses_file_defaults_before_global_defaults():
    inherited = _logic({"StreamKForceDPOnly": 1}, {"StreamK": 3, "GlobalSplitU": 7})
    inherited["Solutions"][0].pop("GlobalSplitU")
    explicit = _logic({"TileProcessingStrategy": "DataParallel", "GlobalSplitU": 7})
    reNameSolutions(inherited)
    reNameSolutions(explicit)
    assert inherited["Solutions"][0]["SolutionNameMin"] == explicit["Solutions"][0]["SolutionNameMin"]
    assert inherited["Solutions"][0]["KernelNameMin"] == explicit["Solutions"][0]["KernelNameMin"]
    assert "GlobalSplitU" not in inherited["Solutions"][0]




@pytest.fixture
def native_data_parallel_state():
    solution = _parse(_logic({"TileProcessingStrategy": "DataParallel"}))
    assert solution["Valid"] and solution["AssignedDerivedParameters"]
    assert solution["InternalSupportParams"]["PersistentLoopArgsVersion"] == 1
    return dict(solution)


_NATIVE_OVERRIDES = (
    ({"TileProcessingStrategy": "StreamK"}, "StreamK", "StaticGrid"),
    ({"TileProcessingStrategy": "StreamK", "WorkAssignment": "Hybrid"}, "StreamK", "Hybrid"),
    ({"TileProcessingStrategy": "None", "WorkAssignment": "Hybrid", "GlobalSplitU": 1}, "None", "StaticGrid"),
    ({"StreamKForceDPOnly": 0}, "StreamK", "StaticGrid"),
    ({"StreamK": 0, "StreamKForceDPOnly": 0, "GlobalSplitU": 1}, "None", "StaticGrid"),
)


@pytest.mark.parametrize("override,strategy,assignment", _NATIVE_OVERRIDES)
@pytest.mark.parametrize("boundary", ("patch", "defaults"))
def test_native_generated_layout_is_rederived_after_selector_override(
    native_data_parallel_state, override, strategy, assignment, boundary
):
    if boundary == "patch":
        result = deepcopy(native_data_parallel_state)
        _apply_overrides(result, [(name, [value]) for name, value in override.items()])
    else:
        result = normalize_execution_policy_with_defaults(override, native_data_parallel_state)
    assert result["TileProcessingStrategy"] == strategy
    assert result["WorkAssignment"] == assignment
    assert result["InternalSupportParams"]["PersistentLoopArgsVersion"] == 0
    assert result["AssignedDerivedParameters"] is False
    assert result["AssignedProblemIndependentDerivedParameters"] is False

    # Re-enter the library parser to verify that the changed policy derives a
    # usable solution, including the defaults path used by real logic files.
    data = _logic({})
    data["Solutions"] = [result if boundary == "patch" else override]
    if boundary == "defaults":
        data["DefaultSolution"] = native_data_parallel_state
    regenerated = _parse(data)
    assert regenerated["Valid"]
    assert regenerated["AssignedDerivedParameters"]
    assert regenerated["TileProcessingStrategy"] == strategy
    assert regenerated["WorkAssignment"] == assignment
    assert regenerated["InternalSupportParams"]["PersistentLoopArgsVersion"] == 0


@pytest.mark.parametrize("selector", ({"TileProcessingStrategy": "StreamK"}, {"StreamKForceDPOnly": 0}))
@pytest.mark.parametrize("support,reason", (
    ({"KernArgsVersion": 3, "PersistentLoopArgsVersion": 1}, "requires DataParallel/StaticGrid"),
    ({"KernArgsVersion": 3, "PersistentLoopArgsVersion": 2}, "Unsupported PersistentLoopArgsVersion"),
    ({"KernArgsVersion": 4, "PersistentLoopArgsVersion": 0}, "Unsupported KernArgsVersion"),
))
@pytest.mark.parametrize("boundary", ("patch", "defaults"))
def test_native_selector_override_preserves_explicit_layout_errors(
    native_data_parallel_state, selector, support, reason, boundary
):
    override = dict(selector, InternalSupportParams=support)
    source = deepcopy(native_data_parallel_state)
    with pytest.raises(ValueError, match=reason):
        if boundary == "patch":
            _apply_overrides(source, [(name, [value]) for name, value in override.items()])
        else:
            normalize_execution_policy_with_defaults(override, source)
    assert source == native_data_parallel_state


@pytest.mark.parametrize("custom", ({"CustomKernelName": "prebuilt_dp"}, {"CustomKernel": {"name": "prebuilt_dp"}}))
def test_handwritten_native_layout_is_not_changed_by_selector_override(native_data_parallel_state, custom):
    source = dict(native_data_parallel_state, **custom)
    with pytest.raises(ValueError, match="requires DataParallel/StaticGrid"):
        _apply_overrides(source, [("TileProcessingStrategy", ["StreamK"])])
    with pytest.raises(ValueError, match="requires DataParallel/StaticGrid"):
        normalize_execution_policy_with_defaults({"TileProcessingStrategy": "StreamK"}, source)


def test_native_layout_is_strict_for_prebuilt_or_explicit_config(native_data_parallel_state):
    state = dict(native_data_parallel_state, TileProcessingStrategy="StreamK")
    state.pop("_PersistentLoop")
    with pytest.raises(ValueError, match="requires DataParallel/StaticGrid"):
        normalize_execution_policy(state, explicit_keys={"TileProcessingStrategy"}, regenerate=False)
    with pytest.raises(ValueError, match="requires DataParallel/StaticGrid"):
        normalize_execution_policy(state)
