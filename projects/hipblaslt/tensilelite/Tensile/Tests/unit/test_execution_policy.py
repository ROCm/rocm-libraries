# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Public execution-policy contracts, including legacy configuration boundaries."""

import copy
import itertools

import pytest

from Tensile.ExecutionPolicy import (
    normalize_execution_policy,
    normalize_hybrid_assignment_policy,
    resolve_policy,
    hasStaticAssignment,
    hasDynamicAssignment,
    hasHybridAssignment,
)

pytestmark = pytest.mark.unit

STRATEGIES = ("None", "DataParallel", "StreamK")
ASSIGNMENTS = ("StaticGrid", "DynamicWorkQueue", "Hybrid")
LEGACY_PAIRS = (
    ({"StreamK": 0}, "None", "StaticGrid"),
    ({"StreamK": 3}, "StreamK", "StaticGrid"),
    ({"StreamK": 3, "StreamKForceDPOnly": 1}, "DataParallel", "StaticGrid"),
    ({"StreamK": 4}, "StreamK", "DynamicWorkQueue"),
    ({"StreamK": 5}, "StreamK", "Hybrid"),
)


@pytest.mark.parametrize("strategy,assignment", tuple(itertools.product(STRATEGIES, ASSIGNMENTS)))
def test_supported_pair_matrix(strategy, assignment):
    state = {"TileProcessingStrategy": strategy, "WorkAssignment": assignment}
    if strategy == "DataParallel" and assignment != "StaticGrid":
        with pytest.raises(ValueError, match=f"{strategy} supports WorkAssignment=StaticGrid only"):
            normalize_execution_policy(state)
        return
    state = normalize_execution_policy(state)
    policy = resolve_policy(state)
    assert state["_PersistentLoop"] is (strategy != "None")
    assert policy.persistent is (strategy != "None")
    assert policy.data_parallel is (strategy == "DataParallel")
    assert policy.stream_k is (strategy == "StreamK")
    if strategy == "None":
        assert state["WorkAssignment"] == "StaticGrid"
        assert not hasStaticAssignment(state)
        assert not hasDynamicAssignment(state)
        assert not hasHybridAssignment(state)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_omitted_assignment_uses_static_grid(strategy):
    state = normalize_execution_policy({"TileProcessingStrategy": strategy})
    assert state["WorkAssignment"] == "StaticGrid"


def test_omitted_selectors_disable_persistent_processing():
    state = normalize_execution_policy({})
    assert state["TileProcessingStrategy"] == "None"
    assert state["WorkAssignment"] == "StaticGrid"
    assert state["_PersistentLoop"] is False


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_assignment_alone_does_not_infer_strategy(assignment):
    state = normalize_execution_policy({"WorkAssignment": assignment})
    assert state == normalize_execution_policy({})
    raw = {"TileProcessingStrategy": "None", "WorkAssignment": assignment}
    assert not hasStaticAssignment(raw)
    assert not hasDynamicAssignment(raw)
    assert not hasHybridAssignment(raw)


@pytest.mark.parametrize("key,values", (
    ("TileProcessingStrategy", (None, False, 0, "none", "Data Parallel", "SK3")),
    ("WorkAssignment", (None, False, 0, "Static", "Queue", "SK5")),
))
def test_selectors_require_canonical_string_enum_values(key, values):
    for value in values:
        with pytest.raises(ValueError, match=key):
            normalize_execution_policy({key: value})


@pytest.mark.parametrize("value", (False, True, 0, 1))
def test_persistence_is_not_an_independent_tuning_switch(value):
    with pytest.raises(ValueError, match="PersistentLoop is derived"):
        normalize_execution_policy({"PersistentLoop": value})


@pytest.mark.parametrize("strategy,cached", (("None", True), ("DataParallel", False)))
def test_cached_persistence_must_match_selectors(strategy, cached):
    with pytest.raises(ValueError, match="_PersistentLoop disagrees"):
        normalize_execution_policy({"TileProcessingStrategy": strategy, "_PersistentLoop": cached})


@pytest.mark.parametrize("legacy,strategy,assignment", LEGACY_PAIRS)
def test_legacy_selectors_normalize_to_canonical_pair(legacy, strategy, assignment):
    original = copy.deepcopy(legacy)
    state = normalize_execution_policy(legacy)
    assert state["TileProcessingStrategy"] == strategy
    assert state["WorkAssignment"] == assignment
    assert "StreamK" not in state
    assert "StreamKForceDPOnly" not in state
    assert legacy == original
    assert normalize_execution_policy(state) == state


@pytest.mark.parametrize("legacy,strategy,assignment", LEGACY_PAIRS)
def test_consistent_explicit_legacy_and_canonical_selectors_are_accepted(legacy, strategy, assignment):
    state = normalize_execution_policy(dict(legacy, TileProcessingStrategy=strategy, WorkAssignment=assignment))
    assert state["TileProcessingStrategy"] == strategy
    assert state["WorkAssignment"] == assignment


@pytest.mark.parametrize("legacy,strategy,assignment", LEGACY_PAIRS)
def test_conflicting_explicit_strategy_fails_before_defaults(legacy, strategy, assignment):
    conflict = "DataParallel" if strategy != "DataParallel" else "StreamK"
    with pytest.raises(ValueError, match="Conflicting legacy StreamK and TileProcessingStrategy"):
        normalize_execution_policy(dict(legacy, TileProcessingStrategy=conflict))


@pytest.mark.parametrize("legacy,strategy,assignment", LEGACY_PAIRS)
def test_conflicting_explicit_assignment_fails_before_defaults(legacy, strategy, assignment):
    conflict = "Hybrid" if assignment != "Hybrid" else "StaticGrid"
    if strategy == "None":
        assert normalize_execution_policy(dict(legacy, WorkAssignment=conflict)) == normalize_execution_policy(legacy)
        return
    with pytest.raises(ValueError, match="Conflicting legacy StreamK and WorkAssignment"):
        normalize_execution_policy(dict(legacy, WorkAssignment=conflict))


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_legacy_disabled_assignment_is_inactive(assignment):
    assert normalize_execution_policy({"StreamK": 0, "WorkAssignment": assignment}) == normalize_execution_policy({"StreamK": 0})


@pytest.mark.parametrize("assignment", (None, False, 0, "Queue", "Static"))
def test_legacy_disabled_assignment_still_validates_spelling(assignment):
    with pytest.raises(ValueError, match="WorkAssignment must be"):
        normalize_execution_policy({"StreamK": 0, "WorkAssignment": assignment})


def test_inherited_legacy_defaults_do_not_override_canonical_request():
    config = {"StreamK": 0, "StreamKForceDPOnly": 0, "TileProcessingStrategy": "DataParallel"}
    state = normalize_execution_policy(config, explicit_keys={"TileProcessingStrategy"})
    assert state["TileProcessingStrategy"] == "DataParallel"


def test_inherited_canonical_defaults_do_not_mask_legacy_request():
    config = {"StreamK": 4, "TileProcessingStrategy": "None", "WorkAssignment": "StaticGrid"}
    state = normalize_execution_policy(config, explicit_keys={"StreamK"})
    assert state["TileProcessingStrategy"] == "StreamK"
    assert state["WorkAssignment"] == "DynamicWorkQueue"


@pytest.mark.parametrize("mode", (1, 2, -1, 6, "3", False, True))
def test_invalid_or_retired_legacy_modes_remain_rejected(mode):
    with pytest.raises(ValueError, match="Legacy StreamK must be"):
        normalize_execution_policy({"StreamK": mode})


@pytest.mark.parametrize("regenerate", (False, True))
@pytest.mark.parametrize("mode,force", ((0, False), (3, False), (4, False), (5, False), (3, True)))
def test_legacy_dp_boolean_flag_matches_integer_flag(mode, force, regenerate):
    boolean = {"StreamK": mode, "StreamKForceDPOnly": force}
    integer = {"StreamK": mode, "StreamKForceDPOnly": int(force)}
    assert normalize_execution_policy(boolean, regenerate=regenerate) == normalize_execution_policy(
        integer, regenerate=regenerate
    )


@pytest.mark.parametrize("force", (-1, 2, 0.0, 1.0, "0", "1", None))
def test_legacy_dp_flag_rejects_other_values(force):
    with pytest.raises(ValueError, match="StreamKForceDPOnly must be 0 or 1"):
        normalize_execution_policy({"StreamK": 3, "StreamKForceDPOnly": force})


@pytest.mark.parametrize("force", (1, True))
@pytest.mark.parametrize("mode,atomic", ((0, 0), (4, 0), (5, 0), (3, 1)))
def test_legacy_dp_rejects_invalid_original_combinations(mode, atomic, force):
    with pytest.raises(ValueError, match="StreamKForceDPOnly requires non-atomic StreamK=3"):
        normalize_execution_policy({"StreamK": mode, "StreamKForceDPOnly": force, "StreamKAtomic": atomic})


@pytest.mark.parametrize("old,new,value", (
    ("StreamKXCCMapping", "PersistentXCCMapping", 8),
    ("StreamKWorkStealing", "WorkQueueStealing", 1),
))
def test_shared_parameter_aliases_preserve_values_and_detect_conflicts(old, new, value):
    base = {"TileProcessingStrategy": "StreamK", "WorkAssignment": "DynamicWorkQueue"}
    state = normalize_execution_policy(dict(base, **{old: value}))
    assert old not in state
    assert state[new] == value
    assert normalize_execution_policy(dict(base, **{old: value, new: value})) == state
    with pytest.raises(ValueError, match=f"Conflicting {old} and {new}"):
        normalize_execution_policy(dict(base, **{old: value, new: 0}))


@pytest.mark.parametrize("option", ("PrefetchAcrossPersistent", "ReuseAcrossPersistent"))
def test_explicit_prefetch_or_reuse_requires_persistence(option):
    with pytest.raises(ValueError, match=f"{option} requires a persistent TileProcessingStrategy"):
        normalize_execution_policy({"TileProcessingStrategy": "None", option: 1})
    # Historical inactive options remain loadable at the legacy boundary.
    assert normalize_execution_policy({"StreamK": 0, option: 1})[option] == 0


@pytest.mark.parametrize("strategy", ("None", "DataParallel"))
@pytest.mark.parametrize("option", ("StreamKAtomic", "StreamKFixupTreeReduction", "DebugStreamK"))
def test_explicit_reduction_controls_require_streamk(strategy, option):
    with pytest.raises(ValueError, match=f"{option} requires TileProcessingStrategy=StreamK"):
        normalize_execution_policy({"TileProcessingStrategy": strategy, option: 1})




@pytest.mark.parametrize("version", (-1, 2, False, True, "1"))
def test_unknown_persistent_argument_layout_is_rejected(version):
    with pytest.raises(ValueError, match="Unsupported PersistentLoopArgsVersion"):
        normalize_execution_policy({"TileProcessingStrategy": "DataParallel", "InternalSupportParams": {"PersistentLoopArgsVersion": version}}, regenerate=False)


@pytest.mark.parametrize("version", (False, True))
def test_boolean_outer_argument_versions_remain_rejected(version):
    with pytest.raises(ValueError, match="Unsupported KernArgsVersion"):
        normalize_execution_policy({"TileProcessingStrategy": "DataParallel", "InternalSupportParams": {"KernArgsVersion": version}}, regenerate=False)


@pytest.mark.parametrize("strategy", ("None", "StreamK"))
def test_native_whole_tile_layout_requires_data_parallel(strategy):
    with pytest.raises(ValueError, match="PersistentLoopArgsVersion=1 requires DataParallel/StaticGrid"):
        normalize_execution_policy({"TileProcessingStrategy": strategy, "InternalSupportParams": {"PersistentLoopArgsVersion": 1}}, regenerate=False)


def test_prebuilt_dp_defaults_to_legacy_layout_and_preserves_explicit_native_layout():
    state = normalize_execution_policy({"StreamK": 3, "StreamKForceDPOnly": 1}, regenerate=False)
    assert state["InternalSupportParams"]["PersistentLoopArgsVersion"] == 0
    native = normalize_execution_policy({"TileProcessingStrategy": "DataParallel", "InternalSupportParams": {"PersistentLoopArgsVersion": 1}}, regenerate=False)
    assert native["InternalSupportParams"]["PersistentLoopArgsVersion"] == 1


@pytest.mark.parametrize("name,value", (("Default", 0), ("DynamicWorkQueue", 1), ("Auto", 2)))
def test_hybrid_runtime_alias_preserves_existing_encoding(name, value):
    canonical = normalize_hybrid_assignment_policy({"HybridAssignmentPolicy": [name]})
    legacy = normalize_hybrid_assignment_policy({"StreamKHybridMode": [value]})
    assert canonical == legacy
    assert canonical["StreamKHybridMode"] == [value]
    assert canonical["HybridAssignmentPolicy"] == [name]
    assert normalize_hybrid_assignment_policy(canonical) == canonical


def test_hybrid_policy_sweep_preserves_order_and_compiled_assignment():
    state = normalize_hybrid_assignment_policy({
        "TileProcessingStrategy": "StreamK", "WorkAssignment": "Hybrid",
        "HybridAssignmentPolicy": ["Default", "DynamicWorkQueue", "Auto"],
    })
    assert state["StreamKHybridMode"] == [0, 1, 2]
    assert state["WorkAssignment"] == "Hybrid"


@pytest.mark.parametrize("canonical,legacy", ((["Default"], [1]), (["Auto"], [0]), (["Default", "DynamicWorkQueue"], [1, 0])))
def test_conflicting_explicit_hybrid_aliases_are_rejected(canonical, legacy):
    with pytest.raises(ValueError, match="Conflicting StreamKHybridMode and HybridAssignmentPolicy"):
        normalize_hybrid_assignment_policy({"HybridAssignmentPolicy": canonical, "StreamKHybridMode": legacy})


@pytest.mark.parametrize("outer", (0, 1, 2, 3))
@pytest.mark.parametrize("regenerate", (False, True))
def test_data_parallel_selects_layout_at_the_generation_boundary(outer, regenerate):
    state = normalize_execution_policy({
        "StreamK": 3,
        "StreamKForceDPOnly": 1,
        "InternalSupportParams": {"KernArgsVersion": outer},
    }, regenerate=regenerate)
    assert state["InternalSupportParams"] == {
        "KernArgsVersion": 3 if regenerate else outer,
        "PersistentLoopArgsVersion": int(regenerate),
    }
