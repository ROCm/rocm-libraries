# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Structural invariants for ScheduleTypes stage aliases.

Each test runs the pipeline up to a stage boundary and asserts the structural
properties callers should be able to rely on for that stage. Since TypeAlias is
erased at runtime, these tests are the runtime safety net that catches "this
object claims to be stage X but actually isn't."
"""

from Tensile.Components.Subtile.LogicalScheduler import (
    Dep,
    GRPlacement,
    LogicalScheduler,
    LRPlacement,
    ReadGranularity,
    SchedulerConfig,
)


def _make_scheduler() -> LogicalScheduler:
    """Minimal scaled config: 2x2 MFMA tiles, 2 subIterK, 1 partition."""
    cfg = SchedulerConfig(
        numMFMATilesM=2,
        numMFMATilesN=2,
        numSubIterK=2,
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        lrSA=ReadGranularity(mn=2, k=2),
        lrSB=ReadGranularity(mn=2, k=2),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
        grSA=ReadGranularity(mn=2, k=2),
        grSB=ReadGranularity(mn=2, k=2),
    )
    return LogicalScheduler(cfg)


def _all_placements(slot):
    out = []
    if slot.mfma is not None:
        out.append(slot.mfma)
    out.extend(slot.lrs)
    out.extend(slot.grs)
    return out


def test_logical_schedule_has_no_deps_or_preops():
    """Post-place_GRs (LogicalSchedule): .deps and .preOps are empty everywhere."""
    sched = _make_scheduler()
    sched.place_GRs()
    s = sched._partitions
    assert s is not None and len(s) >= 1
    for slots in s:
        for slot in slots:
            for p in _all_placements(slot):
                assert p.deps == [], \
                    f"LogicalSchedule has .deps on {type(p).__name__} {p.tensor if hasattr(p, 'tensor') else ''}"
                assert p.preOps == [], \
                    f"LogicalSchedule has .preOps on {type(p).__name__}"


def test_annotated_schedule_has_only_same_slot_deps():
    """Post-remove_cross_deps (AnnotatedSchedule): all .deps are same partition + same slot."""
    sched = _make_scheduler()
    sched.remove_cross_deps()
    s = sched._partitions
    assert s is not None
    for pi, slots in enumerate(s):
        for slot in slots:
            for p in _all_placements(slot):
                for dep in p.deps:
                    assert isinstance(dep, Dep)
                    assert dep.mt_offset == 0, (
                        f"AnnotatedSchedule .deps must be same-MT, got "
                        f"mt_offset={dep.mt_offset} on {type(p).__name__}"
                    )
                    assert dep.ref.partition == pi, (
                        f"AnnotatedSchedule cross-partition dep leaked into .deps: "
                        f"consumer P{pi}, producer P{dep.ref.partition}"
                    )
                    assert dep.ref.subIterK_slot == slot.subIterK, (
                        f"AnnotatedSchedule cross-slot dep leaked into .deps: "
                        f"consumer slot={slot.subIterK}, producer slot={dep.ref.subIterK_slot}"
                    )


def test_augmented_schedule_chains_lr_gr_in_tensor_order():
    """Post-remove_unnecessary_wait_lr_sync (AugmentedSchedule):
    LR/GR within a slot appear in canonical tensor order (A, B, SA, SB)."""
    sched = _make_scheduler()
    sched.remove_unnecessary_wait_lr_sync()
    s = sched._partitions
    assert s is not None
    order = LogicalScheduler._LR_GR_ORDER
    for slots in s:
        for slot in slots:
            lr_indices = [order.index(lr.tensor) for lr in slot.lrs]
            assert lr_indices == sorted(lr_indices), (
                f"AugmentedSchedule LRs not in tensor order at slot "
                f"{slot.subIterK}: {[lr.tensor for lr in slot.lrs]}"
            )
            gr_indices = [order.index(gr.tensor) for gr in slot.grs]
            assert gr_indices == sorted(gr_indices), (
                f"AugmentedSchedule GRs not in tensor order at slot "
                f"{slot.subIterK}: {[gr.tensor for gr in slot.grs]}"
            )


def test_emitted_schedule_is_three_level_with_valid_before_links():
    """Post-emit (EmittedSchedule): 3-level list; every .before is None or a
    valid moduleId in the same subIterK list, and never self-referential."""
    sched = _make_scheduler()
    result = sched.emit()
    assert isinstance(result, list), "EmittedSchedule must be a list (level 1: partitions)"
    for partition in result:
        assert isinstance(partition, list), \
            "EmittedSchedule level 2 (per-subIterK) must be a list"
        for em_list in partition:
            assert isinstance(em_list, list), \
                "EmittedSchedule level 3 (modules) must be a list"
            ids = {em.moduleId for em in em_list}
            for em in em_list:
                if em.before is not None:
                    assert em.before in ids, (
                        f"EmittedModule.before={em.before} not in subIterK "
                        f"module ids {sorted(ids)}"
                    )
                    assert em.before != em.moduleId, (
                        f"EmittedModule {em.moduleId} has self-referential before-link"
                    )
