# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stage-tagged type aliases for the LogicalScheduler pipeline.

These aliases name each intermediate representation produced by the scheduler
passes.

Stage progression
-----------------
    LogicalSchedule    — ``self._partitions`` after the GR pass (all partitions):
        MFMA / LR / GR placements; .deps and .preOps empty.
        Not the return type of ``place_GRs()`` (that is a single partition).

    PartitionSchedule  — one partition's subIterK slots (``List[SubIterKSlot]``);
        returned by ``place_GRs()``.

    AnnotatedSchedule  — output of remove_cross_deps():
        same container; .deps and .preOps populated by dependency analysis.
        Cross-subIterK deps live in .preOps; .deps holds only same-slot refs.

    AugmentedSchedule  — output of remove_unnecessary_wait_lr_sync():
        same container; lr_inc / gr_inc preOps inserted, LR / GR chains
        grouped in tensor order, redundant wait_lr_sync removed.

    EmittedSchedule    — output of emit():
        [partition][subIterK][EmittedModule] with before-link chains.

"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, TypeAlias

if TYPE_CHECKING:
    from .LogicalScheduler import EmittedModule, SubIterKSlot

PartitionSchedule: TypeAlias = "List[SubIterKSlot]"
LogicalSchedule:   TypeAlias = "List[List[SubIterKSlot]]"
AnnotatedSchedule: TypeAlias = "List[List[SubIterKSlot]]"
AugmentedSchedule: TypeAlias = "List[List[SubIterKSlot]]"
EmittedSchedule:   TypeAlias = "List[List[List[EmittedModule]]]"
