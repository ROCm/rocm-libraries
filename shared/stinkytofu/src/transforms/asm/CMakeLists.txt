# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu PRIVATE
    CFGBuilderPass.cpp
    PeepholeOptimizationPass.cpp
    DeadCodeEliminationPass.cpp
    RedundantMovEliminationPass.cpp
    DelayAluInsertionPass.cpp
    LegalizationUtils.cpp
    ScheduleLastLRsPass.cpp
    StinkyDAGSchedulerPass.cpp
    StinkyConfigurableWaitCntPass.cpp
    ScheduleFirstLRsPass.cpp
)
