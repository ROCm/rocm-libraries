# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu PRIVATE
    PassOrderSnapshotJson.cpp
    PhiPlacement.cpp
    BuildDefUseChain.cpp
    CFGBuilderPass.cpp
    PeepholeOptimizationPass.cpp
    DeadCodeEliminationPass.cpp
    RedundantMovEliminationPass.cpp
    LegalizationUtils.cpp
    ScheduleLastLRsPass.cpp
    StinkyDAGSchedulerPass.cpp
    StinkyConfigurableWaitCntPass.cpp
    ScheduleFirstLRsPass.cpp
    StinkyWaitCntInsertionPass.cpp
    StinkyRemoveWaitCntPass.cpp
    StinkyBuildImplicitDependencyPass.cpp
    InsertVgprMsbPass.cpp
    RemoveDelayAluPass.cpp
    InsertDelayAluPass.cpp
)
