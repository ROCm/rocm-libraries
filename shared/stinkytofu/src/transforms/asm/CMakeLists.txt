# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    InstructionSizeCosting.cpp
    AccumulateInstructionSizePass.cpp
    SwPrefetchRelCommon.cpp
    SwInstructionPrefetchRelStaticPass.cpp
    SwInstructionPrefetchRelDynamicPass.cpp
    SwInstructionPrefetchAbsStaticPass.cpp
    SwInstructionPrefetchAbsDynamicPass.cpp
    PhiPlacement.cpp
    BuildDefUseChain.cpp
    DefUseAnalysisCleanup.cpp
    CFGBuilderPass.cpp
    StinkyUnreachableBlockElimPass.cpp
    LongBranchLoweringPass.cpp
    PeepholeOptimizationPass.cpp
    DeadCodeEliminationPass.cpp
    RedundantMovEliminationPass.cpp
    AsmMovePropagationPass.cpp
    LegalizationUtils.cpp
    ExecMaskGrouping.cpp
    StinkyDAGSchedulerPass.cpp
    dag/RegionDAG.cpp
    WaitAwareScheduleRepairPass.cpp
    StinkyMergeBarrierPass.cpp
    SetMatrixReusePass.cpp
    StinkyWaitCntInsertionPass.cpp
    Gfx1250HazardPass.cpp
    waitcnt/WaitDataflow.cpp
    waitcnt/ShallowPredPromotion.cpp
    StinkyRemoveWaitCntPass.cpp
    RemoveDscntPass.cpp
    StinkyRemoveNopPass.cpp
    StinkyBuildImplicitDependencyPass.cpp
    RegionClonePass.cpp
    MemTokenConsistencyCheckPass.cpp
    InsertVgprMsbPass.cpp
    InsertInitialUnclausedVmemPass.cpp
    InsertClusterBarrierPass.cpp
    TDMLoadWaveSyncPass.cpp
    RaiseVgprMsbPass.cpp
    RemoveDelayAluPass.cpp
    RemoveInstructionPass.cpp
    InsertDelayAluPass.cpp
    LoopRegionRemarkPass.cpp
    RemoveWaitAluPass.cpp
    InsertWaitAluPass.cpp
    EstimateAsmCyclesPass.cpp
    InsertCoexecHazardPass.cpp
    EpilogueStoreSinkPass.cpp
    StinkyWmmaVgprReorderPass.cpp
    FlattenCalleesPass.cpp
)

add_subdirectory(ra)
add_subdirectory(ssa)
