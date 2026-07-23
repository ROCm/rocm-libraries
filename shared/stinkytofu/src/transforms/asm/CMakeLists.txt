# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu_objs PRIVATE
    InstructionSizeCosting.cpp
    AccumulateInstructionSizePass.cpp
    SwPrefetchInsertionPass.cpp
    PhiPlacement.cpp
    BuildDefUseChain.cpp
    CFGBuilderPass.cpp
    LongBranchLoweringPass.cpp
    PeepholeOptimizationPass.cpp
    DeadCodeEliminationPass.cpp
    RedundantMovEliminationPass.cpp
    LegalizationUtils.cpp
    ExecMaskGrouping.cpp
    StinkyDAGSchedulerPass.cpp
    SetMatrixReusePass.cpp
    StinkyWaitCntInsertionPass.cpp
    WaitCntValidationCheckPass.cpp
    waitcnt/WaitDataflow.cpp
    waitcnt/WaitCntDataflowUtils.cpp
    waitcnt/WaitCntValidator.cpp
    waitcnt/ShallowPredPromotion.cpp
    StinkyRemoveWaitCntPass.cpp
    StinkyRemoveNopPass.cpp
    StinkyBuildImplicitDependencyPass.cpp
    RegionClonePass.cpp
    MemTokenConsistencyCheckPass.cpp
    InsertVgprMsbPass.cpp
    InsertClusterBarrierPass.cpp
    RaiseVgprMsbPass.cpp
    RemoveDelayAluPass.cpp
    RemoveInstructionPass.cpp
    InsertDelayAluPass.cpp
    LoopRegionRemarkPass.cpp
    RemoveWaitAluPass.cpp
    InsertWaitAluPass.cpp
    RederiveExpertScopePass.cpp
    EstimateAsmCyclesPass.cpp
    StinkyWmmaVgprReorderPass.cpp
    FlattenCalleesPass.cpp
)
