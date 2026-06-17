# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu PRIVATE
    InstructionSizeCosting.cpp
    AccumulateInstructionSizePass.cpp
    SwPrefetchInsertionPass.cpp
    PassOrderSnapshotJson.cpp
    PhiPlacement.cpp
    BuildDefUseChain.cpp
    CFGBuilderPass.cpp
    LongBranchLoweringPass.cpp
    PeepholeOptimizationPass.cpp
    DeadCodeEliminationPass.cpp
    RedundantMovEliminationPass.cpp
    LegalizationUtils.cpp
    StinkyDAGSchedulerPass.cpp
    SetMatrixReusePass.cpp
    StinkyWaitCntInsertionPass.cpp
    waitcnt/WaitDataflow.cpp
    waitcnt/ShallowPredPromotion.cpp
    StinkyRemoveWaitCntPass.cpp
    StinkyRemoveNopPass.cpp
    StinkyBuildImplicitDependencyPass.cpp
    MemTokenConsistencyCheckPass.cpp
    InsertVgprMsbPass.cpp
    InsertClusterBarrierPass.cpp
    RaiseVgprMsbPass.cpp
    RemoveDelayAluPass.cpp
    InsertDelayAluPass.cpp
    LoopRegionRemarkPass.cpp
    RemoveWaitAluPass.cpp
    InsertWaitAluPass.cpp
    RederiveExpertScopePass.cpp
    EstimateAsmCyclesPass.cpp
)
