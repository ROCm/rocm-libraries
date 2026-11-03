/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <filesystem>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/analysis/asm/HazardGapAnalysisPass.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/DumpMemTokenIRStructurePass.hpp"
#include "stinkytofu/ir/DumpStinkyModulePass.hpp"
#include "stinkytofu/pipeline/ScopeAdaptor.hpp"
#include "stinkytofu/support/DebugPrintInstrumentation.hpp"
#include "stinkytofu/support/PassArguments.hpp"
#include "stinkytofu/transforms/asm/AccumulateInstructionSizePass.hpp"
#include "stinkytofu/transforms/asm/AsmMovePropagationPass.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/DeadCodeEliminationPass.hpp"
#include "stinkytofu/transforms/asm/DefUseAnalysisCleanup.hpp"
#include "stinkytofu/transforms/asm/EpilogueStoreSinkPass.hpp"
#include "stinkytofu/transforms/asm/FlattenCFGPass.hpp"
#include "stinkytofu/transforms/asm/Gfx1250HazardPass.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"
#include "stinkytofu/transforms/asm/InsertCoexecHazardPass.hpp"
#include "stinkytofu/transforms/asm/InsertDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/InsertInitialUnclausedVmemPass.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/LongBranchLoweringPass.hpp"
#include "stinkytofu/transforms/asm/LoopRegionRemarkPass.hpp"
#include "stinkytofu/transforms/asm/MemTokenConsistencyCheckPass.hpp"
#include "stinkytofu/transforms/asm/PeepholeOptimizationPass.hpp"
#include "stinkytofu/transforms/asm/PrefetchBridgeSubstitutionPass.hpp"
#include "stinkytofu/transforms/asm/RaiseVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/RedundantMovEliminationPass.hpp"
#include "stinkytofu/transforms/asm/RegionClonePass.hpp"
#include "stinkytofu/transforms/asm/RemoveDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/RemoveDscntPass.hpp"
#include "stinkytofu/transforms/asm/RemoveInstructionPass.hpp"
#include "stinkytofu/transforms/asm/RemoveWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/SetMatrixReusePass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "stinkytofu/transforms/asm/StinkyMergeBarrierPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveNopPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyUnreachableBlockElimPass.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelDynamicPass.hpp"
#include "stinkytofu/transforms/asm/SwInstructionPrefetchRelStaticPass.hpp"
#include "stinkytofu/transforms/asm/TDMLoadWaveSyncPass.hpp"
#include "stinkytofu/transforms/asm/TieExecMaskedWritesPass.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRulesRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationConfig.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationPass.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;

// Structure to hold pass information.
//
// The creator receives the comma-separated argument list that was supplied
// via `--PassName=arg1,arg2`. Passes that don't accept arguments simply
// ignore the vector. Each pass is responsible for documenting and parsing
// its own arguments (typically simple flag-name strings or key=value pairs).
struct PassInfo {
    const char* name;
    std::function<std::unique_ptr<Pass>(const std::vector<std::string>& args)> creator;
};

// List of available passes
const std::vector<PassInfo> availablePasses = {
    {"StinkyDAGSchedulerPass", [](const auto&) { return createStinkyDAGSchedulerPass(); }},
    // HazardGapAnalysisPass accepts optional arg: verbose
    {"HazardGapAnalysisPass",
     [](const std::vector<std::string>& args) {
         return createHazardGapAnalysisPass(hasPassArg(args, "verbose"));
     }},
    {"StinkyMergeBarrierPass", [](const auto&) { return createStinkyMergeBarrierPass(); }},
    {"SetMatrixReusePass", [](const auto&) { return createSetMatrixReusePass(); }},
    {"TieExecMaskedWritesPass", [](const auto&) { return createTieExecMaskedWritesPass(); }},
    {"SwInstructionPrefetchRelStaticPass",
     [](const auto&) { return createSwInstructionPrefetchRelStaticPass(std::string{}); }},
    {"SwInstructionPrefetchRelDynamicPass",
     [](const auto&) { return createSwInstructionPrefetchRelDynamicPass(std::string{}); }},
    {"AccumulateInstructionSizePass",
     [](const auto&) { return createAccumulateInstructionSizePass(""); }},
    {"AccumulateInstructionSizeDebugPass",
     [](const auto&) { return createAccumulateInstructionSizePassWithDebug(); }},
    {"StinkyBuildImplicitDependencyPass",
     [](const auto&) { return createStinkyBuildImplicitDependencyPass(); }},
    // StinkyRemoveWaitCntPass accepts:
    //   keepTensor   — leave s_wait_tensorcnt in place (default strips it)
    //   removeXcnt   — also strip s_wait_xcnt (the O3 backend policy)
    //   removeKmcnt  — also strip s_wait_kmcnt
    {"StinkyRemoveWaitCntPass",
     [](const std::vector<std::string>& args) {
         RemoveWaitCntOptions options;
         options.removeTensor = !hasPassArg(args, "keepTensor");
         options.removeXcnt = hasPassArg(args, "removeXcnt");
         options.removeKmcnt = hasPassArg(args, "removeKmcnt");
         return createStinkyRemoveWaitCntPass(options);
     }},
    {"StinkyRemoveNopPass", [](const auto&) { return createStinkyRemoveNopPass(); }},
    {"RemoveDscntPass", [](const auto&) { return createRemoveDscntPass(); }},
    {"StinkyWaitCntInsertionPass",
     [](const std::vector<std::string>& args) {
         WaitCntInsertionOptions options;
         options.enableLoopCarriedTokenDeps = hasPassArg(args, "enableLoopCarriedTokenDeps");
         options.mergeWaitsInWmmaRuns = hasPassArg(args, "mergeWaitsInWmmaRuns");
         return createStinkyWaitCntInsertionPass(options);
     }},
    // Gfx1250HazardPass accepts:
    //   profile — print the xcnt drain summary (per rule and drain site) to stderr
    {"Gfx1250HazardPass",
     [](const std::vector<std::string>& args) {
         return createGfx1250HazardPass(hasPassArg(args, "profile"));
     }},
    // BuildUseDefChainPass accepts:
    //   includePseudo    — also build chains for pseudo registers (memtokens)
    //   noClearExisting  — keep any existing PHIs/chains
    {"BuildUseDefChainPass",
     [](const std::vector<std::string>& args) {
         bool clearExisting = !hasPassArg(args, "noClearExisting");
         bool includePseudo = hasPassArg(args, "includePseudo");
         return createBuildUseDefChainPass(clearExisting, includePseudo);
     }},
    {"CFGBuilderPass", [](const auto&) { return createCFGBuilderPass(); }},
    // Collapses every block back into the flat entry block, undoing
    // CFGBuilderPass. Labels survive, so a later CFGBuilderPass splits again.
    {"FlattenCFGPass", [](const auto&) { return createFlattenCFGPass(); }},
    // Erases blocks not reachable from the entry along CFG successor edges.
    // Run after CFGBuilderPass / LongBranchLoweringPass; an incomplete CFG
    // would make this delete live targets.
    {"StinkyUnreachableBlockElimPass",
     [](const auto&) { return createStinkyUnreachableBlockElimPass(); }},
    // Discards physical-register PHIs and def-use chains. Lifting rejects a
    // leftover analysis PHI, so this runs immediately before
    // LiftAsmRegistersToSSAPass rather than inside it.
    {"RemoveDefUseAnalysisPass", [](const auto&) { return createRemoveDefUseAnalysisPass(); }},
    // LiftAsmRegistersToSSAPass accepts:
    //   classes=<vs>   — register classes to lift (default vs). Anything left out
    //                    stays physical and no later pass rewrites it, so
    //                    classes=s allocates SGPRs with every VGPR untouched.
    //   strictLiveIns  — reject a read with no reaching definition instead of
    //                    inferring a function live-in
    //   noVerify       — skip attached SSA verification after construction
    {"LiftAsmRegistersToSSAPass",
     [](const std::vector<std::string>& args) -> std::unique_ptr<Pass> {
         LiftAsmRegistersToSSAOptions options;
         const std::optional<RegClassSet> classes =
             parseRegClasses(passArgValue(args, "classes", "vs"));
         if (!classes.has_value()) return nullptr;
         options.classes = *classes;
         options.allowInferredLiveIns = !hasPassArg(args, "strictLiveIns");
         options.verify = !hasPassArg(args, "noVerify");
         return createLiftAsmRegistersToSSAPass(options);
     }},
    // RegisterAllocationPass reads the configuration every caller shares
    // (RegisterAllocationConfig.hpp; docs/developer/register-allocation.md,
    // "Configuring allocation"), with the tool's defaults: shadow, classes=v,
    // allocator=greedy. A mode word may come anywhere in the list:
    //   shadow (default), apply, force, or 0..3 -- apply writes the colouring
    //                       through destroyAttachedSSA (also runs
    //                       syncRegisterSymbols; see
    //                       docs/developer/register-allocation.md §11.1)
    //   allocator=<name>  — registry name: greedy (default), greedy-compact,
    //                       legacy, ... Join several with '+' to try them in
    //                       turn; the first that colours the function is used
    //   classes=<vs>      — classes the policy may move (default v). Anything else
    //                       keeps the register it was lifted from, so classes=s
    //                       reallocates SGPRs and leaves every VGPR alone. Must be
    //                       a subset of what the lift covered.
    //   regionEnd=<label> — only values whose live range lies entirely before this
    //                       block's end slot may move (^ prefix optional)
    //   pinReg=<range>    — leave these registers exactly as found: each keeps
    //                       the value lifted into it and takes no other. One as
    //                       pinReg=s0, an inclusive run as pinReg=s0:19. Repeat
    //                       the key or join with '+' for disjoint runs
    //   unbankable=auto|hold|allocate — what to do about an operand whose field
    //                       cannot select a VGPR bank. allocate places it under
    //                       a ceiling, before the free blocks; hold keeps the
    //                       register the producer chose; auto (default) tries
    //                       allocate and falls back to hold
    //   quadrant=any|prefer — prefer searches the VGPR bank the producer used
    //                       for each block before the rest of the file
    //   report            — emit the shadow comparison (peak, highest, regionPeak)
    //   emitRegisterMap   — after apply, insert a producer→allocated TEXTBLOCK
    //   emitSymbolBreadcrumbs — after apply, append "// s18 was sgprTmp" comments
    //                       on operands whose symbolic name was stripped
    //   dumpBlocks=<prefix> — write each attempt's placement table to
    //                       <prefix><allocator>.<allocate|hold>.tsv, creating
    //                       the directory; only the greedy allocators write one
    //   noVerify          — skip AllocationVerifier
    //   rules=<name>      — force this architecture rule Active, ignoring the
    //                       chip's own capability gate. Repeat the key or join
    //                       with '+' for several; an unknown name is an error,
    //                       because a test that enables nothing still passes
    //   rules=all         — force every rule the triple declares
    //   noRule=<name>     — force this rule Off, so a rule the chip ships
    //                       Active can be compared against without a rebuild.
    //                       Same spelling of lists as rules=
    //   ruleAudit         — force every declared rule to Audit, which reports
    //                       against the producer's colouring without enforcing
    //   noRules           — behave as if the chip declared no rules
    {"RegisterAllocationPass",
     [](const std::vector<std::string>& args) -> std::unique_ptr<Pass> {
         Expected<RegisterAllocationConfig> config =
             parseRegisterAllocationConfig(args, RegisterAllocationDialect::Tool);
         if (config.hasError()) {
             std::cerr << "Error: RegisterAllocationPass: " << config.getError() << "\n";
             return nullptr;
         }
         if (const std::string& prefix = config->options.blockDumpPrefix; !prefix.empty()) {
             std::error_code ignored;
             std::filesystem::create_directories(std::filesystem::path(prefix).parent_path(),
                                                 ignored);
         }
         return createRegisterAllocationPass(std::move(config->options));
     }},
    // DumpStinkyModulePass accepts:
    //   ssaForm  — print attached SSA values instead of physical registers
    //   ssaLive  — also dump SSA live ranges and peak pressure
    //   stdout   — print to stdout instead of dump_module.stir /
    //              ssa_live_intervals.txt
    {"DumpStinkyModulePass",
     [](const std::vector<std::string>& args) {
         const bool toStdout = hasPassArg(args, "stdout");
         DumpStinkyModulePassConfig config;
         config.stirPath = toStdout ? std::string{} : "dump_module.stir";
         config.stirToStdout = toStdout;
         config.printerOptions.ssaForm = hasPassArg(args, "ssaForm");
         if (hasPassArg(args, "ssaLive"))
             config.ssaLiveOut = toStdout ? "-" : "ssa_live_intervals.txt";
         return createDumpStinkyModulePass(std::move(config));
     }},
    {"DumpMemTokenIRStructurePass",
     [](const auto&) {
         return createDumpMemTokenIRStructurePass({.path = "dump_memtoken_ir_structure.txt"});
     }},
    // EpilogueStoreSinkPass accepts: noStoreGroup (maxStoreGroupSize=0),
    // noGuard (avoidMsbXcntDrain=false)
    {"EpilogueStoreSinkPass",
     [](const std::vector<std::string>& args) {
         EpilogueStoreSinkOptions options;
         if (hasPassArg(args, "noStoreGroup")) options.maxStoreGroupSize = 0;
         if (hasPassArg(args, "noGuard")) options.avoidMsbXcntDrain = false;
         return createEpilogueStoreSinkPass(options);
     }},
    {"PeepholeOptimizationPass", [](const auto&) { return createPeepholeOptimizationPass(); }},
    {"DeadCodeEliminationPass", [](const auto&) { return createDeadCodeEliminationPass(); }},
    {"RedundantMovEliminationPass",
     [](const auto&) { return createRedundantMovEliminationPass(); }},
    {"AsmMovePropagationPass", [](const auto&) { return createAsmMovePropagationPass(); }},
    {"StinkyIRVerifierPass", [](const auto&) { return createStinkyIRVerifierPass(); }},
    {"RemoveDelayAluPass", [](const auto&) { return createRemoveDelayAluPass(); }},
    // RemoveInstructionPass accepts one or more mnemonics:
    //   --RemoveInstructionPass=s_wait_alu,tensor_load_to_lds,s_nop
    {"RemoveInstructionPass",
     [](const std::vector<std::string>& args) {
         if (args.empty()) return std::unique_ptr<Pass>{};
         return createRemoveInstructionPass(args);
     }},
    {"InsertDelayAluPass", [](const auto&) { return createInsertDelayAluPass(); }},
    {"LoopRegionRemarkPass", [](const auto&) { return createLoopRegionRemarkPass(); }},
    {"MemTokenConsistencyCheckPass",
     [](const auto&) { return createMemTokenConsistencyCheckPass(); }},
    {"RaiseVgprMsbPass", [](const auto&) { return createRaiseVgprMsbPass(); }},
    {"InsertVgprMsbPass", [](const auto&) { return createInsertVgprMsbPass(); }},
    {"InsertInitialUnclausedVmemPass",
     [](const auto&) { return createInsertInitialUnclausedVmemPass(); }},
    {"PrefetchBridgeSubstitutionPass",
     [](const auto&) { return createPrefetchBridgeSubstitutionPass(); }},
    {"LongBranchLoweringPass", [](const auto&) { return createLongBranchLoweringPass(); }},
    {"InsertClusterBarrierPass", [](const auto&) { return createInsertClusterBarrierPass(); }},
    {"TDMLoadWaveSyncPass", [](const auto&) { return createTDMLoadWaveSyncPass(); }},
    {"RemoveWaitAluPass", [](const auto&) { return createRemoveWaitAluPass(); }},
    {"InsertWaitAluPass",
     [](const std::vector<std::string>& args) {
         return createInsertWaitAluPass(
             {hasPassArg(args, "enableESM2TrackValuVsrc"),
              hasPassArg(args, "sharedOrderCountFollowers"),
              hasPassArg(args, "xdlCountFromNextWmma")});
     }},
    {"InsertCoexecHazardPass", [](const auto&) { return createInsertCoexecHazardPass(); }},
    {"RegionClonePass",
     [](const auto&) {
         return createRegionClonePass({CloneSpec{"InitCIterWmma", "label_LoopBeginL"}});
     }},
};

/**
 * Create default DebugPrintInstrumentation for stinkytofu-opt.
 */
std::shared_ptr<stinkytofu::PassInstrumentation> createDebugPrintInstrumentation() {
    auto streams = std::make_shared<stinkytofu::DebugOutputStreams>();
    auto debugConfig = std::make_unique<stinkytofu::PassManagerDebugConfig>();
    debugConfig->setPrintBeforeAll(true);
    debugConfig->setPrintAfterAll(true);
    debugConfig->setDumpStreamBefore(streams->getOrCreate("before.txt"));
    debugConfig->setDumpStreamAfter(streams->getOrCreate("after.txt"));
    return std::make_shared<stinkytofu::DebugPrintInstrumentation>(std::move(debugConfig));
}

/**
 * Get default PassFeatureConfig configuration.
 */
stinkytofu::PassFeatureConfig getPassFeatureConfig() {
    stinkytofu::PassFeatureConfig config;
    config.loopConfig.unrollGemm = true;
    config.dagFeatures.distributeGlobalRead = true;
    return config;
}
