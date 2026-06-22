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

#include <cstdlib>
#include <functional>
#include <vector>

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/DumpStinkyFunctionPass.hpp"
#include "stinkytofu/pipeline/ScopeAdaptor.hpp"
#include "stinkytofu/support/DebugPrintInstrumentation.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/CFGBuilderPass.hpp"
#include "stinkytofu/transforms/asm/DeadCodeEliminationPass.hpp"
#include "stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp"
#include "stinkytofu/transforms/asm/InsertDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/LongBranchLoweringPass.hpp"
#include "stinkytofu/transforms/asm/LoopRegionRemarkPass.hpp"
#include "stinkytofu/transforms/asm/MemTokenConsistencyCheckPass.hpp"
#include "stinkytofu/transforms/asm/PeepholeOptimizationPass.hpp"
#include "stinkytofu/transforms/asm/RaiseVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/RedundantMovEliminationPass.hpp"
#include "stinkytofu/transforms/asm/RemoveDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/RemoveWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/SetMatrixReusePass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveNopPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"

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

// Helper: returns true if `args` contains the literal flag name `flag`
// (case-sensitive, exact match).
inline bool hasPassArg(const std::vector<std::string>& args, const char* flag) {
    for (const auto& a : args)
        if (a == flag) return true;
    return false;
}

// List of available passes
const std::vector<PassInfo> availablePasses = {
    {"StinkyDAGSchedulerPass", [](const auto&) { return createStinkyDAGSchedulerPass(); }},
    {"SetMatrixReusePass", [](const auto&) { return createSetMatrixReusePass(); }},
    {"StinkyBuildImplicitDependencyPass",
     [](const auto&) { return createStinkyBuildImplicitDependencyPass(); }},
    {"StinkyRemoveWaitCntPass", [](const auto&) { return createStinkyRemoveWaitCntPass(); }},
    {"StinkyRemoveNopPass", [](const auto&) { return createStinkyRemoveNopPass(); }},
    {"StinkyWaitCntInsertionPass",
     [](const std::vector<std::string>& args) {
         WaitCntInsertionOptions options;
         options.enableLoopCarriedTokenDeps = hasPassArg(args, "enableLoopCarriedTokenDeps");
         return createStinkyWaitCntInsertionPass(options);
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
    {"DumpStinkyFunctionPass",
     [](const auto&) { return createDumpStinkyFunctionPass({.stirPath = "dump_function.stir"}); }},
    {"PeepholeOptimizationPass", [](const auto&) { return createPeepholeOptimizationPass(); }},
    {"DeadCodeEliminationPass", [](const auto&) { return createDeadCodeEliminationPass(); }},
    {"RedundantMovEliminationPass",
     [](const auto&) { return createRedundantMovEliminationPass(); }},
    {"StinkyIRVerifierPass", [](const auto&) { return createStinkyIRVerifierPass(); }},
    {"RemoveDelayAluPass", [](const auto&) { return createRemoveDelayAluPass(); }},
    {"InsertDelayAluPass", [](const auto&) { return createInsertDelayAluPass(); }},
    {"LoopRegionRemarkPass", [](const auto&) { return createLoopRegionRemarkPass(); }},
    {"MemTokenConsistencyCheckPass",
     [](const auto&) { return createMemTokenConsistencyCheckPass(); }},
    {"RaiseVgprMsbPass", [](const auto&) { return createRaiseVgprMsbPass(); }},
    {"InsertVgprMsbPass", [](const auto&) { return createInsertVgprMsbPass(); }},
    {"LongBranchLoweringPass", [](const auto&) { return createLongBranchLoweringPass(); }},
    {"InsertClusterBarrierPass",
     [](const auto&) {
         auto geti = [](const char* k, int d) {
             const char* v = std::getenv(k);
             return v != nullptr ? std::atoi(v) : d;
         };
         return createInsertClusterBarrierPass(
             /*isKernelScope=*/true, geti("PrefetchGlobalRead", 1), geti("PrefetchLocalRead", 1));
     }},
    {"RemoveWaitAluPass", [](const auto&) { return createRemoveWaitAluPass(); }},
    {"InsertWaitAluPass", [](const auto&) { return createInsertWaitAluPass(); }},
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
