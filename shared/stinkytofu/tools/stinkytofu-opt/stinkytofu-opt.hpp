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

#include <functional>
#include <vector>

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/DumpStinkyFunctionPass.hpp"
#include "stinkytofu/pipeline/ScopeAdaptor.hpp"
#include "stinkytofu/support/DebugPrintInstrumentation.hpp"
#include "stinkytofu/transforms/asm/BuildDefUseChain.hpp"
#include "stinkytofu/transforms/asm/DeadCodeEliminationPass.hpp"
#include "stinkytofu/transforms/asm/InsertDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/PeepholeOptimizationPass.hpp"
#include "stinkytofu/transforms/asm/RedundantMovEliminationPass.hpp"
#include "stinkytofu/transforms/asm/RemoveDelayAluPass.hpp"
#include "stinkytofu/transforms/asm/ScheduleFirstLRsPass.hpp"
#include "stinkytofu/transforms/asm/ScheduleLastLRsPass.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyConfigurableWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "stinkytofu/transforms/asm/StinkyRemoveWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"

using namespace stinkytofu;

// Structure to hold pass information
struct PassInfo {
    const char* name;
    std::function<std::unique_ptr<Pass>()> creator;
};

// List of available passes
const std::vector<PassInfo> availablePasses = {
    {"StinkyDAGSchedulerPass", []() { return createStinkyDAGSchedulerPass(); }},
    {"StinkyUnrollWaitCntPass", []() { return createStinkyUnrollWaitCntPass(); }},
    {"StinkyBuildImplicitDependencyPass",
     []() { return createStinkyBuildImplicitDependencyPass(); }},
    {"StinkyRemoveWaitCntPass", []() { return createStinkyRemoveWaitCntPass(); }},
    {"StinkyWaitCntInsertionPass", []() { return createStinkyWaitCntInsertionPass(); }},
    {"ScheduleLastLRsPass", []() { return createScheduleLastLRsPass(); }},
    {"ScheduleFirstLRsPass", []() { return createScheduleFirstLRsPass(); }},
    {"BuildUseDefChainPass", []() { return createBuildUseDefChainPass(); }},
    {"DumpStinkyFunctionPass",
     []() { return createDumpStinkyFunctionPass({.stirPath = "dump_function.stir"}); }},
    {"PeepholeOptimizationPass", []() { return createPeepholeOptimizationPass(); }},
    {"DeadCodeEliminationPass", []() { return createDeadCodeEliminationPass(); }},
    {"RedundantMovEliminationPass", []() { return createRedundantMovEliminationPass(); }},
    {"StinkyIRVerifierPass", []() { return createStinkyIRVerifierPass(); }},
    {"RemoveDelayAluPass", []() { return createRemoveDelayAluPass(); }},
    {"InsertDelayAluPass", []() { return createInsertDelayAluPass(); }},
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
    config.barrierConfig.unrollMovableBarrier = true;
    config.loopConfig.unrollGemm = true;
    config.dagFeatures.distributeGlobalRead = true;
    return config;
}

/**
 * Set default kernel configuration for the PassManager.
 */
void setKernelConfig(stinkytofu::PassManager& passManager, const std::array<int, 3>& arch) {
    passManager.setKernelConfig(arch /* arch */, 0 /* ta0 */, 0 /* tb0 */, 0 /* tm0 */,
                                0 /* nGRA */, 0 /* nGRB */, 0 /* nGRM */, 0 /* numWaves */);
}
