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

#include "stinkytofu/core/stinkytofu.hpp"
#include "stinkytofu/transforms/asm/ScheduleFirstLRsPass.hpp"
#include "stinkytofu/transforms/asm/ScheduleLastLRsPass.hpp"
#include "stinkytofu/transforms/asm/StinkyConfigurableWaitCntPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"

#include <functional>
#include <vector>

using namespace stinkytofu;

// Structure to hold pass information
struct PassInfo
{
    const char*                            name;
    std::function<std::unique_ptr<Pass>()> creator;
};

// List of available passes
const std::vector<PassInfo> availablePasses = {
    {"StinkyDAGSchedulerPass", []() { return createStinkyDAGSchedulerPass(); }},
    {"StinkyUnrollWaitCntPass", []() { return createStinkyUnrollWaitCntPass(); }},
    {"ScheduleLastLRsPass", []() { return createScheduleLastLRsPass(); }},
    {"ScheduleFirstLRsPass", []() { return createScheduleFirstLRsPass(); }},
};

/**
 * Get default PassManagerDebugConfig configuration.
 */
std::unique_ptr<stinkytofu::PassManagerDebugConfig> getPassManagerDebugConfig()
{
    auto debugConfig = std::make_unique<stinkytofu::PassManagerDebugConfig>();
    debugConfig->setPrintBeforeAll(true);
    debugConfig->setPrintAfterAll(true);
    debugConfig->setDumpToFileInBefore("before.txt");
    debugConfig->setDumpToFileInAfter("after.txt");
    return debugConfig;
}

/**
 * Get default PassFeatureConfig configuration.
 */
stinkytofu::PassFeatureConfig getPassFeatureConfig()
{
    stinkytofu::PassFeatureConfig config;
    config.barrierConfig.unrollMovableBarrier = true;
    config.loopConfig.unrollGemm              = true;
    config.dagFeatures.distributeGlobalRead   = true;
    return config;
}

/**
 * Set default kernel configuration for the PassManager.
 */
void setKernelConfig(stinkytofu::PassManager& passManager, const std::array<int, 3>& arch)
{
    passManager.setKernelConfig(arch /* arch */,
                                0 /* ta0 */,
                                0 /* tb0 */,
                                0 /* tm0 */,
                                0 /* nGRA */,
                                0 /* nGRB */,
                                0 /* nGRM */,
                                0 /* numWaves */);
}
