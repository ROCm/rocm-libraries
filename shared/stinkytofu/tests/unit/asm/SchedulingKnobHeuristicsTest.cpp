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
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <string>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/transforms/asm/dag/SchedulingKnobHeuristics.hpp"

using namespace stinkytofu;

namespace {

constexpr std::array<int, 3> kGfx1250 = {12, 5, 0};

SchedulingFeatures makeFeatures(int wmma, int ds, int sumWmmaLatency = 0) {
    SchedulingFeatures features;
    features.arch = kGfx1250;
    features.stats.wmmaCount = wmma;
    features.stats.dsLoadCount = ds;
    features.stats.sumWmmaLatencyCycles = sumWmmaLatency;
    return features;
}

}  // namespace

TEST(SchedulingKnobHeuristics, DegenerateFallsBackToStaticDefaults) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;

    for (const auto& features : {makeFeatures(0, 8), makeFeatures(4, 0), makeFeatures(0, 0)}) {
        const ResolvedSchedulingKnobs resolved =
            resolveSchedulingKnobs(features, overrides, policy);
        const ResolvedSchedulingKnobs defaults = staticSchedulingKnobDefaults(kGfx1250);

        EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles,
                  defaults.clusterBarrierRule3SignalLeadCycles);
        EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource,
                  SchedulingKnobSource::StaticDefault);
    }

    EXPECT_EQ(staticSchedulingKnobDefaults(kGfx1250).clusterBarrierRule3SignalLeadCycles,
              kStaticDefaultClusterBarrierRule3SignalLeadCycles);
}

TEST(SchedulingKnobHeuristics, PolicyUsedWhenMainLoopHasBothCounts) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    const SchedulingFeatures features = makeFeatures(/*wmma=*/4, /*ds=*/12);

    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    const ResolvedSchedulingKnobs proposed = policy.propose(features, hwModelForArch(kGfx1250));

    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles,
              proposed.clusterBarrierRule3SignalLeadCycles);
    // sumWmmaLatencyCycles default 0 <= 500 => 100
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles, 100);
}

TEST(SchedulingKnobHeuristics, Rule3LeadUses200WhenSumWmmaLatencyExceeds500) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    const SchedulingFeatures features = makeFeatures(/*wmma=*/4, /*ds=*/4, /*sumWmmaLatency=*/501);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles, 200);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource, SchedulingKnobSource::Policy);
}

TEST(SchedulingKnobHeuristics, Rule3LeadUses100WhenSumWmmaLatencyAtMost500) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    const SchedulingFeatures features = makeFeatures(/*wmma=*/4, /*ds=*/4, /*sumWmmaLatency=*/500);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles, 100);
}

TEST(SchedulingKnobHeuristics, UserOverrideWinsOverPolicy) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    overrides.clusterBarrierRule3SignalLeadCycles = 40;
    const SchedulingFeatures features = makeFeatures(/*wmma=*/2, /*ds=*/10, /*sumWmmaLatency=*/900);

    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);

    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource, SchedulingKnobSource::User);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles, 40);
}

TEST(SchedulingKnobHeuristics, ModuleOptionsSentinelsMapToOverrides) {
    StinkyAsmModule::ModuleOptions opts{};
    // Struct default: Rule3 is -1 (unset).
    EXPECT_EQ(opts.ClusterBarrierRule3SignalLeadCycles, -1);

    SchedulingKnobOverrides empty = schedulingKnobOverridesFromModuleOptions(opts);
    EXPECT_FALSE(empty.clusterBarrierRule3SignalLeadCycles.has_value());

    opts.ClusterBarrierRule3SignalLeadCycles = 0;  // co-locate
    SchedulingKnobOverrides set = schedulingKnobOverridesFromModuleOptions(opts);
    ASSERT_TRUE(set.clusterBarrierRule3SignalLeadCycles.has_value());
    EXPECT_EQ(*set.clusterBarrierRule3SignalLeadCycles, 0);
}

TEST(SchedulingKnobHeuristics, LogResolvedSchedulingKnobsFormat) {
    SchedulingFeatures features;
    features.stats.wmmaCount = 4;
    features.stats.dsLoadCount = 12;
    features.stats.sumWmmaLatencyCycles = 120;

    ResolvedSchedulingKnobs resolved;
    resolved.clusterBarrierRule3SignalLeadCycles = 100;
    resolved.clusterBarrierRule3SignalLeadCyclesSource = SchedulingKnobSource::StaticDefault;

    std::ostringstream oss;
    logResolvedSchedulingKnobs(oss, "Cijk_kernel", features, resolved);
    const std::string line = oss.str();
    EXPECT_NE(line.find("[SchedulingKnobs] module=Cijk_kernel"), std::string::npos);
    EXPECT_NE(line.find("wmma=4"), std::string::npos);
    EXPECT_NE(line.find("dsLoad=12"), std::string::npos);
    EXPECT_NE(line.find("sumWmmaLat=120"), std::string::npos);
    EXPECT_NE(line.find("rule3SignalLeadCycles=100(static)"), std::string::npos);
}

TEST(SchedulingKnobHeuristics, LogResolvedSchedulingKnobsIfDebugHonorsDebugOnly) {
    SchedulingFeatures features;
    features.stats.wmmaCount = 1;
    features.stats.dsLoadCount = 1;
    ResolvedSchedulingKnobs resolved;

    PassManagerDebugConfig::clearDebugOnly();
    {
        std::ostringstream captured;
        std::streambuf* oldBuf = std::cerr.rdbuf(captured.rdbuf());
        logResolvedSchedulingKnobsIfDebug("off", features, resolved);
        std::cerr.rdbuf(oldBuf);
        EXPECT_TRUE(captured.str().empty());
    }

    PassManagerDebugConfig::addDebugOnly("SchedulingKnobHeuristics");
    {
        std::ostringstream captured;
        std::streambuf* oldBuf = std::cerr.rdbuf(captured.rdbuf());
        logResolvedSchedulingKnobsIfDebug("on", features, resolved);
        std::cerr.rdbuf(oldBuf);
        EXPECT_NE(captured.str().find("[SchedulingKnobs] module=on"), std::string::npos);
    }
    PassManagerDebugConfig::clearDebugOnly();
}
