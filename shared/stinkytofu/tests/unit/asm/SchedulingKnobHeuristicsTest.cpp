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

SchedulingFeatures makeFeatures(int wmma, int ds, int firstWmmaLatency = 0,
                                int sumWmmaLatency = 0) {
    SchedulingFeatures features;
    features.arch = kGfx1250;
    features.stats.wmmaCount = wmma;
    features.stats.dsLoadCount = ds;
    features.stats.firstWmmaLatencyCycles = firstWmmaLatency;
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

        EXPECT_EQ(resolved.dsReadThrottleLatency, defaults.dsReadThrottleLatency);
        EXPECT_EQ(resolved.dsReadPerWmma, defaults.dsReadPerWmma);
        EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles,
                  defaults.clusterBarrierRule3SignalLeadCycles);
        EXPECT_EQ(resolved.dsReadThrottleLatencySource, SchedulingKnobSource::StaticDefault);
        EXPECT_EQ(resolved.dsReadPerWmmaSource, SchedulingKnobSource::StaticDefault);
        EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource,
                  SchedulingKnobSource::StaticDefault);
    }

    EXPECT_EQ(staticSchedulingKnobDefaults(kGfx1250).dsReadThrottleLatency,
              hwModelForArch(kGfx1250).lds.readThrottleLatency);
    EXPECT_EQ(staticSchedulingKnobDefaults(kGfx1250).dsReadPerWmma, kStaticDefaultDsReadPerWmma);
    EXPECT_EQ(staticSchedulingKnobDefaults(kGfx1250).clusterBarrierRule3SignalLeadCycles,
              kStaticDefaultClusterBarrierRule3SignalLeadCycles);
}

TEST(SchedulingKnobHeuristics, PolicyUsedWhenMainLoopHasBothCounts) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // firstWmmaLatency=32, perWmma=ceil(12/4)=3, queueDepth=16
    // computed = (32/3)*16 = 10*16 = 160 > 72 => 160
    const SchedulingFeatures features =
        makeFeatures(/*wmma=*/4, /*ds=*/12, /*firstWmmaLatency=*/32);

    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    const ResolvedSchedulingKnobs proposed = policy.propose(features, hwModelForArch(kGfx1250));

    EXPECT_EQ(resolved.dsReadThrottleLatencySource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.dsReadPerWmmaSource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.dsReadThrottleLatency, proposed.dsReadThrottleLatency);
    EXPECT_EQ(resolved.dsReadPerWmma, proposed.dsReadPerWmma);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles,
              proposed.clusterBarrierRule3SignalLeadCycles);
    // ceil(12/4) = 3
    EXPECT_EQ(resolved.dsReadPerWmma, 3);
    EXPECT_EQ(resolved.dsReadThrottleLatency, 160);
    // sumWmmaLatencyCycles default 0 <= 500 => 100
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles, 100);
}

TEST(SchedulingKnobHeuristics, Rule3LeadUses200WhenSumWmmaLatencyExceeds500) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    const SchedulingFeatures features =
        makeFeatures(/*wmma=*/4, /*ds=*/4, /*firstWmmaLatency=*/3, /*sumWmmaLatency=*/501);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles, 200);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource, SchedulingKnobSource::Policy);
}

TEST(SchedulingKnobHeuristics, Rule3LeadUses100WhenSumWmmaLatencyAtMost500) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    const SchedulingFeatures features =
        makeFeatures(/*wmma=*/4, /*ds=*/4, /*firstWmmaLatency=*/3, /*sumWmmaLatency=*/500);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles, 100);
}

TEST(SchedulingKnobHeuristics, DsReadPerWmmaCappedAt3) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // ceil(20/4)=5 would exceed the cap => 3
    const SchedulingFeatures features =
        makeFeatures(/*wmma=*/4, /*ds=*/20, /*firstWmmaLatency=*/32);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadPerWmma, 3);
    EXPECT_EQ(resolved.dsReadPerWmmaSource, SchedulingKnobSource::Policy);
}

TEST(SchedulingKnobHeuristics, ThrottleFlooredAtArchReadThrottleLatency) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // Throttle uses independent ceil(ds/wmma)=1 (not the dsReadPerWmma knob):
    // firstWmmaLatency=3, queueDepth=16 => (3/1)*16=48 < 72 => floor at 72.
    // wmma < 128 => dsReadPerWmma prefers the static default (3), not ceil=1.
    const SchedulingFeatures features = makeFeatures(/*wmma=*/4, /*ds=*/4, /*firstWmmaLatency=*/3);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadThrottleLatency, 72);
    EXPECT_EQ(resolved.dsReadPerWmma, kStaticDefaultDsReadPerWmma);
}
TEST(SchedulingKnobHeuristics, UserOverrideIndependentPerKnob) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    overrides.dsReadThrottleLatency = 40;
    // Leave per-WMMA and Rule3 unset so they still come from policy.
    const SchedulingFeatures features = makeFeatures(/*wmma=*/2, /*ds=*/10);

    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    const ResolvedSchedulingKnobs proposed = policy.propose(features, hwModelForArch(kGfx1250));

    EXPECT_EQ(resolved.dsReadThrottleLatencySource, SchedulingKnobSource::User);
    EXPECT_EQ(resolved.dsReadThrottleLatency, 40);
    EXPECT_EQ(resolved.dsReadPerWmmaSource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.dsReadPerWmma, proposed.dsReadPerWmma);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles,
              proposed.clusterBarrierRule3SignalLeadCycles);
}

TEST(SchedulingKnobHeuristics, ModuleOptionsSentinelsMapToOverrides) {
    StinkyAsmModule::ModuleOptions opts{};
    // Struct defaults: throttle / perWmma / Rule3 are -1 (unset).
    EXPECT_EQ(opts.DsReadThrottleLatency, -1);
    EXPECT_EQ(opts.DsReadPerWmma, -1);
    EXPECT_EQ(opts.ClusterBarrierRule3SignalLeadCycles, -1);

    SchedulingKnobOverrides empty = schedulingKnobOverridesFromModuleOptions(opts);
    EXPECT_FALSE(empty.dsReadThrottleLatency.has_value());
    EXPECT_FALSE(empty.dsReadPerWmma.has_value());
    EXPECT_FALSE(empty.clusterBarrierRule3SignalLeadCycles.has_value());

    opts.DsReadThrottleLatency = 64;
    opts.DsReadPerWmma = 0;                        // extreme but valid
    opts.ClusterBarrierRule3SignalLeadCycles = 0;  // co-locate
    SchedulingKnobOverrides set = schedulingKnobOverridesFromModuleOptions(opts);
    ASSERT_TRUE(set.dsReadThrottleLatency.has_value());
    ASSERT_TRUE(set.dsReadPerWmma.has_value());
    ASSERT_TRUE(set.clusterBarrierRule3SignalLeadCycles.has_value());
    EXPECT_EQ(*set.dsReadThrottleLatency, 64);
    EXPECT_EQ(*set.dsReadPerWmma, 0);
    EXPECT_EQ(*set.clusterBarrierRule3SignalLeadCycles, 0);
}

TEST(SchedulingKnobHeuristics, LogResolvedSchedulingKnobsFormat) {
    SchedulingFeatures features;
    features.stats.wmmaCount = 4;
    features.stats.dsLoadCount = 12;
    features.stats.firstWmmaLatencyCycles = 30;
    features.stats.sumWmmaLatencyCycles = 120;

    ResolvedSchedulingKnobs resolved;
    resolved.dsReadThrottleLatency = 160;
    resolved.dsReadThrottleLatencySource = SchedulingKnobSource::Policy;
    resolved.dsReadPerWmma = 3;
    resolved.dsReadPerWmmaSource = SchedulingKnobSource::Policy;
    resolved.clusterBarrierRule3SignalLeadCycles = 100;
    resolved.clusterBarrierRule3SignalLeadCyclesSource = SchedulingKnobSource::StaticDefault;

    std::ostringstream oss;
    logResolvedSchedulingKnobs(oss, "Cijk_kernel", features, resolved);
    const std::string line = oss.str();
    EXPECT_NE(line.find("[SchedulingKnobs] module=Cijk_kernel"), std::string::npos);
    EXPECT_NE(line.find("wmma=4"), std::string::npos);
    EXPECT_NE(line.find("dsLoad=12"), std::string::npos);
    EXPECT_NE(line.find("dsReadThrottleLatency=160(policy)"), std::string::npos);
    EXPECT_NE(line.find("dsReadPerWmma=3(policy)"), std::string::npos);
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
