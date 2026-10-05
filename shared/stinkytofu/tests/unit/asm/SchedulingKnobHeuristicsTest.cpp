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

SchedulingFeatures makeFeatures(int wmma, int ds, int firstWmmaLatency = 0, int sumWmmaLatency = 0,
                                int unrollLoopCopies = 0, int firstDsLoadLatency = 0) {
    SchedulingFeatures features;
    features.arch = kGfx1250;
    features.stats.wmmaCount = wmma;
    features.stats.dsLoadCount = ds;
    features.stats.firstWmmaLatencyCycles = firstWmmaLatency;
    features.stats.firstDsLoadLatencyCycles = firstDsLoadLatency;
    features.stats.sumWmmaLatencyCycles = sumWmmaLatency;
    features.unrollLoopCopies = unrollLoopCopies;
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
        EXPECT_EQ(resolved.dsReadPerCap, defaults.dsReadPerCap);
        EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles,
                  defaults.clusterBarrierRule3SignalLeadCycles);
        EXPECT_EQ(resolved.dsReadThrottleLatencySource, SchedulingKnobSource::StaticDefault);
        EXPECT_EQ(resolved.dsReadPerCapSource, SchedulingKnobSource::StaticDefault);
        EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource,
                  SchedulingKnobSource::StaticDefault);
        EXPECT_EQ(resolved.optimisticDsReadThrottleLatency, -1);
    }

    EXPECT_EQ(staticSchedulingKnobDefaults(kGfx1250).dsReadThrottleLatency,
              hwModelForArch(kGfx1250).lds.readThrottleLatency);
    EXPECT_EQ(staticSchedulingKnobDefaults(kGfx1250).dsReadPerCap, kStaticDefaultDsReadPerCap);
    EXPECT_EQ(staticSchedulingKnobDefaults(kGfx1250).clusterBarrierRule3SignalLeadCycles,
              kStaticDefaultClusterBarrierRule3SignalLeadCycles);
}

TEST(SchedulingKnobHeuristics, PolicyUsedWhenMainLoopHasBothCounts) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // firstWmmaLatency=32, perCap=ceil(12/4)=3, queueDepth=16
    // computed = (32/3)*16 = 10*16 = 160 > 72 => 160
    const SchedulingFeatures features =
        makeFeatures(/*wmma=*/4, /*ds=*/12, /*firstWmmaLatency=*/32);

    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    const ResolvedSchedulingKnobs proposed = policy.propose(features, hwModelForArch(kGfx1250));

    EXPECT_EQ(resolved.dsReadThrottleLatencySource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.dsReadPerCapSource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.dsReadThrottleLatency, proposed.dsReadThrottleLatency);
    EXPECT_EQ(resolved.dsReadPerCap, proposed.dsReadPerCap);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles,
              proposed.clusterBarrierRule3SignalLeadCycles);
    // ceil(12/4) = 3
    EXPECT_EQ(resolved.dsReadPerCap, 3);
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

TEST(SchedulingKnobHeuristics, DsReadPerCapCappedAt3) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // ceil(20/4)=5 would exceed the cap => 3
    const SchedulingFeatures features =
        makeFeatures(/*wmma=*/4, /*ds=*/20, /*firstWmmaLatency=*/32);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadPerCap, 3);
    EXPECT_EQ(resolved.dsReadPerCapSource, SchedulingKnobSource::Policy);
}

TEST(SchedulingKnobHeuristics, OptimisticThrottleIsLoggedAndNotApplied) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // gfx1250 queueDepth=16, perCapForThrottle=min(3, ceil(32/200))=1
    // applied throttle stays (10/1)*16 = 160
    // dsIssueSpace = 1000 - 4*40 = 840   (first ds_load latency, not first WMMA)
    // basicWmmaUsage = ceil(16/1) = 16
    // throttleSpace = 840 - 16*10 = 680
    // remainingDs = 32-16 = 16
    // cyclePerDs = 680/16 = 42.5
    // optimistic = lround(42.5*16) = 680
    const SchedulingFeatures features = makeFeatures(
        /*wmma=*/200, /*ds=*/32, /*firstWmmaLatency=*/10,
        /*sumWmmaLatency=*/1000, /*unrollLoopCopies=*/4, /*firstDsLoadLatency=*/40);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);

    EXPECT_EQ(resolved.dsReadThrottleLatency, 160);
    EXPECT_EQ(resolved.dsReadThrottleLatencySource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.optimisticDsReadThrottleLatency, 680);

    std::ostringstream oss;
    logResolvedSchedulingKnobs(oss, "optimistic", features, resolved);
    const std::string line = oss.str();
    EXPECT_NE(line.find("dsReadThrottleLatency=160(policy)"), std::string::npos);
    EXPECT_NE(line.find("optimisticDsReadThrottleLatency=680"), std::string::npos);
    EXPECT_NE(line.find("policyOptimisticDsReadThrottleLatency=680"), std::string::npos);
    EXPECT_NE(line.find("optimisticThrottleMatchesPolicy=1"), std::string::npos);
    EXPECT_NE(line.find("firstDsLat=40"), std::string::npos);
    EXPECT_NE(line.find("unrollLoopCopies=4"), std::string::npos);
    EXPECT_NE(line.find("dsIssueSpace=840"), std::string::npos);
    EXPECT_NE(line.find("basicWmmaUsage=16"), std::string::npos);
    EXPECT_NE(line.find("throttleSpace=680"), std::string::npos);
    EXPECT_NE(line.find("remainingDs=16"), std::string::npos);
    EXPECT_NE(line.find("cyclePerDs=42.5"), std::string::npos);
}

TEST(SchedulingKnobHeuristics, CustomPolicyOptimisticThrottleIsLoggedNotApplied) {
    struct CustomPolicy : SchedulingKnobPolicy {
        ResolvedSchedulingKnobs propose(const SchedulingFeatures& features,
                                        const HWModel& hw) const override {
            ResolvedSchedulingKnobs out = HeuristicSchedulingKnobPolicy{}.propose(features, hw);
            // Disagree with the logger's heuristic recompute. Scheduling still
            // uses dsReadThrottleLatency from propose(), not this diagnostic.
            out.optimisticDsReadThrottleLatency = 7;
            return out;
        }
    } policy;
    SchedulingKnobOverrides overrides;
    const SchedulingFeatures features = makeFeatures(
        /*wmma=*/200, /*ds=*/32, /*firstWmmaLatency=*/10,
        /*sumWmmaLatency=*/1000, /*unrollLoopCopies=*/4, /*firstDsLoadLatency=*/40);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);

    EXPECT_EQ(resolved.dsReadThrottleLatency, 160);
    EXPECT_EQ(resolved.optimisticDsReadThrottleLatency, 7);

    std::ostringstream oss;
    logResolvedSchedulingKnobs(oss, "custom-policy", features, resolved);
    const std::string line = oss.str();
    EXPECT_NE(line.find("dsReadThrottleLatency=160(policy)"), std::string::npos);
    EXPECT_NE(line.find("optimisticDsReadThrottleLatency=680"), std::string::npos);
    EXPECT_NE(line.find("policyOptimisticDsReadThrottleLatency=7"), std::string::npos);
    EXPECT_NE(line.find("optimisticThrottleMatchesPolicy=0"), std::string::npos);
}

TEST(SchedulingKnobHeuristics, OptimisticThrottleStaysZeroWhenNoRemainingDs) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // remainingDs = 4-16 < 0, so the estimate does not divide and stays 0.
    // Applied throttle is unchanged: (3/1)*16 = 48, floored at 72.
    const SchedulingFeatures features = makeFeatures(/*wmma=*/4, /*ds=*/4, /*firstWmmaLatency=*/3,
                                                     /*sumWmmaLatency=*/40,
                                                     /*unrollLoopCopies=*/2);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadThrottleLatency, 72);
    EXPECT_EQ(resolved.optimisticDsReadThrottleLatency, 0);

    std::ostringstream oss;
    logResolvedSchedulingKnobs(oss, "no-remaining", features, resolved);
    const std::string line = oss.str();
    EXPECT_NE(line.find("optimisticDsReadThrottleLatency=0"), std::string::npos);
    EXPECT_NE(line.find("remainingDs=-12"), std::string::npos);
    EXPECT_NE(line.find("cyclePerDs=0"), std::string::npos);
}

TEST(SchedulingKnobHeuristics, ThrottleFlooredAtArchReadThrottleLatency) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // Throttle uses independent ceil(ds/wmma)=1 (not the dsReadPerCap knob):
    // firstWmmaLatency=3, queueDepth=16 => (3/1)*16=48 < 72 => floor at 72.
    // wmma < 128 => dsReadPerCap prefers the static default (3), not ceil=1.
    const SchedulingFeatures features = makeFeatures(/*wmma=*/4, /*ds=*/4, /*firstWmmaLatency=*/3);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadThrottleLatency, 72);
    EXPECT_EQ(resolved.dsReadPerCap, kStaticDefaultDsReadPerCap);
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
    EXPECT_EQ(resolved.dsReadPerCapSource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.dsReadPerCap, proposed.dsReadPerCap);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCyclesSource, SchedulingKnobSource::Policy);
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles,
              proposed.clusterBarrierRule3SignalLeadCycles);
}

TEST(SchedulingKnobHeuristics, ModuleOptionsSentinelsMapToOverrides) {
    StinkyAsmModule::ModuleOptions opts{};
    // Struct defaults: throttle / cap / Rule3 are -1 (unset).
    EXPECT_EQ(opts.DsReadThrottleLatency, -1);
    EXPECT_EQ(opts.DsReadPerCap, -1);
    EXPECT_EQ(opts.ClusterBarrierRule3SignalLeadCycles, -1);

    SchedulingKnobOverrides empty = schedulingKnobOverridesFromModuleOptions(opts);
    EXPECT_FALSE(empty.dsReadThrottleLatency.has_value());
    EXPECT_FALSE(empty.dsReadPerCap.has_value());
    EXPECT_FALSE(empty.clusterBarrierRule3SignalLeadCycles.has_value());

    opts.DsReadThrottleLatency = 64;
    opts.DsReadPerCap = 0;                         // extreme but valid
    opts.ClusterBarrierRule3SignalLeadCycles = 0;  // co-locate
    SchedulingKnobOverrides set = schedulingKnobOverridesFromModuleOptions(opts);
    ASSERT_TRUE(set.dsReadThrottleLatency.has_value());
    ASSERT_TRUE(set.dsReadPerCap.has_value());
    ASSERT_TRUE(set.clusterBarrierRule3SignalLeadCycles.has_value());
    EXPECT_EQ(*set.dsReadThrottleLatency, 64);
    EXPECT_EQ(*set.dsReadPerCap, 0);
    EXPECT_EQ(*set.clusterBarrierRule3SignalLeadCycles, 0);
}

// DsReadPerCap was renamed from DsReadPerWmma. Options arrive as string keys,
// so a caller still passing the old spelling gets no error -- it would just
// match nothing and fall back to the default, silently un-tuning a knob that is
// tuned in the field.
TEST(SchedulingKnobHeuristics, DeprecatedDsReadPerWmmaStillSetsTheCap) {
    StinkyAsmModule::ModuleOptions opts{};
    EXPECT_EQ(opts.DsReadPerWmma, -1) << "the deprecated key must also default to unset";

    opts.DsReadPerWmma = 5;
    SchedulingKnobOverrides viaOldName = schedulingKnobOverridesFromModuleOptions(opts);
    ASSERT_TRUE(viaOldName.dsReadPerCap.has_value());
    EXPECT_EQ(*viaOldName.dsReadPerCap, 5);
}

TEST(SchedulingKnobHeuristics, DsReadPerCapWinsOverTheDeprecatedSpelling) {
    StinkyAsmModule::ModuleOptions opts{};
    opts.DsReadPerCap = 2;
    opts.DsReadPerWmma = 5;
    SchedulingKnobOverrides both = schedulingKnobOverridesFromModuleOptions(opts);
    ASSERT_TRUE(both.dsReadPerCap.has_value());
    EXPECT_EQ(*both.dsReadPerCap, 2);
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
    resolved.dsReadPerCap = 3;
    resolved.dsReadPerCapSource = SchedulingKnobSource::Policy;
    resolved.clusterBarrierRule3SignalLeadCycles = 100;
    resolved.clusterBarrierRule3SignalLeadCyclesSource = SchedulingKnobSource::StaticDefault;

    std::ostringstream oss;
    logResolvedSchedulingKnobs(oss, "Cijk_kernel", features, resolved);
    const std::string line = oss.str();
    EXPECT_NE(line.find("[SchedulingKnobs] module=Cijk_kernel"), std::string::npos);
    EXPECT_NE(line.find("wmma=4"), std::string::npos);
    EXPECT_NE(line.find("dsLoad=12"), std::string::npos);
    EXPECT_NE(line.find("dsReadThrottleLatency=160(policy)"), std::string::npos);
    EXPECT_NE(line.find("dsReadPerCap=3(policy)"), std::string::npos);
    EXPECT_NE(line.find("rule3SignalLeadCycles=100(static)"), std::string::npos);
    // unrollLoopCopies unset, queueDepth=16, perCap=ceil(12/4)=3
    // space=120, basicWmma=ceil(16/3)=6, throttleSpace=120-180=-60
    // remainingDs=12-16=-4 => no division, optimistic stays 0
    EXPECT_NE(line.find("optimisticDsReadThrottleLatency=0"), std::string::npos);
    EXPECT_NE(line.find("unrollLoopCopies=0"), std::string::npos);
    EXPECT_NE(line.find("dsIssueSpace=120"), std::string::npos);
    EXPECT_NE(line.find("basicWmmaUsage=6"), std::string::npos);
    EXPECT_NE(line.find("throttleSpace=-60"), std::string::npos);
    EXPECT_NE(line.find("remainingDs=-4"), std::string::npos);
    EXPECT_NE(line.find("cyclePerDs=0"), std::string::npos);
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

// WMMA batch of N: the cap's span is one batch window, so the per-WMMA cap scales
// by N, and the throttle / Rule3 timeline use the batch rate (each joined WMMA
// adds L-I, not L).
TEST(SchedulingKnobHeuristics, WmmaBatchScalesCapAndUsesBatchTimeline) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // {issue 16, latency 64}; cap = 3 (wmma <= 128); sum = 520.
    SchedulingFeatures features = makeFeatures(/*wmma=*/4, /*ds=*/12, /*firstWmmaLatency=*/64,
                                               /*sumWmmaLatency=*/520);
    features.stats.firstWmmaIssueCycles = 16;

    ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadPerCap, 3);
    EXPECT_EQ(resolved.dsReadThrottleLatency, (64 / 3) * 16);      // 336
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles, 200);  // 520 > 500

    features.wmmaBatchSize = 2;
    resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadPerCap, 3 * 2);
    // Batch window 64 + 48 = 112 -> 56 cycles per WMMA.
    EXPECT_EQ(resolved.dsReadThrottleLatency, (56 / 3) * 16);  // 288
    // Two of four WMMAs join a batch: 520 - 2 * 16 = 488 <= 500.
    EXPECT_EQ(resolved.clusterBarrierRule3SignalLeadCycles, 100);
}

TEST(SchedulingKnobHeuristics, WmmaBatchScalesStaticDefaultCap) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    SchedulingFeatures features = makeFeatures(/*wmma=*/0, /*ds=*/8);
    features.wmmaBatchSize = 3;
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadPerCapSource, SchedulingKnobSource::StaticDefault);
    EXPECT_EQ(resolved.dsReadPerCap, kStaticDefaultDsReadPerCap * 3);

    overrides.dsReadPerCap = 12;  // user value is already per batch window
    EXPECT_EQ(resolveSchedulingKnobs(features, overrides, policy).dsReadPerCap, 12);
}
