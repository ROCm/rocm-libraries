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
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/transforms/asm/dag/SchedulingKnobHeuristics.hpp"

using namespace stinkytofu;

namespace {

constexpr std::array<int, 3> kGfx1250 = {12, 5, 0};

SchedulingFeatures makeFeatures(int wmma, int ds, int firstWmmaLatency = 0, int sumWmmaLatency = 0,
                                int dsLoadLatency = 0) {
    SchedulingFeatures features;
    features.arch = kGfx1250;
    features.stats.wmmaCount = wmma;
    features.stats.dsLoadCount = ds;
    features.stats.firstWmmaLatencyCycles = firstWmmaLatency;
    features.stats.sumWmmaLatencyCycles = sumWmmaLatency;
    features.stats.dsLoadLatencyCycles = dsLoadLatency;
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
TEST(SchedulingKnobHeuristics, ThrottleClampedSoEveryDsLoadCanIssue) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // perWmma=min(3, ceil(16/4))=3, computed=(3/3)*16=16, arch floor 72.
    // cycles per ds = 64/16=4, finish budget=4*16=64. 72 cannot issue every
    // ds_load inside the WMMA latency sum, so the budget replaces the floor.
    const SchedulingFeatures features =
        makeFeatures(/*wmma=*/4, /*ds=*/16, /*firstWmmaLatency=*/3, /*sumWmmaLatency=*/64);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadThrottleLatency, 64);
    EXPECT_EQ(resolved.dsReadThrottleLatencySource, SchedulingKnobSource::Policy);
}

TEST(SchedulingKnobHeuristics, ThrottleFinishBudgetAddsOneDsLoadLatency) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // Same shape as ThrottleClampedSoEveryDsLoadCanIssue, plus one ds_load
    // latency of 16. cycles per ds = (64+16)/16=5, finish budget=5*16=80.
    // The arch floor 72 is under that budget, so it is no longer clamped to 64.
    const SchedulingFeatures features = makeFeatures(
        /*wmma=*/4, /*ds=*/16, /*firstWmmaLatency=*/3, /*sumWmmaLatency=*/64,
        /*dsLoadLatency=*/16);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadThrottleLatency, 72);
}

TEST(SchedulingKnobHeuristics, ThrottleKeepsComputedValueBelowFinishBudget) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // computed=(32/3)*16=160. finish budget=(10000/12)*16 stays above 160.
    const SchedulingFeatures features =
        makeFeatures(/*wmma=*/4, /*ds=*/12, /*firstWmmaLatency=*/32, /*sumWmmaLatency=*/10000);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadThrottleLatency, 160);
}

TEST(SchedulingKnobHeuristics, ThrottleFinishBudgetKeepsFractionalCycles) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // Integer (568/162)*16 is 3*16=48. Float 568/162*16 is 56.09, truncated
    // to 56, which replaces the arch floor 72.
    const SchedulingFeatures features = makeFeatures(
        /*wmma=*/128, /*ds=*/162, /*firstWmmaLatency=*/4, /*sumWmmaLatency=*/512,
        /*dsLoadLatency=*/56);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadThrottleLatency, 56);
}

TEST(SchedulingKnobHeuristics, ThrottleBudgetOfZeroUsesSmallestHonoredValue) {
    HeuristicSchedulingKnobPolicy policy;
    SchedulingKnobOverrides overrides;
    // (1/10000)*16 is 0.0016 and truncates to 0. 1 is the smallest positive
    // throttle the accessor will not replace with the arch floor.
    const SchedulingFeatures features =
        makeFeatures(/*wmma=*/4, /*ds=*/10000, /*firstWmmaLatency=*/32, /*sumWmmaLatency=*/1);
    const ResolvedSchedulingKnobs resolved = resolveSchedulingKnobs(features, overrides, policy);
    EXPECT_EQ(resolved.dsReadThrottleLatency, 1);
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
    EXPECT_NE(line.find("dsOverlapsPriorWmmaSrc=0"), std::string::npos);
    EXPECT_NE(line.find("dsReadThrottleLatency=160(policy)"), std::string::npos);
    EXPECT_NE(line.find("dsReadPerCap=3(policy)"), std::string::npos);
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

// loopWithPrefetch program order, not a DAG. ds_load_b128 writes four VGPRs.
// A load counts once when any of those overlap a VGPR read by an earlier WMMA.
TEST(SchedulingKnobHeuristics, DsLoadDestOverlapCountsPriorWmmaSrcOnly) {
    StinkyAsmModule::ModuleOptions opts{};
    opts.OptLevel = 0;
    StinkyAsmModule module("overlap", kGfx1250, opts);
    std::string group(kMainLoopGroupName);
    module.addGroup(group);

    BasicBlock& bb = *module.getFunction().getEntryBlock();
    const GfxArchID archId = getGfxArchID(kGfx1250[0], kGfx1250[1], kGfx1250[2]);
    AsmIRBuilder builder(bb, archId);

    auto addDs = [&](int dest, bool inGroup) {
        const size_t before = bb.size();
        stinkytofu::test::createDsReadB128InBlock(&bb, archId, dest, /*addrReg=*/400);
        std::vector<const std::string*> groups;
        if (inGroup) groups.push_back(&group);
        module.updateInstructionGroups(groups, before);
    };
    auto addWmma = [&](const StinkyRegister& src, bool inGroup) {
        const size_t before = bb.size();
        StinkyInstruction* inst =
            builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x16_bf16, archId));
        EXPECT_NE(inst, nullptr);
        if (!inst) return;
        inst->addSrcReg(src);
        std::vector<const std::string*> groups;
        if (inGroup) groups.push_back(&group);
        module.updateInstructionGroups(groups, before);
    };

    addDs(/*dest=*/10, /*inGroup=*/false);                  // before the group
    addDs(/*dest=*/0, /*inGroup=*/true);                    // before any WMMA
    addWmma(StinkyRegister("s", 16, 1), /*inGroup=*/true);  // SGPR src is ignored
    addDs(/*dest=*/16, /*inGroup=*/true);                   // v[16:20) vs s16
    addWmma(StinkyRegister("v", 10, 8), /*inGroup=*/true);  // v[10:18)
    addDs(/*dest=*/16, /*inGroup=*/true);                   // overlaps v16, v17
    addDs(/*dest=*/40, /*inGroup=*/true);                   // no prior VGPR src
    addWmma(StinkyRegister("v", 40, 1), /*inGroup=*/true);  // too late for the ds above
    addDs(/*dest=*/40, /*inGroup=*/true);                   // overlaps v40

    const SchedulingIRStats stats = countMainLoopSchedulingIRStats(module);
    EXPECT_EQ(stats.wmmaCount, 3);
    EXPECT_EQ(stats.dsLoadCount, 5);
    EXPECT_EQ(stats.dsLoadDestOverlapsPriorWmmaSrc, 2);
    ASSERT_EQ(stats.dsLoadByToken.size(), 1u);
    EXPECT_FALSE(stats.dsLoadByToken[0].token.has_value());
    EXPECT_EQ(stats.dsLoadByToken[0].dsLoadCount, 5);
    EXPECT_EQ(stats.dsLoadByToken[0].dsLoadDestOverlapsPriorWmmaSrc, 2);
    EXPECT_EQ(stats.dsLoadByToken[0].overlappingWmmaCount, 2);
}

// A ds_load is counted in every MemTokenData token it carries. The global
// overlap count still counts that load once.
TEST(SchedulingKnobHeuristics, DsLoadOverlapSplitsByMemToken) {
    StinkyAsmModule::ModuleOptions opts{};
    opts.OptLevel = 0;
    StinkyAsmModule module("tokens", kGfx1250, opts);
    std::string group(kMainLoopGroupName);
    module.addGroup(group);

    BasicBlock& bb = *module.getFunction().getEntryBlock();
    const GfxArchID archId = getGfxArchID(kGfx1250[0], kGfx1250[1], kGfx1250[2]);
    AsmIRBuilder builder(bb, archId);

    auto addDs = [&](int dest, const std::vector<int>& tokens) {
        const size_t before = bb.size();
        StinkyInstruction* inst =
            stinkytofu::test::createDsReadB128InBlock(&bb, archId, dest, /*addrReg=*/400);
        if (!tokens.empty()) inst->addModifier<MemTokenData>(MemTokenData{tokens});
        module.updateInstructionGroups({&group}, before);
    };
    auto addWmma = [&](int src, int num) {
        const size_t before = bb.size();
        StinkyInstruction* inst =
            builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x16_bf16, archId));
        ASSERT_NE(inst, nullptr);
        inst->addSrcReg(StinkyRegister("v", src, num));
        module.updateInstructionGroups({&group}, before);
    };

    addDs(/*dest=*/0, /*tokens=*/{0});         // before any WMMA
    addWmma(/*src=*/10, /*num=*/8);            // v[10:18)
    addDs(/*dest=*/16, /*tokens=*/{0});        // overlaps, token 0
    addDs(/*dest=*/16, /*tokens=*/{0, 1, 0});  // overlaps, both tokens, dup ignored
    addDs(/*dest=*/40, /*tokens=*/{1});        // token 1, no overlap yet
    addWmma(/*src=*/40, /*num=*/4);            // v[40:44)
    addDs(/*dest=*/40, /*tokens=*/{1});        // overlaps the second WMMA only
    addDs(/*dest=*/50, /*tokens=*/{});         // no token, no overlap

    const SchedulingIRStats stats = countMainLoopSchedulingIRStats(module);
    EXPECT_EQ(stats.dsLoadCount, 6);
    EXPECT_EQ(stats.dsLoadDestOverlapsPriorWmmaSrc, 3);
    ASSERT_EQ(stats.dsLoadByToken.size(), 3u);

    EXPECT_EQ(stats.dsLoadByToken[0].token, 0);
    EXPECT_EQ(stats.dsLoadByToken[0].dsLoadCount, 3);
    EXPECT_EQ(stats.dsLoadByToken[0].dsLoadDestOverlapsPriorWmmaSrc, 2);
    EXPECT_EQ(stats.dsLoadByToken[0].overlappingWmmaCount, 1);

    EXPECT_EQ(stats.dsLoadByToken[1].token, 1);
    EXPECT_EQ(stats.dsLoadByToken[1].dsLoadCount, 3);
    EXPECT_EQ(stats.dsLoadByToken[1].dsLoadDestOverlapsPriorWmmaSrc, 2);
    EXPECT_EQ(stats.dsLoadByToken[1].overlappingWmmaCount, 2);

    EXPECT_FALSE(stats.dsLoadByToken[2].token.has_value());
    EXPECT_EQ(stats.dsLoadByToken[2].dsLoadCount, 1);
    EXPECT_EQ(stats.dsLoadByToken[2].dsLoadDestOverlapsPriorWmmaSrc, 0);
    EXPECT_EQ(stats.dsLoadByToken[2].overlappingWmmaCount, 0);
}
