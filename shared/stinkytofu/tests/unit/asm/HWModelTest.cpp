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
// Pins the CDNA5 (Gfx1250) hardware-model facts.
//
// These assertions read the real HWModel through hwModelForArch(), so a change to
// any migrated constant fails here. Scheduling *behaviour* built on top of these
// numbers is covered by DAGSchedulerPassTest.cpp and tests/filecheck/dag_*.stir.
#include <gtest/gtest.h>

#include <string>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

using namespace stinkytofu;

namespace {
constexpr std::array<int, 3> kGfx1250 = {12, 5, 0};
constexpr std::array<int, 3> kGfx1250v0 = {12, 5, 1};
}  // namespace

TEST(HWModel, Gfx1250KnownDefaults) {
    const HWModel& hw = hwModelForArch(kGfx1250);

    EXPECT_EQ(hw.lds.readQueueDepth, 16);
    EXPECT_EQ(hw.lds.readDrainLatency, 0);
    EXPECT_EQ(hw.lds.readCreditLifetime, 72);

    EXPECT_EQ(hw.barrier.signalToWaitLatency, 11);
    EXPECT_EQ(hw.barrier.jumpOverheadCycles, 6);

    EXPECT_EQ(hw.coexec.transToNonCoreSide, 1);
    EXPECT_EQ(hw.coexec.maxSlotBudget, 18);
}

TEST(HWModel, Gfx1250HazardRules) {
    const HWModel& hw = hwModelForArch(kGfx1250);

    ASSERT_EQ(hw.hazards.numRules, kNumCdna5HazardRules);
    ASSERT_EQ(hw.hazards.numRules, 3);
    ASSERT_NE(hw.hazards.rules, nullptr);

    // The model points at the shared table rather than carrying a copy.
    EXPECT_EQ(hw.hazards.rules, kCdna5HazardRules);

    EXPECT_STREQ(hw.hazards.rules[0].name, "SaluSgprToMemAddr");
    EXPECT_EQ(hw.hazards.rules[0].regType, RegType::S);
    EXPECT_EQ(hw.hazards.rules[0].distance, 8);
    EXPECT_EQ(hw.hazards.rules[0].dir, HazardDir::WriteThenRead);
    EXPECT_EQ(hw.hazards.rules[0].unit, HazardUnit::Cycles);

    EXPECT_STREQ(hw.hazards.rules[1].name, "ValuVgprToVmemAddr");
    EXPECT_EQ(hw.hazards.rules[1].regType, RegType::V);
    EXPECT_EQ(hw.hazards.rules[1].distance, 32);
    EXPECT_EQ(hw.hazards.rules[1].dir, HazardDir::WriteThenRead);
    EXPECT_EQ(hw.hazards.rules[1].unit, HazardUnit::Cycles);

    // PipeOps rule: distance 0 in the table means the arch policy supplies it.
    EXPECT_STREQ(hw.hazards.rules[2].name, "WmmaVgprSrcToDsWrite");
    EXPECT_EQ(hw.hazards.rules[2].regType, RegType::V);
    EXPECT_EQ(hw.hazards.rules[2].distance, 0);
    EXPECT_EQ(hw.hazards.rules[2].dir, HazardDir::ReadThenWrite);
    EXPECT_EQ(hw.hazards.rules[2].unit, HazardUnit::PipeOps);
    EXPECT_NE(hw.hazards.rules[2].isPipeOp, nullptr);
}

// gfx1250v0 currently aliases gfx1250 field-for-field. This pins that it is a
// deliberate alias of the same values, not an accidental fallthrough: when
// gfx1250v0 gets its own tuning, this test is the one that should be updated.
TEST(HWModel, Gfx1250v0MatchesGfx1250ForNow) {
    const HWModel& base = hwModelForArch(kGfx1250);
    const HWModel& v0 = hwModelForArch(kGfx1250v0);

    EXPECT_EQ(v0.lds.readQueueDepth, base.lds.readQueueDepth);
    EXPECT_EQ(v0.lds.readDrainLatency, base.lds.readDrainLatency);
    EXPECT_EQ(v0.lds.readCreditLifetime, base.lds.readCreditLifetime);
    EXPECT_EQ(v0.barrier.signalToWaitLatency, base.barrier.signalToWaitLatency);
    EXPECT_EQ(v0.hazards.rules, base.hazards.rules);
}

// An unlisted arch falls back to gfx1250 — and must return the *same object*, so
// callers that cache the reference stay valid.
TEST(HWModel, UnlistedArchFallsBackToGfx1250) {
    EXPECT_EQ(&hwModelForArch({9, 4, 2}), &hwModelForArch(kGfx1250));
}

// Many unit tests construct a bare PassContext, and EstimateAsmCyclesPass builds
// one internally; none of those call setGemmTileConfig. getHWModel() must still
// answer rather than dereference a null cached pointer.
TEST(HWModel, BarePassContextReturnsDefaultModel) {
    PassContext ctx;
    EXPECT_EQ(&ctx.getHWModel(), &hwModelForArch(kGfx1250));
}

TEST(HWModel, ConfiguredPassContextCachesMatchingModel) {
    GemmTileConfig cfg;
    cfg.arch = kGfx1250;
    PassContext ctx;
    ctx.setGemmTileConfig(cfg);
    EXPECT_EQ(&ctx.getHWModel(), &hwModelForArch(kGfx1250));
}

// ---------------------------------------------------------------------------
// ds issue cost vs resident waves. A ds_load's ISA issue cost is quoted for one
// wave; the issue pipe is shared, so resident waves round-robin it and a single
// wave's issues are spaced out by however many share its pipe.
// ---------------------------------------------------------------------------

// gfx1250's pipe-sharing is temporarily disabled (wavesPerDsIssuePipe = 1, see
// HWModel.cpp) after real-hardware measurement found it cost f8_tn_medium
// ~17.5% and mxf4_tn_medium ~12.3% throughput. The tests below that exercise
// the sharing math re-enable it on a local copy of the model, so the model
// logic stays covered independent of whether the arch currently applies it.
TEST(HWModelDsIssue, PipeSharingIsTemporarilyDisabled) {
    const HWModel& hw = hwModelForArch({12, 5, 0});
    EXPECT_EQ(hw.lds.wavesPerDsIssuePipe, 1)
        << "re-enable only after hardware re-validation (see HWModel.cpp)";
}

TEST(HWModelDsIssue, SingleWaveKeepsTheIsaCost) {
    const HWModel& hw = hwModelForArch({12, 5, 0});
    EXPECT_EQ(dsIssueCyclesForWaves(hw, /*issueCycles=*/1, /*numWaves=*/1), 1);
}

TEST(HWModelDsIssue, FourWavesRunAsPairsSoTheCostDoubles) {
    HWModel hw = hwModelForArch({12, 5, 0});
    hw.lds.wavesPerDsIssuePipe = 2;  // re-enable for this test; see HWModel.cpp
    // 4 waves over a 2-wave pipe is 2-2: a wave contends with one partner, not
    // with all three others, so the cost saturates at the share rather than
    // scaling with the wave count.
    EXPECT_EQ(dsIssueCyclesForWaves(hw, /*issueCycles=*/1, /*numWaves=*/4), 2);
    EXPECT_EQ(dsIssueCyclesForWaves(hw, /*issueCycles=*/1, /*numWaves=*/8), 2);
}

TEST(HWModelDsIssue, TwoWavesAreAssumedPaired) {
    HWModel hw = hwModelForArch({12, 5, 0});
    hw.lds.wavesPerDsIssuePipe = 2;  // re-enable for this test; see HWModel.cpp
    // UNVERIFIED on hardware: the conservative reading is that two waves share
    // one pipe. If they turn out to land on separate pipes this becomes 1, and
    // this test is the one to flip.
    EXPECT_EQ(dsIssueCyclesForWaves(hw, /*issueCycles=*/1, /*numWaves=*/2), 2);
}

TEST(HWModelDsIssue, NonsenseWaveCountFallsBackToTheIsaCost) {
    const HWModel& hw = hwModelForArch({12, 5, 0});
    // GemmTileConfig::NumWaves defaults to 1, so this is not the unconfigured
    // path -- it guards a caller that passes something meaningless.
    EXPECT_EQ(dsIssueCyclesForWaves(hw, /*issueCycles=*/1, /*numWaves=*/0), 1);
    EXPECT_EQ(dsIssueCyclesForWaves(hw, /*issueCycles=*/4, /*numWaves=*/-3), 4);
}

TEST(HWModelDsIssue, DefaultConfigIsSingleWave) {
    const HWModel& hw = hwModelForArch({12, 5, 0});
    const GemmTileConfig defaults;
    EXPECT_EQ(defaults.NumWaves, 1u) << "a default config must mean one wave, not a sentinel";
    EXPECT_EQ(dsIssueCyclesForWaves(hw, /*issueCycles=*/1, static_cast<int>(defaults.NumWaves)), 1);
    EXPECT_EQ(defaults.TileA0, 0u) << "0 is not a valid tile, so it marks an unset config";
}

TEST(HWModelDsIssue, ScalesAMultiCycleIssueCost) {
    HWModel hw = hwModelForArch({12, 5, 0});
    hw.lds.wavesPerDsIssuePipe = 2;  // re-enable for this test; see HWModel.cpp
    EXPECT_EQ(dsIssueCyclesForWaves(hw, /*issueCycles=*/4, /*numWaves=*/4), 8);
}

TEST(HWModelDsIssue, UnmodelledShareIsInert) {
    HWModel hw = hwModelForArch({12, 5, 0});
    hw.lds.wavesPerDsIssuePipe = 0;
    EXPECT_EQ(dsIssueCyclesForWaves(hw, /*issueCycles=*/1, /*numWaves=*/4), 1)
        << "an arch that does not model pipe sharing must keep the ISA cost";
}

namespace {
// Fewest matrix ops alone that satisfy a rule (slots per matrix op); -1 when none do.
int matrixOpsAlone(const HWModel::WaitHide::Hide& h, int slots) {
    if (!h.on()) return -1;
    if (h.tail == 0) return h.lead;
    if (slots <= 0) return -1;
    return h.lead + (h.tail + slots - 1) / slots;
}
int matrixOpsAlone(const HWModel::WaitHide::RoleHide& rh, int slots) {
    int best = -1;
    for (const auto* h : {&rh.pure, &rh.mix, &rh.valu}) {
        const int n = matrixOpsAlone(*h, slots);
        if (n >= 0 && (best < 0 || n < best)) best = n;
    }
    return best;
}
// Fewest VALUs alone (no matrix op) that satisfy a rule; -1 when none do.
int valusAlone(const HWModel::WaitHide::RoleHide& rh) {
    int best = -1;
    for (const auto* h : {&rh.pure, &rh.mix, &rh.valu}) {
        if (!h->on() || h->lead > 0) continue;
        if (best < 0 || h->tail < best) best = h->tail;
    }
    return best;
}
// A src rule set may only hide EARLIER than the dst one it stands next to: the source read
// completes before the writeback. Checked where both sides have rules.
void expectSrcNoStricterThanDst(const HWModel::WaitHide::RuleSet& rs, const char* what) {
    if (!rs.src.any() || !rs.dst.any()) return;
    const int wSrc = matrixOpsAlone(rs.src, rs.wmmaSlots);
    const int wDst = matrixOpsAlone(rs.dst, rs.wmmaSlots);
    if (wSrc >= 0 && wDst >= 0) EXPECT_LE(wSrc, wDst) << what << " matrix ops";
    const int vSrc = valusAlone(rs.src);
    const int vDst = valusAlone(rs.dst);
    if (vSrc >= 0 && vDst >= 0) EXPECT_LE(vSrc, vDst) << what << " VALUs";
}
}  // namespace

// The lookup key must stay unique, src rules must not be stricter than dst ones, and gfx1250v0
// must keep mirroring gfx1250; no numbers pinned.
TEST(HWModel, Gfx1250WaitHideFormTable) {
    const HWModel& hw = hwModelForArch(kGfx1250);

    // Two rows sharing a {costLatency, dstVgprs, scaled} key would make the lookup
    // order-dependent.
    for (size_t i = 0; i < hw.waitHide.forms.size(); ++i) {
        for (size_t j = i + 1; j < hw.waitHide.forms.size(); ++j) {
            const auto& a = hw.waitHide.forms[i];
            const auto& b = hw.waitHide.forms[j];
            EXPECT_FALSE(a.costLatency == b.costLatency && a.dstVgprs == b.dstVgprs &&
                         a.scaled == b.scaled)
                << "duplicate key: latency " << a.costLatency << " dst " << a.dstVgprs << " scaled "
                << a.scaled;
        }
    }

    // The pass measures ages against a ring of kAgeRing matrix ordinals (InsertWaitAlu
    // AgeClock::kRing) and takes "older than the ring" as "every matrix rule met": that needs the
    // age cap, 2 * max(lead + tail), inside the ring, and a pure rule beside every mixed one.
    constexpr int kAgeRing = 64;
    auto expectRingFits = [&](const HWModel::WaitHide::RuleSet& rs, const char* what) {
        for (const auto* rh : {&rs.dst, &rs.src}) {
            for (const auto* h : {&rh->pure, &rh->mix, &rh->valu})
                EXPECT_LE(2 * (h->lead + h->tail), kAgeRing) << what;
            // A lead-less mixed rule is a VALU-only count (the capacity sets); a led one needs
            // a pure rule beside it.
            if (rh->mix.lead > 0) EXPECT_TRUE(rh->pure.on()) << what << ": mixed rule without a pure one";
        }
    };
    expectRingFits(hw.waitHide.noLatchCsmacc, "noLatchCsmacc");
    expectRingFits(hw.waitHide.dpmacc, "dpmacc");
    for (const auto& form : hw.waitHide.forms) {
        expectRingFits(form.xdl, "xdl");
        expectRingFits(form.csmacc, "csmacc");
        expectSrcNoStricterThanDst(form.xdl, "xdl");
        expectSrcNoStricterThanDst(form.csmacc, "csmacc");
        // No mixed rule without a lead: the pass counts a mixed rule's VALUs from the lead on.
        for (const auto* rs : {&form.xdl, &form.csmacc})
            for (const auto* rh : {&rs->dst, &rs->src})
                if (rh->mix.on()) EXPECT_GT(rh->mix.lead, 0) << "lead-less mixed rule";
        // s is the form's wmma_scale: a scaled matrix op takes two tail slots, a plain one one.
        EXPECT_EQ(form.xdl.wmmaSlots, form.scaled ? 2 : 1) << "xdl slots of " << form.dstVgprs;
        EXPECT_EQ(form.csmacc.wmmaSlots, form.scaled ? 2 : 1) << "csmacc slots of " << form.dstVgprs;
    }
    expectSrcNoStricterThanDst(hw.waitHide.noLatchCsmacc, "noLatchCsmacc");
    expectSrcNoStricterThanDst(hw.waitHide.dpmacc, "dpmacc");

    // A deliberate alias, not an accidental fallthrough.
    const HWModel& v0 = hwModelForArch(kGfx1250v0);
    ASSERT_EQ(v0.waitHide.forms.size(), hw.waitHide.forms.size());
    EXPECT_EQ(v0.waitHide.vmVsrcLds, hw.waitHide.vmVsrcLds);
    EXPECT_EQ(v0.waitHide.vmVsrcTex, hw.waitHide.vmVsrcTex);
    EXPECT_EQ(v0.waitHide.vmVsrcBridge, hw.waitHide.vmVsrcBridge);
    EXPECT_TRUE(v0.waitHide.noLatchCsmacc == hw.waitHide.noLatchCsmacc);
    EXPECT_TRUE(v0.waitHide.dpmacc == hw.waitHide.dpmacc);
    for (size_t i = 0; i < hw.waitHide.forms.size(); ++i)
        EXPECT_TRUE(v0.waitHide.forms[i] == hw.waitHide.forms[i]) << "row " << i;
}


// The gfx1250 wait-hide table, cell by cell (WMMA producer rules, the derived CSMACC set, the
// capacity sets). A row key that moves or a number that drifts fails here on purpose.
TEST(HWModel, Gfx1250WaitHideFinalEncode) {
    using Hide = HWModel::WaitHide::Hide;
    using RoleHide = HWModel::WaitHide::RoleHide;
    const HWModel& hw = hwModelForArch(kGfx1250);
    auto row = [&](int lat, int dst, bool scaled) -> const HWModel::WaitHide::Form* {
        for (const auto& f : hw.waitHide.forms)
            if (f.costLatency == lat && f.dstVgprs == dst && f.scaled == scaled) return &f;
        return nullptr;
    };
    struct Expect {
        int lat, dst;
        bool scaled;
        RoleHide dstRules;
        RoleHide srcRules;
        int slots;
    };
    const Expect table[] = {
        {4, 8, false, {Hide{13, 0}, Hide{6, 7}, Hide{0, 21}}, {Hide{9, 0}, Hide{2, 7}, Hide{0, 10}}, 1},
        {4, 8, true, {Hide{8, 0}, Hide{5, 6}, Hide{0, 21}}, {Hide{5, 0}, Hide{2, 6}, Hide{0, 10}}, 2},
        {8, 8, false, {Hide{12, 0}, Hide{5, 7}, Hide{0, 21}}, {Hide{9, 0}, Hide{1, 8}, Hide{0, 9}}, 1},
        {8, 8, true, {Hide{7, 0}, Hide{4, 6}, Hide{0, 21}}, {Hide{5, 0}, Hide{1, 8}, Hide{0, 9}}, 2},
        {8, 16, true, {Hide{7, 0}, Hide{3, 8}, Hide{0, 21}}, {Hide{5, 0}, Hide{1, 8}, Hide{0, 10}}, 2},
        {16, 8, false, {Hide{12, 0}, Hide{5, 7}, Hide{0, 21}}, {Hide{9, 0}, Hide{1, 8}, Hide{0, 9}}, 1},
    };
    EXPECT_EQ(hw.waitHide.forms.size(), std::size(table));
    for (const Expect& e : table) {
        const auto* f = row(e.lat, e.dst, e.scaled);
        ASSERT_NE(f, nullptr) << e.lat << "/" << e.dst << "/" << e.scaled;
        EXPECT_TRUE(f->xdl.dst == e.dstRules) << "dst " << e.lat << "/" << e.dst << "/" << e.scaled;
        EXPECT_TRUE(f->xdl.src == e.srcRules) << "src " << e.lat << "/" << e.dst << "/" << e.scaled;
        // A CSMACC producer: the WMMA rules, except VALUs alone on the src role (path capacity).
        HWModel::WaitHide::RuleSet asCsmacc = f->xdl;
        asCsmacc.src.valu = Hide{0, 12};
        EXPECT_TRUE(f->csmacc == asCsmacc) << "csmacc " << e.lat << "/" << e.dst << "/" << e.scaled;
        EXPECT_EQ(f->xdl.wmmaSlots, e.slots);
    }
    // No unscaled 8-pass form writes 16 VGPRs; the old ghost row is gone.
    EXPECT_EQ(row(8, 16, false), nullptr);
    // Form-independent sets, the table's (-1, 12, -1): src 12 VALU-class ops with no order, matrix
    // ops count nothing, dst always waits.
    const RoleHide capacity{Hide{}, Hide{0, 12}, Hide{}};
    const int noWmma = HWModel::WaitHide::RuleSet::kNoWmma;
    EXPECT_TRUE(hw.waitHide.noLatchCsmacc.src == capacity);
    EXPECT_FALSE(hw.waitHide.noLatchCsmacc.dst.any());
    EXPECT_EQ(hw.waitHide.noLatchCsmacc.wmmaSlots, noWmma);
    EXPECT_TRUE(hw.waitHide.dpmacc.src == capacity);
    EXPECT_FALSE(hw.waitHide.dpmacc.dst.any());
    EXPECT_EQ(hw.waitHide.dpmacc.wmmaSlots, noWmma);
}

// archName disambiguates steppings sharing a triple (gfx1250 vs gfx1250v0).
TEST(ArchResolve, ArchNameDisambiguatesSharedTriple) {
    const auto& helper = ArchHelper::getInstance();
    if (helper.getArchInfo(std::string("gfx1250v0")) == nullptr ||
        helper.getArchInfo(std::string("gfx1250")) == nullptr) {
        GTEST_SKIP() << "build does not register both gfx1250 and gfx1250v0";
    }

    const std::array<int, 3> triple = {12, 5, 0};

    // Empty name: resolve by triple alone -> the first-registered stepping.
    EXPECT_EQ(resolveArchId(triple, ""), getGfxArchID(12, 5, 0));

    // A concrete name picks the matching stepping within the same triple.
    EXPECT_EQ(resolveArchId(triple, "gfx1250v0"), getGfxArchID("gfx1250v0"));
    EXPECT_EQ(resolveArchId(triple, "gfx1250"), getGfxArchID("gfx1250"));

    // The two steppings are distinct identities, so v0 is not the triple default.
    EXPECT_NE(resolveArchId(triple, "gfx1250v0"), resolveArchId(triple, ""));
}
