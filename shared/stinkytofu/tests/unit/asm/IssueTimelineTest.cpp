// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/AnalysisManager.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"
#include "transforms/asm/coissue/DamageReport.hpp"
#include "transforms/asm/coissue/IssueTimeline.hpp"
#include "transforms/asm/coissue/TimingProfile.hpp"

using namespace stinkytofu;
using namespace stinkytofu::coissue;

namespace {

constexpr std::array<int, 3> kArch{12, 5, 0};

const char* kScaleWmma =
    "\"st.v_wmma_scale_f32_16x16x128_f8f6f4\"(v[300:315], v[316:331], %ACC%, v220, v221) "
    "{ mod.matrix_fmt = { fmtA = \"MATRIX_FMT_FP8\", fmtB = \"MATRIX_FMT_FP8\" } }";

std::string wmma(const std::string& acc) {
    std::string text = kScaleWmma;
    text.replace(text.find("%ACC%"), 5, acc);
    return acc + " = " + text;
}

class IssueTimelineTest : public ::testing::Test {
   protected:
    const HWModel& hw = hwModelForArch(kArch);
    const GfxArchID arch = getGfxArchID(12, 5, 0);
    StinkyIRConverter converter{kArch};
    std::vector<TimedInst> timed;

    // Parse `ir` and build the timeline's view of every instruction, in order.
    void build(const std::string& ir) {
        Function* func = converter.convertToFunction(ir);
        ASSERT_NE(func, nullptr);
        timed.clear();
        for (BasicBlock& bb : *func)
            for (IRBase& node : bb)
                if (auto* inst = dyn_cast<StinkyInstruction>(&node))
                    timed.push_back(makeTimedInst(*inst, hw));
    }

    std::vector<int> cycles(const TimingProfile& profile) {
        IssueTimeline tl(profile);
        std::vector<int> out;
        for (const TimedInst& t : timed) out.push_back(tl.place(t).cycle);
        out.push_back(tl.now());
        return out;
    }

    TimingProfile compiler() const {
        return compilerProfile(hw);
    }
    TimingProfile measured() const {
        return measuredProfile(hw, arch);
    }
};

// --- compiler preset: the four examples of section 3 of the report ----------------------

// Example 1: the wait lands on a free cycle, so the next v_wmma still issues on cycle 8.
TEST_F(IssueTimelineTest, CompilerWaitOnFreeCycle) {
    build(wmma("v[0:7]") + R"(
        v[420:423] = "st.ds_load_b128"(v38) { mod.ds = { na = 1, offset = 128, gds = false } }
        v[424:427] = "st.ds_load_b128"(v38) { mod.ds = { na = 1, offset = 160, gds = false } }
        s21, SCC0 = "st.s_add_i32"(s21, s6)
        SCC0 = "st.s_cmp_lt_u32"(s69, 136704)
        s6 = "st.s_mov_b32"(-136704)
        "st.s_wait_dscnt"(2)
    )" + wmma("v[8:15]"));
    const std::vector<int> c = cycles(compiler());
    EXPECT_EQ(c, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 8, 9}));

    // Still free at 2 cycles; at 3 the wait runs into the blocked cycle 7.
    TimingProfile two = compiler();
    two.waitcntIssueCycles = 2;
    EXPECT_EQ(cycles(two)[7], 8);
    TimingProfile three = compiler();
    three.waitcntIssueCycles = 3;
    EXPECT_EQ(cycles(three)[7], 9);
}

// Example 2: the window is already full; the wait pushes the v_wmma by its cost.
TEST_F(IssueTimelineTest, CompilerWaitAfterFullWindow) {
    build(wmma("v[0:7]") + R"(
        v[420:423] = "st.ds_load_b128"(v38) { mod.ds = { na = 1, offset = 128, gds = false } }
        v[424:427] = "st.ds_load_b128"(v38) { mod.ds = { na = 1, offset = 160, gds = false } }
        SCC0 = "st.s_cmp_lt_u32"(s21, 136704)
        s6 = "st.s_mov_b32"(-136704)
        s6 = "st.s_cselect_b32"(136704, s6, SCC0)
        "st.s_wait_dscnt"(2)
    )" + wmma("v[8:15]"));
    EXPECT_EQ(cycles(compiler()), (std::vector<int>{0, 1, 2, 3, 4, 6, 8, 9, 10}));
    TimingProfile flat3 = compiler();
    flat3.waitcntIssueCycles = 3;
    EXPECT_EQ(cycles(flat3)[7], 11);
}

// Example 3: a wait in front of a co-issued VALU takes its slot; the VALU waits for slot 6
// and the scalars behind it run past the window.
TEST_F(IssueTimelineTest, CompilerWaitTakesValuSlot) {
    const std::string tail = R"(
        v430 = "st.v_add_nc_u32"(v412, v431)
        s30, SCC0 = "st.s_add_u32"(s30, s8)
        s31, SCC0 = "st.s_addc_u32"(s31, 0, SCC0)
    )";
    const std::string loads = R"(
        v[420:423] = "st.ds_load_b128"(v38) { mod.ds = { na = 1, offset = 128, gds = false } }
        v[424:427] = "st.ds_load_b128"(v38) { mod.ds = { na = 1, offset = 160, gds = false } }
    )";
    const std::string next = wmma("v[8:15]") + "\n v432 = \"st.v_add_nc_u32\"(v432, v433)\n";

    build(wmma("v[0:7]") + loads + tail + next);
    EXPECT_EQ(cycles(compiler()), (std::vector<int>{0, 1, 2, 3, 4, 6, 8, 11, 12}));

    build(wmma("v[0:7]") + loads + "\"st.s_wait_dscnt\"(2)\n" + tail + next);
    EXPECT_EQ(cycles(compiler()), (std::vector<int>{0, 1, 2, 3, 6, 8, 10, 11, 14, 15}));
}

// Example 4: bank switches push the loads, the VALU loses slot 3, and the last switch
// lands on cycle 8 because cycle 7 is blocked.
TEST_F(IssueTimelineTest, CompilerBankSwitches) {
    build(R"(
        "st.s_set_vgpr_msb"(10)
    )" + wmma("v[34:41]") + R"(
        "st.s_set_vgpr_msb"(2754)
        v[874:877] = "st.ds_load_b128"(v550) { mod.ds = { na = 1, offset = 128, gds = false } }
        v[878:881] = "st.ds_load_b128"(v550) { mod.ds = { na = 1, offset = 160, gds = false } }
        "st.s_set_vgpr_msb"(49871)
        v1006 = "st.v_add_nc_u32"(v1006, v1007)
        "st.s_set_vgpr_msb"(53002)
    )" + wmma("v[98:105]"));
    const std::vector<int> c = cycles(compiler());
    // Relative to the first v_wmma.
    std::vector<int> rel;
    for (size_t i = 1; i + 1 < c.size(); ++i) rel.push_back(c[i] - c[1]);
    EXPECT_EQ(rel, (std::vector<int>{0, 1, 2, 3, 4, 6, 8, 9}));
}

// --- measured preset: the gfx1250 facts of HW_COISSUE.md -------------------------------

TEST_F(IssueTimelineTest, MeasuredCompareToBranchTakesNine) {
    build(R"(
        SCC0 = "st.s_cmp_eq_u32"(s60, 1)
        "st.s_cbranch_scc0"("label_end", SCC0)
    )");
    EXPECT_EQ(cycles(measured())[1], 9);
    EXPECT_EQ(cycles(compiler())[1], 2);
}

TEST_F(IssueTimelineTest, MeasuredScalarToScalarTakesOne) {
    build(R"(
        SCC0 = "st.s_cmp_le_i32"(s14, 1)
        s18 = "st.s_cmov_b32"(0, s18, SCC0)
    )");
    EXPECT_EQ(cycles(measured())[1], 1);
    EXPECT_EQ(cycles(compiler())[1], 2);
}

// Window 130 of the mxf8_tn_maf loop: 13 cycles measured (as on hardware), 14 compiler.
TEST_F(IssueTimelineTest, MeasuredClusterWindow130) {
    build(wmma("v[128:135]") + R"(
        s6 = "st.s_cselect_b32"(136704, s6, SCC0)
        "st.s_set_vgpr_msb"(2946)
        v[680:683] = "st.ds_load_b128"(v548) { mod.ds = { na = 1, offset = 128, gds = false } }
        s69, SCC0 = "st.s_add_i32"(s69, s6)
        SCC0 = "st.s_cmp_le_i32"(s14, 1)
        s18 = "st.s_cmov_b32"(0, s18, SCC0)
        v[684:687] = "st.ds_load_b128"(v548) { mod.ds = { na = 1, offset = 160, gds = false } }
        s19 = "st.s_cmov_b32"(0, s19, SCC0)
        s22, SCC0 = "st.s_add_u32"(s22, s18)
        "st.s_set_vgpr_msb"(33291)
        "st.s_wait_dscnt"(52)
    )" + wmma("v[192:199]"));
    const std::vector<int> m = cycles(measured());
    EXPECT_EQ(m, (std::vector<int>{0, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 10, 13, 15}));
    const std::vector<int> c = cycles(compiler());
    EXPECT_EQ(c[6], 8);   // s_cmov: 2 cycles of latency, then the blocked cycle
    EXPECT_EQ(c[12], 14);
}

TEST_F(IssueTimelineTest, MeasuredFirstWaitPaysSettle) {
    build(R"(
        s5 = "st.s_mov_b32"(1)
        "st.s_wait_dscnt"(0)
        "st.s_wait_dscnt"(0)
        s6 = "st.s_mov_b32"(2)
    )");
    EXPECT_EQ(cycles(measured()), (std::vector<int>{0, 1, 4, 5, 6}));
    EXPECT_EQ(cycles(compiler()), (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST_F(IssueTimelineTest, MeasuredBankSwitchCostDependsOnPredecessor) {
    build(R"(
        v[0:3] = "st.ds_load_b128"(v38) { mod.ds = { na = 1, offset = 0, gds = false } }
        "st.s_set_vgpr_msb"(10)
        s5 = "st.s_mov_b32"(1)
        "st.s_set_vgpr_msb"(11)
        s6 = "st.s_mov_b32"(2)
    )");
    // 3 cycles after the load, nothing after the scalar.
    EXPECT_EQ(cycles(measured()), (std::vector<int>{0, 1, 4, 5, 5, 6}));
    EXPECT_EQ(cycles(compiler()), (std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST_F(IssueTimelineTest, MeasuredValuIssuesOnBlockedCycle) {
    build(wmma("v[0:7]") + R"(
        s1 = "st.s_mov_b32"(1)
        s2 = "st.s_mov_b32"(2)
        s3 = "st.s_mov_b32"(3)
        s4 = "st.s_mov_b32"(4)
        s5 = "st.s_mov_b32"(5)
        v430 = "st.v_add_nc_u32"(v431, v432)
    )");
    EXPECT_EQ(cycles(measured())[6], 7);
    TimingProfile blocking = measured();
    blocking.blockedCycleAtIssue = true;
    EXPECT_EQ(cycles(blocking)[6], 8);
}

TEST_F(IssueTimelineTest, MatrixQueueLetsIssueRunAhead) {
    build(wmma("v[0:7]") + wmma("v[8:15]") + wmma("v[16:23]") + wmma("v[24:31]") +
          wmma("v[32:39]"));
    // No queue: one window each. Depth 3: the wave issues 2 cycles apart while the pipe
    // runs one op and three wait; the fifth waits until the second starts at 8.
    EXPECT_EQ(cycles(compiler()), (std::vector<int>{0, 8, 16, 24, 32, 33}));
    IssueTimeline tl(measured());
    std::vector<int> issue;
    for (const TimedInst& t : timed) issue.push_back(tl.place(t).cycle);
    EXPECT_EQ(issue, (std::vector<int>{0, 2, 4, 6, 8}));
    ASSERT_EQ(tl.pipe().size(), 5u);
    EXPECT_EQ(tl.pipe()[4].start, 32);
}

// --- QueueModel, SyncModel and the cost -----------------------------------------------------

// Short windows (two matrix ops issued 2 cycles apart) build a lead over the pipe; a delay is
// free while the lead covers it, and costs in full once the lead is gone.
TEST_F(IssueTimelineTest, QueueLeadHidesDelay) {
    const std::string four = wmma("v[0:7]") + wmma("v[8:15]") + wmma("v[16:23]") + wmma("v[24:31]");
    build(four + "\n\"st.s_nop\"(8)\n" + wmma("v[32:39]"));
    IssueTimeline tl(measured());
    for (const TimedInst& t : timed) tl.place(t);
    const auto& pipe = tl.pipe();
    ASSERT_EQ(pipe.size(), 5u);
    // Leads 0, 6, 12, 18: the issue runs ahead of the 8-cycle ops.
    EXPECT_EQ(pipe[1].start - pipe[1].issue, 6);
    EXPECT_EQ(pipe[3].start - pipe[3].issue, 18);
    // The 9-cycle s_nop is hidden: the fifth op still starts right behind the fourth.
    EXPECT_EQ(pipe[4].start, pipe[3].end);

    // No queue, so no lead: the s_nop starts 2 cycles into the fourth window and runs 3
    // cycles past its end, and the pipe idles for those 3.
    TimingProfile noQueue = measured();
    noQueue.matrixQueueDepth = 0;
    IssueTimeline flat(noQueue);
    for (const TimedInst& t : timed) flat.place(t);
    EXPECT_EQ(flat.pipe()[4].start - flat.pipe()[3].end, 2 + 9 - 8);
}

TEST_F(IssueTimelineTest, BarrierWaitDrainsQueue) {
    build(wmma("v[0:7]") + wmma("v[8:15]") + wmma("v[16:23]") + R"(
        "st.s_barrier_wait"(-1)
    )" + wmma("v[24:31]"));
    // Conservative sync: the op after the barrier wait issues once the queued work is done.
    IssueTimeline tl(measured());
    for (const TimedInst& t : timed) tl.place(t);
    EXPECT_EQ(tl.pipe()[3].issue, tl.pipe()[2].end);
    EXPECT_EQ(tl.pipe()[3].start - tl.pipe()[3].issue, 0);
    // Without the sync model the queue keeps its lead across the barrier.
    TimingProfile free = measured();
    free.sync = SyncModel::None;
    IssueTimeline tf(free);
    for (const TimedInst& t : timed) tf.place(t);
    EXPECT_LT(tf.pipe()[3].issue, tf.pipe()[2].end);
}

TEST_F(IssueTimelineTest, SteadyTripCostIsIdleThenLength) {
    build(wmma("v[0:7]") + R"(
        s1 = "st.s_mov_b32"(1)
        s2 = "st.s_mov_b32"(2)
    )" + wmma("v[8:15]"));
    std::vector<const TimedInst*> body;
    for (const TimedInst& t : timed) body.push_back(&t);
    const TripTiming trip = steadyTrip(body, compiler());
    // No queue: two windows of 8, back to back across the back edge. The trip's issue ends
    // when its second op issues, 8 + 1 cycles after its first.
    EXPECT_EQ(trip.cycles, 9);
    EXPECT_EQ(trip.pipeIdle, 0);
    EXPECT_EQ(trip.pipeIdleWithHandover, 0);
    EXPECT_EQ(trip.cost(), (TripCost{0, 9}));
    // Idle first, issue length as tie-break.
    EXPECT_LT((TripCost{0, 17}), (TripCost{1, 10}));
}

TEST_F(IssueTimelineTest, DamageReportFindsLostSlot) {
    // Plan: the VALU takes slot 3 of the first window. Final: a bank switch in front of it
    // takes cycle 3, so the VALU slips to slot 6.
    const std::string head = wmma("v[0:7]") + R"(
        s1 = "st.s_mov_b32"(1)
        s2 = "st.s_mov_b32"(2)
    )";
    const std::string tail = R"(
        v430 = "st.v_add_nc_u32"(v431, v432)
    )" + wmma("v[8:15]");
    build(head + tail);
    std::vector<const TimedInst*> order;
    for (const TimedInst& t : timed) order.push_back(&t);
    const TripTiming plan = steadyTrip(order, compiler());
    std::vector<TimedInst> planInsts = timed;

    build(head + "\n\"st.s_set_vgpr_msb\"(10)\n" + tail);
    std::vector<const TimedInst*> finalBody;
    for (const TimedInst& t : timed) finalBody.push_back(&t);
    const TripTiming final = steadyTrip(finalBody, compiler());
    // Order position k of the final trip: skip the inserted switch (index 3).
    std::vector<Placement> finalByOrder;
    for (size_t i = 0; i < finalBody.size(); ++i)
        if (i != 3) finalByOrder.push_back(final.placements[i]);
    std::vector<const TimedInst*> planOrder;
    for (const TimedInst& t : planInsts) planOrder.push_back(&t);
    const DamageReport report = buildDamageReport(planOrder, plan.placements, finalByOrder, plan,
                                                  final, DamageTrigger::IssueGrowth);
    ASSERT_EQ(report.all.size(), 2u);
    EXPECT_EQ(plan.placements[3].pos, 3);
    EXPECT_EQ(finalByOrder[3].pos, 6);
    ASSERT_EQ(report.all[0].slipped.size(), 1u);
    EXPECT_EQ(report.all[0].slipped[0], 3u);
    ASSERT_FALSE(report.damaged.empty());
    EXPECT_EQ(report.damaged.front()->window, 0);
}

// --- profile resolution -------------------------------------------------------------------

TEST_F(IssueTimelineTest, SpecsParseStrictly) {
    std::vector<CostRule> rules;
    EXPECT_FALSE(parseIssueCycles("s_wait_tensorcnt=4; s_set_vgpr_msb@lds=3", arch, rules));
    ASSERT_EQ(rules.size(), 2u);
    EXPECT_EQ(rules[1].after.cls, IssueClass::LdsLoad);
    EXPECT_EQ(rules[1].cycles, 3);
    EXPECT_TRUE(parseIssueCycles("s_wait_tensorcnt", arch, rules));
    EXPECT_TRUE(parseIssueCycles("s_not_an_op=1", arch, rules));
    EXPECT_TRUE(parseIssueCycles("salu=-1", arch, rules));
    EXPECT_TRUE(parseIssueCycles("salu=1;;valu=2", arch, rules));

    std::vector<LatencyRule> latency;
    EXPECT_FALSE(parseScalarLatency("salu>salu=1; salu>branch:scc=9", latency));
    ASSERT_EQ(latency.size(), 2u);
    EXPECT_EQ(latency[1].consumer, IssueClass::Branch);
    EXPECT_EQ(latency[1].reg, LatencyReg::Scc);
    EXPECT_TRUE(parseScalarLatency("salu>salu", latency));
    EXPECT_TRUE(parseScalarLatency("salu>branch:xyz=9", latency));
    EXPECT_TRUE(parseScalarLatency("foo>salu=1", latency));
}

TEST_F(IssueTimelineTest, RobustSetAndKnobs) {
    PassFeatureConfig::CoissueFeatures f;
    ProfileSet set;
    ASSERT_FALSE(resolveProfileSet(hw, arch, f, set));
    ASSERT_EQ(set.profiles.size(), 5u);
    EXPECT_EQ(set.primary, 1);
    EXPECT_EQ(set.profiles[0].name, "compiler");
    EXPECT_EQ(set.profiles[1].matrixQueueDepth, 3);
    EXPECT_EQ(set.profiles[2].matrixQueueDepth, 2);
    EXPECT_EQ(set.profiles[3].matrixQueueDepth, 4);
    EXPECT_EQ(set.profiles[4].waitcntIssueCycles, 3);
    EXPECT_EQ(set.profiles[4].waitcntSettleCycles, 0);

    // A knob wins over the fact, in every profile derived from the facts.
    f.waitcntSettleCycles = 1;
    f.matrixQueueDepth = 0;
    f.profileSet = "measured+measured-queue2";
    ASSERT_FALSE(resolveProfileSet(hw, arch, f, set));
    ASSERT_EQ(set.profiles.size(), 2u);
    EXPECT_EQ(set.profiles[0].waitcntSettleCycles, 1);
    EXPECT_EQ(set.profiles[0].matrixQueueDepth, 0);
    EXPECT_EQ(set.profiles[1].matrixQueueDepth, 2);

    // Calibrated scope: only the scale FP8 form.
    const TimingProfile m = measured();
    const int scale = static_cast<int>(GFX::v_wmma_scale_f32_16x16x128_f8f6f4);
    EXPECT_TRUE(m.covers({{scale, false}}));
    EXPECT_FALSE(m.covers({{scale, true}}));
    EXPECT_FALSE(m.covers({}));
    EXPECT_FALSE(compiler().covers({{scale, false}}));
}

// --- the compiler model reproduces CDNA5ReadyQueue's clock ------------------------------

// Run the scheduler with the pick observer installed, then replay its picks in order. Every
// pick must land on the clock the queue had; where the queue skipped ahead by policy, the
// skip is the lower bound.
TEST_F(IssueTimelineTest, SchedulerPresetMatchesCdna5Clock) {
    const std::string ir = wmma("v[0:7]") + wmma("v[8:15]") + wmma("v[16:23]") +
                           wmma("v[24:31]") + R"(
        v[420:423] = "st.ds_load_b128"(v38) { mod.ds = { na = 1, offset = 128, gds = false }, mod.memtoken = { tokens = [0] } }
        v[424:427] = "st.ds_load_b128"(v38) { mod.ds = { na = 1, offset = 160, gds = false }, mod.memtoken = { tokens = [0] } }
        v[428:431] = "st.ds_load_b128"(v39) { mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [0] } }
        s21, SCC0 = "st.s_add_i32"(s21, s6)
        SCC0 = "st.s_cmp_lt_u32"(s69, 136704)
        s6 = "st.s_mov_b32"(-136704)
        s7 = "st.s_cselect_b32"(136704, s6, SCC0)
        s8, SCC0 = "st.s_add_u32"(s8, s7)
        v440 = "st.v_add_nc_u32"(v441, v442)
        v443 = "st.v_xor_b32"(v440, v444)
        v445 = "st.v_add_nc_u32"(v446, v447)
        v448 = "st.v_add_nc_u32"(v445, v449)
    )" + wmma("v[32:39]") + wmma("v[40:47]");
    Function* func = converter.convertToFunction(ir);
    ASSERT_NE(func, nullptr);

    std::vector<SchedulerPickEvent> events;
    setSchedulerPickObserver([&](const SchedulerPickEvent& e) { events.push_back(e); });
    PassContext ctx;
    GemmTileConfig config;
    config.arch = kArch;
    ctx.setGemmTileConfig(config);
    func->setGemmTileConfig(config);
    AnalysisManager am;
    registerAllAnalyses(am);
    createStinkyBuildImplicitDependencyPass()->run(*func, ctx, am);
    createStinkyDAGSchedulerPass()->run(*func, ctx, am);
    setSchedulerPickObserver({});
    ASSERT_EQ(events.size(), 18u);

    const TimingProfile sched = schedulerProfile(hw);
    std::vector<TimedInst> picks;
    for (const SchedulerPickEvent& e : events) picks.push_back(makeTimedInst(*e.inst, hw));
    IssueTimeline tl(sched);
    int skips = 0;
    for (size_t i = 0; i < events.size(); ++i) {
        ASSERT_FALSE(events[i].regionStart && i != 0);
        const int notBefore = events[i].afterSkip ? events[i].issueClock : 0;
        skips += events[i].afterSkip ? 1 : 0;
        const Placement pl = tl.place(picks[i], notBefore);
        EXPECT_EQ(pl.cycle, events[i].issueClock)
            << "pick " << i << " " << events[i].inst->getHwInstDesc()->mnemonic;
        EXPECT_EQ(tl.now(), events[i].clockAfter)
            << "pick " << i << " " << events[i].inst->getHwInstDesc()->mnemonic;
    }
    RecordProperty("policySkips", skips);
}

}  // namespace
