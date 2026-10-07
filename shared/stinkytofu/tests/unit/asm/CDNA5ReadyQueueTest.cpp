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

#include <algorithm>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

#define DEBUG_TYPE "CDNA5ReadyQueueTest"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "transforms/asm/dag/CDNA5.hpp"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

PassContext makeClusterBarrierCtx(bool clusterBarrier) {
    PassContext ctx;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    ctx.setGemmTileConfig(config);
    PassFeatureConfig pfc;
    pfc.dagFeatures.clusterBarrier = clusterBarrier;
    ctx.setPassFeatureConfig(pfc);
    return ctx;
}

StinkyInstruction* makeSCmpDef(BasicBlock& bb) {
    AsmIRBuilder builder(bb, GfxArchID::Gfx1250);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_cmp_eq_u32, GfxArchID::Gfx1250));
    inst->addSrcReg(StinkyRegister("s", 90, 1));
    inst->addSrcReg(StinkyRegister(0));
    inst->addDestReg(StinkyRegister::getSCCRegister());
    return inst;
}

StinkyInstruction* makeWorkgroupBarrierSignal(BasicBlock& bb, int ldsToken) {
    AsmIRBuilder builder(bb, GfxArchID::Gfx1250);
    StinkyInstruction* inst =
        builder.create(getMCIDByUOp(GFX::s_barrier_signal, GfxArchID::Gfx1250));
    inst->addSrcReg(StinkyRegister(-1));
    inst->addSrcReg(StinkyRegister(RegType::LDS, ldsToken, 1));
    inst->addDestReg(StinkyRegister(RegType::LDS, ldsToken, 1));
    return inst;
}

StinkyInstruction* makeWorkgroupBarrierWait(BasicBlock& bb, int ldsToken) {
    AsmIRBuilder builder(bb, GfxArchID::Gfx1250);
    StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_barrier_wait, GfxArchID::Gfx1250));
    inst->addSrcReg(StinkyRegister(-1));
    inst->addSrcReg(StinkyRegister(RegType::LDS, ldsToken, 1));
    inst->addDestReg(StinkyRegister(RegType::LDS, ldsToken, 1));
    return inst;
}

// Synthetic stuck state: SCC def already issued (chain open) but its reader
// never reached the ready queue, while handshake barriers are ready. That
// should be unreachable when applyClusterBarrierSccRule + pickOne invariants
// hold.
void pickWithOpenChainAndOnlyBarriersReady(CDNA5ReadyQueue& queue, BasicBlock& bb) {
    StinkyInstruction* sccDef = makeSCmpDef(bb);
    StinkyInstruction* barrierSignal = makeWorkgroupBarrierSignal(bb, /*ldsToken=*/1);
    StinkyInstruction* barrierWait = makeWorkgroupBarrierWait(bb, /*ldsToken=*/1);

    DAGNode defNode(sccDef, /*id=*/0);
    defNode.sccChainId = 1;
    defNode.sccChainDef = true;
    defNode.sccChainReaders = 1;

    DAGNode signalNode(barrierSignal, /*id=*/1);
    signalNode.handshakeBarrier = true;
    DAGNode waitNode(barrierWait, /*id=*/2);
    waitNode.handshakeBarrier = true;

    queue.push(&defNode);
    ASSERT_NE(queue.pickOne(), nullptr) << "SCC def should issue and open the chain";

    queue.push(&signalNode);
    queue.push(&waitNode);
    (void)queue.pickOne();
}

class CDNA5ReadyQueueTest : public ::testing::Test {
   protected:
    Function func{"cdna5_ready_queue_test"};
    BasicBlock* bb = func.createBasicBlock("entry");

    CDNA5ReadyQueueTest() {
        setFunctionArch(func, GfxArchID::Gfx1250);
    }
};

// Reaches the queue's private timeline state for the WMMA-queue tests below.
struct CDNA5ReadyQueueTestPeer {
    static int& pos(CDNA5ReadyQueue& q) {
        return q.coIssueCyclePos_;
    }
    static int& clock(CDNA5ReadyQueue& q) {
        return q.clock_;
    }
    static int& latency(CDNA5ReadyQueue& q) {
        return q.activeWmmaLatency_;
    }
    static std::vector<uint8_t>& slots(CDNA5ReadyQueue& q) {
        return q.activeWindowSlots_;
    }
    static std::vector<int>& ends(CDNA5ReadyQueue& q) {
        return q.queuedEnds_;
    }
    static std::vector<std::vector<StinkyRegister>>& srcs(CDNA5ReadyQueue& q) {
        return q.queuedSrcs_;
    }
    using TimeKind = CDNA5ReadyQueue::TimeKind;
    static void advance(CDNA5ReadyQueue& q, int cycles, TimeKind kind) {
        q.advanceTime(cycles, kind);
    }
    static void setDsCap(CDNA5ReadyQueue& q, DsIssueCap::Mode mode, int depth) {
        q.dsIssueCap_ = DsIssueCap(mode, depth);
    }
    static void carry(CDNA5ReadyQueue& q) {
        q.carryQueuedWar();
    }
    static void resetWindow(CDNA5ReadyQueue& q) {
        q.resetActiveWindow();
    }
    static bool overlaps(CDNA5ReadyQueue& q, DAGNode* n) {
        return q.destOverlapsActiveWmmaSrc(n);
    }
    static constexpr uint8_t kBlocked = CDNA5ReadyQueue::kBlockedSlot;
};

PassContext makeQueueCtx(int depth, int cover) {
    PassContext ctx;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    ctx.setGemmTileConfig(config);
    PassFeatureConfig pfc;
    pfc.dagFeatures.wmmaQueueDepth = depth;
    pfc.dagFeatures.wmmaQueueCoverCycles = cover;
    ctx.setPassFeatureConfig(pfc);
    return ctx;
}

DsLoadDrainEntry entryFromOpcode(const HWModel& hw, GFX opcode, int latency) {
    const HwInstDesc* desc = getMCIDByUOp(opcode, GfxArchID::Gfx1250);
    return makeDsLoadDrainEntry(hw, latency, desc ? desc->dsThroughput : 0,
                                desc ? desc->dsMaxDrain : 0);
}

}  // namespace

TEST_F(CDNA5ReadyQueueTest, OpenSccChainWithOnlyBarriersReadyAborts) {
    EXPECT_DEATH(
        {
            PassContext ctx = makeClusterBarrierCtx(/*clusterBarrier=*/true);
            CDNA5ReadyQueue queue(ctx);
            pickWithOpenChainAndOnlyBarriersReady(queue, *bb);
        },
        "open SCC chain but only barriers are ready");
}

TEST_F(CDNA5ReadyQueueTest, DsLoadDescCarriesDrainParams) {
    const auto expectDrain = [](GFX opcode, int maxDrain, int throughput) {
        const HwInstDesc* desc = getMCIDByUOp(opcode, GfxArchID::Gfx1250);
        ASSERT_NE(desc, nullptr) << opcode;
        EXPECT_EQ(desc->dsMaxDrain, maxDrain) << desc->mnemonic;
        EXPECT_EQ(desc->dsThroughput, throughput) << desc->mnemonic;
    };

    // <= b32 → (120, 4)
    expectDrain(GFX::ds_load_u8, 120, 4);
    expectDrain(GFX::ds_load_i8, 120, 4);
    expectDrain(GFX::ds_load_u16, 120, 4);
    expectDrain(GFX::ds_load_i16, 120, 4);
    expectDrain(GFX::ds_load_b32, 120, 4);
    expectDrain(GFX::ds_load_2addr_b32, 120, 4);

    // b64 family → (131, 4); tr8 keeps its measured 135 cap
    expectDrain(GFX::ds_load_b64, 131, 4);
    expectDrain(GFX::ds_load_tr4_b64, 131, 4);
    expectDrain(GFX::ds_load_tr8_b64, 135, 4);

    // >= b96 / half-rate / 2addr_b64 (128-bit dest) → (255, 2)
    expectDrain(GFX::ds_load_2addr_b64, 255, 2);
    expectDrain(GFX::ds_load_b96, 255, 2);
    expectDrain(GFX::ds_load_tr6_b96, 255, 2);
    expectDrain(GFX::ds_load_b128, 255, 2);
    expectDrain(GFX::ds_load_b192, 255, 2);
    expectDrain(GFX::ds_load_tr16_b128, 255, 2);

    const auto* valu = getMCIDByUOp(GFX::v_add_f32, GfxArchID::Gfx1250);
    ASSERT_NE(valu, nullptr);
    EXPECT_EQ(valu->dsMaxDrain, 0);
    EXPECT_EQ(valu->dsThroughput, 0);
}

TEST_F(CDNA5ReadyQueueTest, DynamicDrainUsesTypeSpecificThroughput) {
    const HWModel& hw = hwModelForArch({12, 5, 0});
    constexpr int kFirstOverflowLoad = 17;
    constexpr int kLoadLatency = 56;
    constexpr int kNumWaves = 4;

    // Base = 56 + 15*4 = 116. Overflow term = 1*4 / throughput.
    // B128 / Tr16B128 throughput 2 => +2 = 118; default throughput 4 => +1 = 117.
    EXPECT_EQ(computeDynamicDrainLatency(hw, kFirstOverflowLoad, kLoadLatency, /*thr=*/2,
                                         /*maxDrain=*/255, kNumWaves),
              118);
    EXPECT_EQ(computeDynamicDrainLatency(hw, kFirstOverflowLoad, kLoadLatency, /*thr=*/4,
                                         /*maxDrain=*/131, kNumWaves),
              117);
}

TEST_F(CDNA5ReadyQueueTest, DynamicDrainUsesExperimentalTypeSpecificMaximum) {
    const HWModel& hw = hwModelForArch({12, 5, 0});
    constexpr int kLoadsBeyondMaximum = 100;
    constexpr int kLoadLatency = 56;
    constexpr int kNumWaves = 4;

    const auto drain = [&](int thr, int maxDrain) {
        return computeDynamicDrainLatency(hw, kLoadsBeyondMaximum, kLoadLatency, thr, maxDrain,
                                          kNumWaves);
    };

    EXPECT_EQ(drain(4, 120), 120);
    EXPECT_EQ(drain(4, 131), 131);
    EXPECT_EQ(drain(2, 255), 255);
    EXPECT_EQ(drain(4, 135), 135);
    EXPECT_EQ(drain(2, 255), 255);
    // Fallback defaults via makeDsLoadDrainEntry.
    const DsLoadDrainEntry fallback = makeDsLoadDrainEntry(hw, kLoadLatency, 0, 0);
    EXPECT_EQ(fallback.throughput, 4);
    EXPECT_EQ(fallback.maxDrain, 120);
    EXPECT_EQ(computeDynamicDrainLatency(hw, kLoadsBeyondMaximum, fallback.latency,
                                         fallback.throughput, fallback.maxDrain, kNumWaves),
              120);
}

TEST_F(CDNA5ReadyQueueTest, MixedDrainHomogeneousMatchesSingleTypeFormula) {
    const HWModel& hw = hwModelForArch({12, 5, 0});
    constexpr int kLoadLatency = 56;
    constexpr int kNumWaves = 4;
    constexpr int kCount = 17;

    std::vector<DsLoadDrainEntry> b64Loads(kCount,
                                           entryFromOpcode(hw, GFX::ds_load_b64, kLoadLatency));
    std::vector<DsLoadDrainEntry> b128Loads(kCount,
                                            entryFromOpcode(hw, GFX::ds_load_b128, kLoadLatency));

    EXPECT_EQ(computeDynamicDrainLatencyForLoads(hw, b64Loads, kNumWaves),
              computeDynamicDrainLatency(hw, kCount, kLoadLatency, 4, 131, kNumWaves));
    EXPECT_EQ(computeDynamicDrainLatencyForLoads(hw, b128Loads, kNumWaves),
              computeDynamicDrainLatency(hw, kCount, kLoadLatency, 2, 255, kNumWaves));
}

TEST_F(CDNA5ReadyQueueTest, MixedDrainUsesLastLatencyAndCountWeightedRate) {
    const HWModel& hw = hwModelForArch({12, 5, 0});
    constexpr int kB128Latency = 56;
    constexpr int kB64Latency = 40;
    constexpr int kNumWaves = 4;

    // 17 B128 + 1 B64. Rate = (1*4 + 17*2) / 18 = 2.
    // L=40 (last B64); cap = max(255,131) = 255 => 40 + 15*4 + (18-16)*4/2 = 104.
    std::vector<DsLoadDrainEntry> endsWithB64(17,
                                              entryFromOpcode(hw, GFX::ds_load_b128, kB128Latency));
    endsWithB64.push_back(entryFromOpcode(hw, GFX::ds_load_b64, kB64Latency));
    EXPECT_EQ(computeDynamicDrainLatencyForLoads(hw, endsWithB64, kNumWaves), 104);

    // 9 B128 + 9 B64, last B64. Rate = (9*4 + 9*2) / 18 = 3 => 40+60+(2*4)/3 = 102.
    std::vector<DsLoadDrainEntry> grouped(9, entryFromOpcode(hw, GFX::ds_load_b128, kB128Latency));
    grouped.insert(grouped.end(), 9, entryFromOpcode(hw, GFX::ds_load_b64, kB64Latency));
    std::vector<DsLoadDrainEntry> interleaved;
    for (int i = 0; i < 9; ++i) {
        interleaved.push_back(entryFromOpcode(hw, GFX::ds_load_b128, kB128Latency));
        interleaved.push_back(entryFromOpcode(hw, GFX::ds_load_b64, kB64Latency));
    }
    EXPECT_EQ(computeDynamicDrainLatencyForLoads(hw, grouped, kNumWaves), 102);
    EXPECT_EQ(computeDynamicDrainLatencyForLoads(hw, interleaved, kNumWaves), 102);

    // Same mix but last is B128: L=56 => 56+60+4 = 120.
    std::vector<DsLoadDrainEntry> endsWithB128(1,
                                               entryFromOpcode(hw, GFX::ds_load_b64, kB64Latency));
    endsWithB128.insert(endsWithB128.end(), 17,
                        entryFromOpcode(hw, GFX::ds_load_b128, kB128Latency));
    EXPECT_EQ(computeDynamicDrainLatencyForLoads(hw, endsWithB128, kNumWaves), 120);
}

TEST_F(CDNA5ReadyQueueTest, MixedDrainCapUsesMaxDrainInBurst) {
    const HWModel& hw = hwModelForArch({12, 5, 0});
    constexpr int kB128Latency = 56;
    constexpr int kB64Latency = 40;
    constexpr int kNumWaves = 4;

    // 100 B128 + 1 B64. Rate = (1*4 + 100*2) / 101 = 2.
    // Raw = 40 + 15*4 + (101-16)*4/2 = 270. Cap = max(255,131) = 255.
    std::vector<DsLoadDrainEntry> loads(100, entryFromOpcode(hw, GFX::ds_load_b128, kB128Latency));
    loads.push_back(entryFromOpcode(hw, GFX::ds_load_b64, kB64Latency));
    EXPECT_EQ(computeDynamicDrainLatencyForLoads(hw, loads, kNumWaves), 255);
}

// Queued windows are concatenated, so a blocked (LD_SCALE) cycle can lie inside an advance. At
// position 14 with slot 15 blocked, advanceTime() reads its argument by kind:
//  - Issue: two cycles of issue work cross the blocked one, so 3 elapse (queue model on; off keeps
//    the original landing-only rule);
//  - ValuIssue: the same for a VALU (it skips the blocked slot in both modes);
//  - Elapsed: already wall time, so the blocked cycle is not charged a second time.
TEST_F(CDNA5ReadyQueueTest, AdvanceTimeReadsItsArgumentByKind) {
    using Peer = CDNA5ReadyQueueTestPeer;
    for (int on = 0; on < 2; ++on) {
        PassContext ctx = on ? makeQueueCtx(8, 16) : makeQueueCtx(1, 0);
        struct Case {
            Peer::TimeKind kind;
            int cycles;
            int wantOff, wantOn;
            const char* name;
        };
        const Case cases[] = {
            {Peer::TimeKind::Issue, 2, 16, 17, "issue"},
            {Peer::TimeKind::ValuIssue, 2, 17, 17, "valu issue"},
            {Peer::TimeKind::Elapsed, 3, 17, 17, "elapsed (14 -> 17, slot 15 included)"},
        };
        for (const Case& c : cases) {
            CDNA5ReadyQueue queue(ctx);
            Peer::slots(queue).assign(20, 1);
            Peer::slots(queue)[15] = Peer::kBlocked;
            Peer::latency(queue) = 20;
            Peer::pos(queue) = 14;
            Peer::advance(queue, c.cycles, c.kind);
            EXPECT_EQ(Peer::pos(queue), on ? c.wantOn : c.wantOff)
                << c.name << ", queue model " << on;
        }
    }
}

// A queued WMMA keeps reading its sources after a region cut. The WAR gate must still hold a
// ds_load into those registers until the pipe has read them, then release it.
TEST_F(CDNA5ReadyQueueTest, QueuedWmmaWarStateSurvivesARegionCut) {
    PassContext ctx = makeQueueCtx(8, 16);
    CDNA5ReadyQueue queue(ctx);
    // A WMMA reading v[0:7] ends 40 cycles into the window; the region ends at position 10.
    CDNA5ReadyQueueTestPeer::srcs(queue).push_back({StinkyRegister("v", 0, 8)});
    CDNA5ReadyQueueTestPeer::ends(queue).push_back(40);
    CDNA5ReadyQueueTestPeer::pos(queue) = 10;
    CDNA5ReadyQueueTestPeer::clock(queue) = 10;
    CDNA5ReadyQueueTestPeer::carry(queue);  // what onInitRegion does first
    CDNA5ReadyQueueTestPeer::resetWindow(queue);
    CDNA5ReadyQueueTestPeer::clock(queue) = 0;  // new region

    StinkyInstruction* dsLoad = createDsReadB128InBlock(bb, GfxArchID::Gfx1250, /*destReg=*/0, 80);
    StinkyInstruction* dsOther =
        createDsReadB128InBlock(bb, GfxArchID::Gfx1250, /*destReg=*/100, 80);
    DAGNode overwrite(dsLoad, /*id=*/0);
    DAGNode unrelated(dsOther, /*id=*/1);
    EXPECT_TRUE(CDNA5ReadyQueueTestPeer::overlaps(queue, &overwrite)) << "still being read";
    EXPECT_FALSE(CDNA5ReadyQueueTestPeer::overlaps(queue, &unrelated));
    CDNA5ReadyQueueTestPeer::clock(queue) = 30;  // 30 cycles remained when the region began
    EXPECT_FALSE(CDNA5ReadyQueueTestPeer::overlaps(queue, &overwrite)) << "read by now";

    // With the queue model off nothing is carried (the original per-region reset).
    PassContext off = makeQueueCtx(1, 0);
    CDNA5ReadyQueue plain(off);
    CDNA5ReadyQueueTestPeer::srcs(plain).push_back({StinkyRegister("v", 0, 8)});
    CDNA5ReadyQueueTestPeer::ends(plain).push_back(40);
    CDNA5ReadyQueueTestPeer::pos(plain) = 10;
    CDNA5ReadyQueueTestPeer::carry(plain);
    CDNA5ReadyQueueTestPeer::resetWindow(plain);
    EXPECT_FALSE(CDNA5ReadyQueueTestPeer::overlaps(plain, &overwrite));
}

// A WMMA forced out with nothing else to run still waits for its sources: with the queue model
// on that stall elapses on the timeline (a ds_load result is about 50 cycles away). With the
// model off the original schedule is untouched.
TEST_F(CDNA5ReadyQueueTest, ForcedWmmaChargesItsSourceStallWhenTheQueueModelIsOn) {
    int elapsed[2] = {0, 0};
    for (int on = 0; on < 2; ++on) {
        PassContext ctx = on ? makeQueueCtx(8, 16) : makeQueueCtx(1, 0);
        CDNA5ReadyQueue queue(ctx);
        Function f{"forced_wmma"};
        BasicBlock* b = f.createBasicBlock("entry");
        setFunctionArch(f, GfxArchID::Gfx1250);
        StinkyInstruction* ds = createDsReadB128InBlock(b, GfxArchID::Gfx1250, /*destReg=*/100, 80);
        ds->addSrcReg(StinkyRegister(RegType::LDS, 1, 1));
        AsmIRBuilder builder(*b, GfxArchID::Gfx1250);
        StinkyInstruction* wmma =
            builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x16_bf16, GfxArchID::Gfx1250));
        wmma->addDestReg(StinkyRegister("v", 200, 8));
        wmma->addSrcReg(StinkyRegister("v", 100, 8));  // reads the ds_load's v[100:103]
        wmma->addSrcReg(StinkyRegister("v", 100, 8));
        wmma->addSrcReg(StinkyRegister("v", 200, 8));
        DAGNode dsNode(ds, /*id=*/0);
        DAGNode wmmaNode(wmma, /*id=*/1);
        queue.push(&dsNode);
        ASSERT_EQ(queue.pickOne(), &dsNode);
        const int before = CDNA5ReadyQueueTestPeer::clock(queue);
        queue.push(&wmmaNode);
        ASSERT_EQ(queue.pickOne(), &wmmaNode);
        elapsed[on] = CDNA5ReadyQueueTestPeer::clock(queue) - before;
    }
    EXPECT_LT(elapsed[0], 20) << "queue model off: unchanged";
    EXPECT_GE(elapsed[1], 40) << "queue model on: the source stall elapses";
}

// Periodic cap: at most A ds_loads per X-cycle period. With only ds_loads to issue nothing else
// supplies elapsed time, so the cap wait itself must elapse; otherwise the period never ends and
// the ds_loads are all counted into the same full period, exceeding A.
TEST_F(CDNA5ReadyQueueTest, PeriodicDsCapNeverIssuesMoreThanItsLimitInOnePeriod) {
    constexpr int kCap = 4, kSpan = 32, kLoads = 10;
    PassContext ctx;
    GemmTileConfig config;
    config.arch = {12, 5, 0};
    ctx.setGemmTileConfig(config);
    PassFeatureConfig pfc;
    pfc.dagFeatures.dsReadPerCap = kCap;
    pfc.dagFeatures.dsIssueCapSpanCycles = kSpan;
    pfc.dagFeatures.dsIssueCapMode = PassFeatureConfig::DsIssueCapMode::Periodic;
    pfc.dagFeatures.dsReadQueueDepth = 16;
    pfc.dagFeatures.dsReadThrottleLatency = 1;  // the LDS throttle is out of the way
    ctx.setPassFeatureConfig(pfc);
    CDNA5ReadyQueue queue(ctx);
    CDNA5ReadyQueueTestPeer::setDsCap(queue, DsIssueCap::Mode::Periodic, kCap);

    std::vector<DAGNode> nodes;
    nodes.reserve(kLoads);
    for (int i = 0; i < kLoads; ++i) {
        StinkyInstruction* ds =
            createDsReadB128InBlock(bb, GfxArchID::Gfx1250, /*destReg=*/100 + 4 * i, 80);
        ds->addSrcReg(StinkyRegister(RegType::LDS, i + 1, 1));
        nodes.emplace_back(ds, /*id=*/i);
    }
    for (DAGNode& n : nodes) queue.push(&n);

    std::vector<int> issuedAt;  // timeline clock once each ds_load has issued (after any wait)
    for (int i = 0; i < kLoads; ++i) {
        ASSERT_NE(queue.pickOne(), nullptr);
        issuedAt.push_back(CDNA5ReadyQueueTestPeer::clock(queue));
    }
    std::string times;
    for (int t : issuedAt) times += std::to_string(t) + " ";
    // No window of kSpan cycles opened by a ds_load holds more than kCap of them.
    for (int i = 0; i < kLoads; ++i) {
        int inPeriod = 0;
        for (int j = i; j < kLoads; ++j) inPeriod += issuedAt[j] < issuedAt[i] + kSpan;
        EXPECT_LE(inPeriod, kCap) << "period opened by ds_load " << i << ", issue times: " << times;
    }
}
