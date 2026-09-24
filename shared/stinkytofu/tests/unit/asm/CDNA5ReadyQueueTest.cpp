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
