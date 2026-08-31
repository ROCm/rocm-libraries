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

#include "TestHelpers.hpp"
#include "stinkytofu/core/PassManager.hpp"
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

// Synthetic stuck state: SCC def already issued (chain open) but its reader never
// reached the ready queue, while handshake barriers are ready. That should be
// unreachable when applyClusterBarrierSccRule + pickOne invariants hold.
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
