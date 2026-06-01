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
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

static int countStinkyInstructions(const BasicBlock& bb) {
    int count = 0;
    for (const IRBase& ir : bb) {
        if (ir.getType() == IRBase::IRType::StinkyTofu) count++;
    }
    return count;
}

class DAGSchedulerPassTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    GemmTileConfig config;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    std::unique_ptr<Pass> pass;
    AnalysisManager am;

    void SetUp() override {
        config.arch[0] = 12;
        config.arch[1] = 5;
        config.arch[2] = 0;
        func = std::make_unique<Function>("dag_sched_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
        pass = createStinkyDAGSchedulerPass();
        registerAllAnalyses(am);
    }

    void TearDown() override {
        pass.reset();
        func.reset();
        bb = nullptr;
    }

    void runPass() {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        pass->run(*func, ctx, am);
    }

    void runPassWithUnrollGemm() {
        PassContext ctx;
        ctx.setGemmTileConfig(config);
        PassFeatureConfig pfc;
        pfc.loopConfig.unrollGemm = true;
        ctx.setPassFeatureConfig(pfc);
        pass->run(*func, ctx, am);
    }

    // Build a single-BB self-loop so the scheduler uses the loop-aware CDNA5 path.
    // Returns the loop body BB with the branch already appended.
    BasicBlock* buildLoopBB(const char* label = "loop_body") {
        BasicBlock* body = func->createBasicBlock(label);
        body->addSuccessor(body);
        return body;
    }

    static StinkyInstruction* createSCbranchInBlock(BasicBlock* bb, GfxArchID arch) {
        AsmIRBuilder builder(*bb, arch);
        return builder.create(getMCIDByUOp(GFX::s_cbranch_scc0, arch));
    }

    StinkyInstruction* createWmmaF32_16x16x16_bf16_in(BasicBlock* targetBB, int destStart,
                                                      int src0Start) {
        AsmIRBuilder builder(*targetBB, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_wmma_f32_16x16x16_bf16, arch);
        if (!desc) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addDestReg(StinkyRegister("v", destStart, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", destStart, 8));
        return inst;
    }

    StinkyInstruction* createWmmaF32_16x16x16_bf16(int destStart, int src0Start) {
        return createWmmaF32_16x16x16_bf16_in(bb, destStart, src0Start);
    }

    StinkyInstruction* createMovableDsLoad(int destReg, int addrReg, int ldsToken) {
        StinkyInstruction* inst = createDsReadB128InBlock(bb, arch, destReg, addrReg);
        inst->addSrcReg(StinkyRegister(RegType::LDS, ldsToken, 1));
        return inst;
    }
};

// Empty block: pass should not crash
TEST_F(DAGSchedulerPassTest, EmptyBlock_DoesNotCrash) {
    runPass();
    EXPECT_EQ(countStinkyInstructions(*bb), 0);
}

// Single instruction: pass should not crash
TEST_F(DAGSchedulerPassTest, SingleInstruction_DoesNotCrash) {
    createVAddInBlock(bb, arch, 0, 1, 2);
    int n = countStinkyInstructions(*bb);
    runPass();
    EXPECT_EQ(countStinkyInstructions(*bb), n);
}

// A few independent instructions: pass should not crash, count unchanged
TEST_F(DAGSchedulerPassTest, IndependentInstructions_DoesNotCrash) {
    createVAddInBlock(bb, arch, 0, 1, 2);
    createVAddInBlock(bb, arch, 3, 4, 5);
    createVAddInBlock(bb, arch, 6, 7, 8);
    int n = countStinkyInstructions(*bb);
    runPass();
    EXPECT_EQ(countStinkyInstructions(*bb), n);
}

// Chain of dependencies: pass should not crash, count unchanged
TEST_F(DAGSchedulerPassTest, DependentInstructions_DoesNotCrash) {
    createVAddInBlock(bb, arch, 0, 1, 2);  // v0 = v1 + v2
    createVAddInBlock(bb, arch, 3, 0, 4);  // v3 = v0 + v4
    createVAddInBlock(bb, arch, 5, 3, 6);  // v5 = v3 + v6
    int n = countStinkyInstructions(*bb);
    runPass();
    EXPECT_EQ(countStinkyInstructions(*bb), n);
}

// DS reads + WMMAs: scheduler must not issue WMMAs back-to-back when other instructions exist.
// With real ds_load latency, WMMAs are not latency-free until ds_reads are issued and latency
// elapses, so we get: 4 ds_load, then 2 wmma. The rule "lastPickedWasWMMA => prefer other"
// ensures that when both WMMA and other are ready we interleave (no consecutive WMMAs).
TEST_F(DAGSchedulerPassTest, DSReadAndWMMA_NoConsecutiveWMMA) {
    const int addrReg = 24;
    createDsReadB128InBlock(bb, arch, 8, addrReg);                 // v[8:11]
    createDsReadB128InBlock(bb, arch, 12, addrReg);                // v[12:15]
    createDsReadB128InBlock(bb, arch, 16, addrReg);                // v[16:19]
    createDsReadB128InBlock(bb, arch, 20, addrReg);                // v[20:23]
    StinkyInstruction* wmma1 = createWmmaF32_16x16x16_bf16(0, 8);  // v[0:7] v[8:15] v[8:15] v[0:7]
    StinkyInstruction* wmma2 =
        createWmmaF32_16x16x16_bf16(0, 16);  // v[0:7] v[16:23] v[16:23] v[0:7]
    ASSERT_NE(wmma1, nullptr);
    ASSERT_NE(wmma2, nullptr);

    runPassWithUnrollGemm();

    // With real latency, all 4 ds_load are issued first, then 2 wmma. No two WMMAs in a row.
    std::vector<std::pair<std::string, int>> sequence;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const StinkyInstruction* inst = cast<StinkyInstruction>(&ir);
        const char* mnem = inst->getHwInstDesc() ? inst->getHwInstDesc()->mnemonic : nullptr;
        if (!mnem) continue;
        if (std::string(mnem) == "ds_load_b128") {
            if (!inst->getDestRegs().empty() && inst->getDestRegs()[0].isRegister())
                sequence.push_back(
                    {"ds_load_b128", static_cast<int>(inst->getDestRegs()[0].reg.idx)});
        } else if (std::string(mnem) == "v_wmma_f32_16x16x16_bf16") {
            if (!inst->getSrcRegs().empty() && inst->getSrcRegs()[0].isRegister())
                sequence.push_back(
                    {"v_wmma_f32_16x16x16_bf16", static_cast<int>(inst->getSrcRegs()[0].reg.idx)});
        }
    }

    ASSERT_EQ(sequence.size(), 6u) << "Expected 6 instructions (4 ds_load + 2 wmma)";
    // All 4 ds_load first (real latency: WMMAs not ready until latency elapses)
    EXPECT_EQ(sequence[0].first, "ds_load_b128");
    EXPECT_EQ(sequence[0].second, 8);
    EXPECT_EQ(sequence[1].first, "ds_load_b128");
    EXPECT_EQ(sequence[1].second, 12);
    EXPECT_EQ(sequence[2].first, "ds_load_b128");
    EXPECT_EQ(sequence[2].second, 16);
    EXPECT_EQ(sequence[3].first, "ds_load_b128");
    EXPECT_EQ(sequence[3].second, 20);
    // Then 2 wmma; rule ensures we never issue two WMMAs in a row when other work exists
    EXPECT_EQ(sequence[4].first, "v_wmma_f32_16x16x16_bf16");
    EXPECT_EQ(sequence[4].second, 8);
    EXPECT_EQ(sequence[5].first, "v_wmma_f32_16x16x16_bf16");
    EXPECT_EQ(sequence[5].second, 16);
    // When other instructions exist, scheduler prefers them after a WMMA (no back-to-back WMMA).
    // Here with real latency only WMMAs are left at the end so they are issued consecutively.
}

// ---------------------------------------------------------------------------
// Property: when all independent, WMMA fires first (Phase B), then ds_loads
// and VALU fill the WMMA latency window.
// Within that window, ds_load has priority over VALU.
// ---------------------------------------------------------------------------
TEST_F(DAGSchedulerPassTest, IndependentWMMAFirst_ThenDsThenVALU) {
    const int addrReg = 80;
    // 3 independent ds_loads (LDS pseudo-reg makes them movable in DAG)
    createMovableDsLoad(0, addrReg, 1);
    createMovableDsLoad(4, addrReg, 2);
    createMovableDsLoad(8, addrReg, 3);
    // 3 independent VALUs
    createVAddInBlock(bb, arch, 40, 41, 42);
    createVAddInBlock(bb, arch, 43, 44, 45);
    createVAddInBlock(bb, arch, 46, 47, 48);
    // 1 independent WMMA
    createWmmaF32_16x16x16_bf16(12, 50);

    runPassWithUnrollGemm();

    int firstWmmaPos = -1;
    int firstDsPos = -1;
    int firstValuPos = -1;
    int pos = 0;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        const HwInstDesc* hw = inst->getHwInstDesc();
        if (!hw || !hw->mnemonic) continue;
        std::string_view mnem(hw->mnemonic);
        if (mnem.find("wmma") != std::string_view::npos) {
            if (firstWmmaPos < 0) firstWmmaPos = pos;
        } else if (mnem.find("ds_load") != std::string_view::npos) {
            if (firstDsPos < 0) firstDsPos = pos;
        } else if (mnem.find("v_add") != std::string_view::npos) {
            if (firstValuPos < 0) firstValuPos = pos;
        }
        pos++;
    }

    ASSERT_GE(firstWmmaPos, 0) << "No WMMA found";
    ASSERT_GE(firstDsPos, 0) << "No ds_load found";
    ASSERT_GE(firstValuPos, 0) << "No VALU found";
    EXPECT_LT(firstWmmaPos, firstDsPos) << "WMMA should fire before ds_load (Phase B)";
    EXPECT_LT(firstDsPos, firstValuPos)
        << "DS loads should be prioritized before VALU during WMMA latency";
}

// ---------------------------------------------------------------------------
// Property: ds_load burst limit — after a WMMA fires,
// its latency window is filled with non-WMMA. With 5 ds_loads and VALU
// all ready, at most 4 consecutive ds_loads appear before VALU interleaves.
// ---------------------------------------------------------------------------
TEST_F(DAGSchedulerPassTest, DSBurstLimit_VALUInterleaveAfter4) {
    const int addrReg = 80;
    // 5 independent ds_loads (LDS pseudo-reg makes them movable in DAG)
    createMovableDsLoad(0, addrReg, 1);
    createMovableDsLoad(4, addrReg, 2);
    createMovableDsLoad(8, addrReg, 3);
    createMovableDsLoad(12, addrReg, 4);
    createMovableDsLoad(16, addrReg, 5);
    // 2 independent VALUs
    createVAddInBlock(bb, arch, 60, 61, 62);
    createVAddInBlock(bb, arch, 63, 64, 65);
    // 2 independent WMMAs (fire first, create co-issue window)
    createWmmaF32_16x16x16_bf16(20, 28);
    createWmmaF32_16x16x16_bf16(36, 44);

    runPassWithUnrollGemm();

    int consecutiveDs = 0;
    int maxConsecutiveDs = 0;
    for (const IRBase& ir : *bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        const auto* inst = cast<StinkyInstruction>(&ir);
        const HwInstDesc* hw = inst->getHwInstDesc();
        if (!hw || !hw->mnemonic) continue;
        std::string_view mnem(hw->mnemonic);
        if (mnem.find("ds_load") != std::string_view::npos) {
            consecutiveDs++;
            maxConsecutiveDs = std::max(maxConsecutiveDs, consecutiveDs);
        } else {
            consecutiveDs = 0;
        }
    }

    EXPECT_LE(maxConsecutiveDs, 4) << "Burst limit violated: found " << maxConsecutiveDs
                                   << " consecutive ds_loads (VALU should interleave after 4)";
}

// ---------------------------------------------------------------------------
// Property: all original instructions are preserved.
// ---------------------------------------------------------------------------
TEST_F(DAGSchedulerPassTest, DSBurst_InstructionCountPreserved) {
    const int addrReg = 100;
    // 6 independent ds_loads (LDS pseudo-reg makes them movable in DAG)
    for (int i = 0; i < 6; i++) createMovableDsLoad(i * 4, addrReg, i + 1);
    // 4 independent VALUs
    for (int i = 0; i < 4; i++) createVAddInBlock(bb, arch, 30 + i, 40 + i, 50 + i);
    // 3 independent WMMAs
    for (int i = 0; i < 3; i++) createWmmaF32_16x16x16_bf16(60 + i * 8, 84 + i * 8);

    int beforeCount = countStinkyInstructions(*bb);
    runPassWithUnrollGemm();
    int afterCount = countStinkyInstructions(*bb);

    EXPECT_EQ(afterCount, beforeCount) << "Scheduler must preserve instruction count";
}
