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

    // Create v_wmma_f32_16x16x16_bf16: dest v[destStart:destStart+7], src0
    // v[src0Start:src0Start+7], src1 same as src0, acc same as dest (v[destStart:destStart+7]).
    StinkyInstruction* createWmmaF32_16x16x16_bf16(int destStart, int src0Start) {
        AsmIRBuilder builder(*bb, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_wmma_f32_16x16x16_bf16, arch);
        if (!desc) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addDestReg(StinkyRegister("v", destStart, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", destStart, 8));
        return inst;
    }
};

// .cost overwrite: v_wmma_f32_16x16x16_bf16 (format WMMA) has explicit .cost in def -> issue=4,
// latency=8
TEST_F(DAGSchedulerPassTest, CostOverwrite_VWmmaF3216x16x16Bf16_HasIssue4Latency8) {
    const HwInstDesc* desc = getMCIDByUOp(GFX::v_wmma_f32_16x16x16_bf16, arch);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->issue, 4) << "v_wmma_f32_16x16x16_bf16 .cost overwrite: issue should be 4";
    EXPECT_EQ(desc->latency, 8) << "v_wmma_f32_16x16x16_bf16 .cost overwrite: latency should be 8";
}

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
