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
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/LegalizationUtils.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyDAGSchedulerPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

class BarrierTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    GemmTileConfig config;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    AnalysisManager am;

    void SetUp() override {
        config.arch[0] = 12;
        config.arch[1] = 5;
        config.arch[2] = 0;
        func = std::make_unique<Function>("barrier_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
        registerAllAnalyses(am);
    }

    void TearDown() override {
        func.reset();
        bb = nullptr;
    }

    /// Run StinkyBuildImplicitDependencyPass then StinkyDAGSchedulerPass on func.
    /// \p unrollGemm  set true to activate WMMA-aware scheduling (CDNA5 onInit/onInitRegion).
    void runPasses(bool unrollGemm = false) {
        PassContext ctx;
        ctx.setGemmTileConfig(config);

        if (unrollGemm) {
            PassFeatureConfig featureConfig;
            featureConfig.loopConfig.unrollGemm = true;
            ctx.setPassFeatureConfig(featureConfig);
        }

        auto implicitDepPass = createStinkyBuildImplicitDependencyPass();
        implicitDepPass->run(*func, ctx, am);

        auto dagSchedPass = createStinkyDAGSchedulerPass();
        dagSchedPass->run(*func, ctx, am);
    }

    /// Create an s_barrier instruction in bb and return it.
    StinkyInstruction* createSBarrier() {
        AsmIRBuilder builder(*bb, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::s_barrier, arch);
        if (!desc) return nullptr;
        return builder.create(desc);
    }

    /// Create an s_barrier_signal instruction with the given literal operand.
    /// Optionally attach a MemTokenData modifier with \p memTokens.
    StinkyInstruction* createSBarrierSignal(int literal, std::vector<int> memTokens = {}) {
        AsmIRBuilder builder(*bb, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::s_barrier_signal, arch);
        if (!desc) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addSrcReg(StinkyRegister(literal));
        if (!memTokens.empty()) inst->addModifier<MemTokenData>(MemTokenData{memTokens});
        return inst;
    }

    /// Create an s_barrier_wait instruction with the given literal operand.
    /// Optionally attach a MemTokenData modifier with \p memTokens.
    StinkyInstruction* createSBarrierWait(int literal, std::vector<int> memTokens = {}) {
        AsmIRBuilder builder(*bb, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::s_barrier_wait, arch);
        if (!desc) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addSrcReg(StinkyRegister(literal));
        if (!memTokens.empty()) inst->addModifier<MemTokenData>(MemTokenData{memTokens});
        return inst;
    }

    /// Create v_wmma_f32_16x16x32_bf16: dest/acc a[destStart:+7], src0/src1 v[src0Start:+7].
    /// latency is automatically copied from HwInstDesc by builder.create().
    /// Returns nullptr if the opcode is unavailable on \p arch.
    StinkyInstruction* createWMMA(int destStart, int src0Start, int src1Start) {
        AsmIRBuilder builder(*bb, arch);
        const HwInstDesc* desc = getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, arch);
        if (!desc) return nullptr;
        StinkyInstruction* inst = builder.create(desc);
        inst->addDestReg(StinkyRegister("a", destStart, 8));
        inst->addSrcReg(StinkyRegister("v", src0Start, 8));
        inst->addSrcReg(StinkyRegister("v", src1Start, 8));
        inst->addSrcReg(StinkyRegister("a", destStart, 8));  // acc
        return inst;
    }

    /// Count StinkyTofu instructions whose mnemonic matches \p mnem.
    int countByMnemonic(const std::string& mnem) const {
        int count = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            const auto* inst = cast<StinkyInstruction>(&ir);
            const char* m = inst->getHwInstDesc() ? inst->getHwInstDesc()->mnemonic : nullptr;
            if (m && std::string(m) == mnem) count++;
        }
        return count;
    }
};

// s_barrier_signal and s_barrier_wait are recognised as barrier instructions.
// legalizeBarrier() on gfx1250: s_barrier → s_barrier_signal -1 + s_barrier_wait -1.
TEST_F(BarrierTest, LegalizeBarrier_Gfx1250_ExpandsToSignalAndWait) {
    StinkyInstruction* barrier = createSBarrier();
    ASSERT_NE(barrier, nullptr);

    // Before legalization: exactly one s_barrier in the block.
    EXPECT_EQ(countByMnemonic("s_barrier"), 1);
    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 0);
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 0);

    // Run legalization.
    AsmIRBuilder builder(*bb, arch);
    Legalized result = legalizeBarrier(barrier, builder, arch);

    // legalizeBarrier removes the original s_barrier and inserts signal + wait.
    ASSERT_NE(result.first, nullptr);
    ASSERT_NE(result.last, nullptr);

    EXPECT_EQ(countByMnemonic("s_barrier"), 0)
        << "s_barrier should have been removed by legalizeBarrier";
    EXPECT_EQ(countByMnemonic("s_barrier_signal"), 1)
        << "Expected exactly one s_barrier_signal after legalization";
    EXPECT_EQ(countByMnemonic("s_barrier_wait"), 1)
        << "Expected exactly one s_barrier_wait after legalization";

    // Both produced instructions should use the all-wave operand (-1).
    EXPECT_TRUE(isBarrierSignal(*result.first))
        << "First legalized instruction should be s_barrier_signal";
    EXPECT_TRUE(isSplitBarrierAllWave(*result.first))
        << "s_barrier_signal should carry -1 (all-wave) operand";

    EXPECT_TRUE(isBarrierWait(*result.last))
        << "Second legalized instruction should be s_barrier_wait";
    EXPECT_TRUE(isSplitBarrierAllWave(*result.last))
        << "s_barrier_wait should carry -1 (all-wave) operand";
}

// Three independent memory-token groups are emitted into one block.
// After BuildImplicitDependencyPass + DAGSchedulerPass the relative ordering
// of instructions within every token group must be preserved.
//
// Group 0 (token 0): ds_load → s_barrier_signal → s_barrier_wait → tensor_load
// Group 1 (token 1): tensor_load → s_barrier_signal → s_barrier_wait → ds_load
// Group 2 (token 2): tensor_load → ds_load → s_barrier_signal → s_barrier_wait
//                    → ds_load → tensor_load
TEST_F(BarrierTest, TokenGrouped_PassesPreserveOrderWithinGroup) {
    // --- Group 0 (token 0) Group 1 (token 1) Group 2 (token 2) ---
    StinkyInstruction* g0_dsLoad = createDSLoadInBlock(bb, arch, /*dest=*/0, /*addr=*/16, {0});
    StinkyInstruction* g2_tensorLoad0 =
        createTensorLoadInBlock(bb, arch, /*s0=*/24, /*s1=*/28, {2});
    StinkyInstruction* g2_dsLoad0 = createDSLoadInBlock(bb, arch, /*dest=*/8, /*addr=*/24, {2});
    StinkyInstruction* g1_tensorLoad = createTensorLoadInBlock(bb, arch, /*s0=*/12, /*s1=*/16, {1});
    StinkyInstruction* wmma_g1 = createWMMA(/*dest=*/32, /*src0=*/4, 4);
    StinkyInstruction* wmma_g2 = createWMMA(/*dest=*/32, /*src0=*/12, 12);

    StinkyInstruction* g2_signal = createSBarrierSignal(-1, {2});
    StinkyInstruction* g2_wait = createSBarrierWait(-1, {2});
    StinkyInstruction* g0_signal = createSBarrierSignal(-1, {0});
    StinkyInstruction* g0_wait = createSBarrierWait(-1, {0});
    StinkyInstruction* g1_signal = createSBarrierSignal(-1, {1});
    StinkyInstruction* g1_wait = createSBarrierWait(-1, {1});

    (void)createWMMA(/*dest=*/32, /*src0=*/0, 8);
    (void)createWMMA(/*dest=*/32, /*src0=*/8, 0);
    (void)createWMMA(/*dest=*/32, /*src0=*/0, 0);

    StinkyInstruction* g2_dsLoad1 = createDSLoadInBlock(bb, arch, /*dest=*/12, /*addr=*/28, {2});
    StinkyInstruction* g2_tensorLoad1 =
        createTensorLoadInBlock(bb, arch, /*s0=*/36, /*s1=*/40, {2});
    StinkyInstruction* g0_tensorLoad = createTensorLoadInBlock(bb, arch, /*s0=*/0, /*s1=*/4, {0});
    StinkyInstruction* g1_dsLoad = createDSLoadInBlock(bb, arch, /*dest=*/4, /*addr=*/20, {1});

    // WMMA instructions that consume post-barrier ds_load results.
    // wmma_g1: src0/src1 = v[4:11]  — overlaps g1_dsLoad dest v[4:7],  acc/dest = v[32:39]
    // wmma_g2: src0/src1 = v[12:19] — overlaps g2_dsLoad1 dest v[12:15], acc/dest = v[40:47]
    ASSERT_NE(wmma_g1, nullptr) << "v_wmma_f32_16x16x16_bf16 not available for Gfx1250";
    ASSERT_NE(wmma_g2, nullptr) << "v_wmma_f32_16x16x16_bf16 not available for Gfx1250";

    // Verify instruction count before passes (4 + 4 + 6 + 2 wmma = 16).
    int countBefore = 0;
    for (const IRBase& ir : *bb)
        if (ir.getType() == IRBase::IRType::StinkyTofu) countBefore++;
    ASSERT_EQ(countBefore, 19) << "Expected 16 instructions before passes";

    runPasses(/*unrollGemm=*/true);

    // Instruction count must be unchanged.
    int countAfter = 0;
    for (const IRBase& ir : *bb)
        if (ir.getType() == IRBase::IRType::StinkyTofu) countAfter++;
    EXPECT_EQ(countAfter, countBefore) << "Pass pipeline must not add or remove instructions";

    // Return the 0-based position of \p target in bb, or -1 if absent.
    auto getPos = [&](const StinkyInstruction* target) -> int {
        int pos = 0;
        for (const IRBase& ir : *bb) {
            if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
            if (cast<StinkyInstruction>(&ir) == target) return pos;
            ++pos;
        }
        return -1;
    };

    // All instructions must still be present.
    ASSERT_GE(getPos(g0_dsLoad), 0) << "g0_dsLoad missing after passes";
    ASSERT_GE(getPos(g0_signal), 0) << "g0_signal missing after passes";
    ASSERT_GE(getPos(g0_wait), 0) << "g0_wait missing after passes";
    ASSERT_GE(getPos(g0_tensorLoad), 0) << "g0_tensorLoad missing after passes";
    ASSERT_GE(getPos(g1_tensorLoad), 0) << "g1_tensorLoad missing after passes";
    ASSERT_GE(getPos(g1_signal), 0) << "g1_signal missing after passes";
    ASSERT_GE(getPos(g1_wait), 0) << "g1_wait missing after passes";
    ASSERT_GE(getPos(g1_dsLoad), 0) << "g1_dsLoad missing after passes";
    ASSERT_GE(getPos(g2_tensorLoad0), 0) << "g2_tensorLoad0 missing after passes";
    ASSERT_GE(getPos(g2_dsLoad0), 0) << "g2_dsLoad0 missing after passes";
    ASSERT_GE(getPos(g2_signal), 0) << "g2_signal missing after passes";
    ASSERT_GE(getPos(g2_wait), 0) << "g2_wait missing after passes";
    ASSERT_GE(getPos(g2_dsLoad1), 0) << "g2_dsLoad1 missing after passes";
    ASSERT_GE(getPos(g2_tensorLoad1), 0) << "g2_tensorLoad1 missing after passes";
    ASSERT_GE(getPos(wmma_g1), 0) << "wmma_g1 missing after passes";
    ASSERT_GE(getPos(wmma_g2), 0) << "wmma_g2 missing after passes";

    // Group 0: ds_load → signal → wait → tensor_load
    EXPECT_LT(getPos(g0_dsLoad), getPos(g0_signal)) << "g0: ds_load must precede s_barrier_signal";
    EXPECT_LT(getPos(g0_signal), getPos(g0_wait))
        << "g0: s_barrier_signal must precede s_barrier_wait";
    EXPECT_LT(getPos(g0_wait), getPos(g0_tensorLoad))
        << "g0: s_barrier_wait must precede tensor_load";

    // Group 1: tensor_load → signal → wait → ds_load
    EXPECT_LT(getPos(g1_tensorLoad), getPos(g1_signal))
        << "g1: tensor_load must precede s_barrier_signal";
    EXPECT_LT(getPos(g1_signal), getPos(g1_wait))
        << "g1: s_barrier_signal must precede s_barrier_wait";
    EXPECT_LT(getPos(g1_wait), getPos(g1_dsLoad)) << "g1: s_barrier_wait must precede ds_load";

    // Group 2: {tensor_load0, ds_load0} → signal → wait → {ds_load1, tensor_load1}
    // The scheduler may reorder instructions on the same side of the barrier, so
    // we only enforce that every pre-barrier instruction precedes signal and every
    // post-barrier instruction follows wait.
    EXPECT_LT(getPos(g2_tensorLoad0), getPos(g2_signal))
        << "g2: tensor_load0 must precede s_barrier_signal";
    EXPECT_LT(getPos(g2_dsLoad0), getPos(g2_signal))
        << "g2: ds_load0 must precede s_barrier_signal";
    EXPECT_LT(getPos(g2_signal), getPos(g2_wait))
        << "g2: s_barrier_signal must precede s_barrier_wait";
    EXPECT_LT(getPos(g2_wait), getPos(g2_dsLoad1)) << "g2: s_barrier_wait must precede ds_load1";
    EXPECT_LT(getPos(g2_wait), getPos(g2_tensorLoad1))
        << "g2: s_barrier_wait must precede tensor_load1";
}
