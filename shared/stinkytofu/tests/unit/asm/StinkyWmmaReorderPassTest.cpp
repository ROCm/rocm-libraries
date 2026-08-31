// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/StinkyWmmaReorderPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Fixture
//
// A minimal software-pipelined loop: four wmma over four A tiles that share one
// B tile, each A tile fed by one ds_load. Two of the four loads arrive already
// prefetched — a preheader copy at offset O plus a body copy at O + kStride.
// ─────────────────────────────────────────────────────────────────────────────

class WmmaReorderPassTest : public ::testing::Test {
   protected:
    static constexpr GfxArchID kArch = GfxArchID::Gfx1250;
    static constexpr uint16_t kTile = 8;  // VGPRs per wmma operand tile
    static constexpr int kStride = 512;   // per-iteration LDS distance
    static constexpr int kNumWmma = 4;

    void SetUp() override {
        func = std::make_unique<Function>("test");
        setFunctionArch(*func, kArch);
        preheader = func->createBasicBlock("preheader");
        body = func->createBasicBlock("label_LoopBeginL");
        registerAllAnalyses(am);
    }

    StinkyInstruction* addDsLoad(BasicBlock* bb, int destBase, int offset) {
        StinkyInstruction* inst = createDsReadB128InBlock(bb, kArch, destBase, /*addrReg=*/10);
        inst->addModifier(DSModifiers{/*na=*/1, offset});
        return inst;
    }

    StinkyInstruction* addWmma(BasicBlock* bb, int aBase, int cBase) {
        AsmIRBuilder builder(*bb, kArch);
        auto* inst = builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, kArch));
        inst->addDestReg(vgpr(cBase, kTile));
        inst->addSrcReg(vgpr(aBase, kTile));
        inst->addSrcReg(vgpr(60, kTile));
        inst->addSrcReg(vgpr(cBase, kTile));
        return inst;
    }

    /// A tile k occupies v[20 + 8k .. 27 + 8k]; ds_load only writes 4 VGPRs, so
    /// each tile is covered by the load at its base. Tiles 0 and 1 are the ones
    /// the kernel already prefetches.
    void buildLoop() {
        addDsLoad(preheader, /*dest=*/20, /*offset=*/0);
        addDsLoad(preheader, /*dest=*/28, /*offset=*/32);

        for (int k = 0; k < kNumWmma; ++k) {
            const bool prefetched = k < 2;
            const int offset = 32 * k + (prefetched ? kStride : 0);
            loads.push_back(addDsLoad(body, /*dest=*/20 + 8 * k, offset));
        }
        for (int k = 0; k < kNumWmma; ++k)
            wmma.push_back(addWmma(body, /*aBase=*/20 + 8 * k, /*cBase=*/100 + 8 * k));
    }

    const WmmaReorderOutcome* runPass(std::unique_ptr<IWmmaOrderProvider> mode,
                                      WmmaReorderOptions options) {
        PassContext ctx;
        auto pass = createStinkyWmmaReorderPass(std::move(mode), std::move(options));
        pass->run(*func, ctx, am);
        return getWmmaReorderOutcome(*body);
    }

    static std::vector<const StinkyInstruction*> instructionsOf(const BasicBlock& bb) {
        std::vector<const StinkyInstruction*> out;
        for (const IRBase& node : bb)
            if (const auto* inst = dyn_cast<StinkyInstruction>(&node)) out.push_back(inst);
        return out;
    }

    static size_t dsLoadCount(const BasicBlock& bb) {
        size_t n = 0;
        for (const StinkyInstruction* inst : instructionsOf(bb))
            if (isDSRead(*inst)) ++n;
        return n;
    }

    static int dsOffset(const StinkyInstruction& inst) {
        const auto* ds = inst.getModifier<DSModifiers>();
        return ds ? ds->offset : 0;
    }

    std::unique_ptr<Function> func;
    BasicBlock* preheader = nullptr;
    BasicBlock* body = nullptr;
    std::vector<StinkyInstruction*> loads;
    std::vector<StinkyInstruction*> wmma;
    AnalysisManager am;
};

// ─────────────────────────────────────────────────────────────────────────────
// Mode ABI
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(WmmaReorderPassTest, ExplicitPermutation_ReordersWmma) {
    buildLoop();
    WmmaReorderOptions options;
    options.prefetchDistance = 2;

    const auto* res = runPass(
        std::make_unique<ExplicitOrderProvider>(std::vector<unsigned>{3, 2, 1, 0}), options);
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applied) << res->skipReason;

    std::vector<const StinkyInstruction*> seen;
    for (const StinkyInstruction* inst : instructionsOf(*body))
        if (isXDLWMMA(*inst)) seen.push_back(inst);
    EXPECT_EQ(seen, (std::vector<const StinkyInstruction*>{wmma[3], wmma[2], wmma[1], wmma[0]}));
}

TEST_F(WmmaReorderPassTest, NonPermutation_IsRejected) {
    buildLoop();
    // Index 0 twice: not a permutation of the body's wmma.
    const auto* res =
        runPass(std::make_unique<ExplicitOrderProvider>(std::vector<unsigned>{0, 0, 1, 2}), {});
    ASSERT_NE(res, nullptr);
    EXPECT_FALSE(res->applied);
    EXPECT_EQ(res->skipReason, "mode returned a non-permutation");
}

TEST_F(WmmaReorderPassTest, VgprAnalysisMode_UntaggedWmma_IsNoOp) {
    buildLoop();
    // The default mode runs the analysis itself; it declines this block because
    // the wmma carry no WmmaPoolData, so there are no pools to alias across.
    const auto* res = runPass(nullptr, {});
    ASSERT_NE(res, nullptr);
    EXPECT_FALSE(res->applied);
    EXPECT_EQ(res->skipReason, "mode supplied no order");
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-iteration migration
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(WmmaReorderPassTest, ReversedOrder_SwapsWhichLoadsArePrefetched) {
    buildLoop();
    WmmaReorderOptions options;
    options.prefetchDistance = 2;

    const auto* res = runPass(std::make_unique<ReverseOrderProvider>(), options);
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applied) << res->skipReason;

    // Tiles 0 and 1 lose their head start; tiles 2 and 3 gain one.
    EXPECT_EQ(res->prefetchRemoved, 2u);
    EXPECT_EQ(res->prefetchAdded, 2u);
    EXPECT_EQ(res->iterOffsetDelta, kStride);

    // The preheader keeps the same count but now holds the other two tiles.
    EXPECT_EQ(dsLoadCount(*preheader), 2u);
    for (const StinkyInstruction* inst : instructionsOf(*preheader)) {
        if (!isDSRead(*inst)) continue;
        const uint32_t base = inst->getDestReg(0).reg.idx;
        EXPECT_TRUE(base == 36 || base == 44) << "unexpected preheader prefetch v" << base;
    }

    // De-rotated back to this iteration, and rotated forward to the next.
    EXPECT_EQ(dsOffset(*loads[0]), 0);
    EXPECT_EQ(dsOffset(*loads[1]), 32);
    EXPECT_EQ(dsOffset(*loads[2]), 64 + kStride);
    EXPECT_EQ(dsOffset(*loads[3]), 96 + kStride);
}

TEST_F(WmmaReorderPassTest, BodyIsAlwaysAPermutation) {
    buildLoop();
    const size_t before = instructionsOf(*body).size();

    WmmaReorderOptions options;
    options.prefetchDistance = 2;
    const auto* res = runPass(std::make_unique<ReverseOrderProvider>(), options);
    ASSERT_TRUE(res->applied) << res->skipReason;

    const auto after = instructionsOf(*body);
    EXPECT_EQ(after.size(), before);
    for (StinkyInstruction* inst : loads)
        EXPECT_NE(std::find(after.begin(), after.end(), inst), after.end());
}

// ─────────────────────────────────────────────────────────────────────────────
// Refusals
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(WmmaReorderPassTest, NoCrossIteration_RefusesToDemoteAPrefetchedLoad) {
    buildLoop();
    WmmaReorderOptions options;
    options.prefetchDistance = 2;
    options.allowCrossIteration = false;

    // Reversed, tile 0 is read by the last wmma, so it can neither stay wrapped
    // (no slot after its consumer) nor be de-rotated with migration disabled.
    const auto* res = runPass(std::make_unique<ReverseOrderProvider>(), options);
    ASSERT_NE(res, nullptr);
    EXPECT_FALSE(res->applied);
    EXPECT_EQ(dsOffset(*loads[0]), kStride) << "a refused loop must be left untouched";
    EXPECT_EQ(dsLoadCount(*preheader), 2u);
}

TEST_F(WmmaReorderPassTest, WrongLoopLabel_FindsNothing) {
    buildLoop();
    WmmaReorderOptions options;
    options.loopLabel = "label_SomeOtherLoop";
    EXPECT_EQ(runPass(std::make_unique<ReverseOrderProvider>(), options), nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Register conflicts
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(WmmaReorderPassTest, OverlappingReuse_VirtualizesLoserOnSameRegister) {
    // Two ds_loads double-buffer the shared B tile (v[60:68), the operand
    // every addWmma() reads) within one iteration: loadB0 feeds wmma0/wmma1,
    // loadB1 reuses the same register for wmma2/wmma3 -- the software pipeline
    // already relies on program order to keep these apart, since a raw
    // register-overlap scan can't otherwise tell the two values apart.
    //
    // A distance far larger than the body forces every load to the earliest
    // safe slot (0), collapsing loadB0's and loadB1's occupied windows onto
    // each other: the physical register can no longer hold both live ranges.
    StinkyInstruction* loadB0 = addDsLoad(body, /*dest=*/60, /*offset=*/0);
    wmma.push_back(addWmma(body, /*aBase=*/20, /*cBase=*/100));  // wmma0, reads loadB0
    wmma.push_back(addWmma(body, /*aBase=*/28, /*cBase=*/108));  // wmma1, reads loadB0
    StinkyInstruction* loadB1 = addDsLoad(body, /*dest=*/60, /*offset=*/64);
    wmma.push_back(addWmma(body, /*aBase=*/36, /*cBase=*/116));  // wmma2, reads loadB1
    wmma.push_back(addWmma(body, /*aBase=*/44, /*cBase=*/124));  // wmma3, reads loadB1

    WmmaReorderOptions options;
    options.prefetchDistance = 10;

    const auto* res = runPass(
        std::make_unique<ExplicitOrderProvider>(std::vector<unsigned>{0, 1, 2, 3}), options);
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applied) << res->skipReason;
    EXPECT_EQ(res->conflictsVirtualized, 1u);

    // loadB0 (earlier producer) keeps the physical register; loadB1 (the
    // later, colliding producer) is retargeted to a placeholder.
    EXPECT_FALSE(loadB0->getDestReg(0).isVirtualReg());
    EXPECT_TRUE(loadB1->getDestReg(0).isVirtualReg());
    EXPECT_TRUE(loadB0->getDestReg(0).isRegister());
    EXPECT_EQ(loadB0->getDestReg(0).reg.idx, 60u);

    // wmma2/wmma3 (loadB1's own consumers) must have followed it to the
    // virtual register; wmma0/wmma1 (loadB0's) must be untouched.
    auto readsVirtual = [](const StinkyInstruction* w) {
        for (size_t s = 0; s < w->getNumSrcRegs(); ++s)
            if (w->getSrcReg(s).isVirtualReg()) return true;
        return false;
    };
    EXPECT_FALSE(readsVirtual(wmma[0]));
    EXPECT_FALSE(readsVirtual(wmma[1]));
    EXPECT_TRUE(readsVirtual(wmma[2]));
    EXPECT_TRUE(readsVirtual(wmma[3]));
}

TEST_F(WmmaReorderPassTest, DistinctRegisters_NeverVirtualized) {
    // buildLoop()'s four A tiles each own a distinct RegGroup; nothing shares a
    // register within the segment, so the conflict path must never trigger.
    buildLoop();
    WmmaReorderOptions options;
    options.prefetchDistance = 2;

    const auto* res = runPass(std::make_unique<ReverseOrderProvider>(), options);
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applied) << res->skipReason;
    EXPECT_EQ(res->conflictsVirtualized, 0u);
    for (StinkyInstruction* inst : loads) EXPECT_FALSE(inst->getDestReg(0).isVirtualReg());
}

}  // namespace
