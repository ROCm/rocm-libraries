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
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/StinkyWmmaVgprReorderPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

// ─────────────────────────────────────────────────────────────────────────────
// Test fixture
// ─────────────────────────────────────────────────────────────────────────────

class WmmaVgprReorderPassTest : public ::testing::Test {
   protected:
    static constexpr GfxArchID kArch = GfxArchID::Gfx1250;
    static constexpr uint16_t kTileSize = 8;  // VGPRs per wmma tile

    void SetUp() override {
        func = std::make_unique<Function>("test");
        setFunctionArch(*func, kArch);
        bb = func->createBasicBlock("entry");
        registerAllAnalyses(am);
    }

    // Add a v_wmma_f32_16x16x32_bf16 tagged with a pool index.
    // dest=C[cBase:cBase+7], src0=A[aBase:aBase+7], src1=B[bBase:bBase+7], src2=C_in
    StinkyInstruction* addWmma(int aBase, int bBase, int cBase, uint32_t poolIdx) {
        AsmIRBuilder builder(*bb, kArch);
        auto* inst = builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, kArch));
        inst->addDestReg(vgpr(cBase, kTileSize));
        inst->addSrcReg(vgpr(aBase, kTileSize));
        inst->addSrcReg(vgpr(bBase, kTileSize));
        inst->addSrcReg(vgpr(cBase, kTileSize));
        inst->addModifier(WmmaPoolData{poolIdx});
        return inst;
    }

    // Add an untagged wmma (no WmmaPoolData modifier) — used to test that the
    // pass ignores untagged instructions.
    StinkyInstruction* addUntaggedWmma(int aBase, int bBase, int cBase) {
        AsmIRBuilder builder(*bb, kArch);
        auto* inst = builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, kArch));
        inst->addDestReg(vgpr(cBase, kTileSize));
        inst->addSrcReg(vgpr(aBase, kTileSize));
        inst->addSrcReg(vgpr(bBase, kTileSize));
        inst->addSrcReg(vgpr(cBase, kTileSize));
        return inst;
    }

    // Add a ds_load_b128 writing into destBase:destBase+3
    StinkyInstruction* addDsLoad(int destBase, int addrReg = 0) {
        return createDsReadB128InBlock(bb, kArch, destBase, addrReg);
    }

    const WmmaReorderAnalysisResult* runPass() {
        PassContext ctx;
        auto pass = createStinkyWmmaVgprReorderPass();
        pass->run(*func, ctx, am);
        return getWmmaReorderResult(*bb);
    }

    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    AnalysisManager am;
};

// ─────────────────────────────────────────────────────────────────────────────
// Basic applicability
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(WmmaVgprReorderPassTest, EmptyBlock_NotApplicable) {
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    EXPECT_FALSE(res->applicable);
}

TEST_F(WmmaVgprReorderPassTest, UntaggedWmma_NotApplicable) {
    // wmma without WmmaPoolData modifier — pass must not fire
    addUntaggedWmma(/*a=*/0, /*b=*/16, /*c=*/32);
    addUntaggedWmma(/*a=*/8, /*b=*/16, /*c=*/40);
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    EXPECT_FALSE(res->applicable);
}

TEST_F(WmmaVgprReorderPassTest, PartiallyTaggedWmma_NotApplicable) {
    // Mix of tagged and untagged wmma — pass must bail out entirely
    addWmma(0, 16, 32, /*poolIdx=*/0);
    addUntaggedWmma(8, 16, 40);
    addWmma(32, 16, 48, /*poolIdx=*/1);
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    EXPECT_FALSE(res->applicable);
}

TEST_F(WmmaVgprReorderPassTest, SinglePool_NotApplicable) {
    // Only one pool tagged — need at least 2
    addWmma(0, 16, 32, /*poolIdx=*/0);
    addWmma(8, 16, 40, /*poolIdx=*/0);
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    EXPECT_FALSE(res->applicable);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2×2 B-major outer pattern (minimal case)
//
// First half (X0):  A groups {0, 8}, B groups {16, 24}
//   B-major outer order: A[0],B[16]  A[8],B[16]  A[0],B[24]  A[8],B[24]
//
// Second half (X1): A groups {32, 40}, B groups {16, 24}
//   Same B-major outer order
//
// Expected: A_X1 groups {32,40} aliased onto A_X0 groups {0,8}
//           desiredWmmaOrder switches first half to A-major outer
// ─────────────────────────────────────────────────────────────────────────────

class WmmaVgprReorderPass2x2Test : public WmmaVgprReorderPassTest {
   protected:
    // A_X0 bases, B bases, A_X1 bases, C bases (all non-overlapping)
    static constexpr int A0 = 0, A1 = 8;      // first-half A groups
    static constexpr int B0 = 16, B1 = 24;    // B groups (shared across halves)
    static constexpr int AX0 = 32, AX1 = 40;  // second-half A groups (X1)
    static constexpr int C_BASE = 48;         // C tiles start here

    void buildBMajorBlock() {
        // Pool 0 – B-major outer:  for B in {B0,B1}: for A in {A0,A1}: wmma
        w00 = addWmma(A0, B0, C_BASE + 0, /*poolIdx=*/0);
        w10 = addWmma(A1, B0, C_BASE + 8, /*poolIdx=*/0);
        w01 = addWmma(A0, B1, C_BASE + 16, /*poolIdx=*/0);
        w11 = addWmma(A1, B1, C_BASE + 24, /*poolIdx=*/0);
        // Pool 1 – same B-major outer order, different A registers
        wx00 = addWmma(AX0, B0, C_BASE + 32, /*poolIdx=*/1);
        wx10 = addWmma(AX1, B0, C_BASE + 40, /*poolIdx=*/1);
        wx01 = addWmma(AX0, B1, C_BASE + 48, /*poolIdx=*/1);
        wx11 = addWmma(AX1, B1, C_BASE + 56, /*poolIdx=*/1);
    }

    StinkyInstruction *w00, *w10, *w01, *w11;
    StinkyInstruction *wx00, *wx10, *wx01, *wx11;
};

TEST_F(WmmaVgprReorderPass2x2Test, BMajorOuter_IsApplicable) {
    buildBMajorBlock();
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    EXPECT_TRUE(res->applicable);
}

TEST_F(WmmaVgprReorderPass2x2Test, BMajorOuter_SavesExpectedVgprs) {
    buildBMajorBlock();
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applicable);
    // Two A groups aliased, each kTileSize VGPRs
    EXPECT_EQ(res->totalVgprSaved, 2u * kTileSize);
}

TEST_F(WmmaVgprReorderPass2x2Test, BMajorOuter_DesiredOrderIsAMajor) {
    buildBMajorBlock();
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applicable);
    ASSERT_EQ(res->desiredWmmaOrder.size(), 8u);

    // First half in A-major outer order:
    //   A[A0]: B[B0], B[B1]  →  w00, w01
    //   A[A1]: B[B0], B[B1]  →  w10, w11
    EXPECT_EQ(res->desiredWmmaOrder[0], w00);
    EXPECT_EQ(res->desiredWmmaOrder[1], w01);
    EXPECT_EQ(res->desiredWmmaOrder[2], w10);
    EXPECT_EQ(res->desiredWmmaOrder[3], w11);

    // Second half: same A-major outer
    EXPECT_EQ(res->desiredWmmaOrder[4], wx00);
    EXPECT_EQ(res->desiredWmmaOrder[5], wx01);
    EXPECT_EQ(res->desiredWmmaOrder[6], wx10);
    EXPECT_EQ(res->desiredWmmaOrder[7], wx11);
}

TEST_F(WmmaVgprReorderPass2x2Test, BMajorOuter_ReplacementsTargetAliasableWmma) {
    buildBMajorBlock();
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applicable);

    // All replacements must point to second-half wmma (wx00..wx11) which read AX0/AX1.
    // No first-half wmma (w00..w11) should appear: they use A0/A1 which are canonical.
    std::set<StinkyInstruction*> secondHalf = {wx00, wx10, wx01, wx11};
    for (const RegReplacement& r : res->replacements) {
        EXPECT_TRUE(secondHalf.count(r.inst) > 0)
            << "Replacement targets an unexpected instruction";
        EXPECT_TRUE(r.isSrc) << "Wmma A operand must be a src replacement";
    }
}

TEST_F(WmmaVgprReorderPass2x2Test, DsLoadsIntoAliasableGroup_AreReplaced) {
    buildBMajorBlock();
    // Add ds_loads that write into AX0 (32..35) and AX1 (40..43)
    auto* loadAX0 = addDsLoad(AX0);
    auto* loadAX1 = addDsLoad(AX1);

    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applicable);

    // Both ds_loads should appear in replacements as dest rewrites
    bool foundAX0 = false, foundAX1 = false;
    for (const RegReplacement& r : res->replacements) {
        if (r.inst == loadAX0 && !r.isSrc) foundAX0 = true;
        if (r.inst == loadAX1 && !r.isSrc) foundAX1 = true;
    }
    EXPECT_TRUE(foundAX0) << "ds_load into AX0 group not found in replacements";
    EXPECT_TRUE(foundAX1) << "ds_load into AX1 group not found in replacements";
}

TEST_F(WmmaVgprReorderPass2x2Test, Replacements_MapToCanonicalRegisters) {
    buildBMajorBlock();
    addDsLoad(AX0);

    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applicable);

    for (const RegReplacement& r : res->replacements) {
        if (r.inst->is(InstFlag::IF_DSRead) && !r.isSrc) {
            // oldReg must be in AX0 range [32..39]; newReg must be in A0 range [0..7]
            EXPECT_GE(r.oldReg.reg.idx, static_cast<uint32_t>(AX0));
            EXPECT_LT(r.oldReg.reg.idx, static_cast<uint32_t>(AX0 + kTileSize));
            EXPECT_GE(r.newReg.reg.idx, static_cast<uint32_t>(A0));
            EXPECT_LT(r.newReg.reg.idx, static_cast<uint32_t>(A0 + kTileSize));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Already A-major outer → no saving
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(WmmaVgprReorderPassTest, AMajorOuter_NotApplicable) {
    // Already A-major outer — no saving expected
    // Pool 0: for A in {0,8}: for B in {16,24}: wmma
    addWmma(0, 16, 48, /*poolIdx=*/0);
    addWmma(0, 24, 56, /*poolIdx=*/0);
    addWmma(8, 16, 64, /*poolIdx=*/0);
    addWmma(8, 24, 72, /*poolIdx=*/0);
    // Pool 1: same A-major outer
    addWmma(32, 16, 48, /*poolIdx=*/1);
    addWmma(32, 24, 56, /*poolIdx=*/1);
    addWmma(40, 16, 64, /*poolIdx=*/1);
    addWmma(40, 24, 72, /*poolIdx=*/1);

    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    EXPECT_FALSE(res->applicable);
}

// ─────────────────────────────────────────────────────────────────────────────
// B-variant: hardware B (src1) is pool-varying, hardware A (src0) is shared
//
// Pool 0 in A-major outer (hardware A contiguous):
//   wmma(A[0],B[16])  wmma(A[0],B[24])  wmma(A[8],B[16])  wmma(A[8],B[24])
// Pool 1, different B registers:
//   wmma(A[0],B[32])  wmma(A[0],B[40])  wmma(A[8],B[32])  wmma(A[8],B[40])
//
// detectABIndices sees src0 (hardware A) shared across pools → relabels:
//   aGroup = src1 = hardware B (pool-varying)
//   bGroup = src0 = hardware A (pool-shared)
// Hardware B then has inflated liveness; reordering makes it contiguous.
// Expected: hardware B pool-1 registers {32,40} aliased onto {16,24}.
// ─────────────────────────────────────────────────────────────────────────────

class WmmaVgprReorderPassBVariantTest : public WmmaVgprReorderPassTest {
   protected:
    static constexpr int HA0 = 0, HA8 = 8;      // hardware A (pool-shared, src0)
    static constexpr int HB16 = 16, HB24 = 24;  // hardware B pool 0 (pool-varying, src1)
    static constexpr int HB32 = 32, HB40 = 40;  // hardware B pool 1
    static constexpr int C_BASE = 64;

    void buildAMajorOuterBVariant() {
        // Pool 0: A-major outer (hardware A contiguous) → hardware B has inflated liveness
        wA0B16 = addWmma(HA0, HB16, C_BASE + 0, /*poolIdx=*/0);
        wA0B24 = addWmma(HA0, HB24, C_BASE + 8, /*poolIdx=*/0);
        wA8B16 = addWmma(HA8, HB16, C_BASE + 16, /*poolIdx=*/0);
        wA8B24 = addWmma(HA8, HB24, C_BASE + 24, /*poolIdx=*/0);
        // Pool 1: same A-major outer order, different B registers
        wxA0B32 = addWmma(HA0, HB32, C_BASE + 32, /*poolIdx=*/1);
        wxA0B40 = addWmma(HA0, HB40, C_BASE + 40, /*poolIdx=*/1);
        wxA8B32 = addWmma(HA8, HB32, C_BASE + 48, /*poolIdx=*/1);
        wxA8B40 = addWmma(HA8, HB40, C_BASE + 56, /*poolIdx=*/1);
    }

    StinkyInstruction *wA0B16, *wA0B24, *wA8B16, *wA8B24;
    StinkyInstruction *wxA0B32, *wxA0B40, *wxA8B32, *wxA8B40;
};

TEST_F(WmmaVgprReorderPassBVariantTest, PoolVaryingB_IsApplicable) {
    buildAMajorOuterBVariant();
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    EXPECT_TRUE(res->applicable);
}

TEST_F(WmmaVgprReorderPassBVariantTest, PoolVaryingB_SavesExpectedVgprs) {
    buildAMajorOuterBVariant();
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applicable);
    // Two B groups aliased across pools, each kTileSize VGPRs
    EXPECT_EQ(res->totalVgprSaved, 2u * kTileSize);
}

TEST_F(WmmaVgprReorderPassBVariantTest, PoolVaryingB_DesiredOrderMakesBContiguous) {
    buildAMajorOuterBVariant();
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    ASSERT_TRUE(res->applicable);
    ASSERT_EQ(res->desiredWmmaOrder.size(), 8u);

    // Pool 0 reordered so hardware B (relabeled A) is contiguous:
    //   HB16 group: wA0B16, wA8B16
    //   HB24 group: wA0B24, wA8B24
    EXPECT_EQ(res->desiredWmmaOrder[0], wA0B16);
    EXPECT_EQ(res->desiredWmmaOrder[1], wA8B16);
    EXPECT_EQ(res->desiredWmmaOrder[2], wA0B24);
    EXPECT_EQ(res->desiredWmmaOrder[3], wA8B24);
    // Pool 1 same structure
    EXPECT_EQ(res->desiredWmmaOrder[4], wxA0B32);
    EXPECT_EQ(res->desiredWmmaOrder[5], wxA8B32);
    EXPECT_EQ(res->desiredWmmaOrder[6], wxA0B40);
    EXPECT_EQ(res->desiredWmmaOrder[7], wxA8B40);
}

TEST_F(WmmaVgprReorderPassBVariantTest, AlreadyBContiguous_NotApplicable) {
    // Hardware B already contiguous (B-major outer for hardware):
    //   HB16 group first, then HB24 — so hardware B intervals are tight
    addWmma(HA0, HB16, C_BASE + 0, /*poolIdx=*/0);
    addWmma(HA8, HB16, C_BASE + 8, /*poolIdx=*/0);
    addWmma(HA0, HB24, C_BASE + 16, /*poolIdx=*/0);
    addWmma(HA8, HB24, C_BASE + 24, /*poolIdx=*/0);
    addWmma(HA0, HB32, C_BASE + 32, /*poolIdx=*/1);
    addWmma(HA8, HB32, C_BASE + 40, /*poolIdx=*/1);
    addWmma(HA0, HB40, C_BASE + 48, /*poolIdx=*/1);
    addWmma(HA8, HB40, C_BASE + 56, /*poolIdx=*/1);

    // After relabeling: aGroup=hardware B. HB16 at positions 0,1 → width=2 = nB=2 → not inflated.
    const auto* res = runPass();
    ASSERT_NE(res, nullptr);
    EXPECT_FALSE(res->applicable);
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests for individual ABI layers
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Dependency constraint: RAW hazard on C forces partial reorder
//
// A non-wmma instruction writes to C tile of w00 between w00 and w10 in the BB.
// When the pass walks the BB, passedCount=1 at that point → minRank[w00]=1.
// Ideal A-major order puts w00 at position 0, but the dep blocks that.
// EDF scheduler produces: w01, w00, w10, w11 — A0 still contiguous (positions 0-1)
// so the saving still occurs; only the first two slots are swapped.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(WmmaVgprReorderPass2x2Test, RawDepOnC_ConstrainsFirstWmmaButStillApplicable) {
    // Build pool 0 manually (in B-major order) with the dep inserted mid-pool.
    // BB order: w00, [ds_load→C0], w10, w01, w11, wx00, wx10, wx01, wx11
    w00 = addWmma(A0, B0, C_BASE + 0, /*poolIdx=*/0);
    addDsLoad(C_BASE);  // writes to C0 (C tile of w00) — RAW: minRank[w00]=1
    w10 = addWmma(A1, B0, C_BASE + 8, /*poolIdx=*/0);
    w01 = addWmma(A0, B1, C_BASE + 16, /*poolIdx=*/0);
    w11 = addWmma(A1, B1, C_BASE + 24, /*poolIdx=*/0);
    wx00 = addWmma(AX0, B0, C_BASE + 32, /*poolIdx=*/1);
    wx10 = addWmma(AX1, B0, C_BASE + 40, /*poolIdx=*/1);
    wx01 = addWmma(AX0, B1, C_BASE + 48, /*poolIdx=*/1);
    wx11 = addWmma(AX1, B1, C_BASE + 56, /*poolIdx=*/1);

    const auto* res = runPass();
    ASSERT_NE(res, nullptr);

    // Dep must not kill the optimization entirely.
    EXPECT_TRUE(res->applicable);
    EXPECT_GT(res->totalVgprSaved, 0u);

    // Pool 0 EDF result: w00 blocked at pos 0 (minRank=1), EDF picks w01 first.
    //   pos 0: w01 (no dep)
    //   pos 1: w00 (minRank=1 satisfied)
    //   pos 2: w10
    //   pos 3: w11
    // A0 group (w01, w00) is at positions 0-1, A1 group (w10, w11) at 2-3 —
    // intervals [0,1] and [2,3] are non-overlapping, so saving is preserved.
    ASSERT_GE(res->desiredWmmaOrder.size(), 4u);
    EXPECT_EQ(res->desiredWmmaOrder[0], w01) << "w01 must be first (no dep)";
    EXPECT_EQ(res->desiredWmmaOrder[1], w00) << "w00 slides to pos 1 (RAW dep)";
    EXPECT_EQ(res->desiredWmmaOrder[2], w10);
    EXPECT_EQ(res->desiredWmmaOrder[3], w11);
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests for individual ABI layers
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(WmmaVgprReorderPassTest, WmmaIntervalLiveness_ComputesCorrectIntervals) {
    // B-major outer 2×2, first half only (indices 0-3):
    //   wmma(A[0], B[16])  → index 0
    //   wmma(A[8], B[16])  → index 1
    //   wmma(A[0], B[24])  → index 2
    //   wmma(A[8], B[24])  → index 3
    // Expected: A[0] interval [0,2], A[8] interval [1,3], B[16] [0,1], B[24] [2,3]

    std::vector<WmmaNode> seq = {
        {nullptr, {0u, kTileSize}, {16u, kTileSize}, {48u, kTileSize}},
        {nullptr, {8u, kTileSize}, {16u, kTileSize}, {56u, kTileSize}},
        {nullptr, {0u, kTileSize}, {24u, kTileSize}, {64u, kTileSize}},
        {nullptr, {8u, kTileSize}, {24u, kTileSize}, {72u, kTileSize}},
    };

    WmmaIntervalLiveness liveness;
    auto intervals = liveness.computeLiveness(*bb, seq);

    RegGroup g0{0u, kTileSize}, g8{8u, kTileSize}, g16{16u, kTileSize}, g24{24u, kTileSize};
    ASSERT_TRUE(intervals.count(g0));
    EXPECT_EQ(intervals[g0].first, 0u);
    EXPECT_EQ(intervals[g0].last, 2u);

    ASSERT_TRUE(intervals.count(g8));
    EXPECT_EQ(intervals[g8].first, 1u);
    EXPECT_EQ(intervals[g8].last, 3u);

    ASSERT_TRUE(intervals.count(g16));
    EXPECT_EQ(intervals[g16].first, 0u);
    EXPECT_EQ(intervals[g16].last, 1u);

    ASSERT_TRUE(intervals.count(g24));
    EXPECT_EQ(intervals[g24].first, 2u);
    EXPECT_EQ(intervals[g24].last, 3u);
}

TEST_F(WmmaVgprReorderPassTest, PoolVaryingReorderAlgorithm_FindsAliasesForBMajorInput) {
    // Two pools, each 2A × 2B in B-major outer — passed as explicit pool vectors.
    std::vector<std::vector<WmmaNode>> pools = {
        // Pool 0
        {
            {nullptr, {0u, kTileSize}, {16u, kTileSize}, {48u, kTileSize}},
            {nullptr, {8u, kTileSize}, {16u, kTileSize}, {56u, kTileSize}},
            {nullptr, {0u, kTileSize}, {24u, kTileSize}, {64u, kTileSize}},
            {nullptr, {8u, kTileSize}, {24u, kTileSize}, {72u, kTileSize}},
        },
        // Pool 1 (different A groups)
        {
            {nullptr, {32u, kTileSize}, {16u, kTileSize}, {48u, kTileSize}},
            {nullptr, {40u, kTileSize}, {16u, kTileSize}, {56u, kTileSize}},
            {nullptr, {32u, kTileSize}, {24u, kTileSize}, {64u, kTileSize}},
            {nullptr, {40u, kTileSize}, {24u, kTileSize}, {72u, kTileSize}},
        },
    };

    // Build flat seq for liveness
    std::vector<WmmaNode> seq;
    for (const auto& p : pools)
        for (const auto& n : p) seq.push_back(n);

    WmmaIntervalLiveness liveness;
    const auto intervals = liveness.computeLiveness(*bb, seq);

    PoolVaryingReorderAlgorithm algo;
    const auto result = algo.solve(pools, intervals);

    ASSERT_FALSE(result.aliases.empty());
    EXPECT_EQ(result.aliases.size(), 2u);
    for (const AliasCandidate& c : result.aliases) {
        EXPECT_LT(c.canonical.base, 16u);
        EXPECT_GE(c.aliasable.base, 32u);
        EXPECT_EQ(c.vgprSaved, kTileSize);
    }
    EXPECT_EQ(result.desiredOrder.size(), 8u);
}

TEST_F(WmmaVgprReorderPassTest, PoolVaryingReorderAlgorithm_TripleBufferFindsAliases) {
    // Triple buffer: 3 explicit pools, each 2A × 2B in B-major outer.
    std::vector<std::vector<WmmaNode>> pools = {
        {{nullptr, {0u, kTileSize}, {16u, kTileSize}, {96u, kTileSize}},
         {nullptr, {8u, kTileSize}, {16u, kTileSize}, {104u, kTileSize}},
         {nullptr, {0u, kTileSize}, {24u, kTileSize}, {112u, kTileSize}},
         {nullptr, {8u, kTileSize}, {24u, kTileSize}, {120u, kTileSize}}},
        {{nullptr, {32u, kTileSize}, {16u, kTileSize}, {96u, kTileSize}},
         {nullptr, {40u, kTileSize}, {16u, kTileSize}, {104u, kTileSize}},
         {nullptr, {32u, kTileSize}, {24u, kTileSize}, {112u, kTileSize}},
         {nullptr, {40u, kTileSize}, {24u, kTileSize}, {120u, kTileSize}}},
        {{nullptr, {64u, kTileSize}, {16u, kTileSize}, {96u, kTileSize}},
         {nullptr, {72u, kTileSize}, {16u, kTileSize}, {104u, kTileSize}},
         {nullptr, {64u, kTileSize}, {24u, kTileSize}, {112u, kTileSize}},
         {nullptr, {72u, kTileSize}, {24u, kTileSize}, {120u, kTileSize}}},
    };

    std::vector<WmmaNode> seq;
    for (const auto& p : pools)
        for (const auto& n : p) seq.push_back(n);
    WmmaIntervalLiveness liveness;
    const auto intervals = liveness.computeLiveness(*bb, seq);

    PoolVaryingReorderAlgorithm algo;
    const auto result = algo.solve(pools, intervals);

    // 2 non-canonical pools × 2 A groups each = 4 alias pairs
    ASSERT_EQ(result.aliases.size(), 4u);
    for (const AliasCandidate& c : result.aliases) {
        EXPECT_LT(c.canonical.base, 16u);
        EXPECT_GE(c.aliasable.base, 32u);
        EXPECT_EQ(c.vgprSaved, kTileSize);
    }
    EXPECT_EQ(result.desiredOrder.size(), 12u);
}

TEST_F(WmmaVgprReorderPassTest, PoolVaryingReorderAlgorithm_NoAliasForAMajorInput) {
    // Already A-major outer — no saving expected
    std::vector<std::vector<WmmaNode>> pools = {
        {{nullptr, {0u, kTileSize}, {16u, kTileSize}, {48u, kTileSize}},
         {nullptr, {0u, kTileSize}, {24u, kTileSize}, {56u, kTileSize}},
         {nullptr, {8u, kTileSize}, {16u, kTileSize}, {64u, kTileSize}},
         {nullptr, {8u, kTileSize}, {24u, kTileSize}, {72u, kTileSize}}},
        {{nullptr, {32u, kTileSize}, {16u, kTileSize}, {48u, kTileSize}},
         {nullptr, {32u, kTileSize}, {24u, kTileSize}, {56u, kTileSize}},
         {nullptr, {40u, kTileSize}, {16u, kTileSize}, {64u, kTileSize}},
         {nullptr, {40u, kTileSize}, {24u, kTileSize}, {72u, kTileSize}}},
    };

    std::vector<WmmaNode> seq;
    for (const auto& p : pools)
        for (const auto& n : p) seq.push_back(n);
    WmmaIntervalLiveness liveness;
    const auto intervals = liveness.computeLiveness(*bb, seq);

    PoolVaryingReorderAlgorithm algo;
    const auto result = algo.solve(pools, intervals);

    EXPECT_TRUE(result.aliases.empty());
}
