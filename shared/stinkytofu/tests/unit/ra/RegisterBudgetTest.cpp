// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <memory>
#include <optional>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterBudget.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class RegisterBudgetTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
        entry = func->createBasicBlock("entry");
    }

    /// sX = s_mov_b32(sY), with widths so tuple spans can be exercised.
    void mov(uint32_t dst, uint32_t src, uint16_t dstWidth = 1, uint16_t srcWidth = 1) {
        AsmIRBuilder builder(*entry, kRaTestArch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_mov_b32, kRaTestArch));
        inst->addDestReg(StinkyRegister("s", dst, dstWidth));
        inst->addSrcReg(StinkyRegister("s", src, srcWidth));
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
};

}  // namespace

TEST_F(RegisterBudgetTest, NextEvenBaseSitsOnTheFirstFreeEvenIndex) {
    mov(/*dst=*/7, /*src=*/3);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 8u);
    EXPECT_EQ(nextEvenRegisterBase(*func, RegType::S), 8u);

    mov(/*dst=*/8, /*src=*/0);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 9u);
    EXPECT_EQ(nextEvenRegisterBase(*func, RegType::S), 10u);
}

TEST_F(RegisterBudgetTest, AReusableBlockEndsInsideWhatTheKernelAlreadyDeclares) {
    func->setMetaData(kSigDispatchFilledSgprsMetaKey, 32);
    mov(/*dst=*/57, /*src=*/0);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 58u);

    // A pair lands on s[56:57] and a pair-plus-scratch on s[54:56]: both end at
    // the 58 already declared, so neither costs a register.
    EXPECT_EQ(reusableEvenSgprBase(*func, /*width=*/2), 56u);
    EXPECT_EQ(reusableEvenSgprBase(*func, /*width=*/3), 54u);

    // Capping at the pair above stacks a second block clear of the first.
    EXPECT_EQ(reusableEvenSgprBase(*func, /*width=*/3, /*limit=*/56), 52u);
}

TEST_F(RegisterBudgetTest, NothingIsReusableWhenNoBlockIsProvablyFree) {
    mov(/*dst=*/57, /*src=*/0);

    // No published dispatch line, and unpublished cannot read as zero: s[54:56]
    // may be preloaded kernargs the kernel reads but never names, so the caller
    // has to stay above everything instead.
    EXPECT_EQ(reusableEvenSgprBase(*func, /*width=*/3), std::nullopt);
    EXPECT_EQ(nextEvenRegisterBase(*func, RegType::S), 58u);

    // Published, but the whole range sits in the prefix the dispatch fills.
    func->setMetaData(kSigDispatchFilledSgprsMetaKey, 58);
    EXPECT_EQ(reusableEvenSgprBase(*func, /*width=*/3), std::nullopt);

    // No room for the block at all.
    EXPECT_EQ(reusableEvenSgprBase(*func, /*width=*/3, /*limit=*/2), std::nullopt);
}

TEST_F(RegisterBudgetTest, CountsOnePastTheHighestIndexUsed) {
    mov(/*dst=*/7, /*src=*/3);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 8u);
    // A class the function never names needs nothing declared.
    EXPECT_EQ(highestRegisterCount(*func, RegType::V), 0u);
}

TEST_F(RegisterBudgetTest, CountsTheWholeSpanOfATupleOperand) {
    // s[8:11] names only its base, so the count has to follow the width.
    mov(/*dst=*/8, /*src=*/0, /*dstWidth=*/4);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 12u);
}

TEST_F(RegisterBudgetTest, SourcesCountAsWellAsDestinations) {
    mov(/*dst=*/1, /*src=*/40);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 41u);
}

TEST_F(RegisterBudgetTest, UsageWinsWhenItExceedsWhatTheAbiFills) {
    mov(/*dst=*/60, /*src=*/0);
    // 4 preloaded + 2 for the kernarg pointer + 3 workgroup ids = 9, well under.
    EXPECT_EQ(requiredSgprCount(*func, /*numSgprPreload=*/4, {1, 1, 1}), 61u);
}

TEST_F(RegisterBudgetTest, NeverDeclaresFewerThanTheHardwareFills) {
    // The whole point of the floor: a kernel can compact its own registers far
    // below the preloaded arguments, which the dispatch still writes whether or
    // not any operand names them.
    mov(/*dst=*/1, /*src=*/0);
    EXPECT_EQ(highestRegisterCount(*func, RegType::S), 2u);
    EXPECT_EQ(requiredSgprCount(*func, /*numSgprPreload=*/27, {1, 1, 1}), 32u);
}

TEST_F(RegisterBudgetTest, EachEnabledWorkgroupIdCostsOneRegister) {
    mov(/*dst=*/1, /*src=*/0);
    EXPECT_EQ(requiredSgprCount(*func, /*numSgprPreload=*/27, {1, 0, 0}), 30u);
    EXPECT_EQ(requiredSgprCount(*func, /*numSgprPreload=*/27, {0, 0, 0}), 29u);
}

TEST_F(RegisterBudgetTest, NoPreloadMeansNoKernargPointerToAccountFor) {
    mov(/*dst=*/1, /*src=*/0);
    // numSgprPreload of 0 suppresses the .amdhsa_user_sgpr_count line entirely,
    // so there is no +2 to carry either.
    EXPECT_EQ(requiredSgprCount(*func, /*numSgprPreload=*/0, {1, 1, 1}), 3u);
}

TEST(SettledDispatchFilledSgprCountTest, TheLineIsThePreloadedFloorOrNothingAtAll) {
    EXPECT_EQ(settledDispatchFilledSgprCount(/*numSgprPreload=*/27, {1, 1, 1}), 32u);

    // With no preload the floor above drops the kernarg segment pointer, which
    // requiredSgprCount can absorb and a pin boundary cannot: reading 3 when the
    // pointer does take s[0:1] would free s3 and s4, which the dispatch wrote.
    EXPECT_EQ(settledDispatchFilledSgprCount(/*numSgprPreload=*/0, {1, 1, 1}), std::nullopt);
}

TEST(DispatchFilledVgprCountTest, PackingDecidesTheCountRatherThanTheField) {
    // The field counts enabled dimensions. On a packed target they share v0, so
    // every value of it means one register.
    for (int workItem = -1; workItem <= 2; ++workItem) {
        EXPECT_EQ(dispatchFilledVgprCount(workItem, /*packedWorkitemId=*/true), 1u)
            << "workItem=" << workItem;
    }

    // Unpacked, one register per dimension from v0 up, the field counting the
    // extras -- so x alone is 0 and reaches v0.
    EXPECT_EQ(dispatchFilledVgprCount(/*vgprWorkItem=*/-1, /*packedWorkitemId=*/false), 1u);
    EXPECT_EQ(dispatchFilledVgprCount(/*vgprWorkItem=*/0, /*packedWorkitemId=*/false), 1u);
    EXPECT_EQ(dispatchFilledVgprCount(/*vgprWorkItem=*/2, /*packedWorkitemId=*/false), 3u);
}

TEST(SettledDispatchFilledVgprCountTest, APackedTargetFillsV0AloneAndAnUnknownOneNothing) {
    ASSERT_TRUE(hasPackedWorkitemId(kRaTestArch)) << "this test needs a packed target";
    EXPECT_EQ(settledDispatchFilledVgprCount(kRaTestArch), 1u);

    // An id past the end of what this build registered. Nothing is known about
    // its packing, so the line is unsettled and every vector live-in stays
    // pinned -- the same direction of caution as a missing scalar boundary.
    EXPECT_EQ(settledDispatchFilledVgprCount(static_cast<GfxArchID>(1u << 20)), std::nullopt);
}
