// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <memory>

#include "TestHelpers.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class VgprMsbEncodingTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func = std::make_unique<Function>("vgpr_msb_encoding_test");
    BasicBlock* bb = nullptr;

    void SetUp() override {
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
    }
};

constexpr int kSrc0 = 0x03;
constexpr int kSrc1 = 0x0c;
constexpr int kDst = 0xc0;

TEST_F(VgprMsbEncodingTest, NeedIsTheBankOfEachSlotsOperand) {
    // v300 = v_add_f32(v10, v600): dst bank 1, src0 bank 0, src1 bank 2.
    const StinkyInstruction* add = createVAddInBlock(bb, arch, 300, 10, 600);

    const VgprMsbNeed need = vgprMsbNeed(*add);
    EXPECT_EQ(need.value, 72);
    EXPECT_EQ(need.usedSlots, kSrc0 | kSrc1 | kDst);
    EXPECT_EQ(computeRequiredMsb(add), std::make_pair(72, true));
}

TEST_F(VgprMsbEncodingTest, TheLookupDecidesTheBanks) {
    // What a colouring that moved every operand up one bank would need.
    const StinkyInstruction* add = createVAddInBlock(bb, arch, 300, 10, 600);

    const VgprMsbNeed need = vgprMsbNeed(
        *add, [](const StinkyRegister& reg, size_t, bool) { return reg.reg.idx + 256; });
    EXPECT_EQ(decodeVgprMsbForSlot(need.value, 3), 2);
    EXPECT_EQ(decodeVgprMsbForSlot(need.value, 0), 1);
    EXPECT_EQ(decodeVgprMsbForSlot(need.value, 1), 3);
    EXPECT_EQ(need.usedSlots, kSrc0 | kSrc1 | kDst);
}

TEST_F(VgprMsbEncodingTest, AnInstructionWithoutVgprsNeedsNothing) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* wait = builder.create(getMCIDByUOp(GFX::s_waitcnt, arch));
    wait->addSrcReg(StinkyRegister(0));

    EXPECT_FALSE(vgprMsbNeed(*wait).any());
    EXPECT_EQ(computeRequiredMsb(wait), std::make_pair(-1, false));
}

TEST_F(VgprMsbEncodingTest, StateAsksForASetOnlyWhenTheByteChanges) {
    const VgprMsbNeed high = vgprMsbNeed(*createVAddInBlock(bb, arch, 300, 301, 302));
    const VgprMsbNeed low = vgprMsbNeed(*createVAddInBlock(bb, arch, 3, 4, 5));

    VgprMsbState state;
    EXPECT_FALSE(state.known());
    EXPECT_TRUE(state.needsSet(high));
    state.set(high);
    EXPECT_FALSE(state.needsSet(high));
    EXPECT_TRUE(state.needsSet(low));
    EXPECT_FALSE(state.needsSet(VgprMsbNeed{}));

    // Bank 0 everywhere still needs a set once the byte in force is unknown.
    state.forget();
    EXPECT_TRUE(state.needsSet(low));
}

}  // namespace
