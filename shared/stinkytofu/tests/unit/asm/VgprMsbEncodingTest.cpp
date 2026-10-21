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

/// `v<dst> = v_mov_b32 v<src>`: src0 and dst, nothing in src1 or src2.
VgprMsbNeed movNeed(BasicBlock* bb, GfxArchID arch, int dst, int src) {
    AsmIRBuilder builder(*bb, arch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, arch));
    mov->addDestReg(StinkyRegister("v", dst, 1));
    mov->addSrcReg(StinkyRegister("v", src, 1));
    return vgprMsbNeed(*mov);
}

TEST_F(VgprMsbEncodingTest, UnderZeroAnUnusedSlotHasToReadBankZero) {
    const VgprMsbNeed add = vgprMsbNeed(*createVAddInBlock(bb, arch, 300, 301, 302));
    const VgprMsbNeed mov = movNeed(bb, arch, 303, 304);
    EXPECT_EQ(mov.usedSlots, kSrc0 | kDst);

    VgprMsbState state;
    state.set(add);
    EXPECT_TRUE(state.needsSet(mov)) << "src1 is in bank 1, and Zero wants it at 0";
}

TEST_F(VgprMsbEncodingTest, UnderKeepOnlyTheUsedSlotsHaveToMatch) {
    const VgprMsbNeed add = vgprMsbNeed(*createVAddInBlock(bb, arch, 300, 301, 302));
    const VgprMsbNeed mov = movNeed(bb, arch, 303, 304);

    VgprMsbState state(UnusedVgprMsbSlots::Keep);
    EXPECT_EQ(state.set(add), add.value);
    EXPECT_FALSE(state.needsSet(mov));
    EXPECT_TRUE(state.needsSet(movNeed(bb, arch, 3, 304))) << "dst is used, and differs";
}

TEST_F(VgprMsbEncodingTest, UnderKeepANewByteFillsUnusedSlotsFromWhatComesNext) {
    // The mov leaves src1 open; the add after it wants bank 1 there.
    const VgprMsbNeed mov = movNeed(bb, arch, 303, 304);
    const VgprMsbNeed add = vgprMsbNeed(*createVAddInBlock(bb, arch, 300, 301, 302));
    const VgprMsbNeed ahead[] = {add};

    VgprMsbState state(UnusedVgprMsbSlots::Keep);
    EXPECT_EQ(state.set(mov, ahead), add.value);
    EXPECT_FALSE(state.needsSet(add));

    // Zero ignores what comes next.
    VgprMsbState zero;
    EXPECT_EQ(zero.set(mov, ahead), mov.value);
}

TEST_F(VgprMsbEncodingTest, UnderKeepASlotNothingAheadUsesKeepsItsBank) {
    const VgprMsbNeed add = vgprMsbNeed(*createVAddInBlock(bb, arch, 300, 301, 302));
    const VgprMsbNeed mov = movNeed(bb, arch, 3, 4);

    VgprMsbState state(UnusedVgprMsbSlots::Keep);
    state.set(add);
    const int byte = state.set(mov);
    EXPECT_EQ(decodeVgprMsbForSlot(byte, 1), 1) << "src1 keeps the bank in force";
    EXPECT_EQ(decodeVgprMsbForSlot(byte, 0), 0);
    EXPECT_EQ(decodeVgprMsbForSlot(byte, 3), 0);
}

}  // namespace
