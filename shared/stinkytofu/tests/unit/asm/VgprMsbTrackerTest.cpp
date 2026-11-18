// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The bank-word state machine InsertVgprMsbPass emits by, the allocator's
// prediction replays and the scheduler orders by, and the requirement each
// instruction brings to it.

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

using namespace stinkytofu;

namespace {

constexpr int kWordA = 0x5A;
constexpr int kWordB = 0x5B;

/// A requirement on every slot of \p word.
VgprMsbRequirement needs(int word) {
    return {word, kVgprMsbFullMask};
}

/// What an instruction with no VGPR in a slot brings: nothing.
constexpr VgprMsbRequirement kNoWord{};

/// The bits of \p slot in a bank word. Disjoint for distinct slots, so a sum of
/// them is the mask of those slots.
int slotBits(int slot) {
    return encodeVgprMsbForSlot(slot, 0x3);
}

/// \p bank in \p slot, and 0 in the others.
int inSlot(int slot, int bank) {
    return encodeVgprMsbForSlot(slot, bank);
}

}  // namespace

TEST(VgprMsbTrackerTest, AWordIsSetOnceAndKept) {
    VgprMsbTracker tracker;
    const VgprMsbStep first = tracker.step(needs(kWordA));
    EXPECT_TRUE(first.emit);
    EXPECT_EQ(first.word, kWordA);
    EXPECT_FALSE(first.firstAfterLabel);
    EXPECT_FALSE(first.previousWord.has_value());

    EXPECT_FALSE(tracker.step(needs(kWordA)).emit);

    const VgprMsbStep changed = tracker.step(needs(kWordB));
    EXPECT_TRUE(changed.emit);
    EXPECT_EQ(changed.word, kWordB);
    EXPECT_EQ(changed.previousWord, kWordA);
}

TEST(VgprMsbTrackerTest, AnInstructionWithoutVgprsLeavesTheWord) {
    VgprMsbTracker tracker;
    tracker.step(needs(kWordA));
    EXPECT_FALSE(tracker.step(kNoWord).emit);
    EXPECT_FALSE(tracker.step(needs(kWordA)).emit);
}

TEST(VgprMsbTrackerTest, BankZeroEverywhereIsAWordToo) {
    // Word 0 still has to be set at the start, and is the previous word of the
    // next switch, which Msb16 packs.
    VgprMsbTracker tracker;
    EXPECT_TRUE(tracker.step(needs(0)).emit);
    EXPECT_FALSE(tracker.step(needs(0)).emit);
    EXPECT_EQ(tracker.step(needs(kWordA)).previousWord, 0);
}

TEST(VgprMsbTrackerTest, ALabelForgetsTheWordAndAsksForAnSNop) {
    VgprMsbTracker tracker;
    tracker.step(needs(kWordA));
    tracker.label();
    const VgprMsbStep again = tracker.step(needs(kWordA));
    EXPECT_TRUE(again.emit);
    EXPECT_TRUE(again.firstAfterLabel);
    EXPECT_FALSE(again.previousWord.has_value());
}

TEST(VgprMsbTrackerTest, AnInstructionAfterTheLabelDropsTheSNop) {
    VgprMsbTracker tracker;
    tracker.label();
    EXPECT_FALSE(tracker.step(kNoWord).emit);
    const VgprMsbStep set = tracker.step(needs(kWordA));
    EXPECT_TRUE(set.emit);
    EXPECT_FALSE(set.firstAfterLabel);
    EXPECT_FALSE(set.previousWord.has_value());
}

TEST(VgprMsbTrackerTest, AResetForgetsTheWordWithoutAnSNop) {
    VgprMsbTracker tracker;
    tracker.step(needs(kWordA));
    tracker.reset();
    const VgprMsbStep again = tracker.step(needs(kWordA));
    EXPECT_TRUE(again.emit);
    EXPECT_FALSE(again.firstAfterLabel);
    EXPECT_FALSE(again.previousWord.has_value());
}

TEST(VgprMsbTrackerTest, UnderExactWordOnlyTheWordInForceSatisfies) {
    VgprMsbTracker tracker;
    EXPECT_TRUE(tracker.satisfies(kNoWord)) << "an instruction that needs no word is served";
    EXPECT_FALSE(tracker.satisfies(needs(kWordA))) << "nothing is in force yet";
    tracker.step(needs(kWordA));
    EXPECT_TRUE(tracker.satisfies(needs(kWordA)));
    // The slots a requirement does not depend on still count: the word in force
    // has the src0 the requirement asks for, but differs elsewhere.
    EXPECT_FALSE(tracker.satisfies({inSlot(0, decodeVgprMsbForSlot(kWordA, 0)), slotBits(0)}));
    tracker.label();
    EXPECT_FALSE(tracker.satisfies(needs(kWordA))) << "a label forgets the word";
}

namespace {

/// Instructions on gfx1250, the target that selects VGPR banks this way.
class VgprMsbRequirementTest : public ::testing::Test {
   protected:
    void SetUp() override {
        arch = getGfxArchID(12, 5, 0);
        bb = func.createBasicBlock("entry");
        builder = std::make_unique<AsmIRBuilder>(*bb, arch);
    }

    StinkyInstruction* create(GFX opcode) {
        return builder->create(getMCIDByUOp(opcode, arch));
    }

    Function func;
    BasicBlock* bb = nullptr;
    GfxArchID arch{};
    std::unique_ptr<AsmIRBuilder> builder;
};

}  // namespace

TEST_F(VgprMsbRequirementTest, ADsLoadDependsOnItsDestinationAndAddressSlots) {
    StinkyInstruction* load = create(GFX::ds_load_b128);
    load->addDestReg(StinkyRegister("v", 256, 4));
    load->addSrcReg(StinkyRegister("v", 4, 1));
    const VgprMsbRequirement required = requiredVgprMsb(*load);
    EXPECT_EQ(required.word, inSlot(3, 1));
    EXPECT_EQ(required.mask, slotBits(0) + slotBits(3));
}

TEST_F(VgprMsbRequirementTest, AValuOpWithoutSrc2LeavesThatSlotOut) {
    StinkyInstruction* add = create(GFX::v_add_f32);
    add->addDestReg(StinkyRegister("v", 512, 1));
    add->addSrcReg(StinkyRegister("v", 1, 1));
    add->addSrcReg(StinkyRegister("v", 300, 1));
    const VgprMsbRequirement required = requiredVgprMsb(*add);
    EXPECT_EQ(required.word, inSlot(3, 2) + inSlot(1, 1));
    EXPECT_EQ(required.mask, slotBits(0) + slotBits(1) + slotBits(3));
}

TEST_F(VgprMsbRequirementTest, AMatrixInstructionDependsOnEverySlot) {
    StinkyInstruction* wmma = create(GFX::v_wmma_f32_16x16x32_bf16);
    wmma->addDestReg(StinkyRegister("v", 256, 8));
    wmma->addSrcReg(StinkyRegister("v", 0, 8));
    wmma->addSrcReg(StinkyRegister("v", 512, 8));
    wmma->addSrcReg(StinkyRegister("v", 256, 8));
    const VgprMsbRequirement required = requiredVgprMsb(*wmma);
    EXPECT_EQ(required.word, inSlot(3, 1) + inSlot(1, 2) + inSlot(2, 1));
    EXPECT_EQ(required.mask, kVgprMsbFullMask);
}

TEST_F(VgprMsbRequirementTest, ASlotlessScaleFieldMakesEverySlotCount) {
    // C is a literal, so only dst, src0 and src1 fill a slot. The two scale
    // fields select none, and which bank they read is not settled, so the
    // whole word counts.
    StinkyInstruction* wmma = create(GFX::v_wmma_scale16_f32_16x16x128_f8f6f4);
    wmma->addDestReg(StinkyRegister("v", 256, 8));
    wmma->addSrcReg(StinkyRegister("v", 0, 8));
    wmma->addSrcReg(StinkyRegister("v", 512, 8));
    wmma->addSrcReg(StinkyRegister(0));
    wmma->addSrcReg(StinkyRegister("v", 10, 2));
    wmma->addSrcReg(StinkyRegister("v", 12, 2));
    const VgprMsbRequirement required = requiredVgprMsb(*wmma);
    EXPECT_EQ(required.word, inSlot(3, 1) + inSlot(1, 2));
    EXPECT_EQ(required.mask, kVgprMsbFullMask);
}

TEST_F(VgprMsbRequirementTest, AScalarInstructionNeedsNoWord) {
    StinkyInstruction* mov = create(GFX::s_mov_b32);
    mov->addDestReg(StinkyRegister("s", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 1, 1));
    const VgprMsbRequirement required = requiredVgprMsb(*mov);
    EXPECT_FALSE(required.needed());
    EXPECT_EQ(required.mask, 0);
    EXPECT_EQ(computeRequiredMsb(mov), std::make_pair(-1, false));
}

TEST_F(VgprMsbRequirementTest, TheWordIsWhatComputeRequiredMsbReturns) {
    StinkyInstruction* load = create(GFX::ds_load_b128);
    load->addDestReg(StinkyRegister("v", 768, 4));
    load->addSrcReg(StinkyRegister("v", 260, 1));
    EXPECT_EQ(computeRequiredMsb(load), std::make_pair(requiredVgprMsb(*load).word, true));
}
