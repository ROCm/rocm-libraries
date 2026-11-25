// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The bank-word state machine InsertVgprMsbPass emits by, the allocator's
// prediction replays and the scheduler orders by, and the requirement each
// instruction brings to it.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

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

TEST(VgprMsbTrackerTest, OnlyASetterWithNothingSinceTheLabelDirectlyFollowsIt) {
    // What InsertVgprMsbPass asks before a setter already in the stream, which
    // needs the s_nop its own setters get there.
    VgprMsbTracker tracker;
    EXPECT_FALSE(tracker.directlyAfterLabel());
    tracker.label();
    EXPECT_TRUE(tracker.directlyAfterLabel());
    tracker.step(kNoWord);
    EXPECT_FALSE(tracker.directlyAfterLabel()) << "an instruction comes between";
    tracker.label();
    tracker.observeSetter(kWordA);
    EXPECT_FALSE(tracker.directlyAfterLabel()) << "the setter itself comes between";
    tracker.label();
    tracker.reset();
    EXPECT_FALSE(tracker.directlyAfterLabel()) << "a call comes between";
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

/// A WMMA's word: A in bank 0, B in bank 2, C and D in bank 1. Every slot.
const VgprMsbRequirement kWmma{inSlot(1, 2) + inSlot(2, 1) + inSlot(3, 1), kVgprMsbFullMask};

/// A ds_load writing bank 1 from an address in bank 0: src0 and dst only.
const VgprMsbRequirement kLoad{inSlot(3, 1), slotBits(0) + slotBits(3)};

}  // namespace

TEST(VgprMsbTrackerTest, UnusedSlotsDontCareAWordServesWhatAgreesOnTheSlotsUsed) {
    VgprMsbTracker tracker(VgprMsbPolicy::UnusedSlotsDontCare);
    EXPECT_TRUE(tracker.step(kWmma).emit);
    // The load's src0 and dst are what the WMMA's word holds there; src1 and
    // src2 are not the load's business.
    EXPECT_TRUE(tracker.satisfies(kLoad));
    const VgprMsbStep load = tracker.step(kLoad);
    EXPECT_FALSE(load.emit);
    EXPECT_FALSE(load.rewrite);
    EXPECT_FALSE(tracker.step(kWmma).emit) << "the word in force is still the WMMA's";

    // Under the exact rule the same window switches at every instruction.
    VgprMsbTracker exact;
    EXPECT_TRUE(exact.step(kWmma).emit);
    EXPECT_TRUE(exact.step(kLoad).emit);
    EXPECT_TRUE(exact.step(kWmma).emit);
}

TEST(VgprMsbTrackerTest, UnusedSlotsDontCareASetterNothingHasReadTakesTheNewBanks) {
    VgprMsbTracker tracker(VgprMsbPolicy::UnusedSlotsDontCare);
    const VgprMsbStep first = tracker.step(kLoad);
    EXPECT_TRUE(first.emit);
    EXPECT_EQ(first.word, kLoad.word) << "with no word known the unused slots are 0";

    // The WMMA needs src1 and src2 in banks 2 and 1, which the load did not
    // read: the load's setter takes them rather than a second one following.
    EXPECT_FALSE(tracker.satisfies(kWmma));
    EXPECT_FALSE(tracker.needsSetter(kWmma));
    const VgprMsbStep wmma = tracker.step(kWmma);
    EXPECT_FALSE(wmma.emit);
    EXPECT_TRUE(wmma.rewrite);
    EXPECT_EQ(wmma.word, kWmma.word);
    EXPECT_TRUE(tracker.satisfies(kLoad)) << "the rewritten word still serves the load";

    // Every slot has now been read, so a change of address bank needs a setter,
    // whose Msb16 high byte is the rewritten word.
    const VgprMsbRequirement load{inSlot(0, 1) + inSlot(3, 1), slotBits(0) + slotBits(3)};
    EXPECT_TRUE(tracker.needsSetter(load));
    const VgprMsbStep next = tracker.step(load);
    EXPECT_TRUE(next.emit);
    EXPECT_FALSE(next.rewrite);
    EXPECT_EQ(next.word, kWmma.word + inSlot(0, 1)) << "a setter keeps the slots it does not set";
    EXPECT_EQ(next.previousWord, kWmma.word);
}

TEST(VgprMsbTrackerTest, UnusedSlotsDontCareASlotReadSinceTheSetterNeedsANewOne) {
    VgprMsbTracker tracker(VgprMsbPolicy::UnusedSlotsDontCare);
    // dst in bank 1, src0 and src1 in bank 0; then the same with src1 in bank 1.
    const int used = slotBits(0) + slotBits(1) + slotBits(3);
    const VgprMsbRequirement first{inSlot(3, 1), used};
    const VgprMsbRequirement second{inSlot(3, 1) + inSlot(1, 1), used};
    EXPECT_TRUE(tracker.step(first).emit);
    // The first read src1 in bank 0, so changing it under that setter would
    // move the first's operand.
    EXPECT_TRUE(tracker.needsSetter(second));
    const VgprMsbStep step = tracker.step(second);
    EXPECT_TRUE(step.emit);
    EXPECT_EQ(step.word, second.word);
    EXPECT_EQ(step.previousWord, first.word);
}

TEST(VgprMsbTrackerTest, UnusedSlotsDontCareRewritesOnlyItsOwnSetterInTheSameStretch) {
    for (int boundary = 0; boundary < 3; ++boundary) {
        SCOPED_TRACE(boundary);
        VgprMsbTracker tracker(VgprMsbPolicy::UnusedSlotsDontCare);
        tracker.step(kLoad);
        if (boundary == 0) tracker.label();
        if (boundary == 1) tracker.reset();
        // A setter already in the stream, with the same word as the one above.
        if (boundary == 2) tracker.observeSetter(kLoad.word);
        // Nothing has read src1 or src2, but the setter that would change is
        // gone, or not this tracker's.
        EXPECT_TRUE(tracker.needsSetter(kWmma));
        const VgprMsbStep step = tracker.step(kWmma);
        EXPECT_TRUE(step.emit);
        EXPECT_FALSE(step.rewrite);
        EXPECT_EQ(step.firstAfterLabel, boundary == 0);
        if (boundary == 2) {
            EXPECT_EQ(step.previousWord, kLoad.word);
            EXPECT_EQ(step.word, kWmma.word);
        } else {
            EXPECT_FALSE(step.previousWord.has_value());
        }
    }
}

TEST(VgprMsbTrackerTest, AnObservedSetterServesWhatItsWordServes) {
    VgprMsbTracker tracker(VgprMsbPolicy::UnusedSlotsDontCare);
    tracker.observeSetter(kWmma.word);
    EXPECT_TRUE(tracker.satisfies(kLoad));
    EXPECT_FALSE(tracker.step(kLoad).emit);
    // A setter whose word is not a literal leaves nothing known.
    tracker.observeSetter(-1);
    EXPECT_FALSE(tracker.satisfies(kLoad));
    EXPECT_TRUE(tracker.step(kLoad).emit);

    VgprMsbTracker exact;
    exact.observeSetter(kWordA);
    EXPECT_TRUE(exact.satisfies(needs(kWordA)));
}

TEST(VgprMsbTrackerTest, UnusedSlotsDontCareAFullMaskIsComparedExactly) {
    // An instruction with a slotless VGPR field depends on every slot: after a
    // setter whose slots have all been read, it needs its own word.
    VgprMsbTracker tracker(VgprMsbPolicy::UnusedSlotsDontCare);
    tracker.step(kWmma);
    const VgprMsbRequirement scaled{inSlot(3, 1), kVgprMsbFullMask};
    EXPECT_FALSE(tracker.satisfies(scaled));
    const VgprMsbStep step = tracker.step(scaled);
    EXPECT_TRUE(step.emit);
    EXPECT_EQ(step.word, scaled.word);
}

TEST(VgprMsbTrackerTest, ThePolicyComesFromTheCapabilities) {
    AsmCapsConfig caps;
    EXPECT_EQ(vgprMsbPolicyOf(caps), VgprMsbPolicy::ExactWord);
    caps.vgprMsbUnusedSlotsDontCare = true;
    EXPECT_EQ(vgprMsbPolicyOf(caps), VgprMsbPolicy::UnusedSlotsDontCare);
}

namespace {

/// A random requirement: each slot used with a random bank, or not; now and
/// then a slotless field, which makes every slot count; now and then nothing.
VgprMsbRequirement randomRequirement(std::mt19937& rng) {
    std::uniform_int_distribution<int> percent(0, 99);
    std::uniform_int_distribution<int> bank(0, 3);
    if (percent(rng) < 10) return kNoWord;
    int word = 0;
    int mask = 0;
    for (int slot = 0; slot < 4; ++slot) {
        if (percent(rng) < 50) continue;
        word += inSlot(slot, bank(rng));
        mask += slotBits(slot);
    }
    if (mask == 0) return kNoWord;
    return {word, percent(rng) < 10 ? kVgprMsbFullMask : mask};
}

/// The fewest setters that serve \p sequence in order, counted directly: cut
/// it into the longest stretches in which every slot one of them uses is the
/// same bank wherever it is used. A stretch of a stretch qualifies too, so
/// taking each as long as it goes is as few as there can be.
int fewestSetters(const std::vector<VgprMsbRequirement>& sequence) {
    int setters = 0;
    bool open = false;
    uint32_t fixedMask = 0;
    uint32_t fixedWord = 0;
    for (const VgprMsbRequirement& required : sequence) {
        if (!required.needed()) continue;
        const uint32_t mask = static_cast<uint32_t>(required.mask);
        const uint32_t word = static_cast<uint32_t>(required.word) & mask;
        if (open && ((fixedWord ^ word) & mask & fixedMask) == 0) {
            fixedWord = (fixedWord & ~mask) | word;
            fixedMask |= mask;
            continue;
        }
        ++setters;
        open = true;
        fixedMask = mask;
        fixedWord = word;
    }
    return setters;
}

}  // namespace

TEST(VgprMsbTrackerTest, UnusedSlotsDontCareEmitsTheFewestSettersTheOrderAllows) {
    std::uniform_int_distribution<int> length(1, 40);
    for (int round = 0; round < 2000; ++round) {
        // One seed per round, so a failing round can be replayed alone.
        std::mt19937 rng(static_cast<std::mt19937::result_type>(round));
        std::vector<VgprMsbRequirement> sequence(static_cast<size_t>(length(rng)));
        for (VgprMsbRequirement& required : sequence) required = randomRequirement(rng);

        VgprMsbTracker dontCare(VgprMsbPolicy::UnusedSlotsDontCare);
        VgprMsbTracker exact;
        int dontCareSetters = 0;
        int exactSetters = 0;
        // The word the last setter now writes, and everything it has to serve.
        int setterWord = -1;
        std::vector<VgprMsbRequirement> sinceSetter;
        for (const VgprMsbRequirement& required : sequence) {
            if (exact.step(required).emit) ++exactSetters;

            const bool needsSetter = dontCare.needsSetter(required);
            const VgprMsbStep step = dontCare.step(required);
            EXPECT_EQ(step.emit, needsSetter) << "round " << round;
            EXPECT_FALSE(step.emit && step.rewrite) << "round " << round;
            if (step.emit) {
                ++dontCareSetters;
                setterWord = step.word;
                sinceSetter.clear();
            } else if (step.rewrite) {
                setterWord = step.word;
            }
            if (!required.needed()) continue;
            sinceSetter.push_back(required);
            for (const VgprMsbRequirement& served : sinceSetter)
                ASSERT_EQ(vgprMsbChangedFields(setterWord, served), 0U) << "round " << round;
        }

        EXPECT_EQ(dontCareSetters, fewestSetters(sequence)) << "round " << round;
        EXPECT_LE(dontCareSetters, exactSetters) << "round " << round;
    }
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
