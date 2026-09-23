/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
// The one shipped rule table. Unlike AllocationRulesTest, which builds throwaway
// tables to exercise the framework, this asserts what gfx1250 actually declares.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRules.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRulesRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationVerifier.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "transforms/asm/ra/allocators/GreedyAllocator.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

constexpr const char* kSmemRule = "SmemSelfOverlapUnderXnackReplay";
constexpr const char* kAlignRule = "ScalarTupleAlignment";

/// Looked up by name rather than by index, so adding a rule does not renumber
/// every other test.
const AllocationRule* findRule(const AllocationRules& rules, std::string_view name) {
    for (const AllocationRule& rule : rules.all()) {
        if (rule.name == name) return &rule;
    }
    return nullptr;
}

AsmCapsConfig xnackReplay(bool on) {
    AsmCapsConfig caps;
    caps.enableXnackReplay = on;
    return caps;
}

AllocationRules gfx1250Rules(bool xnack) {
    AllocationRulesRegistry::registerAll();
    return AllocationRulesRegistry::forArch(raTestTriple(), xnackReplay(xnack));
}

/// s[dst:dst+1] = s_load_b64(s[addr:addr+1]) -- a two-DWORD scalar load, which
/// is the family the rule is about.
StinkyInstruction* createSLoadB64(BasicBlock* bb, int dstReg, int addrReg) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* load = builder.create(getMCIDByUOp(GFX::s_load_b64, kRaTestArch));
    load->addDestReg(StinkyRegister("s", dstReg, 2));
    load->addSrcReg(StinkyRegister("s", addrReg, 2));
    return load;
}

/// s[dst] = s_load_b32(s[addr:addr+1]) -- one DWORD, so it returns all or
/// nothing and can always replay.
StinkyInstruction* createSLoadB32(BasicBlock* bb, int dstReg, int addrReg) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* load = builder.create(getMCIDByUOp(GFX::s_load_b32, kRaTestArch));
    load->addDestReg(StinkyRegister("s", dstReg, 1));
    load->addSrcReg(StinkyRegister("s", addrReg, 2));
    return load;
}

/// s[dst:dst+3] = s_load_b128(s[addr:addr+1]) -- a four-DWORD destination, which
/// the assembler requires to be 4-aligned.
StinkyInstruction* createSLoadB128(BasicBlock* bb, int dstReg, int addrReg) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* load = builder.create(getMCIDByUOp(GFX::s_load_b128, kRaTestArch));
    load->addDestReg(StinkyRegister("s", dstReg, 4));
    load->addSrcReg(StinkyRegister("s", addrReg, 2));
    return load;
}

class Gfx1250AllocationRulesTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
    }

    BasicBlock* block(const std::string& label) {
        return func->createBasicBlock(label);
    }

    std::unique_ptr<Function> func;
};

}  // namespace

TEST_F(Gfx1250AllocationRulesTest, DeclaresBothRulesWithTheRightKinds) {
    const AllocationRules rules = gfx1250Rules(/*xnack=*/true);
    EXPECT_TRUE(rules.problems().empty()) << rules.toString();

    const AllocationRule* smem = findRule(rules, kSmemRule);
    ASSERT_NE(smem, nullptr) << rules.toString();
    EXPECT_FALSE(smem->description.empty());
    // Interference, not placement: the hardware constraint is about when the
    // access reads versus writes.
    EXPECT_EQ(smem->kind(), RuleKind::Interference);

    const AllocationRule* align = findRule(rules, kAlignRule);
    ASSERT_NE(align, nullptr) << rules.toString();
    EXPECT_FALSE(align->description.empty());
    // Placement: it is about which index a tuple may start on, full stop.
    EXPECT_EQ(align->kind(), RuleKind::Placement);
}

TEST_F(Gfx1250AllocationRulesTest, BothRulesAreActive) {
    // The SMEM one was promoted after its audit came back silent; alignment is an
    // encoding requirement that was never optional. Demoting either should have
    // to change this line and say why.
    const AllocationRules rules = gfx1250Rules(/*xnack=*/true);
    ASSERT_NE(findRule(rules, kSmemRule), nullptr);
    ASSERT_NE(findRule(rules, kAlignRule), nullptr);
    EXPECT_EQ(findRule(rules, kSmemRule)->status, RuleStatus::Active);
    EXPECT_EQ(findRule(rules, kAlignRule)->status, RuleStatus::Active);
}

TEST_F(Gfx1250AllocationRulesTest, WithoutXnackReplayOnlyTheGatedRuleGoesInert) {
    // The gate is per rule, not per table. "Rule present, capability unset" and
    // "no rule" colour identically and want different fixes, so the gated one
    // must stay visible at Off.
    const AllocationRules rules = gfx1250Rules(/*xnack=*/false);
    ASSERT_NE(findRule(rules, kSmemRule), nullptr) << rules.toString();
    EXPECT_EQ(findRule(rules, kSmemRule)->status, RuleStatus::Off);

    // Alignment is an encoding requirement of every gfx1250 module, so no
    // capability can switch it off.
    ASSERT_NE(findRule(rules, kAlignRule), nullptr);
    EXPECT_EQ(findRule(rules, kAlignRule)->status, RuleStatus::Active);
}

TEST_F(Gfx1250AllocationRulesTest, WithoutTheCapabilityNoRangeIsWidened) {
    // The gate is the whole reason status is resolved at construction: a module
    // built without XNACK replay keeps the ranges it always had.
    BasicBlock* entry = block("entry");
    createSLoadB64(entry, 4, 0);
    ASSERT_TRUE(liftForAllocation(*func));

    const AllocationRules rules = gfx1250Rules(/*xnack=*/false);
    const SSALiveIntervals base = computeSSALiveIntervals(*func);
    EXPECT_EQ(applyEarlyClobber(*func, base, rules).toString(), base.toString());
}

TEST_F(Gfx1250AllocationRulesTest, ForcedActiveItSeparatesTheDestinationFromTheAddress) {
    // What promoting the rule would buy, exercised through the override hatch so
    // the assertion does not depend on the declared status.
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createSLoadB64(entry, 4, 0);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* address = ssaSourceValue(*load, 0);
    const StinkySSAValue* result = ssaDefinedValue(*load, 0);
    ASSERT_NE(address, nullptr);
    ASSERT_NE(result, nullptr);

    AllocationRules rules = gfx1250Rules(/*xnack=*/true);
    RuleOverrides force;
    force.activate.emplace_back(kSmemRule);
    ASSERT_TRUE(rules.unknownNames(force).empty());
    rules.force(force);

    const SSALiveIntervals base = computeSSALiveIntervals(*func);
    // The address dies at the load, so without the rule it and the result may
    // share a register. With it they may not.
    EXPECT_FALSE(base.overlap(address->valueId(), result->valueId()));
    EXPECT_TRUE(
        applyEarlyClobber(*func, base, rules).overlap(address->valueId(), result->valueId()));
}

TEST_F(Gfx1250AllocationRulesTest, ForcedActiveTheColouringKeepsThemApart) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createSLoadB64(entry, 4, 0);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* address = ssaSourceValue(*load, 0);
    const StinkySSAValue* result = ssaDefinedValue(*load, 0);
    ASSERT_NE(address, nullptr);
    ASSERT_NE(result, nullptr);

    AllocationRules rules = gfx1250Rules(/*xnack=*/true);
    RuleOverrides force;
    force.activateAll = true;
    rules.force(force);

    AllocationSetup setup(*func, RegClassSet::only(RegType::S), {}, std::move(rules));
    GreedyAllocator allocator;
    Expected<AllocationResult> result_ = allocator.allocate(setup.context());
    ASSERT_TRUE(result_.hasValue()) << result_.getError();
    EXPECT_TRUE(verifyAllocation(*func, *result_, setup.context()).ok());
    EXPECT_NE(result_->assignmentOf(address->valueId()), result_->assignmentOf(result->valueId()));
}

TEST_F(Gfx1250AllocationRulesTest, WithoutTheRuleCompactionIntroducesTheOverlap) {
    // Why the rule exists, stated as a test. The address is a live-in dying at
    // the load and the destination is defined there, so half-open ranges let
    // them share -- and compaction takes the offer, producing exactly the
    // unrepairable case Gfx1250HazardPass asserts on.
    //
    // This is the *current* behaviour, not the desired one. Promoting the rule
    // to Active is what makes it go away; this test then becomes the record of
    // what promotion bought.
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createSLoadB64(entry, 4, 0);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* address = ssaSourceValue(*load, 0);
    const StinkySSAValue* result = ssaDefinedValue(*load, 0);
    ASSERT_NE(address, nullptr);
    ASSERT_NE(result, nullptr);

    AllocationSetup setup(*func, RegClassSet::only(RegType::S));
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> coloured = allocator.allocate(setup.context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    EXPECT_EQ(coloured->assignmentOf(address->valueId()), coloured->assignmentOf(result->valueId()))
        << "if this stops overlapping, the rule may have become redundant";
}

TEST_F(Gfx1250AllocationRulesTest, ASingleDwordLoadIsNotInTheFamily) {
    // One DWORD returns all or nothing, so it can always replay and the rule
    // must not constrain it. Asserted through the rule rather than against the
    // predicate, which is private to the arch's TU.
    BasicBlock* entry = block("entry");
    StinkyInstruction* narrow = createSLoadB32(entry, 4, 0);
    StinkyInstruction* wide = createSLoadB64(entry, 6, 0);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationRules rules = gfx1250Rules(/*xnack=*/true);
    RuleOverrides force;
    force.activateAll = true;
    rules.force(force);

    EXPECT_EQ(rules.clobbersEarly(*narrow), nullptr);
    EXPECT_NE(rules.clobbersEarly(*wide), nullptr);
}

// ---------------------------------------------------------------------------
// ScalarTupleAlignment
// ---------------------------------------------------------------------------

TEST_F(Gfx1250AllocationRulesTest, AlignmentForbidsExactlyTheBasesTheAssemblerRejects) {
    const AllocationRules rules = gfx1250Rules(/*xnack=*/true);

    // A single SGPR sits anywhere.
    EXPECT_EQ(rules.forbidsBase(RegType::S, 1, 1), nullptr);
    EXPECT_EQ(rules.forbidsBase(RegType::S, 7, 1), nullptr);

    // A pair must be even.
    EXPECT_EQ(rules.forbidsBase(RegType::S, 2, 2), nullptr);
    ASSERT_NE(rules.forbidsBase(RegType::S, 1, 2), nullptr);
    EXPECT_EQ(rules.forbidsBase(RegType::S, 1, 2)->name, kAlignRule);

    // Quads and wider must be 4-aligned.
    EXPECT_EQ(rules.forbidsBase(RegType::S, 4, 4), nullptr);
    EXPECT_NE(rules.forbidsBase(RegType::S, 2, 4), nullptr);
    EXPECT_EQ(rules.forbidsBase(RegType::S, 8, 8), nullptr);
    EXPECT_NE(rules.forbidsBase(RegType::S, 6, 8), nullptr);
    EXPECT_EQ(rules.forbidsBase(RegType::S, 16, 16), nullptr);
    EXPECT_NE(rules.forbidsBase(RegType::S, 2, 16), nullptr);
}

TEST_F(Gfx1250AllocationRulesTest, AlignmentDoesNotConstrainVectorRegisters) {
    // gfx1250 has no alignment requirement on VGPR tuples, and the pipeline does
    // not allocate them anyway. Constraining them would cost registers for
    // nothing.
    const AllocationRules rules = gfx1250Rules(/*xnack=*/true);
    EXPECT_EQ(rules.forbidsBase(RegType::V, 1, 2), nullptr);
    EXPECT_EQ(rules.forbidsBase(RegType::V, 3, 4), nullptr);
}

TEST_F(Gfx1250AllocationRulesTest, AlignmentSurvivesWithoutAnyCapability) {
    // It is an encoding requirement, so no module configuration turns it off.
    const AllocationRules rules = gfx1250Rules(/*xnack=*/false);
    EXPECT_NE(rules.forbidsBase(RegType::S, 1, 2), nullptr);
}

TEST_F(Gfx1250AllocationRulesTest, ThePairMovesOffAnOddBase) {
    // The reported failure: compaction packs a scalar pair against a pinned
    // live-in and lands on an odd index, which the assembler rejects with
    // "invalid register alignment".
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createSLoadB64(entry, 40, 24);
    AsmIRBuilder builder(*entry, kRaTestArch);
    StinkyInstruction* use = builder.create(getMCIDByUOp(GFX::s_add_u32, kRaTestArch));
    use->addDestReg(StinkyRegister("s", 5, 1));
    use->addSrcReg(StinkyRegister("s", 40, 1));
    use->addSrcReg(StinkyRegister("s", 0, 1));  // s0 is a live-in, pinned, live across the load
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* first = ssaDefinedValue(*load, 0);
    ASSERT_NE(first, nullptr);

    AllocationSetup setup(*func, RegClassSet::only(RegType::S), {}, gfx1250Rules(/*xnack=*/true));
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> coloured = allocator.allocate(setup.context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    EXPECT_TRUE(verifyAllocation(*func, *coloured, setup.context()).ok());

    const RegKey base = coloured->assignmentOf(first->valueId());
    EXPECT_EQ(base.idx % 2, 0u) << "the pair landed on odd " << regKeyToString(base);
}

TEST_F(Gfx1250AllocationRulesTest, TheVerifierRejectsAMisalignedPair) {
    // The enforcement point: a policy that ignored alignment produces a refused
    // colouring rather than assembly the assembler will not accept.
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createSLoadB64(entry, 40, 24);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* first = ssaDefinedValue(*load, 0);
    const StinkySSAValue* second = ssaDefinedValue(*load, 1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    AllocationResult misaligned = createLegacyColoring(*func);
    misaligned.assign(first->valueId(), RegKey{RegType::S, 1, RegHalf::NONE});
    misaligned.assign(second->valueId(), RegKey{RegType::S, 2, RegHalf::NONE});

    AllocationSetup setup(*func, RegClassSet::only(RegType::S), {}, gfx1250Rules(/*xnack=*/true));
    const AllocationVerificationResult checked =
        verifyAllocation(*func, misaligned, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), kAlignRule)) << checked.toString();
}

TEST_F(Gfx1250AllocationRulesTest, TheAuditIsSilentOnAnAlignedProducerColouring) {
    // The producer's own registers assemble today, so activating alignment
    // refuses nothing that used to work.
    BasicBlock* entry = block("entry");
    createSLoadB64(entry, 40, 24);
    createSLoadB128(entry, 44, 26);
    ASSERT_TRUE(liftForAllocation(*func));

    EXPECT_TRUE(
        auditRules(*func, createLegacyColoring(*func), gfx1250Rules(/*xnack=*/true)).empty());
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

TEST_F(Gfx1250AllocationRulesTest, TheAuditReportsAProducerColouringThatAlreadyOverlaps) {
    // s[0:1] = s_load_b64(s[0:1]) -- the producer writes the register it reads,
    // which is exactly what the rule exists to prevent. Audit is how you find
    // out whether a corpus does this before switching the rule on.
    BasicBlock* entry = block("entry");
    createSLoadB64(entry, 0, 0);
    ASSERT_TRUE(liftForAllocation(*func));

    const std::vector<std::string> findings =
        auditRules(*func, createLegacyColoring(*func), gfx1250Rules(/*xnack=*/true));
    ASSERT_FALSE(findings.empty()) << "the self-overlap should have been reported";
    EXPECT_TRUE(findings.front().find(kSmemRule) != std::string::npos) << findings.front();
}

TEST_F(Gfx1250AllocationRulesTest, TheAuditIsSilentOnADisjointProducerColouring) {
    // What TensileLite already emits when EnableXnackReplay is set: the address
    // and the destination are deliberately kept apart.
    BasicBlock* entry = block("entry");
    createSLoadB64(entry, 4, 0);
    ASSERT_TRUE(liftForAllocation(*func));

    EXPECT_TRUE(
        auditRules(*func, createLegacyColoring(*func), gfx1250Rules(/*xnack=*/true)).empty());
}
