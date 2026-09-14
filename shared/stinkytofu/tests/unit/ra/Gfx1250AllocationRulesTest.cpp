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

#include <cstddef>
#include <limits>
#include <memory>
#include <set>
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
constexpr const char* kVectorAlignRule = "VectorTupleAlignment";

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

TEST_F(Gfx1250AllocationRulesTest, DeclaresEveryRuleWithTheRightKind) {
    const AllocationRules rules = gfx1250Rules(/*xnack=*/true);
    EXPECT_TRUE(rules.problems().empty()) << rules.toString();

    const AllocationRule* smem = findRule(rules, kSmemRule);
    ASSERT_NE(smem, nullptr) << rules.toString();
    EXPECT_FALSE(smem->description.empty());
    // Interference, not placement: the hardware constraint is about when the
    // access reads versus writes.
    EXPECT_EQ(smem->kind(), RuleKind::Interference);

    // Both alignment rows are about which index a tuple may start on, full stop.
    for (const char* name : {kAlignRule, kVectorAlignRule}) {
        const AllocationRule* align = findRule(rules, name);
        ASSERT_NE(align, nullptr) << name << " missing from " << rules.toString();
        EXPECT_FALSE(align->description.empty()) << name;
        EXPECT_EQ(align->kind(), RuleKind::Placement) << name;
    }
}

TEST_F(Gfx1250AllocationRulesTest, EveryRuleIsActive) {
    // The SMEM one was promoted after its audit came back silent; both alignment
    // rows are encoding requirements that were never optional. Demoting any of
    // them should have to change this line and say why.
    const AllocationRules rules = gfx1250Rules(/*xnack=*/true);
    for (const char* name : {kSmemRule, kAlignRule, kVectorAlignRule}) {
        ASSERT_NE(findRule(rules, name), nullptr) << name;
        EXPECT_EQ(findRule(rules, name)->status, RuleStatus::Active) << name;
    }
}

TEST_F(Gfx1250AllocationRulesTest, WithoutXnackReplayOnlyTheGatedRuleGoesInert) {
    // The gate is per rule, not per table. "Rule present, capability unset" and
    // "no rule" colour identically and want different fixes, so the gated one
    // must stay visible at Off.
    const AllocationRules rules = gfx1250Rules(/*xnack=*/false);
    ASSERT_NE(findRule(rules, kSmemRule), nullptr) << rules.toString();
    EXPECT_EQ(findRule(rules, kSmemRule)->status, RuleStatus::Off);

    // Both alignment rows are encoding requirements of every gfx1250 module, so
    // no capability can switch either off.
    for (const char* name : {kAlignRule, kVectorAlignRule}) {
        ASSERT_NE(findRule(rules, name), nullptr) << name;
        EXPECT_EQ(findRule(rules, name)->status, RuleStatus::Active) << name;
    }
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

// ---------------------------------------------------------------------------
// VectorTupleAlignment
// ---------------------------------------------------------------------------

TEST_F(Gfx1250AllocationRulesTest, VectorAlignmentForbidsExactlyTheBasesTheAssemblerRejects) {
    const AllocationRules rules = gfx1250Rules(/*xnack=*/true);

    // A single VGPR sits anywhere.
    EXPECT_EQ(rules.forbidsBase(RegType::V, 1, 1), nullptr);

    // Every width above one needs an even base and nothing more. Probed against
    // the assembler: v[2:9] and ds_load_b96 v[2:4] assemble, v[3:10] and v[3:5]
    // are rejected with "vgpr tuples must be 64 bit aligned". The odd bases are
    // the two the bf16 kernel emitted -- v[3:10] for a WMMA destination, v[9:12]
    // for a buffer_load_b128.
    for (uint32_t width : {2u, 3u, 4u, 8u, 16u}) {
        EXPECT_EQ(rules.forbidsBase(RegType::V, 2, width), nullptr) << "even, width " << width;
        for (uint32_t base : {3u, 9u}) {
            ASSERT_NE(rules.forbidsBase(RegType::V, base, width), nullptr)
                << "odd base " << base << ", width " << width;
            EXPECT_EQ(rules.forbidsBase(RegType::V, base, width)->name, kVectorAlignRule);
        }
    }

    // Flat where the scalar rule is width-based: the 4-DWORD tuple at base 2
    // accepted above is illegal for scalars, so neither rule's answer can stand
    // in for the other's. That is why these are two rows and not one predicate.
    EXPECT_NE(rules.forbidsBase(RegType::S, 2, 4), nullptr);
}

TEST_F(Gfx1250AllocationRulesTest, TheVectorRangeMovesOffAnOddBase) {
    // The reported failure, reduced. Lifting VGPRs put multi-DWORD vector ranges
    // in front of an allocator with no vector placement rule, and a 4-DWORD load
    // landed on v[9:12], which the assembler rejects. v0 is a live-in, so it is
    // pinned and leaves v1 as the lowest free index for compaction to reach for.
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createDsReadB128InBlock(entry, kRaTestArch, /*destReg=*/40,
                                                      /*addrReg=*/24);
    AsmIRBuilder builder(*entry, kRaTestArch);
    StinkyInstruction* use = builder.create(getMCIDByUOp(GFX::v_add_f32, kRaTestArch));
    use->addDestReg(StinkyRegister("v", 5, 1));
    use->addSrcReg(StinkyRegister("v", 40, 1));
    use->addSrcReg(StinkyRegister("v", 0, 1));  // live-in, pinned, live across the load
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* first = ssaDefinedValue(*load, 0);
    ASSERT_NE(first, nullptr);

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, gfx1250Rules(/*xnack=*/true));
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> coloured = allocator.allocate(setup.context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    EXPECT_TRUE(verifyAllocation(*func, *coloured, setup.context()).ok());

    const RegKey base = coloured->assignmentOf(first->valueId());
    EXPECT_EQ(base.idx % 2, 0u) << "the range landed on odd " << regKeyToString(base);
}

TEST_F(Gfx1250AllocationRulesTest, TheVerifierRejectsAMisalignedVectorRange) {
    // The enforcement point. Before this rule existed the verifier read the same
    // empty specification as the allocator and reported nothing, so a misaligned
    // range reached the assembler instead of being refused here.
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createDsReadB128InBlock(entry, kRaTestArch, /*destReg=*/40,
                                                      /*addrReg=*/24);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationResult misaligned = createLegacyColoring(*func);
    for (unsigned unit = 0; unit < 4; ++unit) {
        const StinkySSAValue* value = ssaDefinedValue(*load, unit);
        ASSERT_NE(value, nullptr) << "unit " << unit;
        misaligned.assign(value->valueId(), RegKey{RegType::V, 9 + unit, RegHalf::NONE});
    }

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, gfx1250Rules(/*xnack=*/true));
    const AllocationVerificationResult checked =
        verifyAllocation(*func, misaligned, setup.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), kVectorAlignRule)) << checked.toString();
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

// ---------------------------------------------------------------------------
// WmmaAccumulatorReuse
// ---------------------------------------------------------------------------

namespace {

/// `v_wmma_f32_16x16x32_bf16 dst, a, b, c` -- c is the src2 accumulator.
StinkyInstruction* createWmmaBf16(BasicBlock* bb, int dst, int a, int b, int c) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* wmma =
        builder.create(getMCIDByUOp(GFX::v_wmma_f32_16x16x32_bf16, kRaTestArch));
    wmma->addDestReg(StinkyRegister("v", dst, 8));
    wmma->addSrcReg(StinkyRegister("v", a, 8));
    wmma->addSrcReg(StinkyRegister("v", b, 8));
    wmma->addSrcReg(StinkyRegister("v", c, 8));
    return wmma;
}

/// Eight `v<base+i> = v_mov_b32 0`, so a tuple is function-defined and free to
/// move rather than a pinned live-in.
void defineTuple(BasicBlock* bb, int base) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    for (int unit = 0; unit < 8; ++unit) {
        StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
        mov->addDestReg(StinkyRegister("v", base + unit, 1));
        mov->addSrcReg(StinkyRegister(0));
    }
}

/// One read per DWORD of `v[base:base+count-1]`, so every member of a tuple has
/// a use and the block carries weight.
void useTuple(BasicBlock* bb, int base, int sink, int count = 8) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    for (int unit = 0; unit < count; ++unit) {
        StinkyInstruction* add = builder.create(getMCIDByUOp(GFX::v_add_f32, kRaTestArch));
        add->addDestReg(StinkyRegister("v", sink + unit, 1));
        add->addSrcReg(StinkyRegister("v", base + unit, 1));
        add->addSrcReg(StinkyRegister("v", base + unit, 1));
    }
}

/// `v[base:base+3] = ds_load_b128 v<addr>` -- a four-wide vector tuple that
/// competes for registers without asking for any pairing of its own.
void loadQuad(BasicBlock* bb, int base, int addr) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* load = builder.create(getMCIDByUOp(GFX::ds_load_b128, kRaTestArch));
    load->addDestReg(StinkyRegister("v", base, 4));
    load->addSrcReg(StinkyRegister("v", addr, 1));
}

/// The shipped table with the accumulator rule forced to a known status, so
/// these tests keep measuring the rule rather than whatever it ships as.
AllocationRules gfx1250RulesWithAccumulatorReuse(bool on = true) {
    AllocationRules rules = gfx1250Rules(/*xnack=*/true);
    RuleOverrides forced;
    if (on)
        forced.activate = {"WmmaAccumulatorReuse"};
    else
        forced.disable = {"WmmaAccumulatorReuse"};
    rules.force(forced);
    return rules;
}

}  // namespace

TEST_F(Gfx1250AllocationRulesTest, TheAccumulatorPairingNamesEveryDwordOfTheTuple) {
    // One preference per DWORD, destination unit i against accumulator unit i.
    // Pairing only the bases would leave the other seven registers free to
    // drift, and the rule would report itself satisfied having achieved little.
    BasicBlock* entry = block("entry");
    StinkyInstruction* wmma = createWmmaBf16(entry, /*dst=*/100, /*a=*/300, /*b=*/320, /*c=*/340);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {},
                          gfx1250RulesWithAccumulatorReuse());
    const std::span<const Preference> prefs = setup.constraints().preferences();
    ASSERT_EQ(prefs.size(), 8u) << "expected one pairing per DWORD of the tuple";

    const std::vector<StinkySSAValue*> accumulator = ssaSourceUnits(*wmma, 2);
    ASSERT_EQ(accumulator.size(), 8u);
    for (unsigned unit = 0; unit < 8; ++unit) {
        const StinkySSAValue* dest = ssaDefinedValue(*wmma, unit);
        ASSERT_NE(dest, nullptr) << "unit " << unit;
        ASSERT_NE(accumulator[unit], nullptr) << "unit " << unit;
        EXPECT_EQ(prefs[unit].a, dest->valueId()) << "unit " << unit;
        EXPECT_EQ(prefs[unit].b, accumulator[unit]->valueId()) << "unit " << unit;
    }
}

TEST_F(Gfx1250AllocationRulesTest, TheDestinationReusesTheAccumulatorOnlyWithTheRuleOn) {
    // The accumulator is a live-in pinned at v340 and dies at the WMMA, so its
    // registers are free for the destination afterwards. Packing from the
    // bottom has no reason to take them and puts the destination at v0; the
    // pairing is what sends it to v340 instead.
    BasicBlock* entry = block("entry");
    StinkyInstruction* wmma = createWmmaBf16(entry, /*dst=*/100, /*a=*/300, /*b=*/320, /*c=*/340);
    AsmIRBuilder builder(*entry, kRaTestArch);
    StinkyInstruction* use = builder.create(getMCIDByUOp(GFX::v_add_f32, kRaTestArch));
    use->addDestReg(StinkyRegister("v", 40, 1));
    use->addSrcReg(StinkyRegister("v", 100, 1));
    use->addSrcReg(StinkyRegister("v", 101, 1));
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* dest = ssaDefinedValue(*wmma, 0);
    ASSERT_NE(dest, nullptr);
    CompactingGreedyAllocator allocator;

    AllocationSetup off(*func, RegClassSet::only(RegType::V), {},
                        gfx1250RulesWithAccumulatorReuse(/*on=*/false));
    Expected<AllocationResult> without = allocator.allocate(off.context());
    ASSERT_TRUE(without.hasValue()) << without.getError();
    EXPECT_EQ(without->assignmentOf(dest->valueId()).idx, 0u);

    AllocationSetup on(*func, RegClassSet::only(RegType::V), {},
                       gfx1250RulesWithAccumulatorReuse());
    Expected<AllocationResult> with = allocator.allocate(on.context());
    ASSERT_TRUE(with.hasValue()) << with.getError();
    EXPECT_TRUE(verifyAllocation(*func, *with, on.context()).ok());
    EXPECT_EQ(with->assignmentOf(dest->valueId()).idx, 340u) << with->toString();
}

TEST_F(Gfx1250AllocationRulesTest, FoldingPairsAnAccumulatorScoringWouldHaveLost) {
    // The shape the rule loses to when it only scores. The accumulator is
    // placed first and packs low; a later tuple takes those registers once it
    // dies and outlives the destination; so by the time the destination is
    // placed, the register its preference names is no longer free for its
    // range and the pair comes apart.
    //
    // Folding the two before anything is placed removes the window: one block
    // covers both lives, so the later tuple never gets the chance.
    BasicBlock* entry = block("entry");
    defineTuple(entry, /*base=*/340);
    StinkyInstruction* wmma = createWmmaBf16(entry, /*dst=*/100, /*a=*/300, /*b=*/320, /*c=*/340);
    // Two loads, born where the accumulator dies and gone before the
    // destination is. Short and much used, so they are placed first.
    loadQuad(entry, /*base=*/200, /*addr=*/400);
    loadQuad(entry, /*base=*/204, /*addr=*/400);
    useTuple(entry, /*base=*/200, /*sink=*/700, /*count=*/4);
    useTuple(entry, /*base=*/204, /*sink=*/704, /*count=*/4);
    useTuple(entry, /*base=*/320, /*sink=*/740);  // filler, to lengthen the destination
    useTuple(entry, /*base=*/100, /*sink=*/720);  // the destination dies last
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* dest = ssaDefinedValue(*wmma, 0);
    const std::vector<StinkySSAValue*> accumulator = ssaSourceUnits(*wmma, 2);
    ASSERT_NE(dest, nullptr);
    ASSERT_EQ(accumulator.size(), 8u);
    ASSERT_NE(accumulator[0], nullptr);
    CompactingGreedyAllocator allocator;

    AllocationSetup off(*func, RegClassSet::only(RegType::V), {},
                        gfx1250RulesWithAccumulatorReuse(/*on=*/false));
    Expected<AllocationResult> without = allocator.allocate(off.context());
    ASSERT_TRUE(without.hasValue()) << without.getError();
    // If this ever ties on its own the test has stopped measuring anything.
    ASSERT_NE(without->assignmentOf(dest->valueId()).idx,
              without->assignmentOf(accumulator[0]->valueId()).idx)
        << without->toString();

    AllocationSetup on(*func, RegClassSet::only(RegType::V), {},
                       gfx1250RulesWithAccumulatorReuse());
    Expected<AllocationResult> with = allocator.allocate(on.context());
    ASSERT_TRUE(with.hasValue()) << with.getError();
    EXPECT_TRUE(verifyAllocation(*func, *with, on.context()).ok());
    EXPECT_EQ(with->assignmentOf(dest->valueId()).idx,
              with->assignmentOf(accumulator[0]->valueId()).idx)
        << with->toString();
}

namespace {

/// Three WMMAs accumulating in turn, with loads competing for the low
/// registers in between.
///
/// The competitors matter. Without them each accumulator dies exactly where
/// the next one is born, so first-fit hands base 0 down the whole chain and a
/// test would pass with the rule switched off. The loads take those registers
/// first, so a scattered chain is what happens unless the rule prevents it.
void buildCompetedChain(BasicBlock* entry, std::vector<StinkyInstruction*>& chain) {
    defineTuple(entry, /*base=*/340);
    defineTuple(entry, /*base=*/360);
    chain.push_back(createWmmaBf16(entry, /*dst=*/100, /*a=*/300, /*b=*/320, /*c=*/340));
    // A second, unrelated accumulator. It is eight wide, so both placement
    // policies weigh it against the chain rather than leaving it until last,
    // and it lives and dies inside the first link alone. That sends the first
    // link high while the two after it stay low.
    createWmmaBf16(entry, /*dst=*/200, /*a=*/300, /*b=*/320, /*c=*/360);
    useTuple(entry, /*base=*/200, /*sink=*/700);
    useTuple(entry, /*base=*/320, /*sink=*/740);  // filler, to lengthen the link
    chain.push_back(createWmmaBf16(entry, /*dst=*/108, /*a=*/300, /*b=*/320, /*c=*/100));
    chain.push_back(createWmmaBf16(entry, /*dst=*/116, /*a=*/300, /*b=*/320, /*c=*/108));
    useTuple(entry, /*base=*/116, /*sink=*/720);
}

/// How many distinct registers the chain head and every destination land on.
/// One when the whole chain folded, more when it came apart.
size_t basesUsedBy(const AllocationResult& coloured, StinkySSAValue* head,
                   const std::vector<StinkyInstruction*>& chain) {
    std::set<uint32_t> bases{coloured.assignmentOf(head->valueId()).idx};
    for (const StinkyInstruction* wmma : chain) {
        const StinkySSAValue* dest = ssaDefinedValue(*wmma, 0);
        if (dest != nullptr) bases.insert(coloured.assignmentOf(dest->valueId()).idx);
    }
    return bases.size();
}

}  // namespace

TEST_F(Gfx1250AllocationRulesTest, AnAccumulatorChainFoldsIntoOneBlock) {
    // Each pairing is checked against the block as it stands, so the second
    // link joins the block the first one built. That is what turns a run of
    // separate tuples back into the one register range the producer uses, and
    // pairing only the first link would not.
    BasicBlock* entry = block("entry");
    std::vector<StinkyInstruction*> chain;
    buildCompetedChain(entry, chain);
    ASSERT_TRUE(liftForAllocation(*func));

    const std::vector<StinkySSAValue*> head = ssaSourceUnits(*chain.front(), 2);
    ASSERT_EQ(head.size(), 8u);
    ASSERT_NE(head[0], nullptr);
    CompactingGreedyAllocator allocator;

    AllocationSetup off(*func, RegClassSet::only(RegType::V), {},
                        gfx1250RulesWithAccumulatorReuse(/*on=*/false));
    Expected<AllocationResult> without = allocator.allocate(off.context());
    ASSERT_TRUE(without.hasValue()) << without.getError();
    // If the chain ever lands on one register by itself, this test has stopped
    // measuring the rule.
    ASSERT_GT(basesUsedBy(*without, head[0], chain), 1u) << without->toString();

    AllocationSetup on(*func, RegClassSet::only(RegType::V), {},
                       gfx1250RulesWithAccumulatorReuse());
    Expected<AllocationResult> coloured = allocator.allocate(on.context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    EXPECT_TRUE(verifyAllocation(*func, *coloured, on.context()).ok());
    EXPECT_EQ(basesUsedBy(*coloured, head[0], chain), 1u) << coloured->toString();
}

TEST_F(Gfx1250AllocationRulesTest, TheChainFoldsUnderTheShippingAllocatorToo) {
    // greedy-compact-freedom is what the pipeline runs, and folding rewrites
    // placementFreedom, which is the number that policy orders on. Without this
    // the allocator that ships is covered by nothing.
    BasicBlock* entry = block("entry");
    std::vector<StinkyInstruction*> chain;
    buildCompetedChain(entry, chain);
    ASSERT_TRUE(liftForAllocation(*func));

    const std::vector<StinkySSAValue*> head = ssaSourceUnits(*chain.front(), 2);
    ASSERT_EQ(head.size(), 8u);
    ASSERT_NE(head[0], nullptr);
    FreedomOrderedGreedyAllocator allocator;

    AllocationSetup off(*func, RegClassSet::only(RegType::V), {},
                        gfx1250RulesWithAccumulatorReuse(/*on=*/false));
    Expected<AllocationResult> without = allocator.allocate(off.context());
    ASSERT_TRUE(without.hasValue()) << without.getError();
    ASSERT_GT(basesUsedBy(*without, head[0], chain), 1u) << without->toString();

    AllocationSetup on(*func, RegClassSet::only(RegType::V), {},
                       gfx1250RulesWithAccumulatorReuse());
    Expected<AllocationResult> coloured = allocator.allocate(on.context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    EXPECT_TRUE(verifyAllocation(*func, *coloured, on.context()).ok());
    EXPECT_EQ(basesUsedBy(*coloured, head[0], chain), 1u) << coloured->toString();
}

TEST_F(Gfx1250AllocationRulesTest, AnAccumulatorUnderACeilingIsNotFolded) {
    // The accumulator doubles as a scale operand, whose field selects no
    // s_set_vgpr_msb slot and so reaches the first bank only. That ceiling
    // belongs to those two registers. Folding would hand it to the
    // destination as well, and in a real kernel to a whole accumulator chain,
    // confining all of it to v0-v255 to buy one pairing. Decline instead and
    // let scoring try.
    BasicBlock* entry = block("entry");
    defineTuple(entry, /*base=*/340);
    // Before the WMMA, not after, so the accumulator still dies at the WMMA.
    // Read afterwards it would simply be live, and interference rather than
    // the ceiling would be what declined the fold.
    AsmIRBuilder builder(*entry, kRaTestArch);
    StinkyInstruction* scaled =
        builder.create(getMCIDByUOp(GFX::v_wmma_scale16_f32_16x16x128_f8f6f4, kRaTestArch));
    scaled->addDestReg(StinkyRegister("v", 500, 8));
    scaled->addSrcReg(StinkyRegister("v", 510, 8));
    scaled->addSrcReg(StinkyRegister("v", 520, 8));
    scaled->addSrcReg(StinkyRegister(0));
    scaled->addSrcReg(StinkyRegister("v", 340, 2));  // the accumulator, as a scale
    // Low, because a live-in scale operand has to start somewhere its own
    // field can reach or the colouring refuses before folding is even asked.
    scaled->addSrcReg(StinkyRegister("v", 10, 2));
    StinkyInstruction* wmma = createWmmaBf16(entry, /*dst=*/100, /*a=*/300, /*b=*/320, /*c=*/340);
    // The same competitors as the test above. Without them the destination
    // reuses the accumulator's registers by ordinary first-fit, and the two
    // colourings look alike whether folding happened or not.
    loadQuad(entry, /*base=*/200, /*addr=*/400);
    loadQuad(entry, /*base=*/204, /*addr=*/400);
    useTuple(entry, /*base=*/200, /*sink=*/700, /*count=*/4);
    useTuple(entry, /*base=*/204, /*sink=*/704, /*count=*/4);
    useTuple(entry, /*base=*/320, /*sink=*/740);
    useTuple(entry, /*base=*/100, /*sink=*/720);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* dest = ssaDefinedValue(*wmma, 0);
    const std::vector<StinkySSAValue*> accumulator = ssaSourceUnits(*wmma, 2);
    ASSERT_NE(dest, nullptr);
    ASSERT_EQ(accumulator.size(), 8u);
    ASSERT_NE(accumulator[0], nullptr);

    AllocationSetup on(*func, RegClassSet::only(RegType::V), {},
                       gfx1250RulesWithAccumulatorReuse());
    // The fixture is only worth anything while the ceiling is really there.
    ASSERT_LT(on.constraints().maxIndexFor(accumulator[0]->valueId()), 1024u);
    ASSERT_EQ(on.constraints().maxIndexFor(dest->valueId()), std::numeric_limits<uint32_t>::max());

    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> coloured = allocator.allocate(on.context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    EXPECT_TRUE(verifyAllocation(*func, *coloured, on.context()).ok());
    EXPECT_NE(coloured->assignmentOf(dest->valueId()).idx,
              coloured->assignmentOf(accumulator[0]->valueId()).idx)
        << coloured->toString();
}

TEST_F(Gfx1250AllocationRulesTest, AnAccumulatorReadAgainLaterIsNotFolded) {
    // The accumulator is read after the WMMA, so the two are live together and
    // one register cannot hold both. Folding has to decline this: the rule is a
    // preference, and a preference that overrode liveness would miscompile.
    BasicBlock* entry = block("entry");
    defineTuple(entry, /*base=*/340);
    StinkyInstruction* wmma = createWmmaBf16(entry, /*dst=*/100, /*a=*/300, /*b=*/320, /*c=*/340);
    useTuple(entry, /*base=*/340, /*sink=*/700);  // the accumulator, read again
    useTuple(entry, /*base=*/100, /*sink=*/720);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* dest = ssaDefinedValue(*wmma, 0);
    const std::vector<StinkySSAValue*> accumulator = ssaSourceUnits(*wmma, 2);
    ASSERT_NE(dest, nullptr);
    ASSERT_EQ(accumulator.size(), 8u);
    ASSERT_NE(accumulator[0], nullptr);

    CompactingGreedyAllocator allocator;
    AllocationSetup on(*func, RegClassSet::only(RegType::V), {},
                       gfx1250RulesWithAccumulatorReuse());
    Expected<AllocationResult> coloured = allocator.allocate(on.context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    EXPECT_TRUE(verifyAllocation(*func, *coloured, on.context()).ok());
    EXPECT_NE(coloured->assignmentOf(dest->valueId()).idx,
              coloured->assignmentOf(accumulator[0]->valueId()).idx)
        << coloured->toString();
}

TEST_F(Gfx1250AllocationRulesTest, FollowingHintsStillReproducesTheProducer) {
    // Folding is off wherever hints are on. The producer's numbering is what
    // that mode exists to reproduce, and a folded pair the producer kept apart
    // has no agreed hint left to follow.
    BasicBlock* entry = block("entry");
    defineTuple(entry, /*base=*/340);
    StinkyInstruction* wmma = createWmmaBf16(entry, /*dst=*/100, /*a=*/300, /*b=*/320, /*c=*/340);
    useTuple(entry, /*base=*/100, /*sink=*/700);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* dest = ssaDefinedValue(*wmma, 0);
    ASSERT_NE(dest, nullptr);

    GreedyAllocator allocator;
    AllocationSetup on(*func, RegClassSet::only(RegType::V), {},
                       gfx1250RulesWithAccumulatorReuse());
    Expected<AllocationResult> coloured = allocator.allocate(on.context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    EXPECT_EQ(coloured->assignmentOf(dest->valueId()).idx, 100u) << coloured->toString();
}
