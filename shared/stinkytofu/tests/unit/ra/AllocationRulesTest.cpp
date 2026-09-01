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
// Every case builds its own one- or two-row table. No architecture declares
// rules, and even once one does, a test leaning on it would break the moment
// that rule was promoted or retired.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRules.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationVerifier.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "transforms/asm/ra/allocators/GreedyAllocator.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

// --- The tables under test. Each is one row: a name, a status, and the single
// function that says what the rule does.

/// Two placement rules, so the first in declaration order is the one named.
AllocationRules twoPlacementRules(RuleStatus first, RuleStatus second) {
    AllocationRule even;
    even.name = "EvenVBase";
    even.description = "a V block must start even";
    even.status = first;
    even.forbidsBase = [](RegType regClass, uint32_t base, uint32_t) {
        return regClass == RegType::V && base % 2 != 0;
    };

    AllocationRule low;
    low.name = "LowVBase";
    low.description = "a V block must not start below 4";
    low.status = second;
    low.forbidsBase = [](RegType regClass, uint32_t base, uint32_t) {
        return regClass == RegType::V && base < 4;
    };
    return AllocationRules({even, low});
}

AllocationRules nothingIsLegal() {
    AllocationRule rule;
    rule.name = "NothingLegal";
    rule.description = "no base is ever legal";
    rule.status = RuleStatus::Active;
    rule.forbidsBase = [](RegType, uint32_t, uint32_t) { return true; };
    return AllocationRules({rule});
}

AllocationRules forcedAffinity(SSAValueID a, SSAValueID b) {
    AllocationRule rule;
    rule.name = "MustShare";
    rule.description = "these two values sit on one register";
    rule.status = RuleStatus::Active;
    rule.addRelations = [a, b](const Function&, std::vector<TupleRun>&,
                               std::vector<AffinitySet>& affinitySets) {
        affinitySets.push_back(AffinitySet{{a, b}});
    };
    return AllocationRules({rule});
}

AllocationRules earlyClobberOf(const StinkyInstruction* target,
                               RuleStatus status = RuleStatus::Active) {
    AllocationRule rule;
    rule.name = "EarlyClobber";
    rule.description = "this instruction writes before it finishes reading";
    rule.status = status;
    rule.clobbersEarly = [target](const StinkyInstruction& inst) { return &inst == target; };
    return AllocationRules({rule});
}

/// A preference, not a veto: every base is legal, odd ones just cost more.
AllocationRules prefersEvenBases(RuleStatus status = RuleStatus::Active) {
    AllocationRule rule;
    rule.name = "PreferEvenVBase";
    rule.description = "a V block is cheaper on an even index";
    rule.status = status;
    rule.baseCost = [](RegType regClass, uint32_t base, uint32_t) {
        if (regClass != RegType::V) return 0.0;
        return base % 2 == 0 ? 0.0 : 1.0;
    };
    return AllocationRules({rule});
}

class AllocationRulesTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
    }

    BasicBlock* block(const std::string& label) {
        return func->createBasicBlock(label);
    }

    AllocationResult colour(AllocationSetup& setup) {
        GreedyAllocator allocator;
        Expected<AllocationResult> result = allocator.allocate(setup.context());
        EXPECT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
        if (!result.hasValue()) return AllocationResult{};
        const AllocationVerificationResult checked =
            verifyAllocation(*func, *result, setup.context());
        EXPECT_TRUE(checked.ok()) << checked.toString();
        return std::move(*result);
    }

    std::string colourError(AllocationSetup& setup) {
        GreedyAllocator allocator;
        Expected<AllocationResult> result = allocator.allocate(setup.context());
        EXPECT_TRUE(result.hasError());
        return result.hasError() ? result.getError() : std::string{};
    }

    std::unique_ptr<Function> func;
};

}  // namespace

// ---------------------------------------------------------------------------
// The empty table, which is what an unregistered triple still gets
// ---------------------------------------------------------------------------

TEST_F(AllocationRulesTest, AnEmptyTableRequiresNothing) {
    const AllocationRules none;
    EXPECT_TRUE(none.empty());
    EXPECT_TRUE(none.problems().empty());
    EXPECT_EQ(none.forbidsBase(RegType::V, 3, 2), nullptr);
    EXPECT_EQ(none.baseCost(RegType::V, 3, 2), 0.0);
    EXPECT_FALSE(none.prices());
}

TEST_F(AllocationRulesTest, WithNoRulesTheColouringIsUnchanged) {
    // The acceptance criterion for the whole framework: with nothing declared,
    // greedy reproduces exactly what it produced before.
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 8, 4, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func);
    EXPECT_EQ(colour(setup).toString(), createLegacyColoring(*func).toString());
}

// ---------------------------------------------------------------------------
// The table itself
// ---------------------------------------------------------------------------

TEST_F(AllocationRulesTest, KindIsDerivedFromWhichFunctionIsSet) {
    // Derived rather than declared, so it cannot disagree with what the rule
    // actually does.
    EXPECT_EQ(evenVBasesOnly().all()[0].kind(), RuleKind::Placement);
    EXPECT_EQ(earlyClobberOf(nullptr).all()[0].kind(), RuleKind::Interference);
    EXPECT_EQ(forcedAffinity(1, 2).all()[0].kind(), RuleKind::Offset);
    EXPECT_EQ(prefersEvenBases().all()[0].kind(), RuleKind::Preference);
}

TEST_F(AllocationRulesTest, ARowWithNoFunctionIsReportedNotSilentlyKept) {
    AllocationRule empty;
    empty.name = "DoesNothing";
    empty.description = "no function filled in";
    empty.status = RuleStatus::Active;

    const AllocationRules rules({empty});
    EXPECT_TRUE(rules.empty());
    ASSERT_EQ(rules.problems().size(), 1u);
    EXPECT_TRUE(contains(rules.problems()[0], "DoesNothing")) << rules.problems()[0];
}

TEST_F(AllocationRulesTest, ARowWithTwoFunctionsIsReported) {
    // A rule that is both a veto and a price would be invisible to the
    // lifecycle and to diagnostics, so the table refuses it.
    AllocationRule both;
    both.name = "VetoAndPrice";
    both.description = "fills in two functions";
    both.status = RuleStatus::Active;
    both.forbidsBase = [](RegType, uint32_t, uint32_t) { return false; };
    both.baseCost = [](RegType, uint32_t, uint32_t) { return 1.0; };

    const AllocationRules rules({both});
    EXPECT_TRUE(rules.empty());
    ASSERT_EQ(rules.problems().size(), 1u);
    EXPECT_TRUE(contains(rules.problems()[0], "exactly one")) << rules.problems()[0];
}

TEST_F(AllocationRulesTest, APreferenceCannotBeAudited) {
    // Audit asks "does the input already violate this", which presumes a
    // violation. A preference has none, so Audit would be silently inert.
    const AllocationRules rules = prefersEvenBases(RuleStatus::Audit);
    ASSERT_EQ(rules.all().size(), 1u);
    EXPECT_EQ(rules.all()[0].status, RuleStatus::Off);
    ASSERT_EQ(rules.problems().size(), 1u);
    EXPECT_TRUE(contains(rules.problems()[0], "no Audit state")) << rules.problems()[0];
}

TEST_F(AllocationRulesTest, OnlyActiveRulesAnswerAQuery) {
    // The single place status is consulted, which is why no rule tests its own.
    EXPECT_NE(evenVBasesOnly(RuleStatus::Active).forbidsBase(RegType::V, 3, 1), nullptr);
    EXPECT_EQ(evenVBasesOnly(RuleStatus::Audit).forbidsBase(RegType::V, 3, 1), nullptr);
    EXPECT_EQ(evenVBasesOnly(RuleStatus::Off).forbidsBase(RegType::V, 3, 1), nullptr);
}

TEST_F(AllocationRulesTest, AQueryNamesTheFirstMatchInDeclarationOrder) {
    const AllocationRules rules = twoPlacementRules(RuleStatus::Active, RuleStatus::Active);
    // Base 3 breaks both; the earlier row is named, so switching that rule off
    // is a change the user can actually observe.
    ASSERT_NE(rules.forbidsBase(RegType::V, 3, 1), nullptr);
    EXPECT_EQ(rules.forbidsBase(RegType::V, 3, 1)->name, "EvenVBase");
    ASSERT_NE(rules.forbidsBase(RegType::V, 2, 1), nullptr);
    EXPECT_EQ(rules.forbidsBase(RegType::V, 2, 1)->name, "LowVBase");
    EXPECT_EQ(rules.forbidsBase(RegType::V, 6, 1), nullptr);
}

TEST_F(AllocationRulesTest, AnInactiveRuleCannotMaskAnActiveOne) {
    const AllocationRules rules = twoPlacementRules(RuleStatus::Off, RuleStatus::Active);
    // Base 3 breaks both, but only the second is on, so that is what answers.
    ASSERT_NE(rules.forbidsBase(RegType::V, 3, 1), nullptr);
    EXPECT_EQ(rules.forbidsBase(RegType::V, 3, 1)->name, "LowVBase");
}

TEST_F(AllocationRulesTest, ToStringReportsKindAndStatus) {
    const std::string text = evenVBasesOnly(RuleStatus::Off).toString();
    EXPECT_TRUE(contains(text, "EvenVBase")) << text;
    EXPECT_TRUE(contains(text, "placement")) << text;
    EXPECT_TRUE(contains(text, "off")) << text;
}

// ---------------------------------------------------------------------------
// Placement rules
// ---------------------------------------------------------------------------

TEST_F(AllocationRulesTest, PlacementSkipsAForbiddenBase) {
    BasicBlock* entry = block("entry");
    // Sources defined inside the function, so nothing is pinned at an odd index
    // and only the destination has to move off v3.
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 3, 4, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* defined = ssaDefinedValue(*add);
    ASSERT_NE(defined, nullptr);

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, evenVBasesOnly());
    const RegKey destination = colour(setup).assignmentOf(defined->valueId());
    EXPECT_NE(destination.idx, 3u) << "the forbidden hint must be passed over";
    EXPECT_EQ(destination.idx % 2, 0u) << regKeyToString(destination);
}

TEST_F(AllocationRulesTest, PlacementAtAuditVetoesNothing) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 3, 4, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {},
                          evenVBasesOnly(RuleStatus::Audit));
    EXPECT_EQ(colour(setup).toString(), createLegacyColoring(*func).toString());
}

TEST_F(AllocationRulesTest, PlacementBlocksEvictionToo) {
    // reachableAt is the single funnel, so a forbidden base must be unreachable
    // by eviction as well, not just by placement.
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 3, 0, 2);
    createVAddInBlock(entry, kRaTestArch, 5, 3, 0);
    createVAddInBlock(entry, kRaTestArch, 7, 5, 3);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, evenVBasesOnly());
    const uint32_t count = setup.target().indexCount(RegType::V);
    ASSERT_GT(count, 8u);
    setup.target().reserve(RegType::V, 8, count - 8);

    // Every block here is width 1, so every assignment is also a block base.
    const AllocationResult result = colour(setup);
    for (StinkySSAValue* value : func->ssaArena().values()) {
        if (value == nullptr || !result.isAssigned(value->valueId())) continue;
        const RegKey physical = result.assignmentOf(value->valueId());
        if (physical.type != RegType::V) continue;
        EXPECT_EQ(physical.idx % 2, 0u) << regKeyToString(physical);
    }
}

TEST_F(AllocationRulesTest, PlacementConstrainsABlockBaseNotEveryMember) {
    // A placement rule speaks about where a block *starts*. A 4-DWORD tuple at
    // an even base still puts members on odd indexes, and that is not a
    // violation -- the rule never claimed otherwise.
    BasicBlock* entry = block("entry");
    StinkyInstruction* load = createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 8, 4, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* firstUnit = ssaDefinedValue(*load, 0);
    const StinkySSAValue* secondUnit = ssaDefinedValue(*load, 1);
    ASSERT_NE(firstUnit, nullptr);
    ASSERT_NE(secondUnit, nullptr);

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, evenVBasesOnly());
    const AllocationResult result = colour(setup);

    const uint32_t base = result.assignmentOf(firstUnit->valueId()).idx;
    EXPECT_EQ(base % 2, 0u) << "the run must start even";
    EXPECT_EQ(result.assignmentOf(secondUnit->valueId()).idx, base + 1)
        << "and its members stay consecutive regardless";
}

TEST_F(AllocationRulesTest, ABlockNoBaseCanSatisfyNamesTheRule) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, nothingIsLegal());
    const std::string error = colourError(setup);
    EXPECT_TRUE(contains(error, "NothingLegal")) << error;
    EXPECT_TRUE(contains(error, "base is legal")) << error;
}

TEST_F(AllocationRulesTest, PlacementAgainstAPinnedLiveInLeavesTheKernelUncoloured) {
    // A function live-in cannot move, so a rule forbidding the register it
    // arrives in has no repair. That is why a new rule audits before it goes
    // Active.
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 2, 0, 1);  // v1 is a live-in at an odd index
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, evenVBasesOnly());
    const std::string error = colourError(setup);
    EXPECT_TRUE(contains(error, "live-in")) << error;
}

TEST_F(AllocationRulesTest, TheVerifierRejectsAPlacementViolation) {
    // The enforcement point. Honouring lives in the allocator, but a policy that
    // ignores a rule has to be caught, or the rule would not bind every policy.
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 3, 0, 2);
    ASSERT_TRUE(liftForAllocation(*func));

    const AllocationResult producer = createLegacyColoring(*func);

    AllocationSetup permissive(*func);
    EXPECT_TRUE(verifyAllocation(*func, producer, permissive.context()).ok());

    AllocationSetup strict(*func, RegClassSet::only(RegType::V), {}, evenVBasesOnly());
    const AllocationVerificationResult checked =
        verifyAllocation(*func, producer, strict.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "EvenVBase")) << checked.toString();
}

TEST_F(AllocationRulesTest, TheVerifierAcceptsOddTupleMembers) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, evenVBasesOnly());
    const AllocationVerificationResult checked =
        verifyAllocation(*func, createLegacyColoring(*func), setup.context());
    EXPECT_TRUE(checked.ok()) << checked.toString();
}

// ---------------------------------------------------------------------------
// Offset rules
// ---------------------------------------------------------------------------

TEST_F(AllocationRulesTest, OffsetRelationIsHonouredAndVerified) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* first = createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    StinkyInstruction* second = createVAddInBlock(entry, kRaTestArch, 3, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* a = ssaDefinedValue(*first);
    const StinkySSAValue* b = ssaDefinedValue(*second);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {},
                          forcedAffinity(a->valueId(), b->valueId()));
    // The relation reached the constraints through build().
    EXPECT_EQ(setup.constraints().affinitySets().size(), 1u);

    const AllocationResult result = colour(setup);
    EXPECT_EQ(result.assignmentOf(a->valueId()), result.assignmentOf(b->valueId()));
}

TEST_F(AllocationRulesTest, AnOffsetRuleAtOffContributesNothing) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* first = createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    StinkyInstruction* second = createVAddInBlock(entry, kRaTestArch, 3, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationRules rules =
        forcedAffinity(ssaDefinedValue(*first)->valueId(), ssaDefinedValue(*second)->valueId());
    RuleOverrides off;
    off.disableAll = true;
    rules.force(off);

    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {}, std::move(rules));
    EXPECT_TRUE(setup.constraints().affinitySets().empty());
}

TEST_F(AllocationRulesTest, AnUnsatisfiableOffsetRelationIsReported) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* src0 = ssaSourceValue(*add, 0);
    const StinkySSAValue* src1 = ssaSourceValue(*add, 1);
    ASSERT_NE(src0, nullptr);
    ASSERT_NE(src1, nullptr);

    // Both live into the same instruction, so demanding they share is
    // impossible. A rule-imposed relation gets the same treatment an
    // IR-derived one would.
    AllocationSetup setup(*func, RegClassSet::only(RegType::V), {},
                          forcedAffinity(src0->valueId(), src1->valueId()));
    const std::string error = colourError(setup);
    EXPECT_TRUE(contains(error, "live at the same point") || contains(error, "one register"))
        << error;
}

// ---------------------------------------------------------------------------
// Interference rules
// ---------------------------------------------------------------------------

TEST_F(AllocationRulesTest, EarlyClobberMakesADyingSourceOverlapTheDestination) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 8, 4, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* dying = ssaSourceValue(*add, 0);
    const StinkySSAValue* defined = ssaDefinedValue(*add);
    ASSERT_NE(dying, nullptr);
    ASSERT_NE(defined, nullptr);

    const SSALiveIntervals base = computeSSALiveIntervals(*func);
    // Half-open ranges make these touch without overlapping, which is what lets
    // a normal instruction reuse its dying source's register.
    ASSERT_FALSE(base.overlap(dying->valueId(), defined->valueId()));

    const SSALiveIntervals widened = applyEarlyClobber(*func, base, earlyClobberOf(add));
    EXPECT_TRUE(widened.overlap(dying->valueId(), defined->valueId()));

    // The pure set is untouched, which keeps the shadow report's pressure floor
    // a property of the program.
    EXPECT_EQ(base.toString(), computeSSALiveIntervals(*func).toString());
}

TEST_F(AllocationRulesTest, EarlyClobberAtAuditWidensNothing) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 8, 4, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    const SSALiveIntervals base = computeSSALiveIntervals(*func);
    EXPECT_EQ(applyEarlyClobber(*func, base, earlyClobberOf(add, RuleStatus::Audit)).toString(),
              base.toString());
}

TEST_F(AllocationRulesTest, EarlyClobberRefusesTheReuseItForbids) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* defined = ssaDefinedValue(*add);
    ASSERT_NE(defined, nullptr);

    // Only v0 and v1 exist, and both are pinned live-ins dying at the add, so
    // the destination has to reuse one of their registers: fine normally,
    // impossible once the rule makes it overlap them.
    auto squeeze = [](AllocationSetup& setup) {
        const uint32_t count = setup.target().indexCount(RegType::V);
        setup.target().reserve(RegType::V, 2, count - 2);
    };

    AllocationSetup permissive(*func);
    squeeze(permissive);
    GreedyAllocator allocator;
    const Expected<AllocationResult> reused = allocator.allocate(permissive.context());
    ASSERT_TRUE(reused.hasValue()) << reused.getError();
    EXPECT_LT(reused->assignmentOf(defined->valueId()).idx, 2u);

    AllocationSetup strict(*func, RegClassSet::only(RegType::V), {}, earlyClobberOf(add));
    squeeze(strict);
    EXPECT_TRUE(contains(colourError(strict), "register is free")) << colourError(strict);
}

TEST_F(AllocationRulesTest, TheVerifierRejectsAnEarlyClobberOverlap) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* defined = ssaDefinedValue(*add);
    const StinkySSAValue* src0 = ssaSourceValue(*add, 0);
    ASSERT_NE(defined, nullptr);
    ASSERT_NE(src0, nullptr);

    AllocationResult sharing = createLegacyColoring(*func);
    sharing.assign(defined->valueId(), sharing.assignmentOf(src0->valueId()));

    AllocationSetup permissive(*func);
    EXPECT_TRUE(verifyAllocation(*func, sharing, permissive.context()).ok());

    AllocationSetup strict(*func, RegClassSet::only(RegType::V), {}, earlyClobberOf(add));
    const AllocationVerificationResult checked = verifyAllocation(*func, sharing, strict.context());
    EXPECT_FALSE(checked.ok());
    EXPECT_TRUE(contains(checked.toString(), "overlaps")) << checked.toString();
}

// ---------------------------------------------------------------------------
// Preferences
// ---------------------------------------------------------------------------

TEST_F(AllocationRulesTest, APreferenceIsAdvisoryNotALegalityCheck) {
    const AllocationRules rules = prefersEvenBases();
    EXPECT_TRUE(rules.prices());
    EXPECT_EQ(rules.forbidsBase(RegType::V, 3, 1), nullptr) << "a price must never veto";
    EXPECT_EQ(rules.baseCost(RegType::V, 2, 1), 0.0);
    EXPECT_EQ(rules.baseCost(RegType::V, 3, 1), 1.0);
}

TEST_F(AllocationRulesTest, PricesIsFalseUnlessAPreferenceIsActive) {
    // What lets a chip with no preference keep plain first-fit, and today's
    // colouring, at no cost.
    EXPECT_FALSE(AllocationRules().prices());
    EXPECT_FALSE(evenVBasesOnly().prices());
    EXPECT_FALSE(prefersEvenBases(RuleStatus::Off).prices());
    EXPECT_TRUE(prefersEvenBases().prices());
}

TEST_F(AllocationRulesTest, APreferenceChangesPlacementWithoutForbiddingAnything) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 3, 4, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    const StinkySSAValue* defined = ssaDefinedValue(*add);
    ASSERT_NE(defined, nullptr);

    // Hints win outright over a preference: following the producer reproduces
    // its numbering, which the whole shadow workflow rests on.
    AllocationSetup hinted(*func, RegClassSet::only(RegType::V), {}, prefersEvenBases());
    EXPECT_EQ(colour(hinted).assignmentOf(defined->valueId()).idx, 3u);

    // Without hints the price decides, and it picks an even index rather than
    // the lowest free one.
    AllocationSetup compact(*func, RegClassSet::only(RegType::V), {}, prefersEvenBases());
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> result = allocator.allocate(compact.context());
    ASSERT_TRUE(result.hasValue()) << result.getError();
    EXPECT_EQ(result->assignmentOf(defined->valueId()).idx % 2, 0u)
        << regKeyToString(result->assignmentOf(defined->valueId()));
}

TEST_F(AllocationRulesTest, AZeroCostPreferenceReproducesTheFirstFitColouring) {
    // Pricing must not perturb a colouring by itself: with a flat cost the
    // min-cost scan has to land exactly where first-fit did.
    AllocationRule flat;
    flat.name = "FlatCost";
    flat.description = "every base costs the same";
    flat.status = RuleStatus::Active;
    flat.baseCost = [](RegType, uint32_t, uint32_t) { return 2.0; };

    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 3, 4, 5);
    ASSERT_TRUE(liftForAllocation(*func));

    AllocationSetup priced(*func, RegClassSet::only(RegType::V), {}, AllocationRules({flat}));
    AllocationSetup plain(*func);
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> withPrice = allocator.allocate(priced.context());
    Expected<AllocationResult> without = allocator.allocate(plain.context());
    ASSERT_TRUE(withPrice.hasValue() && without.hasValue());
    EXPECT_EQ(withPrice->toString(), without->toString());
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

TEST_F(AllocationRulesTest, AuditFindsAProducerViolationWhateverTheStatus) {
    BasicBlock* entry = block("entry");
    // v2 = v_add(v2, v1): the producer already writes a register it reads.
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 2, 2, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    const std::vector<std::string> findings =
        auditRules(*func, createLegacyColoring(*func), earlyClobberOf(add, RuleStatus::Off));
    ASSERT_FALSE(findings.empty());
    EXPECT_TRUE(contains(findings.front(), "EarlyClobber")) << findings.front();
    EXPECT_TRUE(contains(findings.front(), "share")) << findings.front();
}

TEST_F(AllocationRulesTest, AuditFindsAPlacementViolation) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 3, 0, 2);
    ASSERT_TRUE(liftForAllocation(*func));

    const std::vector<std::string> findings =
        auditRules(*func, createLegacyColoring(*func), evenVBasesOnly(RuleStatus::Off));
    ASSERT_FALSE(findings.empty());
    EXPECT_TRUE(contains(findings.front(), "EvenVBase")) << findings.front();
}

TEST_F(AllocationRulesTest, AuditIsSilentOnACleanProducerColouring) {
    BasicBlock* entry = block("entry");
    StinkyInstruction* add = createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    EXPECT_TRUE(auditRules(*func, createLegacyColoring(*func), earlyClobberOf(add)).empty());
}

TEST_F(AllocationRulesTest, AuditSaysNothingAboutAPreference) {
    // Paying a preference produces a working kernel that is merely slower, which
    // the shadow report already measures.
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 3, 0, 2);
    ASSERT_TRUE(liftForAllocation(*func));

    EXPECT_TRUE(auditRules(*func, createLegacyColoring(*func), prefersEvenBases()).empty());
}

TEST_F(AllocationRulesTest, AuditIsSilentWithNoRules) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 2, 2, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    EXPECT_TRUE(auditRules(*func, createLegacyColoring(*func), AllocationRules{}).empty());
}

// ---------------------------------------------------------------------------
// Overrides
// ---------------------------------------------------------------------------

TEST_F(AllocationRulesTest, OverridesActivateByName) {
    AllocationRules rules = evenVBasesOnly(RuleStatus::Off);
    RuleOverrides overrides;
    overrides.activate.push_back("EvenVBase");

    EXPECT_TRUE(rules.unknownNames(overrides).empty());
    rules.force(overrides);
    EXPECT_EQ(rules.all()[0].status, RuleStatus::Active);
    EXPECT_NE(rules.forbidsBase(RegType::V, 3, 1), nullptr);
}

TEST_F(AllocationRulesTest, OverridesRejectAnUnknownName) {
    // A misspelling has to be an error: a test that enables nothing still
    // passes, which is the worst possible way to find out.
    const AllocationRules rules = evenVBasesOnly(RuleStatus::Off);
    RuleOverrides overrides;
    overrides.activate.push_back("EvenVBse");

    const std::vector<std::string> unknown = rules.unknownNames(overrides);
    ASSERT_EQ(unknown.size(), 1u);
    EXPECT_EQ(unknown.front(), "EvenVBse");
}

TEST_F(AllocationRulesTest, OverridesCanActivateOrAuditEverything) {
    AllocationRules activated = twoPlacementRules(RuleStatus::Off, RuleStatus::Off);
    RuleOverrides all;
    all.activateAll = true;
    activated.force(all);
    EXPECT_EQ(activated.all()[0].status, RuleStatus::Active);
    EXPECT_EQ(activated.all()[1].status, RuleStatus::Active);

    AllocationRules audited = twoPlacementRules(RuleStatus::Off, RuleStatus::Off);
    RuleOverrides audit;
    audit.auditAll = true;
    audited.force(audit);
    EXPECT_EQ(audited.all()[0].status, RuleStatus::Audit);
    EXPECT_EQ(audited.all()[1].status, RuleStatus::Audit);
}

TEST_F(AllocationRulesTest, AuditAllSkipsPreferences) {
    AllocationRules rules = prefersEvenBases(RuleStatus::Off);
    RuleOverrides audit;
    audit.auditAll = true;
    rules.force(audit);
    EXPECT_EQ(rules.all()[0].status, RuleStatus::Off) << "a preference has no Audit state";
}

TEST_F(AllocationRulesTest, DisableAllEmptiesTheTable) {
    AllocationRules rules = evenVBasesOnly();
    RuleOverrides overrides;
    overrides.disableAll = true;
    rules.force(overrides);

    EXPECT_TRUE(rules.empty());
    EXPECT_EQ(rules.forbidsBase(RegType::V, 3, 1), nullptr);
    EXPECT_FALSE(rules.prices());
}
