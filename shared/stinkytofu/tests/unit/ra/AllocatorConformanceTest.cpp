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

// Every registered policy must colour these shapes legally.
//
// Parameterized over registeredAllocatorNames(), so a later allocator inherits
// the whole suite from its one registration line, and a failure names the policy
// that produced it.

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationVerifier.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class AllocatorConformanceTest : public ::testing::TestWithParam<std::string> {
   protected:
    void SetUp() override {
        allocator_ = AllocatorRegistry::createAllocator(GetParam());
        ASSERT_NE(allocator_, nullptr) << "allocator '" << GetParam() << "' is not registered";
        func_ = std::make_unique<Function>("kernel");
        setFunctionArch(*func_, kRaTestArch);
    }

    BasicBlock* block(const std::string& label) {
        return func_->createBasicBlock(label);
    }

    Function& function() {
        return *func_;
    }

    /// Shadow-colour the current function and assert the result is legal. Shadow
    /// rather than apply, because conformance is about the colouring a policy
    /// produces, not about lowering it.
    void expectLegalColoring() {
        ASSERT_TRUE(liftForAllocation(*func_));

        RegisterAllocationOptions options;
        options.allocator = GetParam();
        options.applyToOperands = false;
        options.verify = true;

        Expected<AllocationResult> result = allocateRegisters(*func_, *allocator_, options);
        ASSERT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
        EXPECT_EQ(result->shape(), func_->ssaArena().shape());
        EXPECT_EQ(result->unassignedCount(), 0u);
        EXPECT_TRUE(func_->hasAttachedSSA());

        // Verify again outside the driver, so the suite still checks legality
        // even if a caller passes verify=false.
        AllocationSetup setup(*func_);
        const AllocationVerificationResult checked =
            verifyAllocation(*func_, *result, setup.context());
        EXPECT_TRUE(checked.ok()) << checked.toString();
    }

    std::unique_ptr<Function> func_;
    std::unique_ptr<RegisterAllocator> allocator_;
};

bool isTupleInterior(const AllocationSetup& setup, SSAValueID id) {
    for (const TupleRun& run : setup.constraints().tupleRuns()) {
        for (size_t unit = 1; unit < run.units.size(); ++unit) {
            if (run.units[unit] == id) return true;
        }
    }
    return false;
}

/// gtest requires an identifier-safe suffix, and a registry name need not be one.
std::string testNameFor(const ::testing::TestParamInfo<std::string>& info) {
    std::string name = info.param;
    for (char& c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) == 0) c = '_';
    }
    return name.empty() ? "unnamed" : name;
}

}  // namespace

TEST(AllocatorRegistryTest, LegacyIsRegistered) {
    const std::vector<std::string> names = AllocatorRegistry::registeredAllocatorNames();
    EXPECT_NE(std::find(names.begin(), names.end(), "legacy"), names.end());
}

TEST(AllocatorRegistryTest, AnUnknownNameYieldsNoAllocator) {
    EXPECT_EQ(AllocatorRegistry::createAllocator("does-not-exist"), nullptr);
}

TEST_P(AllocatorConformanceTest, DeclaresOnlyCapabilitiesLoweringSupports) {
    // The driver refuses anything else, so a registered policy declaring either
    // of these could never be applied.
    const AllocatorCapabilities caps = allocator_->capabilities();
    EXPECT_FALSE(caps.mayRecolourMerges);
    EXPECT_FALSE(caps.maySpill);
    EXPECT_STREQ(allocator_->name(), GetParam().c_str());
}

TEST_P(AllocatorConformanceTest, ColoursAStraightLine) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    expectLegalColoring();
}

TEST_P(AllocatorConformanceTest, ColoursADiamond) {
    BasicBlock* entry = block("entry");
    BasicBlock* left = block("left");
    BasicBlock* right = block("right");
    BasicBlock* join = block("join");
    function().addEdge(entry, left);
    function().addEdge(entry, right);
    function().addEdge(left, join);
    function().addEdge(right, join);
    createVAddInBlock(left, kRaTestArch, 5, 20, 21);
    createVAddInBlock(right, kRaTestArch, 5, 22, 23);
    createVAddInBlock(join, kRaTestArch, 6, 5, 5);
    expectLegalColoring();
}

TEST_P(AllocatorConformanceTest, ColoursALoop) {
    BasicBlock* entry = block("entry");
    BasicBlock* header = block("header");
    BasicBlock* body = block("body");
    BasicBlock* exit = block("exit");
    function().addEdge(entry, header);
    function().addEdge(header, body);
    function().addEdge(body, header);
    function().addEdge(header, exit);
    createVAddInBlock(entry, kRaTestArch, 5, 20, 21);
    createVAddInBlock(body, kRaTestArch, 5, 5, 22);
    createVAddInBlock(exit, kRaTestArch, 6, 5, 5);
    expectLegalColoring();
}

TEST_P(AllocatorConformanceTest, ColoursAMultiDwordOperand) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 10, 4);
    createVAddInBlock(entry, kRaTestArch, 20, 10, 13);
    expectLegalColoring();
}

TEST_P(AllocatorConformanceTest, ColoursAPartialRedefinition) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 4, 0, 0);
    createDSWriteInBlock(entry, kRaTestArch, 0, 4);
    expectLegalColoring();
}

TEST_P(AllocatorConformanceTest, ColoursAReadModifyWrite) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 2, 1);
    expectLegalColoring();
}

TEST_P(AllocatorConformanceTest, ColoursScalarAndVectorTogether) {
    BasicBlock* entry = block("entry");
    AsmIRBuilder builder(*entry, kRaTestArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));
    expectLegalColoring();
}

TEST_P(AllocatorConformanceTest, RefusesAFunctionWithoutAttachedSSA) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);

    RegisterAllocationOptions options;
    options.allocator = GetParam();
    Expected<AllocationResult> result = allocateRegisters(function(), *allocator_, options);

    EXPECT_TRUE(result.hasError());
}

TEST_P(AllocatorConformanceTest, NeverProducesAColouringAnActiveRuleForbids) {
    // The property worth machine-checking: a policy queries the table and never
    // sees a rule, so every present and future policy is bound by whatever the
    // chip forbids. A policy that ignored the rule would be caught here as a
    // refusal, never as wrong code.
    //
    // legacy reproduces the producer's numbering and cannot move anything, so it
    // is expected to *refuse* rather than comply -- either outcome is conformant,
    // an accepted-but-violating colouring is not.
    const ScopedArchRules registered(evenVBasesOnly());

    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 3, 4, 5);
    ASSERT_TRUE(liftForAllocation(function()));

    RegisterAllocationOptions options;
    options.allocator = GetParam();
    options.verify = true;
    Expected<AllocationResult> result = allocateRegisters(function(), *allocator_, options);
    if (result.hasError()) return;  // refused, which is conformant

    AllocationSetup setup(function(), RegClassSet::only(RegType::V), {}, evenVBasesOnly());
    for (StinkySSAValue* value : function().ssaArena().values()) {
        if (value == nullptr || !result->isAssigned(value->valueId())) continue;
        const RegKey physical = result->assignmentOf(value->valueId());
        if (physical.type != RegType::V) continue;
        // Interior tuple members are legitimately odd; only run bases are ruled.
        if (value->type().dwordWidth <= 1 && !isTupleInterior(setup, value->valueId())) {
            EXPECT_EQ(physical.idx % 2, 0u)
                << GetParam() << " accepted %" << value->valueId() << " on "
                << regKeyToString(physical) << ", which EvenVBase forbids";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(RegisteredAllocators, AllocatorConformanceTest,
                         ::testing::ValuesIn(AllocatorRegistry::registeredAllocatorNames()),
                         testNameFor);
