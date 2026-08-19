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
#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/transforms/asm/ra/LegacyColoring.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationPass.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"
#include "transforms/asm/ra/allocators/GreedyAllocator.hpp"
#include "transforms/asm/ra/allocators/LegacyIdentityAllocator.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

class RegisterAllocationPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
    }

    BasicBlock* block(const std::string& label) {
        return func->createBasicBlock(label);
    }

    RegisterAllocationOptions legacyApply() {
        RegisterAllocationOptions options;
        options.allocator = "legacy";
        options.applyToOperands = true;
        options.verify = true;
        return options;
    }

    std::unique_ptr<Function> func;
};

class RecolouringAllocator : public RegisterAllocator {
   public:
    const char* name() const override {
        return "recolour-merges";
    }
    AllocatorCapabilities capabilities() const override {
        AllocatorCapabilities caps;
        caps.mayRecolourMerges = true;
        return caps;
    }
    Expected<AllocationResult> allocate(const AllocationContext& context) override {
        allocated = true;
        return createLegacyColoring(context.function);
    }
    bool allocated = false;
};

class SpillingAllocator : public RegisterAllocator {
   public:
    const char* name() const override {
        return "spill";
    }
    AllocatorCapabilities capabilities() const override {
        AllocatorCapabilities caps;
        caps.maySpill = true;
        return caps;
    }
    Expected<AllocationResult> allocate(const AllocationContext& context) override {
        allocated = true;
        return createLegacyColoring(context.function);
    }
    bool allocated = false;
};

}  // namespace

TEST_F(RegisterAllocationPassTest, ApplyWithLegacyLeavesThePhysicalProgramUnchanged) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 8, 4, 5);
    const std::string before = physicalIR(*func);

    ASSERT_TRUE(liftForAllocation(*func));
    LegacyIdentityAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, legacyApply());

    ASSERT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
    EXPECT_EQ(physicalIR(*func), before);
    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(RegisterAllocationPassTest, PassApplyIsAnIdentityTransformThroughThePassManager) {
    BasicBlock* entry = block("entry");
    createDsReadB128InBlock(entry, kRaTestArch, 4, 0);
    createVAddInBlock(entry, kRaTestArch, 8, 4, 5);
    const std::string before = physicalIR(*func);

    PassManager pm;
    registerAllAnalyses(pm.getAnalysisManager());
    pm.setGemmTileConfig(func->getGemmTileConfig());
    pm.addPass(createLiftAsmRegistersToSSAPass());
    pm.addPass(createRegisterAllocationPass(legacyApply()));
    pm.run(*func);

    EXPECT_EQ(physicalIR(*func), before);
    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(RegisterAllocationPassTest, ShadowLeavesAttachedSSAAndOperands) {
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, 2, 0, 1);
    const std::string before = physicalIR(*func);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.allocator = "legacy";
    options.applyToOperands = false;
    LegacyIdentityAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
    EXPECT_TRUE(func->hasAttachedSSA());
    EXPECT_EQ(physicalIR(*func), before);
}

TEST_F(RegisterAllocationPassTest, RefusesAnAllocatorThatMayRecolourMerges) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    const std::string before = physicalIR(*func);
    ASSERT_TRUE(liftForAllocation(*func));

    RecolouringAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, legacyApply());

    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(contains(result.getError(), "copy insertion")) << result.getError();
    EXPECT_FALSE(allocator.allocated);
    EXPECT_TRUE(func->hasAttachedSSA());
    EXPECT_EQ(physicalIR(*func), before);
}

TEST_F(RegisterAllocationPassTest, RefusesAnAllocatorThatMaySpill) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    SpillingAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, legacyApply());

    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(contains(result.getError(), "spilling")) << result.getError();
    EXPECT_FALSE(allocator.allocated);
}

TEST_F(RegisterAllocationPassTest, RefusesToAllocateAClassTheLiftLeftPhysical) {
    // Colouring a class with no SSA values would assign registers nothing can
    // rewrite, so the mismatch is reported instead of quietly doing nothing.
    BasicBlock* entry = block("entry");
    AsmIRBuilder builder(*entry, kRaTestArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));

    LiftAsmRegistersToSSAOptions scalarOnly;
    scalarOnly.classes = RegClassSet::only(RegType::S);
    ASSERT_TRUE(liftAsmRegistersToAttachedSSA(*func, scalarOnly).hasValue());

    RegisterAllocationOptions options;  // defaults to VGPRs, which were not lifted
    options.allocator = "greedy";
    GreedyAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasError());
    EXPECT_TRUE(
        contains(result.getError(), "asked to allocate v but this function was lifted for s"))
        << result.getError();
}

TEST_F(RegisterAllocationPassTest, AllocatesTheClassTheLiftCovered) {
    BasicBlock* entry = block("entry");
    AsmIRBuilder builder(*entry, kRaTestArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
    mov->addDestReg(StinkyRegister("v", 0, 1));
    mov->addSrcReg(StinkyRegister("s", 4, 1));

    LiftAsmRegistersToSSAOptions scalarOnly;
    scalarOnly.classes = RegClassSet::only(RegType::S);
    ASSERT_TRUE(liftAsmRegistersToAttachedSSA(*func, scalarOnly).hasValue());

    RegisterAllocationOptions options;
    options.allocator = "greedy";
    options.allocate = RegClassSet::only(RegType::S);
    options.applyToOperands = true;
    GreedyAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
    // The scalar was colourable and the vector operand was never in SSA, so it
    // comes out exactly as written.
    EXPECT_TRUE(contains(physicalIR(*func), "v0 = \"st.v_mov_b32\"(s4)")) << physicalIR(*func);
}

TEST_F(RegisterAllocationPassTest, PassReportsAMissingGraph) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    const std::string before = physicalIR(*func);

    PassContext passCtx;
    passCtx.setRemarksEnabled(true);
    AnalysisManager am;
    registerAllAnalyses(am);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    createRegisterAllocationPass(legacyApply())->run(*func, passCtx, am);
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "missed: RegisterAllocation")) << text;
    EXPECT_TRUE(contains(text, "no attached SSA")) << text;
    // Reporting is all it does: a function with nothing to colour comes out
    // exactly as it went in.
    EXPECT_EQ(physicalIR(*func), before);
}

TEST_F(RegisterAllocationPassTest, PassReportsAnUnknownAllocator) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.allocator = "does-not-exist";
    options.applyToOperands = true;

    PassContext passCtx;
    passCtx.setRemarksEnabled(true);
    AnalysisManager am;
    registerAllAnalyses(am);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    createRegisterAllocationPass(options)->run(*func, passCtx, am);
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "is not registered")) << text;
    EXPECT_TRUE(func->hasAttachedSSA());
}

TEST_F(RegisterAllocationPassTest, InjectedAllocatorIsUsedInsteadOfTheRegistry) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));
    const std::string before = physicalIR(*func);

    // The named allocator does not exist, so the run can only succeed by using
    // the injected one. Naming a registered allocator here would leave the test
    // passing even if injection were ignored.
    RegisterAllocationOptions options = legacyApply();
    options.allocator = "does-not-exist";

    PassContext passCtx;
    AnalysisManager am;
    registerAllAnalyses(am);
    createRegisterAllocationPass(options, std::make_unique<LegacyIdentityAllocator>())
        ->run(*func, passCtx, am);

    EXPECT_EQ(physicalIR(*func), before);
    EXPECT_FALSE(func->hasAttachedSSA());
}

TEST_F(RegisterAllocationPassTest, RefusesAnUnknownRegionEndBlock) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.regionEnd = "missing";
    GreedyAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasError());
    EXPECT_TRUE(contains(result.getError(), "region end block 'missing' was not found"))
        << result.getError();
}

TEST_F(RegisterAllocationPassTest, ShadowReportIncludesRegionPeak) {
    BasicBlock* entry = block("entry");
    BasicBlock* tail = block("tail");
    func->addEdge(entry, tail);
    createVAddInBlock(entry, kRaTestArch, 50, 0, 1);
    createVAddInBlock(tail, kRaTestArch, 60, 50, 2);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.allocator = "greedy-compact";
    options.regionEnd = "entry";
    options.report = true;

    CompactingGreedyAllocator allocator;
    std::string report;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options, &report);

    ASSERT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
    EXPECT_TRUE(contains(report, "regionPeak=")) << report;
}
