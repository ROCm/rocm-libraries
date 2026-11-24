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
#include <vector>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/transforms/asm/ra/AllocatorRegistry.hpp"
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

/// Every instruction writes early, so any producer self-overwrite is a finding.
AllocationRules selfOverwriteTable(RuleStatus status) {
    AllocationRule rule;
    rule.name = "SelfOverwrite";
    rule.description = "this instruction must not write a register it reads";
    rule.status = status;
    rule.clobbersEarly = [](const StinkyInstruction&) { return true; };
    return AllocationRules({rule});
}

/// `v<dest> = v_mov_b32 v<src>` -- one source, so it asks for no pairing and
/// can scaffold a fixture without adding to the count under test.
StinkyInstruction* vMov(BasicBlock* bb, int dest, int src) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
    mov->addDestReg(StinkyRegister("v", dest, 1));
    mov->addSrcReg(StinkyRegister("v", src, 1));
    return mov;
}

/// Pairs a destination with its second source, wherever there is one.
///
/// A stand-in for the shipped WMMA rule. What the report has to explain is why
/// a pairing went unmet, and that reasoning does not depend on which rule asked
/// or on an instruction only one chip has.
AllocationRules destReusesSrc1() {
    AllocationRule rule;
    rule.name = "DestReusesSrc1";
    rule.description = "a destination should reuse its second source's register";
    rule.status = RuleStatus::Active;
    rule.satisfiedBy = [](RegKey dest, RegKey source) { return dest == source; };
    rule.addPreferences = [](const StinkyInstruction&, const OperandValues& values,
                             std::vector<Preference>& preferences) {
        const std::span<const SSAValueID> dest = values(0, true);
        const std::span<const SSAValueID> source = values(1, false);
        if (dest.empty() || source.empty()) return;
        preferences.push_back({dest[0], source[0], 0, 1.0});
    };
    return AllocationRules({rule});
}

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

TEST_F(RegisterAllocationPassTest, TheNotRunPassReportsAsTheAllocationPassWould) {
    // For a pipeline that decides while being built not to allocate: the same
    // NotAllocated remark, and nothing else.
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    const std::string before = physicalIR(*func);

    PassContext passCtx;
    passCtx.setRemarksEnabled(true);
    AnalysisManager am;
    registerAllAnalyses(am);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    createRegisterAllocationNotRunPass("register allocation config 'x' does not parse")
        ->run(*func, passCtx, am);
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text,
                         "missed: RegisterAllocation: @kernel: register allocation config "
                         "'x' does not parse"))
        << text;
    EXPECT_EQ(physicalIR(*func), before);
}

TEST_F(RegisterAllocationPassTest, AuditRemarksOnAProducerViolationAndStillColours) {
    // Audit is the pre-flight check that keeps a new rule from silently
    // uncolouring kernels: it reports against the producer's own registers and
    // changes nothing. Here v2 = v_add(v2, v1) already writes what it reads.
    const ScopedArchRules rules(selfOverwriteTable(RuleStatus::Audit));
    createVAddInBlock(block("entry"), kRaTestArch, 2, 2, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    PassContext passCtx;
    passCtx.setRemarksEnabled(true);
    AnalysisManager am;
    registerAllAnalyses(am);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    createRegisterAllocationPass(legacyApply())->run(*func, passCtx, am);
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    EXPECT_TRUE(contains(text, "RuleAudit") || contains(text, "SelfOverwrite")) << text;
    EXPECT_TRUE(contains(text, "producer colouring already violates")) << text;
    // Reported, not enforced: the colouring still applied.
    EXPECT_TRUE(contains(text, "AllocatedRegisters") || contains(text, "coloured")) << text;
}

TEST_F(RegisterAllocationPassTest, AuditIsSilentWhenRemarksAreOff) {
    const ScopedArchRules rules(selfOverwriteTable(RuleStatus::Audit));
    createVAddInBlock(block("entry"), kRaTestArch, 2, 2, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    PassContext passCtx;  // remarks disabled
    AnalysisManager am;
    registerAllAnalyses(am);

    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    createRegisterAllocationPass(legacyApply())->run(*func, passCtx, am);
    std::cerr.rdbuf(previous);

    EXPECT_TRUE(captured.str().empty()) << captured.str();
}

TEST_F(RegisterAllocationPassTest, AnUnknownForcedRuleNameIsAnError) {
    // A misspelling must fail. A test that enables nothing still passes, which
    // is the worst way to discover the name was wrong.
    const ScopedArchRules rules(selfOverwriteTable(RuleStatus::Off));
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.rules.activate.push_back("SelfOverwrit");

    LegacyIdentityAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);
    ASSERT_TRUE(result.hasError());
    EXPECT_TRUE(contains(result.getError(), "no such allocation rule")) << result.getError();
    EXPECT_TRUE(contains(result.getError(), "SelfOverwrit")) << result.getError();
}

TEST_F(RegisterAllocationPassTest, ForcingARuleBothOnAndOffIsAnError) {
    const ScopedArchRules rules(selfOverwriteTable(RuleStatus::Off));
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.rules.activate.push_back("SelfOverwrite");
    options.rules.disable.push_back("SelfOverwrite");

    LegacyIdentityAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);
    ASSERT_TRUE(result.hasError());
    EXPECT_TRUE(contains(result.getError(), "activate and to disable")) << result.getError();
    EXPECT_TRUE(contains(result.getError(), "SelfOverwrite")) << result.getError();
}

TEST_F(RegisterAllocationPassTest, ForcingARuleActiveIgnoresTheArchGate) {
    // The testing hatch: a standalone run has no rocisa capabilities, so a
    // filecheck test has to be able to switch a rule on by name.
    const ScopedArchRules rules(selfOverwriteTable(RuleStatus::Off));
    createVAddInBlock(block("entry"), kRaTestArch, 2, 2, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.rules.activate.push_back("SelfOverwrite");

    // The producer overwrites what it reads, and legacy cannot move anything, so
    // an Active rule has to refuse the colouring.
    LegacyIdentityAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);
    EXPECT_TRUE(result.hasError()) << "an Active rule the input violates must refuse";
}

TEST_F(RegisterAllocationPassTest, NoRulesRestoresThePreFrameworkBehaviour) {
    const ScopedArchRules rules(selfOverwriteTable(RuleStatus::Active));
    createVAddInBlock(block("entry"), kRaTestArch, 2, 2, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.rules.disableAll = true;

    LegacyIdentityAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);
    EXPECT_TRUE(result.hasValue()) << result.getError();
}

// ---------------------------------------------------------------------------
// Why a pairing went unmet
// ---------------------------------------------------------------------------
//
// The legacy allocator hands back the registers as written, so these fixtures
// set the colouring the report has to read rather than hoping an allocator
// produces one. Both shapes leave the pairing unmet; they differ only in
// whether the shared register was there to be had.

TEST_F(RegisterAllocationPassTest, AnUnmetPairingWithRoomReportsAsMissed) {
    // v1 dies at the add and nothing else ever names it, so the destination
    // could have been put there. Nothing was in the way, so the colouring is
    // the only thing left to blame.
    const ScopedArchRules rules(destReusesSrc1());
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, /*dest=*/2, /*src0=*/0, /*src1=*/1);
    vMov(entry, /*dest=*/3, /*src=*/2);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.report = true;
    std::string report;
    LegacyIdentityAllocator allocator;
    ASSERT_TRUE(allocateRegisters(*func, allocator, options, &report).hasValue());

    EXPECT_TRUE(contains(report, "pref[DestReusesSrc1=active 0/1 unmet 0 blocked 1 missed]"))
        << report;
}

TEST_F(RegisterAllocationPassTest, AnUnmetPairingWithNoRoomEitherWayReportsAsBlocked) {
    // Both ends walled in. A live-in on v2 reaches past the point where v1 is
    // born, so v1 cannot move up; and a fresh value takes v1 while the add's v2
    // is still live, so v2 cannot move down.
    const ScopedArchRules rules(destReusesSrc1());
    BasicBlock* entry = block("entry");
    vMov(entry, /*dest=*/4, /*src=*/2);
    createVAddInBlock(entry, kRaTestArch, /*dest=*/2, /*src0=*/0, /*src1=*/1);
    vMov(entry, /*dest=*/1, /*src=*/0);
    vMov(entry, /*dest=*/5, /*src=*/2);
    vMov(entry, /*dest=*/6, /*src=*/1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.report = true;
    std::string report;
    LegacyIdentityAllocator allocator;
    ASSERT_TRUE(allocateRegisters(*func, allocator, options, &report).hasValue());

    EXPECT_TRUE(contains(report, "pref[DestReusesSrc1=active 0/1 unmet 1 blocked 0 missed]"))
        << report;
}

TEST_F(RegisterAllocationPassTest, ASatisfiedPairingAddsNoUnmetBreakdown) {
    // The breakdown explains a loss, so with nothing lost it must stay out of
    // the line. Otherwise every clean report carries two zeroes.
    const ScopedArchRules rules(destReusesSrc1());
    BasicBlock* entry = block("entry");
    createVAddInBlock(entry, kRaTestArch, /*dest=*/1, /*src0=*/0, /*src1=*/1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.report = true;
    std::string report;
    LegacyIdentityAllocator allocator;
    ASSERT_TRUE(allocateRegisters(*func, allocator, options, &report).hasValue());

    EXPECT_TRUE(contains(report, "pref[DestReusesSrc1=active 1/1]")) << report;
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

namespace {

RegisterAllocationOptions compactApply() {
    RegisterAllocationOptions options;
    options.allocator = "greedy-compact";
    options.applyToOperands = true;
    return options;
}

}  // namespace

TEST_F(RegisterAllocationPassTest, HoldsARangeWithoutEvictingWhatIsInIt) {
    BasicBlock* entry = block("entry");
    AsmIRBuilder builder(*entry, kRaTestArch);
    // s0 arrives as a live-in, which is what reserve() would reject.
    StinkyInstruction* def = builder.create(getMCIDByUOp(GFX::s_mov_b32, kRaTestArch));
    def->addDestReg(StinkyRegister("s", 40, 1));
    def->addSrcReg(StinkyRegister("s", 0, 1));
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = compactApply();
    options.allocate = RegClassSet::only(RegType::S);
    options.pinRegisters = {{RegType::S, 0, 3}};

    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasValue()) << (result.hasValue() ? "" : result.getError());
    const std::string ir = physicalIR(*func);
    // Neither can pass vacuously: unheld, compaction puts the definition in s0.
    EXPECT_TRUE(contains(ir, "s4 = \"st.s_mov_b32\"")) << ir;
    EXPECT_TRUE(contains(ir, "(s0)")) << ir;
}

TEST_F(RegisterAllocationPassTest, RefusesABackwardsHeldRange) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.pinRegisters = {{RegType::V, 20, 19}};
    GreedyAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasError());
    EXPECT_TRUE(contains(result.getError(), "asked to hold v20 through v19")) << result.getError();
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

namespace {

/// `v_wmma_scale16 dst, a, b, 0, scaleA, scaleB`. Its two scale fields select
/// no s_set_vgpr_msb slot, so they reach the first bank only.
StinkyInstruction* createWmmaScale16(BasicBlock* bb, int dst, int a, int b, int scaleA,
                                     int scaleB) {
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* wmma =
        builder.create(getMCIDByUOp(GFX::v_wmma_scale16_f32_16x16x128_f8f6f4, kRaTestArch));
    wmma->addDestReg(StinkyRegister("v", dst, 8));
    wmma->addSrcReg(StinkyRegister("v", a, 8));
    wmma->addSrcReg(StinkyRegister("v", b, 8));
    wmma->addSrcReg(StinkyRegister(0));
    wmma->addSrcReg(StinkyRegister("v", scaleA, 2));
    wmma->addSrcReg(StinkyRegister("v", scaleB, 2));
    return wmma;
}

/// A scale operand the producer left at v300, which no bank selector can name,
/// defined in the function so that a policy is free to move it.
StinkyInstruction* outOfReachScaleOperand(Function& func, BasicBlock* entry) {
    AsmIRBuilder builder(*entry, kRaTestArch);
    for (int i = 0; i < 2; ++i) {
        StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
        mov->addDestReg(StinkyRegister("v", 300 + i, 1));
        mov->addSrcReg(StinkyRegister(0));
    }
    return createWmmaScale16(entry, /*dst=*/100, /*a=*/110, /*b=*/120, /*scaleA=*/300,
                             /*scaleB=*/130);
}

}  // namespace

TEST_F(RegisterAllocationPassTest, OnlyAllocateBringsAnOutOfReachScaleOperandBackIntoTheBank) {
    // The case a hold cannot fix, and the reason the Allocate policy exists.
    //
    // Hold keeps whatever register the producer chose, which is only right while
    // the producer chooses reachable ones. Asked to freeze v300, a register no
    // selector can name, it now refuses instead: the ceiling is collected under
    // either policy, so the contradiction is caught rather than emitted. That is
    // as far as holding can get, since it has no other register to offer.
    //
    // Allocate owes the producer nothing. It places the value under the ceiling
    // like any other constrained block, which brings it back into the bank.
    BasicBlock* entry = block("entry");
    StinkyInstruction* wmma = outOfReachScaleOperand(*func, entry);
    ASSERT_TRUE(liftForAllocation(*func));

    const std::vector<StinkySSAValue*> scale = ssaSourceUnits(*wmma, 3);
    ASSERT_EQ(scale.size(), 2u);
    ASSERT_NE(scale[0], nullptr);
    const SSAValueID outOfReach = scale[0]->valueId();

    RegisterAllocationOptions options;
    options.allocator = "greedy-compact";
    options.allocate = RegClassSet::only(RegType::V);

    GreedyAllocator allocator;
    options.unbankableOperands = RegisterAllocationOptions::UnbankableOperands::Hold;
    Expected<AllocationResult> held = allocateRegisters(*func, allocator, options);
    ASSERT_TRUE(held.hasError()) << "holding an unreachable register cannot be honoured";
    EXPECT_TRUE(contains(held.getError(), "v300")) << held.getError();

    options.unbankableOperands = RegisterAllocationOptions::UnbankableOperands::Allocate;
    Expected<AllocationResult> allocated = allocateRegisters(*func, allocator, options);
    ASSERT_TRUE(allocated.hasValue()) << allocated.getError();
    EXPECT_LT(allocated->assignmentOf(outOfReach).idx, kVgprBankSize) << allocated->toString();
}

TEST_F(RegisterAllocationPassTest, AutoAllocatesAnOutOfReachScaleOperandBeforeTryingToHoldIt) {
    // Hold refuses this kernel, so Auto has to reach the colouring through
    // its first attempt, and the report says the treatment was found by trying.
    BasicBlock* entry = block("entry");
    StinkyInstruction* wmma = outOfReachScaleOperand(*func, entry);
    ASSERT_TRUE(liftForAllocation(*func));
    const SSAValueID outOfReach = ssaSourceUnits(*wmma, 3)[0]->valueId();

    RegisterAllocationOptions options;
    options.allocate = RegClassSet::only(RegType::V);
    options.report = true;
    ASSERT_EQ(options.unbankableOperands, RegisterAllocationOptions::UnbankableOperands::Auto);
    CompactingGreedyAllocator allocator;
    std::string report;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options, &report);

    ASSERT_TRUE(result.hasValue()) << result.getError();
    EXPECT_LT(result->assignmentOf(outOfReach).idx, kVgprBankSize) << result->toString();
    EXPECT_TRUE(contains(report, "unbankable[auto allocated")) << report;
    EXPECT_FALSE(contains(report, "was refused")) << report;
}

namespace {

/// A scale operand at v140, which its field can reach, defined in the function
/// so a policy is free to move it. The other scale field reads v130.
void inReachScaleOperand(Function& func, BasicBlock* entry) {
    AsmIRBuilder builder(*entry, kRaTestArch);
    for (int i = 0; i < 2; ++i) {
        StinkyInstruction* mov = builder.create(getMCIDByUOp(GFX::v_mov_b32, kRaTestArch));
        mov->addDestReg(StinkyRegister("v", 140 + i, 1));
        mov->addSrcReg(StinkyRegister(0));
    }
    createWmmaScale16(entry, /*dst=*/100, /*a=*/110, /*b=*/120, /*scaleA=*/140, /*scaleB=*/130);
}

/// Colours only while \p heldRegister is held, as the unbankable operands are
/// under Hold, and refuses otherwise. The shape of the bank-0 conflict: the
/// matrix pins leave Allocate no room under the ceiling, while the producer's
/// own registers for the scale operands still fit.
class ColoursOnlyWhenHeld : public RegisterAllocator {
   public:
    explicit ColoursOnlyWhenHeld(uint32_t heldRegister) : heldRegister_(heldRegister) {}
    const char* name() const override {
        return "colours-only-when-held";
    }
    AllocatorCapabilities capabilities() const override {
        return {};
    }
    Expected<AllocationResult> allocate(const AllocationContext& context) override {
        ++attempts;
        if (!context.scope.isPinnedRegister(RegType::V, heldRegister_)) {
            return Expected<AllocationResult>::Error("@" + context.function.getName() +
                                                     ": no room under the ceiling");
        }
        return createLegacyColoring(context.function);
    }
    int attempts = 0;

   private:
    uint32_t heldRegister_;
};

class AlwaysRefuses : public RegisterAllocator {
   public:
    const char* name() const override {
        return "always-refuses";
    }
    AllocatorCapabilities capabilities() const override {
        return {};
    }
    Expected<AllocationResult> allocate(const AllocationContext& context) override {
        ++attempts;
        dumpPaths.push_back(context.blockDumpPath);
        return Expected<AllocationResult>::Error("@" + context.function.getName() + ": refused");
    }
    int attempts = 0;
    std::vector<std::string> dumpPaths;
};

std::string runPassCapturingRemarks(Function& func, RegisterAllocationOptions options) {
    PassContext passCtx;
    passCtx.setRemarksEnabled(true);
    AnalysisManager am;
    registerAllAnalyses(am);
    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    createRegisterAllocationPass(std::move(options))->run(func, passCtx, am);
    std::cerr.rdbuf(previous);
    return captured.str();
}

}  // namespace

TEST_F(RegisterAllocationPassTest, AutoHoldsTheScaleOperandsWhenAllocatingThemIsRefused) {
    BasicBlock* entry = block("entry");
    inReachScaleOperand(*func, entry);
    const std::string before = physicalIR(*func);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.allocate = RegClassSet::only(RegType::V);
    options.applyToOperands = true;
    options.report = true;
    ColoursOnlyWhenHeld allocator(/*heldRegister=*/140);
    std::string report;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options, &report);

    ASSERT_TRUE(result.hasValue()) << result.getError();
    EXPECT_EQ(allocator.attempts, 2);
    EXPECT_TRUE(contains(report, "unbankable[auto held v130:131 v140:141]")) << report;
    EXPECT_TRUE(contains(report,
                         "held them because the allocate attempt was refused: no room under the "
                         "ceiling"))
        << report;
    // The refused attempt changed nothing; the identity colouring that won
    // was applied once.
    EXPECT_FALSE(func->hasAttachedSSA());
    EXPECT_EQ(physicalIR(*func), before);
}

TEST_F(RegisterAllocationPassTest, AnExplicitPolicyIsTheOnlyAttempt) {
    inReachScaleOperand(*func, block("entry"));
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.allocate = RegClassSet::only(RegType::V);
    options.unbankableOperands = RegisterAllocationOptions::UnbankableOperands::Allocate;
    ColoursOnlyWhenHeld allocator(/*heldRegister=*/140);
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasError());
    EXPECT_EQ(result.getError(), "@kernel: no room under the ceiling");
    EXPECT_EQ(allocator.attempts, 1);
}

TEST_F(RegisterAllocationPassTest, AutoNamesBothRefusalsWhenNeitherColours) {
    inReachScaleOperand(*func, block("entry"));
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.allocate = RegClassSet::only(RegType::V);
    AlwaysRefuses allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasError());
    EXPECT_EQ(allocator.attempts, 2);
    EXPECT_EQ(result.getError(),
              "@kernel: refused whether the operands that cannot name a bank are allocated or "
              "held\n  allocate: refused\n  hold: refused");
}

TEST_F(RegisterAllocationPassTest, EachAttemptWritesItsOwnPlacementTable) {
    inReachScaleOperand(*func, block("entry"));
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.allocate = RegClassSet::only(RegType::V);
    options.blockDumpPrefix = "dump/blocks.";
    AlwaysRefuses allocator;
    ASSERT_TRUE(allocateRegisters(*func, allocator, options).hasError());

    EXPECT_EQ(allocator.dumpPaths,
              (std::vector<std::string>{"dump/blocks.always-refuses.allocate.tsv",
                                        "dump/blocks.always-refuses.hold.tsv"}));
}

TEST_F(RegisterAllocationPassTest, AutoIsOneAttemptWhenNoOperandIsUnbankable) {
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    AlwaysRefuses allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasError());
    EXPECT_EQ(allocator.attempts, 1) << "holding nothing is the same attempt again";
    EXPECT_EQ(result.getError(), "@kernel: refused");
}

TEST_F(RegisterAllocationPassTest, AChainAppliesTheFirstAllocatorThatColours) {
    // Registered for the rest of the run. The conformance suite took its list
    // of allocators before any test ran, so it never sees this one.
    AllocatorRegistry::registerAllocator("test-always-refuses",
                                         [] { return std::make_unique<AlwaysRefuses>(); });
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    const std::string before = physicalIR(*func);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.allocator = "test-always-refuses+legacy";
    const std::string text = runPassCapturingRemarks(*func, options);

    EXPECT_TRUE(contains(text, "with legacy after always-refuses refused")) << text;
    EXPECT_FALSE(contains(text, "NotAllocated")) << text;
    EXPECT_FALSE(func->hasAttachedSSA());
    EXPECT_EQ(physicalIR(*func), before);
}

TEST_F(RegisterAllocationPassTest, AChainThatEveryAllocatorRefusesNamesEachRefusal) {
    AllocatorRegistry::registerAllocator("test-always-refuses",
                                         [] { return std::make_unique<AlwaysRefuses>(); });
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.allocator = "test-always-refuses+test-always-refuses";
    const std::string text = runPassCapturingRemarks(*func, options);

    EXPECT_TRUE(contains(text,
                         "every allocator in test-always-refuses+test-always-refuses "
                         "refused\n  always-refuses: refused\n  always-refuses: refused"))
        << text;
    EXPECT_TRUE(func->hasAttachedSSA()) << "nothing was applied";
}

TEST_F(RegisterAllocationPassTest, TheReportPredictsVgprMsbSwitchesOnlyWhereBanksAreSelected) {
    // In a loop, src1 in bank 2 and then in bank 3: the first instruction sets
    // the word and the second changes it. The legacy colouring is the
    // producer's, so both sides read 2. A target without a VGPR MSB mode has
    // nothing to count, and its report stays as it was.
    BasicBlock* entry = block("entry");
    BasicBlock* loop = block("loop");
    BasicBlock* exit = block("exit");
    func->addEdge(entry, loop);
    func->addEdge(loop, loop);
    func->addEdge(loop, exit);
    createWmmaScale16(loop, /*dst=*/0, /*a=*/256, /*b=*/512, /*scaleA=*/10, /*scaleB=*/12);
    createWmmaScale16(loop, /*dst=*/8, /*a=*/264, /*b=*/768, /*scaleA=*/10, /*scaleB=*/12);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.applyToOperands = false;
    options.report = true;
    LegacyIdentityAllocator allocator;

    std::string withoutMode;
    ASSERT_TRUE(allocateRegisters(*func, allocator, options, &withoutMode).hasValue());
    EXPECT_FALSE(contains(withoutMode, "msb[")) << withoutMode;

    options.caps.vgprMsbMode = VgprMsbMode::Msb16;
    std::string withMode;
    ASSERT_TRUE(allocateRegisters(*func, allocator, options, &withMode).hasValue());
    EXPECT_TRUE(contains(withMode, " msb[loop=2->2]")) << withMode;
}

// ---------------------------------------------------------------------------
// Placing matrix operands: matrix=
// ---------------------------------------------------------------------------

namespace {

using Matrix = RegisterAllocationOptions::Matrix;

/// One rule per matrix role, each doing nothing, so a test sees which rules a
/// placement chose without their effects in the way. Shipped as gfx1250 ships
/// its own: the pin Active, the other two Off.
AllocationRules matrixRoleTable() {
    AllocationRule pin;
    pin.name = "TestPin";
    pin.description = "matrix operands keep the producer's registers";
    pin.status = RuleStatus::Active;
    pin.role = MatrixRole::Pin;
    pin.pinToProducer = [](const StinkyInstruction&, const OperandValues&,
                           std::vector<SSAValueID>&) {};
    AllocationRule shape;
    shape.name = "TestShape";
    shape.description = "matrix operands keep the producer's grouping";
    shape.role = MatrixRole::Shape;
    shape.addRelations = [](const Function&, std::vector<TupleRun>&, std::vector<AffinitySet>&) {};
    AllocationRule pinAccumulator;
    pinAccumulator.name = "TestPinAccumulator";
    pinAccumulator.description = "accumulators keep the producer's registers";
    pinAccumulator.role = MatrixRole::PinAccumulator;
    pinAccumulator.pinToProducer = [](const StinkyInstruction&, const OperandValues&,
                                      std::vector<SSAValueID>&) {};
    return AllocationRules({pin, shape, pinAccumulator});
}

/// Refuses whenever a Shape-role rule is Active and colours as the producer did
/// otherwise: a kernel the pin colours and shape mode does not.
class RefusesShapePlacement : public RegisterAllocator {
   public:
    const char* name() const override {
        return "refuses-shape";
    }
    AllocatorCapabilities capabilities() const override {
        return {};
    }
    Expected<AllocationResult> allocate(const AllocationContext& context) override {
        ++attempts;
        for (const AllocationRule& rule : context.rules.all()) {
            if (rule.role == MatrixRole::Shape && rule.status == RuleStatus::Active) {
                return Expected<AllocationResult>::Error("@" + context.function.getName() +
                                                         ": no room for the tiles");
            }
        }
        return createLegacyColoring(context.function);
    }
    int attempts = 0;
};

}  // namespace

TEST_F(RegisterAllocationPassTest, AutoPinsTheMatrixOperandsWhenPlacingThemByShapeIsRefused) {
    const ScopedArchRules rules(matrixRoleTable());
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    const std::string before = physicalIR(*func);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.report = true;
    options.matrix = Matrix::Auto;
    RefusesShapePlacement allocator;
    std::string report;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options, &report);

    ASSERT_TRUE(result.hasValue()) << result.getError();
    EXPECT_EQ(allocator.attempts, 2);
    EXPECT_TRUE(contains(report, " matrix[auto pin]")) << report;
    EXPECT_TRUE(
        contains(report, "rule[TestPin=active] rule[TestShape=off] rule[TestPinAccumulator=off]"))
        << report;
    EXPECT_TRUE(contains(report,
                         "\n  pinned the matrix operands because placing them by shape was "
                         "refused: no room for the tiles"))
        << report;
    // The refused attempt changed nothing; the pin's colouring was applied once.
    EXPECT_FALSE(func->hasAttachedSSA());
    EXPECT_EQ(physicalIR(*func), before);
}

TEST_F(RegisterAllocationPassTest, AutoKeepsTheShapeColouringWhenItColours) {
    const ScopedArchRules rules(matrixRoleTable());
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.report = true;
    options.matrix = Matrix::Auto;
    LegacyIdentityAllocator allocator;
    std::string report;
    ASSERT_TRUE(allocateRegisters(*func, allocator, options, &report).hasValue());

    EXPECT_TRUE(contains(report, " matrix[auto shape]")) << report;
    EXPECT_TRUE(
        contains(report, "rule[TestPin=off] rule[TestShape=active] rule[TestPinAccumulator=off]"))
        << report;
    EXPECT_FALSE(contains(report, "because placing them by shape was refused")) << report;
}

TEST_F(RegisterAllocationPassTest, EachMatrixPlacementSetsTheRulesItChoosesByRole) {
    const ScopedArchRules rules(matrixRoleTable());
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    const struct {
        Matrix placement;
        const char* field;
        const char* statuses;
    } cases[] = {
        {Matrix::Pin, "", "rule[TestPin=active] rule[TestShape=off] rule[TestPinAccumulator=off]"},
        {Matrix::Shape, " matrix[shape]",
         "rule[TestPin=off] rule[TestShape=active] rule[TestPinAccumulator=off]"},
        {Matrix::Accumulators, " matrix[accumulators]",
         "rule[TestPin=off] rule[TestShape=active] rule[TestPinAccumulator=active]"},
    };
    for (const auto& c : cases) {
        RegisterAllocationOptions options = legacyApply();
        options.applyToOperands = false;
        options.report = true;
        options.matrix = c.placement;
        LegacyIdentityAllocator allocator;
        std::string report;
        ASSERT_TRUE(allocateRegisters(*func, allocator, options, &report).hasValue());
        EXPECT_TRUE(contains(report, c.statuses)) << report;
        if (c.placement == Matrix::Pin)
            EXPECT_FALSE(contains(report, "matrix[")) << "the default stays silent: " << report;
        else
            EXPECT_TRUE(contains(report, c.field)) << report;
    }
}

TEST_F(RegisterAllocationPassTest, AutoNamesBothRefusalsWhenNoPlacementColours) {
    const ScopedArchRules rules(matrixRoleTable());
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.matrix = Matrix::Auto;
    AlwaysRefuses allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(result.hasError());
    EXPECT_EQ(allocator.attempts, 2);
    EXPECT_EQ(result.getError(),
              "@kernel: refused whether the matrix operands are placed by shape or pinned\n"
              "  shape: refused\n  pin: refused");
}

TEST_F(RegisterAllocationPassTest, AMatrixPlacementCannotAlsoNameAMatrixRule) {
    // Whichever won, one of the two requests would go silently undone.
    const ScopedArchRules rules(matrixRoleTable());
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.matrix = Matrix::Shape;
    options.rules.activate.push_back("TestShape");
    LegacyIdentityAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);
    ASSERT_TRUE(result.hasError());
    EXPECT_EQ(result.getError(),
              "@kernel: matrix=shape sets the matrix rules itself, so rules= and noRule= cannot "
              "name TestShape");

    // Naming them with the pin is the spelling from before matrix=, and stands.
    options.matrix = Matrix::Pin;
    options.rules.disable.push_back("TestPin");
    EXPECT_TRUE(allocateRegisters(*func, allocator, options).hasValue());
}

TEST_F(RegisterAllocationPassTest, WithoutAShapeRuleShapeIsRefusedAndAutoIsThePin) {
    const ScopedArchRules rules(selfOverwriteTable(RuleStatus::Off));
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    LegacyIdentityAllocator allocator;
    for (const Matrix placement : {Matrix::Shape, Matrix::Accumulators}) {
        options.matrix = placement;
        Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);
        ASSERT_TRUE(result.hasError());
        EXPECT_TRUE(contains(result.getError(), "needs a rule that keeps the producer's grouping"))
            << result.getError();
        EXPECT_TRUE(contains(result.getError(), "this architecture has none")) << result.getError();
    }

    options.matrix = Matrix::Auto;
    AlwaysRefuses refuses;
    Expected<AllocationResult> refused = allocateRegisters(*func, refuses, options);
    ASSERT_TRUE(refused.hasError());
    EXPECT_EQ(refuses.attempts, 1) << "nothing to place by shape, so the pin is the one attempt";
    EXPECT_EQ(refused.getError(), "@kernel: refused");

    options.applyToOperands = false;
    options.report = true;
    std::string report;
    ASSERT_TRUE(allocateRegisters(*func, allocator, options, &report).hasValue());
    EXPECT_TRUE(contains(report, " matrix[auto pin]")) << report;
    EXPECT_FALSE(contains(report, "was refused")) << report;
}

TEST_F(RegisterAllocationPassTest, AShapePlacementUnderNoRulesBlamesNoRules) {
    // This table has a shape rule, so a refusal blaming the architecture would
    // send the reader looking for a rule that is there.
    const ScopedArchRules rules(matrixRoleTable());
    createVAddInBlock(block("entry"), kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options = legacyApply();
    options.rules.disableAll = true;
    LegacyIdentityAllocator allocator;
    const struct {
        Matrix placement;
        const char* spelling;
    } cases[] = {{Matrix::Shape, "shape"}, {Matrix::Accumulators, "accumulators"}};
    for (const auto& c : cases) {
        options.matrix = c.placement;
        Expected<AllocationResult> result = allocateRegisters(*func, allocator, options);
        ASSERT_TRUE(result.hasError());
        EXPECT_EQ(result.getError(),
                  std::string("@kernel: matrix=") + c.spelling +
                      " needs a rule that keeps the producer's grouping of matrix operands, and "
                      "noRules removes every rule");
    }
}

TEST_F(RegisterAllocationPassTest, MatrixShapeColoursAsNamingTheGfx1250RulesDoes) {
    // On gfx1250, matrix=shape is a spelling of
    // noRule=ProducerPinMatrix,rules=ProducerShapeMatrix, not a new colouring.
    AllocationRulesRegistry::registerAll();
    ASSERT_TRUE(AllocationRulesRegistry::forArch(raTestTriple(), AsmCapsConfig{})
                    .hasRole(MatrixRole::Shape));
    inReachScaleOperand(*func, block("entry"));
    ASSERT_TRUE(liftForAllocation(*func));

    RegisterAllocationOptions options;
    options.allocate = RegClassSet::only(RegType::V);
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> pinned = allocateRegisters(*func, allocator, options);
    options.matrix = Matrix::Shape;
    Expected<AllocationResult> byShape = allocateRegisters(*func, allocator, options);
    options.matrix = Matrix::Pin;
    options.rules.activate = {"ProducerShapeMatrix"};
    options.rules.disable = {"ProducerPinMatrix"};
    Expected<AllocationResult> byName = allocateRegisters(*func, allocator, options);

    ASSERT_TRUE(pinned.hasValue()) << pinned.getError();
    ASSERT_TRUE(byShape.hasValue()) << byShape.getError();
    ASSERT_TRUE(byName.hasValue()) << byName.getError();
    EXPECT_EQ(byShape->toString(), byName->toString());
    // If the pin ever colours this the same way, the test has stopped telling
    // the placements apart.
    EXPECT_NE(byShape->toString(), pinned->toString());
}
