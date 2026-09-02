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

#include <memory>
#include <string>
#include <vector>

#include "AllocationTestUtils.hpp"
#include "stinkytofu/analysis/asm/ssa/SSAFunctionShape.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocationPass.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterSymbolSync.hpp"
#include "stinkytofu/transforms/asm/ssa/SSADestruction.hpp"
#include "transforms/asm/ra/allocators/GreedyAllocator.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

void appendSetDirective(BasicBlock* bb, const std::string& symbol, const std::string& value) {
    AsmDirective* d = IRBase::createIR<AsmDirective>();
    d->kind = AsmDirectiveKind::SET;
    d->name = ".set";
    d->symbol = symbol;
    d->value = value;
    bb->appendIR(d);
}

StinkyRegister namedReg(RegType type, uint32_t idx, uint16_t num, const std::string& name) {
    StinkyRegister reg(type, idx, num);
    reg.setSymbolicName(name);
    return reg;
}

StinkyRegister placeholderNamedReg(const std::string& name) {
    StinkyRegister reg(RegType::S, 0u, 1u);
    reg.setSymbolicName(name);
    return reg;
}

std::string emitSymbolicAssembly(const Function& function) {
    AsmEmitterOptions options;
    options.useSymbolicNames = true;
    options.emitComments = true;
    StinkyAsmEmitter emitter(options);
    return emitter.emit(function);
}

RegisterAllocationOptions compactApply() {
    RegisterAllocationOptions options;
    options.allocator = "greedy-compact";
    options.applyToOperands = true;
    options.verify = true;
    return options;
}

RegisterAllocationOptions greedyApply() {
    RegisterAllocationOptions options;
    options.allocator = "greedy";
    options.applyToOperands = true;
    options.verify = true;
    return options;
}

/// Scalar compaction, which is what the production pipeline actually runs.
RegisterAllocationOptions scalarCompactApply() {
    RegisterAllocationOptions options;
    options.allocator = "greedy-compact";
    options.allocate = RegClassSet::only(RegType::S);
    options.applyToOperands = true;
    options.verify = true;
    return options;
}

class RegisterSymbolSyncTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kRaTestArch);
    }

    BasicBlock* entry() {
        return func->createBasicBlock("entry");
    }

    std::unique_ptr<Function> func;
};

TEST_F(RegisterSymbolSyncTest, RejectedDestructionLeavesRewrittenEmpty) {
    BasicBlock* bb = entry();
    createVAddInBlock(bb, kRaTestArch, 2, 0, 1);
    ASSERT_TRUE(liftForAllocation(*func));

    // Nothing coloured, so destruction has to reject the result rather than
    // rewrite half a program.
    const AllocationResult uncoloured(*func);

    const SSADestructionResult destroyed = destroyAttachedSSA(*func, uncoloured);
    EXPECT_FALSE(destroyed.ok());
    EXPECT_TRUE(destroyed.rewritten.empty());
}

TEST_F(RegisterSymbolSyncTest, GreedyLeavesSymbolicNamesUntouched) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "vgprTmp", "40");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* add = builder.create(getMCIDByUOp(GFX::v_add_f32, kRaTestArch));
    add->addDestReg(namedReg(RegType::V, 40, 1, "vgprTmp"));
    add->addSrcReg(StinkyRegister("v", 0, 1));
    add->addSrcReg(StinkyRegister("v", 1, 1));
    const std::string before = emitSymbolicAssembly(*func);

    ASSERT_TRUE(liftForAllocation(*func));
    GreedyAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, greedyApply());
    ASSERT_TRUE(result.hasValue()) << result.getError();

    const std::string after = emitSymbolicAssembly(*func);
    EXPECT_EQ(after, before);
}

TEST_F(RegisterSymbolSyncTest, CompactMovedTempRewritesSetAndKeepsName) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "vgprTmp", "40");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* first = builder.create(getMCIDByUOp(GFX::v_add_f32, kRaTestArch));
    first->addDestReg(namedReg(RegType::V, 40, 1, "vgprTmp"));
    first->addSrcReg(StinkyRegister("v", 0, 1));
    first->addSrcReg(StinkyRegister("v", 1, 1));
    StinkyInstruction* second = builder.create(getMCIDByUOp(GFX::v_add_f32, kRaTestArch));
    second->addDestReg(StinkyRegister("v", 41, 1));
    second->addSrcReg(namedReg(RegType::V, 40, 1, "vgprTmp"));
    second->addSrcReg(StinkyRegister("v", 2, 1));

    ASSERT_TRUE(liftForAllocation(*func));
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, compactApply());
    ASSERT_TRUE(result.hasValue()) << result.getError();

    const std::string asmText = emitSymbolicAssembly(*func);
    EXPECT_TRUE(contains(asmText, "vgprTmp"));
    EXPECT_FALSE(contains(asmText, "s[vgprTmp]"));
    EXPECT_FALSE(contains(asmText, ".set vgprTmp, 40"));

    bool foundRewrittenSet = false;
    for (const BasicBlock& block : *func) {
        for (const IRBase& ir : block) {
            const auto* directive = dyn_cast<AsmDirective>(&ir);
            if (directive == nullptr || directive->kind != AsmDirectiveKind::SET) continue;
            if (directive->symbol != "vgprTmp") continue;
            EXPECT_NE(directive->value, "40");
            foundRewrittenSet = true;
        }
    }
    EXPECT_TRUE(foundRewrittenSet);
}

TEST_F(RegisterSymbolSyncTest, SplitKeepsLiveInNameAndStripsMovedDest) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "sgprWorkGroup0", "0");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* sub = builder.create(getMCIDByUOp(GFX::s_sub_u32, kRaTestArch));
    sub->addDestReg(namedReg(RegType::S, 12, 1, "sgprWorkGroup0"));
    sub->addSrcReg(namedReg(RegType::S, 0, 1, "sgprWorkGroup0"));
    sub->addSrcReg(StinkyRegister("s", 4, 1));

    // The live-in stays at s0 where `.set sgprWorkGroup0` points; the
    // redefinition moved to s12, so one base now names two registers.
    std::vector<RewrittenOperand> rewritten{
        {sub, /*isDestination=*/true, 0, RegType::S, 0, RegType::S, 12},
        {sub, /*isDestination=*/false, 0, RegType::S, 0, RegType::S, 0},
    };

    SymbolSyncReport report;
    syncRegisterSymbols(*func, rewritten, {}, &report);

    EXPECT_EQ(report.split, 1u);
    EXPECT_EQ(report.setsRewritten, 0u);
    EXPECT_EQ(sub->getSrcRegs()[0].getSymbolicName(), "sgprWorkGroup0");
    EXPECT_FALSE(sub->getDestRegs()[0].hasSymbolicName());

    const std::string asmText = emitSymbolicAssembly(*func);
    EXPECT_TRUE(contains(asmText, "s[sgprWorkGroup0]"));
    EXPECT_TRUE(contains(asmText, "s12"));
    EXPECT_TRUE(contains(asmText, ".set sgprWorkGroup0, 0"));
    // Breadcrumbs were not asked for, so the strip is silent here.
    EXPECT_FALSE(contains(asmText, "was sgprWorkGroup0")) << asmText;
}

// Two names lost on one instruction must annotate in operand order, or the
// output stops being reproducible run to run.
TEST_F(RegisterSymbolSyncTest, TwoStrippedOperandsAnnotateInOperandOrder) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "sgprWorkGroup0", "0");
    appendSetDirective(bb, "sgprTmp", "7");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* keep = builder.create(getMCIDByUOp(GFX::s_mul_i32, kRaTestArch));
    keep->addDestReg(namedReg(RegType::S, 7, 1, "sgprTmp"));
    keep->addSrcReg(namedReg(RegType::S, 0, 1, "sgprWorkGroup0"));
    keep->addSrcReg(StinkyRegister("s", 4, 1));
    StinkyInstruction* sub = builder.create(getMCIDByUOp(GFX::s_sub_u32, kRaTestArch));
    sub->addDestReg(namedReg(RegType::S, 12, 1, "sgprWorkGroup0"));
    sub->addSrcReg(namedReg(RegType::S, 20, 1, "sgprTmp"));
    sub->addSrcReg(StinkyRegister("s", 5, 1));

    std::vector<RewrittenOperand> rewritten{
        {keep, /*isDestination=*/true, 0, RegType::S, 7, RegType::S, 7},
        {keep, /*isDestination=*/false, 0, RegType::S, 0, RegType::S, 0},
        {sub, /*isDestination=*/true, 0, RegType::S, 0, RegType::S, 12},
        {sub, /*isDestination=*/false, 0, RegType::S, 7, RegType::S, 20},
    };

    SymbolSyncOptions options;
    options.emitBreadcrumbs = true;
    SymbolSyncReport report;
    syncRegisterSymbols(*func, rewritten, options, &report);

    EXPECT_EQ(report.namesCleared, 2u);
    const CommentData* comment = sub->getModifier<CommentData>();
    ASSERT_NE(comment, nullptr) << "both stripped operands sit on this instruction";
    EXPECT_EQ(comment->comment, "s12 was sgprWorkGroup0 (split), s20 was sgprTmp (split)");
}

// The note has to name the whole operand, not just its first register, or it
// will not match the `v[20:23]` the emitter prints beside it.
TEST_F(RegisterSymbolSyncTest, RangeOperandNoteNamesTheWholeRange) {
    BasicBlock* bb = entry();
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* read = builder.create(getMCIDByUOp(GFX::ds_load_b128, kRaTestArch));
    read->addDestReg(namedReg(RegType::V, 20, 4, "vgprValuA+0:vgprValuA+3"));
    read->addSrcReg(StinkyRegister("v", 1, 1));

    // No `.set vgprValuA` anywhere in the function, so the name cannot resolve.
    std::vector<RewrittenOperand> rewritten{
        {read, /*isDestination=*/true, 0, RegType::V, 20, RegType::V, 20},
    };

    SymbolSyncOptions options;
    options.emitBreadcrumbs = true;
    syncRegisterSymbols(*func, rewritten, options);

    const CommentData* comment = read->getModifier<CommentData>();
    ASSERT_NE(comment, nullptr);
    EXPECT_EQ(comment->comment, "v[20:23] was vgprValuA+0:vgprValuA+3 (unresolved .set)");
}

TEST_F(RegisterSymbolSyncTest, ASymbolReleasedWithUndefNeverKeepsAStaleName) {
    // TensileLite releases a register name at the end of its scope with a second
    // `.set NAME, UNDEF`. That makes the symbol both redefined *and*
    // unresolvable, and a moved operand must not keep a name that still points
    // at the old index -- the emitter would print the wrong register.
    BasicBlock* bb = entry();
    appendSetDirective(bb, "sgprtdmAGroup0", "20");

    AsmIRBuilder builder(*bb, kRaTestArch);
    // Defined here, so the value is free to move -- a live-in would be pinned and
    // keeping its name would be correct.
    StinkyInstruction* def = builder.create(getMCIDByUOp(GFX::s_mov_b32, kRaTestArch));
    def->addDestReg(namedReg(RegType::S, 20, 1, "sgprtdmAGroup0"));
    def->addSrcReg(StinkyRegister(1));

    StinkyInstruction* use = builder.create(getMCIDByUOp(GFX::s_add_u32, kRaTestArch));
    use->addDestReg(StinkyRegister(RegType::S, 30u, 1u));
    use->addSrcReg(namedReg(RegType::S, 20, 1, "sgprtdmAGroup0"));
    use->addSrcReg(StinkyRegister(RegType::S, 21u, 1u));

    appendSetDirective(bb, "sgprtdmAGroup0", "UNDEF");

    ASSERT_TRUE(liftForAllocation(*func));
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, scalarCompactApply());
    ASSERT_TRUE(result.hasValue()) << result.getError();

    // Compaction moved the operand off s20, so the name must be gone rather than
    // resolving through the stale `.set` to a register the value no longer uses.
    const std::string asmText = emitSymbolicAssembly(*func);
    EXPECT_FALSE(contains(asmText, "sgprtdmAGroup0:")) << asmText;
    EXPECT_FALSE(contains(asmText, "s[sgprtdmAGroup0]")) << asmText;
}

TEST_F(RegisterSymbolSyncTest, AnOperandWhoseNameDisagreesWithItsRegisterIsReported) {
    // The detector exists; for a long time nothing read it. An operand whose
    // name resolves somewhere other than where it sits is the shape of a
    // wrong-register bug -- the emitter prints the name, the allocator reasoned
    // about the index -- so it has to reach a human rather than a discarded
    // struct.
    BasicBlock* bb = entry();
    appendSetDirective(bb, "sgprGSU", "5");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* add = builder.create(getMCIDByUOp(GFX::s_add_u32, kRaTestArch));
    add->addDestReg(StinkyRegister(RegType::S, 30u, 1u));
    add->addSrcReg(placeholderNamedReg("sgprGSU"));  // sits at s0, name says s5
    add->addSrcReg(StinkyRegister(RegType::S, 31u, 1u));

    ASSERT_TRUE(liftForAllocation(*func));
    CompactingGreedyAllocator allocator;
    std::string report;
    Expected<AllocationResult> result =
        allocateRegisters(*func, allocator, scalarCompactApply(), &report);
    ASSERT_TRUE(result.hasValue()) << result.getError();

    EXPECT_TRUE(contains(report, "symbol sync")) << report;
    EXPECT_TRUE(contains(report, "sgprGSU")) << report;
}

TEST_F(RegisterSymbolSyncTest, PlaceholderNameOnlyOperandSurvivesCompactingRun) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "sgprGSU", "5");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* add = builder.create(getMCIDByUOp(GFX::v_add_f32, kRaTestArch));
    add->addDestReg(StinkyRegister("v", 10, 1));
    add->addSrcReg(StinkyRegister("v", 0, 1));
    add->addSrcReg(placeholderNamedReg("sgprGSU"));

    ASSERT_TRUE(liftForAllocation(*func));
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> result = allocateRegisters(*func, allocator, compactApply());
    ASSERT_TRUE(result.hasValue()) << result.getError();

    const std::string asmText = emitSymbolicAssembly(*func);
    EXPECT_TRUE(contains(asmText, "s[sgprGSU]"));
    EXPECT_FALSE(contains(asmText, "s0"));
}

TEST_F(RegisterSymbolSyncTest, BreadcrumbAppendsToExistingCommentOnSplit) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "sgprWorkGroup0", "0");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* sub = builder.create(getMCIDByUOp(GFX::s_sub_u32, kRaTestArch));
    sub->addDestReg(namedReg(RegType::S, 12, 1, "sgprWorkGroup0"));
    sub->addSrcReg(namedReg(RegType::S, 0, 1, "sgprWorkGroup0"));
    sub->addSrcReg(StinkyRegister("s", 4, 1));
    sub->addModifier<CommentData>(CommentData{"producer note"});

    std::vector<RewrittenOperand> rewritten{
        {sub, /*isDestination=*/true, 0, RegType::S, 0, RegType::S, 12},
        {sub, /*isDestination=*/false, 0, RegType::S, 0, RegType::S, 0},
    };

    SymbolSyncOptions options;
    options.emitBreadcrumbs = true;
    syncRegisterSymbols(*func, rewritten, options);

    const CommentData* comment = sub->getModifier<CommentData>();
    ASSERT_NE(comment, nullptr);
    EXPECT_TRUE(contains(comment->comment, "producer note"));
    EXPECT_TRUE(contains(comment->comment, "was sgprWorkGroup0"));
}

TEST_F(RegisterSymbolSyncTest, RegisterMapInsertsTextblock) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "vgprTmp", "40");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* add = builder.create(getMCIDByUOp(GFX::v_add_f32, kRaTestArch));
    add->addDestReg(namedReg(RegType::V, 40, 1, "vgprTmp"));
    add->addSrcReg(StinkyRegister("v", 0, 1));
    add->addSrcReg(StinkyRegister("v", 1, 1));

    ASSERT_TRUE(liftForAllocation(*func));
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> coloured = allocator.allocate(AllocationSetup(*func).context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    SSADestructionResult destroyed = destroyAttachedSSA(*func, *coloured);
    ASSERT_TRUE(destroyed.ok());

    SymbolSyncOptions options;
    options.emitRegisterMap = true;
    syncRegisterSymbols(*func, destroyed.rewritten, options);

    bool foundMap = false;
    for (const IRBase& ir : *bb) {
        const auto* directive = dyn_cast<AsmDirective>(&ir);
        if (directive == nullptr || directive->kind != AsmDirectiveKind::TEXTBLOCK) continue;
        if (contains(directive->value, "register-map")) foundMap = true;
    }
    EXPECT_TRUE(foundMap);
}

TEST_F(RegisterSymbolSyncTest, ImmediateSetUntouched) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "MT0", "64");
    appendSetDirective(bb, "vgprTmp", "40");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* add = builder.create(getMCIDByUOp(GFX::v_add_f32, kRaTestArch));
    add->addDestReg(namedReg(RegType::V, 40, 1, "vgprTmp"));
    add->addSrcReg(StinkyRegister("v", 0, 1));
    add->addSrcReg(StinkyRegister("v", 1, 1));

    ASSERT_TRUE(liftForAllocation(*func));
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> coloured = allocator.allocate(AllocationSetup(*func).context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    SSADestructionResult destroyed = destroyAttachedSSA(*func, *coloured);
    ASSERT_TRUE(destroyed.ok());
    syncRegisterSymbols(*func, destroyed.rewritten);

    for (const BasicBlock& block : *func) {
        for (const IRBase& ir : block) {
            const auto* directive = dyn_cast<AsmDirective>(&ir);
            if (directive == nullptr || directive->kind != AsmDirectiveKind::SET) continue;
            if (directive->symbol == "MT0") {
                EXPECT_EQ(directive->value, "64");
            }
        }
    }
}

// A `.set` base shared by several offsets lands on several indices by construction.
// `sgprSrdD+1` sitting at s21 under `.set sgprSrdD, 20` is accurate, not split.
TEST_F(RegisterSymbolSyncTest, OffsetNamesSurviveWhenTheOperandsDoNotMove) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "sgprSrdD", "20");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* lo = builder.create(getMCIDByUOp(GFX::s_add_u32, kRaTestArch));
    lo->addDestReg(namedReg(RegType::S, 20, 1, "sgprSrdD+0"));
    lo->addSrcReg(namedReg(RegType::S, 20, 1, "sgprSrdD+0"));
    lo->addSrcReg(StinkyRegister("s", 4, 1));
    StinkyInstruction* hi = builder.create(getMCIDByUOp(GFX::s_addc_u32, kRaTestArch));
    hi->addDestReg(namedReg(RegType::S, 21, 1, "sgprSrdD+1"));
    hi->addSrcReg(namedReg(RegType::S, 21, 1, "sgprSrdD+1"));
    hi->addSrcReg(StinkyRegister("s", 5, 1));

    std::vector<RewrittenOperand> rewritten{
        {lo, /*isDestination=*/true, 0, RegType::S, 20, RegType::S, 20},
        {lo, /*isDestination=*/false, 0, RegType::S, 20, RegType::S, 20},
        {hi, /*isDestination=*/true, 0, RegType::S, 21, RegType::S, 21},
        {hi, /*isDestination=*/false, 0, RegType::S, 21, RegType::S, 21},
    };

    SymbolSyncReport report;
    syncRegisterSymbols(*func, rewritten, {}, &report);

    EXPECT_EQ(report.split, 0u);
    EXPECT_EQ(report.namesCleared, 0u);
    EXPECT_EQ(report.setsRewritten, 0u);
    EXPECT_EQ(lo->getDestRegs()[0].getSymbolicName(), "sgprSrdD+0");
    EXPECT_EQ(hi->getDestRegs()[0].getSymbolicName(), "sgprSrdD+1");
    EXPECT_EQ(hi->getSrcRegs()[0].getSymbolicName(), "sgprSrdD+1");

    const std::string asmText = emitSymbolicAssembly(*func);
    EXPECT_TRUE(contains(asmText, "s[sgprSrdD+1]"));
    EXPECT_FALSE(contains(asmText, "s21"));
}

// Moving a tuple shifts every offset by the same delta, so one `.set` rewrite
// keeps the whole group correct and no name has to be stripped.
TEST_F(RegisterSymbolSyncTest, MovedTupleShiftsEveryOffsetByTheSameDelta) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "sgprSrdD", "20");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* lo = builder.create(getMCIDByUOp(GFX::s_add_u32, kRaTestArch));
    lo->addDestReg(namedReg(RegType::S, 40, 1, "sgprSrdD+0"));
    lo->addSrcReg(StinkyRegister("s", 4, 1));
    lo->addSrcReg(StinkyRegister("s", 5, 1));
    StinkyInstruction* hi = builder.create(getMCIDByUOp(GFX::s_addc_u32, kRaTestArch));
    hi->addDestReg(namedReg(RegType::S, 41, 1, "sgprSrdD+1"));
    hi->addSrcReg(StinkyRegister("s", 6, 1));
    hi->addSrcReg(StinkyRegister("s", 7, 1));

    std::vector<RewrittenOperand> rewritten{
        {lo, /*isDestination=*/true, 0, RegType::S, 20, RegType::S, 40},
        {hi, /*isDestination=*/true, 0, RegType::S, 21, RegType::S, 41},
    };

    SymbolSyncReport report;
    syncRegisterSymbols(*func, rewritten, {}, &report);

    EXPECT_EQ(report.movedUniquely, 1u);
    EXPECT_EQ(report.setsRewritten, 1u);
    EXPECT_EQ(report.split, 0u);
    EXPECT_EQ(report.namesCleared, 0u);
    EXPECT_EQ(lo->getDestRegs()[0].getSymbolicName(), "sgprSrdD+0");
    EXPECT_EQ(hi->getDestRegs()[0].getSymbolicName(), "sgprSrdD+1");

    for (const BasicBlock& block : *func) {
        for (const IRBase& ir : block) {
            const auto* directive = dyn_cast<AsmDirective>(&ir);
            if (directive == nullptr || directive->kind != AsmDirectiveKind::SET) continue;
            if (directive->symbol == "sgprSrdD") {
                EXPECT_EQ(directive->value, "40");
            }
        }
    }
}

// Only part of a tuple moving is a genuine split: the `.set` stays put and just
// the operand whose name no longer resolves loses it.
TEST_F(RegisterSymbolSyncTest, PartiallyMovedTupleStripsOnlyTheDivergentOffset) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "sgprSrdD", "20");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* lo = builder.create(getMCIDByUOp(GFX::s_add_u32, kRaTestArch));
    lo->addDestReg(namedReg(RegType::S, 20, 1, "sgprSrdD+0"));
    lo->addSrcReg(StinkyRegister("s", 4, 1));
    lo->addSrcReg(StinkyRegister("s", 5, 1));
    StinkyInstruction* hi = builder.create(getMCIDByUOp(GFX::s_addc_u32, kRaTestArch));
    hi->addDestReg(namedReg(RegType::S, 60, 1, "sgprSrdD+1"));
    hi->addSrcReg(StinkyRegister("s", 6, 1));
    hi->addSrcReg(StinkyRegister("s", 7, 1));

    std::vector<RewrittenOperand> rewritten{
        {lo, /*isDestination=*/true, 0, RegType::S, 20, RegType::S, 20},
        {hi, /*isDestination=*/true, 0, RegType::S, 21, RegType::S, 60},
    };

    SymbolSyncReport report;
    syncRegisterSymbols(*func, rewritten, {}, &report);

    EXPECT_EQ(report.split, 1u);
    EXPECT_EQ(report.namesCleared, 1u);
    EXPECT_EQ(report.setsRewritten, 0u);
    EXPECT_EQ(lo->getDestRegs()[0].getSymbolicName(), "sgprSrdD+0");
    EXPECT_FALSE(hi->getDestRegs()[0].hasSymbolicName());
}

TEST_F(RegisterSymbolSyncTest, RedefinedSymbolClearsEligibleNames) {
    BasicBlock* bb = entry();
    appendSetDirective(bb, "vgprTmp", "40");
    appendSetDirective(bb, "vgprTmp", "50");
    AsmIRBuilder builder(*bb, kRaTestArch);
    StinkyInstruction* add = builder.create(getMCIDByUOp(GFX::v_add_f32, kRaTestArch));
    add->addDestReg(namedReg(RegType::V, 50, 1, "vgprTmp"));
    add->addSrcReg(StinkyRegister("v", 0, 1));
    add->addSrcReg(StinkyRegister("v", 1, 1));

    ASSERT_TRUE(liftForAllocation(*func));
    CompactingGreedyAllocator allocator;
    Expected<AllocationResult> coloured = allocator.allocate(AllocationSetup(*func).context());
    ASSERT_TRUE(coloured.hasValue()) << coloured.getError();
    SSADestructionResult destroyed = destroyAttachedSSA(*func, *coloured);
    ASSERT_TRUE(destroyed.ok());
    SymbolSyncReport report;
    syncRegisterSymbols(*func, destroyed.rewritten, {}, &report);
    EXPECT_GE(report.unresolvable, 1u);
    EXPECT_FALSE(add->getDestRegs()[0].hasSymbolicName());
}

}  // namespace
