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
#include <utility>

#include "CanonicalSSATestUtils.hpp"
#include "TestHelpers.hpp"
#include "stinkytofu/analysis/ssa/CanonicalSSA.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/ssa/CanonicalSSAPrinter.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

class CanonicalSSAPrinterTest : public ::testing::Test {
   protected:
    void SetUp() override {
        func = std::make_unique<Function>("kernel");
        setFunctionArch(*func, kArch);
        entry = func->createBasicBlock("entry");
    }

    std::string dump(const CanonicalSSAPrinterOptions& options = {}) {
        ssa = builder.take();
        return canonicalSSAToString(*func, ssa, options);
    }

    std::unique_ptr<Function> func;
    BasicBlock* entry = nullptr;
    CanonicalSSABuilder builder;
    CanonicalSSA ssa;
};

}  // namespace

TEST_F(CanonicalSSAPrinterTest, EmptyGraphPrintsOnlyTheFunctionAndBlock) {
    const std::string text = dump();
    EXPECT_EQ(text,
              "ssa.func @kernel {\n"
              "  ^entry:\n"
              "}\n");
}

TEST_F(CanonicalSSAPrinterTest, StraightLineDumpIsExact) {
    StinkyInstruction* add = createVAddInBlock(entry, kArch, /*dest=*/2, /*src0=*/0, /*src1=*/1);
    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addUndef(builder, 1);
    bindInstruction(builder, *add, {{in0}, {in1}});

    EXPECT_EQ(dump(),
              "ssa.func @kernel {\n"
              "  initial_values:\n"
              "    %1:v = livein { origin = v0 }\n"
              "    %2:v = undef { origin = v1 }\n"
              "  ^entry:\n"
              "    %3:v = \"st.v_add_f32\"(src0 = [%1:v], src1 = [%2:v]) "
              "{ inst = #0, origin = [v2] }\n"
              "      // physical: v2 = \"st.v_add_f32\"(v0, v1)\n"
              "}\n");
}

TEST_F(CanonicalSSAPrinterTest, RepeatedDefinitionsPrintDistinctValues) {
    StinkyInstruction* first = createVAddInBlock(entry, kArch, 2, 0, 1);
    StinkyInstruction* second = createVAddInBlock(entry, kArch, 2, 2, 1);
    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addLiveIn(builder, 1);
    const std::vector<SSAValueID> firstDef = bindInstruction(builder, *first, {{in0}, {in1}});
    bindInstruction(builder, *second, {{firstDef[0]}, {in1}});

    const std::string text = dump();
    EXPECT_TRUE(contains(text, "%3:v = \"st.v_add_f32\"(src0 = [%1:v], src1 = [%2:v])")) << text;
    EXPECT_TRUE(contains(text, "%4:v = \"st.v_add_f32\"(src0 = [%3:v], src1 = [%2:v])")) << text;
}

TEST_F(CanonicalSSAPrinterTest, PartialDefinitionKeepsEveryUnitVisible) {
    StinkyInstruction* load = createDsReadB128InBlock(entry, kArch, /*dest=*/4, /*addr=*/0);
    const SSAValueID addr = addLiveIn(builder, 0);
    const std::vector<SSAValueID> loaded = bindInstruction(builder, *load, {{addr}});

    StinkyInstruction* store = createDSWriteInBlock(entry, kArch, /*addr=*/0, /*data=*/4);
    bindInstruction(builder, *store, {{addr}, {loaded[0], loaded[1]}});

    const std::string text = dump();
    EXPECT_TRUE(contains(text, "%2:v, %3:v, %4:v, %5:v = \"st.ds_load_b128\"(src0 = [%1:v])"))
        << text;
    EXPECT_TRUE(contains(text, "origin = [v4, v5, v6, v7]")) << text;
    EXPECT_TRUE(contains(text, "\"st.ds_store_b64\"(src0 = [%1:v], src1 = [%2:v, %3:v])")) << text;
}

TEST_F(CanonicalSSAPrinterTest, PhiPrintsPredecessorOrderedEdges) {
    BasicBlock* left = func->createBasicBlock("left");
    BasicBlock* right = func->createBasicBlock("right");
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, left);
    func->addEdge(entry, right);
    func->addEdge(left, join);
    func->addEdge(right, join);

    StinkyInstruction* leftDef = createVAddInBlock(left, kArch, 5, 0, 1);
    StinkyInstruction* rightDef = createVAddInBlock(right, kArch, 5, 1, 0);
    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addLiveIn(builder, 1);
    const SSAValueID leftValue = bindInstruction(builder, *leftDef, {{in0}, {in1}}).front();
    const SSAValueID rightValue = bindInstruction(builder, *rightDef, {{in1}, {in0}}).front();
    addPhi(builder, *join, vgprKey(5), {leftValue, rightValue});

    const std::string text = dump();
    EXPECT_TRUE(contains(text,
                         "  ^join:\n    %5:v = phi(^left: %3:v, ^right: %4:v) "
                         "{ origin = v5 }\n"))
        << text;
}

TEST_F(CanonicalSSAPrinterTest, ProvenanceCanBeDisabled) {
    StinkyInstruction* add = createVAddInBlock(entry, kArch, 2, 0, 1);
    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addLiveIn(builder, 1);
    bindInstruction(builder, *add, {{in0}, {in1}});

    CanonicalSSAPrinterOptions options;
    options.printProvenance = false;
    const std::string text = dump(options);
    EXPECT_FALSE(contains(text, "origin")) << text;
    EXPECT_TRUE(contains(text, "%1:v = livein\n")) << text;
    EXPECT_TRUE(contains(text, "{ inst = #0 }")) << text;
}

TEST_F(CanonicalSSAPrinterTest, PhysicalCommentCanBeDisabled) {
    StinkyInstruction* add = createVAddInBlock(entry, kArch, 2, 0, 1);
    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addLiveIn(builder, 1);
    bindInstruction(builder, *add, {{in0}, {in1}});

    CanonicalSSAPrinterOptions options;
    options.printPhysicalInstruction = false;
    EXPECT_FALSE(contains(dump(options), "// physical:"));
}

TEST_F(CanonicalSSAPrinterTest, UseListsArePrintedInFunctionOrder) {
    StinkyInstruction* first = createVAddInBlock(entry, kArch, 2, 0, 0);
    StinkyInstruction* second = createVAddInBlock(entry, kArch, 3, 0, 1);
    const SSAValueID in0 = addLiveIn(builder, 0);
    const SSAValueID in1 = addLiveIn(builder, 1);
    bindInstruction(builder, *first, {{in0}, {in0}});
    bindInstruction(builder, *second, {{in0}, {in1}});

    CanonicalSSAPrinterOptions options;
    options.printUses = true;
    const std::string text = dump(options);
    EXPECT_TRUE(contains(text,
                         "uses = [{ inst = #0, src = 0, unit = 0 }, "
                         "{ inst = #0, src = 1, unit = 0 }, "
                         "{ inst = #1, src = 0, unit = 0 }]"))
        << text;
}

TEST_F(CanonicalSSAPrinterTest, PhiUsesArePrintedAfterInstructionUses) {
    BasicBlock* join = func->createBasicBlock("join");
    func->addEdge(entry, join);

    StinkyInstruction* consumer = createVAddInBlock(entry, kArch, 2, 5, 5);
    const SSAValueID in5 = addLiveIn(builder, 5);
    bindInstruction(builder, *consumer, {{in5}, {in5}});
    addPhi(builder, *join, vgprKey(5), {in5});

    CanonicalSSAPrinterOptions options;
    options.printUses = true;
    const std::string text = dump(options);
    EXPECT_TRUE(contains(text,
                         "uses = [{ inst = #0, src = 0, unit = 0 }, "
                         "{ inst = #0, src = 1, unit = 0 }, "
                         "{ phi#1, pred = ^entry }]"))
        << text;
}

TEST_F(CanonicalSSAPrinterTest, DuplicateAndEmptyBlockLabelsFallBackToPositionalNames) {
    func->createBasicBlock("dup");
    func->createBasicBlock("dup");
    func->createBasicBlock("");

    const std::string text = dump();
    EXPECT_TRUE(contains(text, "  ^entry:\n")) << text;
    EXPECT_TRUE(contains(text, "  ^bb1:\n")) << text;
    EXPECT_TRUE(contains(text, "  ^bb2:\n")) << text;
    EXPECT_TRUE(contains(text, "  ^bb3:\n")) << text;
    EXPECT_FALSE(contains(text, "^dup")) << text;
}

TEST_F(CanonicalSSAPrinterTest, UnmappedInstructionIsMarked) {
    AsmIRBuilder irBuilder(*entry, kArch);
    irBuilder.create(getMCIDByUOp(GFX::s_nop, kArch));

    const std::string text = dump();
    EXPECT_TRUE(contains(text, "\"st.s_nop\"() { inst = #0, unmapped }")) << text;
}

TEST_F(CanonicalSSAPrinterTest, LabelsAreNotPrintedAsDataflow) {
    AsmIRBuilder irBuilder(*entry, kArch);
    irBuilder.createLabel("loop");

    EXPECT_FALSE(contains(dump(), "LABEL"));
}

TEST_F(CanonicalSSAPrinterTest, InvalidValueReferencePrintsAMarkerInsteadOfCrashing) {
    StinkyInstruction* add = createVAddInBlock(entry, kArch, 2, 0, 1);
    SSAInstructionInfo info;
    info.sources.resize(2);
    info.sources[0].units = {42};  // never created
    info.destinations.resize(1);
    builder.setInstructionInfo(*add, std::move(info));

    const std::string text = dump();
    EXPECT_TRUE(contains(text, "<invalid-ssa:%42>")) << text;
}

TEST_F(CanonicalSSAPrinterTest, ForeignReferencesPrintMarkers) {
    Function other("other");
    setFunctionArch(other, kArch);
    BasicBlock* otherBlock = other.createBasicBlock("elsewhere");
    StinkyInstruction* foreign = createVAddInBlock(otherBlock, kArch, 2, 0, 1);

    SSAValue value;
    value.kind = SSAValueKind::InstructionDef;
    value.origin = vgprKey(2);
    value.definingInstruction = foreign;
    builder.addValue(std::move(value));

    const std::string text = dump();
    EXPECT_TRUE(contains(text, "unprinted_values:")) << text;
    EXPECT_TRUE(contains(text, "<foreign-instruction>")) << text;
}

TEST_F(CanonicalSSAPrinterTest, MissingGraphPrintsPlaceholder) {
    // What a dump of a function nobody could lift looks like. The printer offers
    // this separately because deciding that a function has no graph belongs to
    // the caller, not to the printer.
    std::ostringstream out;
    CanonicalSSAPrinter(out).printMissing(*func);

    EXPECT_EQ(out.str(),
              "ssa.func @kernel {\n"
              "  <no canonical SSA attached>\n"
              "}\n");
}

TEST_F(CanonicalSSAPrinterTest, OutputIsByteIdenticalAcrossRepeatedPrints) {
    StinkyInstruction* load = createDsReadB128InBlock(entry, kArch, 4, 0);
    const SSAValueID addr = addLiveIn(builder, 0);
    const std::vector<SSAValueID> loaded = bindInstruction(builder, *load, {{addr}});
    StinkyInstruction* store = createDSWriteInBlock(entry, kArch, 0, 4);
    bindInstruction(builder, *store, {{addr}, {loaded[0], loaded[1]}});

    CanonicalSSAPrinterOptions options;
    options.printUses = true;
    ssa = builder.take();

    const std::string first = canonicalSSAToString(*func, ssa, options);
    const std::string second = canonicalSSAToString(*func, ssa, options);
    EXPECT_EQ(first, second);
}
