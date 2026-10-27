// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/support/Casting.hpp"

using namespace stinkytofu;

namespace {
struct Converted {
    StinkyErrorCode code;
    std::vector<Diagnostic> diagnostics;
    size_t entryInstructions;
};

Converted convert(const std::string& text, Function& func) {
    MultiParseResult parsed = parseAllSourceStringsWithDiagnostics(text);
    EXPECT_FALSE(parsed.hasErrors());
    EXPECT_EQ(parsed.functions.size(), 1u);
    Converted out{};
    out.code = StinkyIRConverter::populateFunctionFromParsed(
        *parsed.functions.front(), func, getGfxArchID(12, 5, 0), out.diagnostics);
    out.entryInstructions = func.getEntryBlock()->size();
    return out;
}

std::vector<const BasicBlock*> blocksOf(const Function& func) {
    std::vector<const BasicBlock*> blocks;
    for (const BasicBlock& bb : func) blocks.push_back(&bb);
    return blocks;
}

std::vector<const LabelData*> labelsIn(const BasicBlock& bb) {
    std::vector<const LabelData*> labels;
    for (const IRBase& ir : bb) {
        const auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst != nullptr && inst->getUnifiedOpcode() == GFX::LABEL)
            labels.push_back(inst->getModifier<LabelData>());
    }
    return labels;
}

std::string print(const Function& func) {
    std::ostringstream out;
    AsmPrinter(out).print(func);
    return out.str();
}

AsmDirective* addDirective(AsmIRBuilder& builder, AsmDirectiveKind kind, const std::string& name,
                           const std::string& symbol, const std::string& value) {
    AsmDirective* directive = builder.createIR<AsmDirective>();
    directive->kind = kind;
    directive->name = name;
    directive->symbol = symbol;
    directive->value = value;
    return directive;
}
}  // namespace

TEST(IRConverterTest, UnknownMnemonicFailsAndIsLeftOut) {
    const std::string text = R"(st.func @f() {
^entry:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  "st.s_nopp"(0) { issueCycles = 1, latencyCycles = 1 }
}
)";
    Function func("f");
    const Converted result = convert(text, func);

    EXPECT_EQ(result.code, StinkyErrorCode::CONVERT_ERROR);
    EXPECT_EQ(result.entryInstructions, 1u);
    ASSERT_EQ(result.diagnostics.size(), 1u);
    EXPECT_EQ(result.diagnostics[0].getLevel(), Diagnostic::Level::Error);
    EXPECT_EQ(result.diagnostics[0].getMessage(), "Unknown mnemonic 's_nopp' for gfx1250");
    EXPECT_EQ(result.diagnostics[0].getLine(), 4u);
}

TEST(IRConverterTest, DanglingSuccessorFailsAndIsLeftOut) {
    const std::string text = R"(st.func @f() {
^entry:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^loop, ^lopo
^loop:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
}
)";
    Function func("f");
    const Converted result = convert(text, func);

    EXPECT_EQ(result.code, StinkyErrorCode::CONVERT_ERROR);
    ASSERT_EQ(func.getEntryBlock()->getSuccessors().size(), 1u);
    EXPECT_EQ(func.getEntryBlock()->getSuccessors()[0]->getLabel(), "loop");
    ASSERT_EQ(result.diagnostics.size(), 1u);
    EXPECT_EQ(result.diagnostics[0].getMessage(),
              "Successor '^lopo' of block '^entry' is not a block in this function");
    EXPECT_EQ(result.diagnostics[0].getLine(), 4u);
}

TEST(IRConverterTest, FenceConvertsWithoutAnError) {
    const std::string text = R"(st.func @f() {
^entry:
  "st.FENCE"() { issueCycles = 0, latencyCycles = 0 }
}
)";
    Function func("f");
    const Converted result = convert(text, func);

    EXPECT_EQ(result.code, StinkyErrorCode::SUCCESS);
    EXPECT_TRUE(result.diagnostics.empty());
    EXPECT_EQ(result.entryInstructions, 1u);
}

TEST(IRConverterTest, MarkedOrNamedBlocksStartWithTheirLabel) {
    const std::string text = R"(st.func @f() {
^entry:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^loop
^loop:
  "st.s_cbranch_scc1"(loop) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^loop, ^kept
^kept: { label, alignment = 16 }
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^plain
^plain:
  "st.s_endpgm"() { issueCycles = 1, latencyCycles = 1 }
}
)";
    Function func("f");
    const Converted result = convert(text, func);

    ASSERT_EQ(result.code, StinkyErrorCode::SUCCESS);
    const std::vector<const BasicBlock*> blocks = blocksOf(func);
    ASSERT_EQ(blocks.size(), 4u);
    EXPECT_TRUE(labelsIn(*blocks[0]).empty());
    // ^loop is named by the branch, so it gets a label without the attribute.
    ASSERT_EQ(labelsIn(*blocks[1]).size(), 1u);
    EXPECT_EQ(labelsIn(*blocks[1])[0]->label, "loop");
    EXPECT_EQ(labelsIn(*blocks[1])[0]->alignment, 1);
    ASSERT_EQ(labelsIn(*blocks[2]).size(), 1u);
    EXPECT_EQ(labelsIn(*blocks[2])[0]->label, "kept");
    EXPECT_EQ(labelsIn(*blocks[2])[0]->alignment, 16);
    EXPECT_TRUE(labelsIn(*blocks[3]).empty());
}

TEST(IRConverterTest, SpelledOutLeadingLabelIsNotDoubled) {
    const std::string text = R"(st.func @f() {
^loop: { label }
  loop:
  "st.s_cbranch_scc1"(loop) { issueCycles = 1, latencyCycles = 1 }
}
)";
    Function func("f");
    const Converted result = convert(text, func);

    ASSERT_EQ(result.code, StinkyErrorCode::SUCCESS);
    EXPECT_EQ(labelsIn(*func.getEntryBlock()).size(), 1u);
}

TEST(IRConverterTest, TextDirectivesKeepTheirKind) {
    const std::string text = R"(st.func @f() {
^entry:
  "st.asm_directive"("TEXTBLOCK", ".text\n")
  "st.asm_directive"("MACRO", ".macro M\n.endm\n")
  "st.asm_directive"("IF", ".if 1\n")
  "st.asm_directive"("ENDIF", ".endif\n")
  "st.asm_directive"(".set", "x", "0")
}
)";
    Function func("f");
    const Converted result = convert(text, func);

    ASSERT_EQ(result.code, StinkyErrorCode::SUCCESS);
    std::vector<const AsmDirective*> directives;
    for (const IRBase& ir : *func.getEntryBlock()) {
        if (const auto* directive = dyn_cast<AsmDirective>(&ir)) directives.push_back(directive);
    }
    ASSERT_EQ(directives.size(), 5u);
    EXPECT_EQ(directives[0]->kind, AsmDirectiveKind::TEXTBLOCK);
    EXPECT_EQ(directives[0]->value, ".text\n");
    EXPECT_EQ(directives[1]->kind, AsmDirectiveKind::MACRO);
    EXPECT_EQ(directives[1]->value, ".macro M\n.endm\n");
    EXPECT_EQ(directives[2]->kind, AsmDirectiveKind::IF);
    EXPECT_EQ(directives[3]->kind, AsmDirectiveKind::ENDIF);
    EXPECT_EQ(directives[4]->kind, AsmDirectiveKind::SET);
    EXPECT_EQ(directives[4]->symbol, "x");
    EXPECT_EQ(directives[4]->value, "0");
}

TEST(IRConverterTest, PrintedFunctionReloadsWhole) {
    const GfxArchID arch = getGfxArchID(12, 5, 0);
    Function func("f");
    BasicBlock* entry = func.createBasicBlock("entry");
    BasicBlock* loop = func.createBasicBlock("label_loop");

    AsmIRBuilder head(*entry, arch);
    addDirective(head, AsmDirectiveKind::TEXTBLOCK, "", "",
                 ".amdgcn_target \"amdgcn-amd-amdhsa--gfx1250\"\n  // C:\\path\n");
    addDirective(head, AsmDirectiveKind::SET, ".set", "vgprBase", "vgprMXSBase+0");
    addDirective(head, AsmDirectiveKind::ALIGN, ".align", "16", "")->intValue = 16;
    addDirective(head, AsmDirectiveKind::MACRO, ".macro", "M",
                 ".macro M a\n  v_mov_b32 v[\\a], 0\n.endm\n");
    head.createLabel("label_mid", 16);
    head.create(getMCIDByUOp(GFX::s_nop, arch))->addSrcReg(StinkyRegister(0));

    AsmIRBuilder body(*loop, arch);
    body.createLabel("label_loop");
    body.create(getMCIDByUOp(GFX::s_cbranch_scc1, arch))
        ->addSrcReg(StinkyRegister(std::string("label_loop")));
    func.addEdge(entry, loop);
    func.addEdge(loop, loop);

    const std::string printed = print(func);
    Function reloaded("f");
    const Converted result = convert(printed, reloaded);

    ASSERT_EQ(result.code, StinkyErrorCode::SUCCESS);
    EXPECT_TRUE(result.diagnostics.empty());
    EXPECT_EQ(print(reloaded), printed);
    EXPECT_NE(printed.find("^label_loop: { label }"), std::string::npos) << printed;
    EXPECT_NE(printed.find("label_mid: { alignment = 16 }"), std::string::npos) << printed;
}
