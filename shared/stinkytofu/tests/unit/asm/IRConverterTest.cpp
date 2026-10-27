// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"

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
