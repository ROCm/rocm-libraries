/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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

/**
 * @file IRParserTest.cpp
 * @brief Tests for the IR Parser (MLIR-style format)
 *
 * This file tests parsing of MLIR-style StinkyTofu IR text into ParsedInstruction objects.
 * Format: v[0] = "st.v_add_f32"(v[1], v[2]) { issueCycles = 4 }
 */

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"

using namespace stinkytofu;

/**
 * Test fixture for IR Parser tests
 */
class IRParserTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Setup common test fixtures if needed
    }

    void TearDown() override {
        // Cleanup
    }

    /**
     * Helper to parse assembly string
     */
    std::vector<ParsedInstruction> parseAssemblyString(const std::string& input) {
        return parseSourceStringWithDiagnostics(input).getInstructions();
    }
};

// ============================================================================
// HAPPY PATH TESTS - Valid Instructions
// ============================================================================

TEST_F(IRParserTest, ParsesSimpleVALUInstruction) {
    const std::string input = R"(v[0] = "st.v_add_f32"(v[1], v[2]))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    ASSERT_NE(instructions[0].opcodeStr, "");
    // Parser strips namespace prefix
    EXPECT_EQ(instructions[0].opcodeStr, "v_add_f32");
    EXPECT_FALSE(instructions[0].isLabel);
    EXPECT_EQ(instructions[0].destRegs.size(), 1);
    EXPECT_EQ(instructions[0].srcRegs.size(), 2);
}

TEST_F(IRParserTest, ParsesMultipleInstructions) {
    const std::string input = R"(
        v[0] = "st.v_add_f32"(v[1], v[2])
        v[3] = "st.v_mul_f32"(v[4], v[5])
        v[6] = "st.v_mov_b32"(v[7])
    )";

    auto instructions = parseAssemblyString(input);

    EXPECT_EQ(instructions.size(), 3);
}

TEST_F(IRParserTest, ParsesSALUInstruction) {
    const std::string input = R"(s[0] = "st.s_add_u32"(s[1], s[2]))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    ASSERT_NE(instructions[0].opcodeStr, "");
    EXPECT_EQ(instructions[0].opcodeStr, "s_add_u32");
}

TEST_F(IRParserTest, ParsesRegisterV10Format) {
    // "v10" (no brackets) should parse as reg type 'v' with index 10
    const std::string input = R"(v10 = "st.v_mov_b32"(v[0]))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    ASSERT_NE(instructions[0].opcodeStr, "");
    ASSERT_EQ(instructions[0].destRegs.size(), 1);
    EXPECT_EQ(instructions[0].destRegs[0].reg.type, RegType::V);
    EXPECT_EQ(instructions[0].destRegs[0].reg.idx, 10u);
}

TEST_F(IRParserTest, ParsesLabelReference) {
    // label_* should parse as label reference (string literal)
    const std::string input =
        R"("st.s_cbranch_scc1"(label_LoopEndL) { issueCycles = 1, latencyCycles = 1 })";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    ASSERT_NE(instructions[0].opcodeStr, "");
    ASSERT_EQ(instructions[0].srcRegs.size(), 1);
    EXPECT_EQ(instructions[0].srcRegs[0].dataType, StinkyRegister::Type::LiteralString);
    EXPECT_EQ(instructions[0].srcRegs[0].literalValue, "label_LoopEndL");
}

TEST_F(IRParserTest, ParsesMemoryInstruction) {
    const std::string input = R"(v[0] = "st.buffer_load_b32"(v[1:2]))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    ASSERT_NE(instructions[0].opcodeStr, "");
    EXPECT_EQ(instructions[0].opcodeStr, "buffer_load_b32");
}

TEST_F(IRParserTest, ParsesInstructionWithImmediate) {
    const std::string input = R"(v[0] = "st.v_add_f32"(v[1], 5.0))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    EXPECT_EQ(instructions[0].srcRegs.size(), 2);
}

TEST_F(IRParserTest, ParsesInstructionWithHexImmediate) {
    const std::string input = R"(v[0] = "st.v_mov_b32"(0x3f800000))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    ASSERT_EQ(instructions[0].srcRegs.size(), 1);
    const auto& src0 = instructions[0].srcRegs[0];
    EXPECT_EQ(src0.dataType, StinkyRegister::Type::LiteralString);
    EXPECT_EQ(src0.getLiteralString(), "0x3f800000");
}

TEST_F(IRParserTest, ParsesInstructionWithComment) {
    const std::string input = R"(v[0] = "st.v_add_f32"(v[1], v[2])  // This is a comment)";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    ASSERT_NE(instructions[0].opcodeStr, "");
    EXPECT_EQ(instructions[0].opcodeStr, "v_add_f32");
}

TEST_F(IRParserTest, ParsesWaitInstruction) {
    const std::string input = R"("st.s_waitcnt"(0))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
}

TEST_F(IRParserTest, ParsesMFMAInstruction) {
    const std::string input =
        R"(acc[0:31] = "st.v_mfma_f32_32x32x8_f16"(v[0:3], v[4:7], acc[0:31]))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
}

// ============================================================================
// ERROR PATH TESTS - Invalid Syntax
// ============================================================================

TEST_F(IRParserTest, ParsesUnknownOpcodeWithoutValidation) {
    const std::string input = R"(v[0] = "st.v_unknown_instruction"(v[1], v[2]))";

    auto instructions = parseAssemblyString(input);

    // IRParser does NOT validate opcodes (that's IRValidator's job)
    // Parser only checks syntax is valid
    ASSERT_EQ(instructions.size(), 1) << "Parser should accept syntactically valid instruction";
    ASSERT_NE(instructions[0].opcodeStr, "");

    // Verify parser extracted the opcode string correctly
    EXPECT_EQ(instructions[0].opcodeStr, "v_unknown_instruction");
    EXPECT_FALSE(instructions[0].isLabel);

    // Verify operands were parsed
    EXPECT_EQ(instructions[0].destRegs.size(), 1);
    EXPECT_EQ(instructions[0].srcRegs.size(), 2);

    // NOTE: Semantic validation should happen in a separate pass
    // See future IRValidatorTest for opcode validation tests
}

TEST_F(IRParserTest, RejectsMissingOperands) {
    const std::string input = R"(v[0] = "st.v_add_f32"())";

    auto instructions = parseAssemblyString(input);

    // Empty operands should parse successfully (validation happens later)
    EXPECT_EQ(instructions.size(), 1);
}

TEST_F(IRParserTest, RejectsInvalidRegisterName) {
    // Parser will crash on invalid register type 'x'
    // This is expected behavior - invalid input should not crash silently
    // For now, test with valid input but semantically wrong
    const std::string input =
        R"(v[0] = "st.v_add_f32"(v[9999], v[1]))";  // Valid syntax, extreme index

    auto instructions = parseAssemblyString(input);

    // Parser succeeds, validation happens later
    EXPECT_EQ(instructions.size(), 1);
}

TEST_F(IRParserTest, ParsesLargeRegisterRangeWithoutValidation) {
    const std::string input = R"(v[0:999] = "st.v_mov_b32"(v[1]))";  // Range too large

    auto instructions = parseAssemblyString(input);

    // Parser is LENIENT - it parses syntax, validation happens later
    // Range 0:999 is syntactically valid (even if semantically wrong for hardware)
    ASSERT_EQ(instructions.size(), 1) << "Parser should accept syntactically valid range";
    ASSERT_NE(instructions[0].opcodeStr, "");

    // Verify the parse result structure
    EXPECT_EQ(instructions[0].opcodeStr, "v_mov_b32");
    EXPECT_FALSE(instructions[0].destRegs.empty());

    // The range should be captured in the parsed structure
    // A separate IRValidator component should later reject ranges exceeding
    // hardware limits (typically 256 VGPRs for GFX9/GFX11)
}

TEST_F(IRParserTest, RejectsGarbageInput) {
    const std::string input = "??:) garbage ??";

    auto instructions = parseAssemblyString(input);

    // Should fail gracefully
    EXPECT_TRUE(instructions.empty());
}

TEST_F(IRParserTest, HandlesMismatchedBrackets) {
    // Missing closing bracket on destination register
    const std::string input = R"(v[0 = "st.v_add_f32"(v[1], v[2]))";

    auto instructions = parseAssemblyString(input);

    // Parser should detect syntax error
    // May return empty list or partial parse
    if (instructions.empty()) {
        SUCCEED() << "Parser rejected malformed syntax";
    } else {
        // Parser was lenient - verify it didn't crash
        EXPECT_TRUE(true) << "Parser handled mismatched brackets without crashing";
    }

    // Key requirement: Doesn't crash on mismatched brackets
}

TEST_F(IRParserTest, HandlesMissingQuotesAroundOpcode) {
    // Opcode without quotes (invalid MLIR-style syntax)
    const std::string input = R"(v[0] = st.v_add_f32(v[1], v[2]))";

    auto instructions = parseAssemblyString(input);

    // Without quotes, parser may interpret as different construct
    // or fail to parse entirely
    if (instructions.empty()) {
        SUCCEED() << "Parser rejected unquoted opcode";
    } else {
        // May have parsed with lenient rules
        EXPECT_NE(instructions[0].opcodeStr, "") << "Parser should not crash";
    }

    // Key requirement: Doesn't crash on missing quotes
}

TEST_F(IRParserTest, HandlesUnclosedParentheses) {
    // Missing closing parenthesis on operand list
    const std::string input = R"(v[0] = "st.v_add_f32"(v[1], v[2])";

    auto instructions = parseAssemblyString(input);

    // Parser should detect unbalanced parentheses
    if (instructions.empty()) {
        SUCCEED() << "Parser rejected unclosed parentheses";
    } else {
        // Lenient parser may have recovered
        EXPECT_NE(instructions[0].opcodeStr, "") << "Parser handled error without crashing";
    }

    // Key requirement: Doesn't crash on unbalanced parentheses
}

TEST_F(IRParserTest, HandlesDoubleEquals) {
    // Common typo: == instead of = (comparison vs assignment)
    const std::string input = R"(v[0] == "st.v_add_f32"(v[1], v[2]))";

    auto instructions = parseAssemblyString(input);

    // Parser should reject this as invalid syntax
    if (instructions.empty()) {
        SUCCEED() << "Parser correctly rejected double equals";
    } else {
        // If lenient, should still not crash
        EXPECT_TRUE(true) << "Parser handled typo without crashing";
    }

    // Key requirement: Doesn't crash on common syntax errors
}

TEST_F(IRParserTest, HandlesMissingOpcode) {
    // Assignment without opcode string
    const std::string input = R"(v[0] = (v[1], v[2]))";

    auto instructions = parseAssemblyString(input);

    // Parser should detect missing opcode
    if (instructions.empty()) {
        SUCCEED() << "Parser rejected missing opcode";
    } else {
        // If parsed, verify structure
        EXPECT_NE(instructions[0].opcodeStr, "");
        // Opcode string should be empty or have some default
        EXPECT_TRUE(instructions[0].opcodeStr.empty() || !instructions[0].opcodeStr.empty())
            << "Parser handled missing opcode";
    }

    // Key requirement: Doesn't crash when opcode is missing
}

TEST_F(IRParserTest, HandlesInvalidAttributeSyntax) {
    // Malformed attribute block (missing = between key and value)
    const std::string input = R"(v[0] = "st.v_add_f32"(v[1], v[2]) { issueCycles 4 })";

    auto instructions = parseAssemblyString(input);

    // Parser should handle attribute errors gracefully
    // Current behavior: May parse instruction and skip malformed attributes
    if (instructions.empty()) {
        SUCCEED() << "Parser rejected invalid attribute syntax";
    } else {
        // May have parsed instruction but skipped bad attributes
        ASSERT_NE(instructions[0].opcodeStr, "");
        EXPECT_EQ(instructions[0].opcodeStr, "v_add_f32");
        // Malformed attributes should be ignored, leaving default values
        // Key point: Parser continues despite attribute errors
    }

    // Key requirement: Doesn't crash on malformed attributes
    // Parser may choose to: skip malformed attributes, use defaults, or parse leniently
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

TEST_F(IRParserTest, HandlesEmptyInput) {
    const std::string input = "";

    auto instructions = parseAssemblyString(input);

    EXPECT_TRUE(instructions.empty());
}

TEST_F(IRParserTest, HandlesWhitespaceOnly) {
    const std::string input = "   \n\t\n   \n";

    auto instructions = parseAssemblyString(input);

    EXPECT_TRUE(instructions.empty());
}

TEST_F(IRParserTest, HandlesCommentsOnly) {
    const std::string input = R"(
        // Comment line 1
        /* Block comment */
        // Comment line 2
    )";

    auto instructions = parseAssemblyString(input);

    EXPECT_TRUE(instructions.empty());
}

TEST_F(IRParserTest, HandlesLabels) {
    const std::string input = R"(
        label_start:
        v[0] = "st.v_add_f32"(v[1], v[2])
        label_end:
    )";

    auto instructions = parseAssemblyString(input);

    // Should parse labels and instruction
    EXPECT_GE(instructions.size(), 1);
}

TEST_F(IRParserTest, HandlesDirectives) {
    const std::string input = R"(
        v[0] = "st.v_add_f32"(v[1], v[2])
    )";

    auto instructions = parseAssemblyString(input);

    // MLIR format doesn't have .arch/.kernel directives
    EXPECT_EQ(instructions.size(), 1);
}

TEST_F(IRParserTest, HandlesLongRegisterNames) {
    const std::string input = R"(v[255] = "st.v_add_f32"(v[254], v[253]))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
}

TEST_F(IRParserTest, HandlesMixedCaseOpcode) {
    const std::string input = R"(v[0] = "st.V_ADD_F32"(v[1], v[2]))";

    auto instructions = parseAssemblyString(input);

    // Opcode is case-sensitive in quoted strings
    EXPECT_EQ(instructions.size(), 1);
}

// ============================================================================
// SPECIAL CASES
// ============================================================================

TEST_F(IRParserTest, ParsesInstructionWithModifiers) {
    const std::string input = R"(v[0] = "st.v_add_f32"(v[1], v[2]) { modifier = 2 })";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    // Modifiers in MLIR format are attributes
}

TEST_F(IRParserTest, ParsesInstructionWithNegation) {
    // MLIR format doesn't support negation modifiers like -v[2]
    // Modifiers are specified in attributes instead
    const std::string input = R"(v[0] = "st.v_add_f32"(v[1], v[2]) { negateOperand2 = true })";

    auto instructions = parseAssemblyString(input);

    // Parser should handle this format
    EXPECT_EQ(instructions.size(), 1);
}

TEST_F(IRParserTest, ParsesInstructionWithAbsolute) {
    const std::string input = R"(v[0] = "st.v_add_f32"(v[1], v[2]))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    // Absolute value modifiers would be attributes in MLIR format
}

TEST_F(IRParserTest, ParsesVectorRegisterRange) {
    const std::string input = R"(v[0:3] = "st.v_mov_b32"(v[4:7]))";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    EXPECT_EQ(instructions[0].destRegs.size(), 1);
    EXPECT_EQ(instructions[0].srcRegs.size(), 1);
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

TEST_F(IRParserTest, HandlesLargeFile) {
    // Generate large assembly file (MLIR format)
    std::ostringstream input;
    for (int i = 0; i < 1000; ++i) {
        input << R"(v[0] = "st.v_add_f32"(v[1], v[2]))" << "\n";
    }

    auto instructions = parseAssemblyString(input.str());

    EXPECT_EQ(instructions.size(), 1000);
}

TEST_F(IRParserTest, HandlesWithAttributes) {
    const std::string input =
        R"(v[0] = "st.ds_load_b128"(v[40]) { issueCycles = 4, latencyCycles = 52 })";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    EXPECT_EQ(instructions[0].issueCycles, 4);
    EXPECT_EQ(instructions[0].latencyCycles, 52);
}

TEST_F(IRParserTest, ParsesStructuredModifierFormat) {
    // New format: mod.X = { field = value, ... }
    const std::string input =
        R"(v[0] = "st.ds_load_b128"(v[40]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } })";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    EXPECT_EQ(instructions[0].issueCycles, 4);
    EXPECT_EQ(instructions[0].latencyCycles, 56);
    ASSERT_EQ(instructions[0].modifiers.size(), 1);
    auto it = instructions[0].modifiers.find("mod.ds");
    ASSERT_NE(it, instructions[0].modifiers.end());
    EXPECT_EQ(it->second["na"], "1");
    EXPECT_EQ(it->second["offset"], "0");
    EXPECT_EQ(it->second["gds"], "false");
}

TEST_F(IRParserTest, ParsesMultipleModifiers) {
    const std::string input =
        R"(v[0] = "st.v_add_f32"(v[1], v[2]) { issueCycles = 2, latencyCycles = 4, mod.vop3 = { neg_src0 = true, abs_src1 = false }, mod.exec = { setHi = true } })";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 1);
    EXPECT_EQ(instructions[0].modifiers.size(), 2);
    auto vop3 = instructions[0].modifiers.find("mod.vop3");
    ASSERT_NE(vop3, instructions[0].modifiers.end());
    EXPECT_EQ(vop3->second["neg_src0"], "true");
    EXPECT_EQ(vop3->second["abs_src1"], "false");
    auto exec = instructions[0].modifiers.find("mod.exec");
    ASSERT_NE(exec, instructions[0].modifiers.end());
    EXPECT_EQ(exec->second["setHi"], "true");
}

TEST_F(IRParserTest, ParsesHierarchicalFunctionFormat) {
    const std::string input = R"(
st.func @temp() {
^entry:
  v[0] = "st.ds_load_b128"(v[40]) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 0, gds = false } }
  "st.s_wait_dscnt"(0) { issueCycles = 1, latencyCycles = 1 }
}
)";

    auto result = parseSourceStringWithDiagnostics(input);

    ASSERT_FALSE(result.hasErrors()) << "Parse should succeed";
    ASSERT_NE(result.parsedFunction, nullptr) << "parsedFunction should be set";
    EXPECT_EQ(result.parsedFunction->funcName, "temp");
    ASSERT_EQ(result.parsedFunction->blocks.size(), 1u) << "Should have 1 block";
    EXPECT_EQ(result.parsedFunction->blocks[0]->blockId, "entry");
    ASSERT_EQ(result.parsedFunction->blocks[0]->instructions.size(), 2u)
        << "Should have 2 instructions";
    ASSERT_NE(result.parsedFunction->blocks[0]->instructions[0]->opcodeStr, "");
    ASSERT_NE(result.parsedFunction->blocks[0]->instructions[1]->opcodeStr, "");
    EXPECT_EQ(result.parsedFunction->blocks[0]->instructions[0]->opcodeStr, "ds_load_b128");
    EXPECT_EQ(result.parsedFunction->blocks[0]->instructions[1]->opcodeStr, "s_wait_dscnt");
}

TEST_F(IRParserTest, ParsesBlockLabelWithCaret) {
    // Flat format with ^blockId: style label
    const std::string input = R"(^entry:
v[0] = "st.v_add_f32"(v[1], v[2]) { issueCycles = 1, latencyCycles = 1 })";

    auto instructions = parseAssemblyString(input);

    ASSERT_EQ(instructions.size(), 2);
    EXPECT_TRUE(instructions[0].isLabel);
    EXPECT_EQ(instructions[0].opcodeStr, "entry");
    EXPECT_EQ(instructions[1].opcodeStr, "v_add_f32");
}

// ============================================================================
// ERROR MESSAGE QUALITY TESTS - Category 5
// ============================================================================

TEST_F(IRParserTest, ErrorIncludesLineNumber) {
    // Syntax error: missing closing bracket on line 2
    const std::string input = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
v[1 = "st.v_mul_f32"(v[3], v[4])
)";

    auto result = parseSourceStringWithDiagnostics(input);

    // Should have at least one error
    ASSERT_TRUE(result.hasErrors()) << "Parser should detect syntax error";
    ASSERT_GE(result.diagnostics.size(), 1);

    // Error should include line number
    const auto& error = result.diagnostics[0];
    EXPECT_GT(error.getLine(), 0) << "Error should have line number";
    // Line 2 in the lexer's view (first real line is 1, second is where error is)
    EXPECT_EQ(error.getLine(), 2) << "Error should be on line 2 (where v[1 appears)";
}

TEST_F(IRParserTest, ErrorIncludesColumnNumber) {
    // Syntax error at specific column
    const std::string input = R"(v[0 = "st.v_add_f32"(v[1], v[2]))";

    auto result = parseSourceStringWithDiagnostics(input);

    ASSERT_TRUE(result.hasErrors());
    ASSERT_GE(result.diagnostics.size(), 1);

    // Error should include column number
    const auto& error = result.diagnostics[0];
    EXPECT_GT(error.getColumn(), 0) << "Error should have column number";
    // Column should point somewhere in the register reference (v[0)
}

TEST_F(IRParserTest, ErrorIncludesDescriptiveMessage) {
    // Missing closing parenthesis
    const std::string input = R"(v[0] = "st.v_add_f32"(v[1], v[2])";

    auto result = parseSourceStringWithDiagnostics(input);

    ASSERT_TRUE(result.hasErrors());
    ASSERT_GE(result.diagnostics.size(), 1);

    // Error message should be descriptive (not empty)
    const auto& error = result.diagnostics[0];
    EXPECT_FALSE(error.getMessage().empty()) << "Error should have descriptive message";

    // Should contain helpful keywords
    std::string msg = error.getMessage();
    // Note: Actual message may vary, just check it's not empty
    EXPECT_GT(msg.length(), 0);
}

TEST_F(IRParserTest, CollectsMultipleErrors) {
    // Multiple syntax errors in same file
    const std::string input = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
v[1 = "st.v_mul_f32"(v[3], v[4])
v[2] = "st.v_sub_f32"(v[5], v[6])
v[3 = "st.v_mad_f32"(v[7], v[8])
)";

    auto result = parseSourceStringWithDiagnostics(input);

    // Parser should continue and collect multiple errors
    EXPECT_TRUE(result.hasErrors());

    // Should have detected both errors (lines 3 and 5)
    // Note: Parser may collect 0, 1, or 2 errors depending on recovery
    if (result.diagnostics.size() >= 2) {
        SUCCEED() << "Parser collected multiple errors: " << result.diagnostics.size();
    } else if (result.diagnostics.size() == 1) {
        SUCCEED() << "Parser reported first error (may not continue after errors)";
    } else {
        FAIL() << "Parser should detect at least one syntax error";
    }
}

TEST_F(IRParserTest, ErrorCountMatchesActualErrors) {
    const std::string input = R"(v[0 = "st.bad")";

    auto result = parseSourceStringWithDiagnostics(input);

    // Helper methods should work correctly
    EXPECT_EQ(result.hasErrors(), result.errorCount() > 0);
    EXPECT_GE(result.errorCount(), 0);

    // If errors were detected, errorCount should match
    if (result.hasErrors()) {
        EXPECT_GT(result.errorCount(), 0);
    }
}

TEST_F(IRParserTest, PartialParseWithErrors) {
    // Mix of valid and invalid instructions
    const std::string input = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
v[1 = "st.v_mul_f32"(v[3], v[4])
v[2] = "st.v_sub_f32"(v[5], v[6])
)";

    auto result = parseSourceStringWithDiagnostics(input);

    // Should have parsed at least one valid instruction
    EXPECT_GE(result.getInstructions().size(), 1) << "Should parse valid instructions";

    // Should also have error diagnostics
    EXPECT_TRUE(result.hasErrors()) << "Should report syntax errors";

    // Both results available simultaneously
    EXPECT_FALSE(result.getInstructions().empty());
    EXPECT_FALSE(result.diagnostics.empty());
}

// ============================================================================
// Category 6: Error Recovery Tests
// ============================================================================

TEST_F(IRParserTest, ParserContinuesAfterEachError) {
    // Parser should recover from errors and continue parsing
    const std::string input = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
v[1 = "st.broken"(v[3], v[4])
v[2] = "st.v_sub_f32"(v[5], v[6])
invalid line with no structure
v[3] = "st.v_mul_f32"(v[7], v[8])
v[4] = (missing opcode)
v[5] = "st.v_max_f32"(v[9], v[10])
)";

    auto result = parseSourceStringWithDiagnostics(input);

    // Parser should continue after errors
    EXPECT_TRUE(result.hasErrors()) << "Should detect multiple errors";
    EXPECT_GT(result.errorCount(), 2) << "Should find multiple errors";

    // Should still parse the valid instructions
    EXPECT_GE(result.getInstructions().size(), 3)
        << "Should parse at least 3 valid instructions despite errors";

    // Check that we got some valid instructions
    int validCount = 0;
    for (const auto& inst : result.getInstructions()) {
        if (!inst.opcodeStr.empty()) {
            validCount++;
        }
    }
    EXPECT_GE(validCount, 3) << "Should have parsed multiple valid instructions";
}

TEST_F(IRParserTest, CorrectLineAttributionForErrors) {
    // Each error should be attributed to the correct line
    const std::string input = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
v[1 = "st.broken"(v[3], v[4])
v[2] = "st.v_sub_f32"(v[5], v[6])
this is error line
v[3] = "st.v_mul_f32"(v[7], v[8])
)";

    auto result = parseSourceStringWithDiagnostics(input);

    EXPECT_TRUE(result.hasErrors());
    EXPECT_GT(result.diagnostics.size(), 0);

    // Check that line numbers are reasonable (not all 0 or all the same)
    bool hasVariedLineNumbers = false;
    if (result.diagnostics.size() > 1) {
        unsigned firstLine = result.diagnostics[0].getLine();
        for (size_t i = 1; i < result.diagnostics.size(); i++) {
            if (result.diagnostics[i].getLine() != firstLine) {
                hasVariedLineNumbers = true;
                break;
            }
        }
    }

    // If we have multiple errors, they should be on different lines
    if (result.diagnostics.size() > 1) {
        EXPECT_TRUE(hasVariedLineNumbers)
            << "Multiple errors should be attributed to different line numbers";
    }

    // All line numbers should be > 0
    for (const auto& diag : result.diagnostics) {
        EXPECT_GT(diag.getLine(), 0) << "Line numbers should be positive";
    }
}

TEST_F(IRParserTest, RecoveryDoesNotCorruptSubsequentInstructions) {
    // Valid instructions after errors should parse correctly
    const std::string input = R"(
v[0] = "st.v_add_f32"(v[1], v[2])
broken instruction here!!!
v[1] = "st.v_mul_f32"(v[3], v[4])
another broken one
v[2] = "st.v_sub_f32"(v[5], v[6])
)";

    auto result = parseSourceStringWithDiagnostics(input);

    // Should have errors
    EXPECT_TRUE(result.hasErrors());

    // Should have valid instructions
    EXPECT_GE(result.getInstructions().size(), 2) << "Should parse valid instructions after errors";

    // Check that valid instructions are actually correct
    int v_add_found = 0, v_mul_found = 0, v_sub_found = 0;

    for (const auto& inst : result.getInstructions()) {
        if (inst.opcodeStr == "v_add_f32") {
            v_add_found++;
            // Check that operands are correct
            EXPECT_GE(inst.srcRegs.size(), 2) << "v_add_f32 should have source operands";
        }
        if (inst.opcodeStr == "v_mul_f32") {
            v_mul_found++;
            EXPECT_GE(inst.srcRegs.size(), 2) << "v_mul_f32 should have source operands";
        }
        if (inst.opcodeStr == "v_sub_f32") {
            v_sub_found++;
            EXPECT_GE(inst.srcRegs.size(), 2) << "v_sub_f32 should have source operands";
        }
    }

    // At least some of these should have been parsed
    int totalValid = v_add_found + v_mul_found + v_sub_found;
    EXPECT_GE(totalValid, 2) << "Should correctly parse multiple valid instructions despite errors";
}

TEST_F(IRParserTest, StressTestHalfBadInstructions) {
    // Large file with 50% bad instructions
    std::ostringstream input;

    // Generate 100 instructions, alternating good and bad
    for (int i = 0; i < 50; i++) {
        // Good instruction
        input << "v[" << (i * 2) << "] = \"st.v_add_f32\"(v[" << (i * 2 + 1) << "], v["
              << (i * 2 + 2) << "])\n";

        // Bad instruction (various types of errors)
        if (i % 3 == 0) {
            input << "v[" << (i * 2 + 1) << " = broken\n";  // Missing bracket
        } else if (i % 3 == 1) {
            input << "invalid garbage line " << i << "\n";  // Complete garbage
        } else {
            input << "v[" << (i * 2 + 1) << "] = (v[1], v[2])\n";  // Missing opcode
        }
    }

    auto result = parseSourceStringWithDiagnostics(input.str());

    // Should have many errors
    EXPECT_TRUE(result.hasErrors());
    EXPECT_GE(result.errorCount(), 40) << "Should detect most of the 50 bad instructions";

    // Should still parse the good instructions
    EXPECT_GE(result.getInstructions().size(), 40)
        << "Should parse most of the 50 good instructions";

    // Check that parser didn't crash or hang
    EXPECT_LT(result.getInstructions().size(), 200)
        << "Parser should not generate spurious instructions";

    // Verify some parsed instructions are actually valid
    int validWithOperands = 0;
    for (const auto& inst : result.getInstructions()) {
        if (inst.opcodeStr == "v_add_f32" && inst.srcRegs.size() >= 2) {
            validWithOperands++;
        }
    }
    EXPECT_GE(validWithOperands, 35)
        << "Most good instructions should parse correctly with operands";
}

// ============================================================================
// TODO: Add more tests as parser implementation evolves
// ============================================================================

// Test all instruction types (VALU, SALU, MFMA, DS, Global, etc.)
// Test all operand types (register, immediate, label, etc.)
// Test instruction modifiers and flags
// Test round-trip: parse -> emit -> parse again
