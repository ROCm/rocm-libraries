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

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "stinkytofu/ir/logical/IntrinsicPatternConverter.hpp"
#include "stinkytofu/serialization/asm/PatternParser.hpp"
#include "stinkytofu/serialization/logical/IRSerializer.hpp"

// Include IRLexer implementation (it's not in public headers)
#include "../../src/serialization/asm/IRLexer.hpp"

using namespace stinkytofu;

/**
 * Test hex literal parsing and round-trip conversion
 */
TEST(HexLiteralTest, ParseHexLiteral) {
    // Define a test intrinsic with hex literal
    std::string testDef = R"(
intrinsic TestHexLiteral {
  arguments {
    dest: vgpr
    src: vgpr
    temp: vgpr
  }

  body {
    temp = v_mul_f32(src, 0x40ec7326)
    dest = v_exp_f32(temp)
  }

  comment \"Test hex literal\"
  python_binding false
}
)";

    // Parse the definition
    IRLexer lexer(testDef);
    lexer.lex();
    PatternParser parser(lexer);
    auto patterns = parser.parsePatterns();

    ASSERT_FALSE(parser.hasErrors()) << "Parser should not have errors";
    ASSERT_EQ(patterns.size(), 1) << "Should parse one pattern";

    const auto& pattern = patterns[0];
    EXPECT_EQ(pattern.type, PatternType::Intrinsic);
    EXPECT_EQ(pattern.name, "TestHexLiteral");
    EXPECT_EQ(pattern.body.size(), 2) << "Should have 2 instructions";

    // Check first instruction
    const auto& inst1 = pattern.body[0];
    EXPECT_EQ(inst1.destReg, "temp");
    EXPECT_EQ(inst1.operation, "v_mul_f32");
    EXPECT_EQ(inst1.operands.size(), 2) << "Should have 2 operands";

    // First operand is a register
    EXPECT_EQ(inst1.operands[0].type, IntrinsicOperand::Register);
    EXPECT_EQ(inst1.operands[0].registerName, "src");

    // Second operand is a hex literal
    EXPECT_EQ(inst1.operands[1].type, IntrinsicOperand::HexLiteral);

    // Verify the hex literal was parsed correctly
    // 0x40ec7326 as float should be approximately 7.3890562
    float expectedFloat = 7.3890562f;
    uint32_t expectedBits = 0x40ec7326;
    float actualFloat = static_cast<float>(inst1.operands[1].floatValue);
    uint32_t actualBits = std::bit_cast<uint32_t>(actualFloat);

    EXPECT_EQ(actualBits, expectedBits) << "Hex literal bits should match exactly";
    EXPECT_NEAR(actualFloat, expectedFloat, 0.0001f) << "Float value should be approximately 7.389";
}

/**
 * Test round-trip conversion: Text -> IR -> Text
 */
TEST(HexLiteralTest, RoundTripConversion) {
    // Define a test intrinsic with hex literal
    std::string testDef = R"(
intrinsic RoundTripTest {
  arguments {
    dest: vgpr
    src: vgpr
  }

  body {
    dest = v_mul_f32(src, 0x40ec7326)
  }

  comment \"Test round trip\"
  python_binding false
}
)";

    // Parse
    IRLexer lexer(testDef);
    lexer.lex();
    PatternParser parser(lexer);
    auto patterns = parser.parsePatterns();

    ASSERT_FALSE(parser.hasErrors()) << "Parser should not have errors";
    ASSERT_EQ(patterns.size(), 1);

    // Convert to IR
    auto irModules = IntrinsicPatternConverter::patternsToIR(patterns);
    ASSERT_EQ(irModules.size(), 1);

    // Convert back to pattern
    auto backToPatterns = IntrinsicPatternConverter::irToPatterns(irModules);
    ASSERT_EQ(backToPatterns.size(), 1);

    // Verify operand is still a hex literal
    const auto& backPattern = backToPatterns[0];
    ASSERT_EQ(backPattern.body.size(), 1);
    const auto& backInst = backPattern.body[0];
    ASSERT_EQ(backInst.operands.size(), 2);

    EXPECT_EQ(backInst.operands[1].type, IntrinsicOperand::HexLiteral)
        << "Operand should remain a hex literal after round-trip";

    // Verify bits are preserved
    float floatVal = static_cast<float>(backInst.operands[1].floatValue);
    uint32_t bits = std::bit_cast<uint32_t>(floatVal);
    EXPECT_EQ(bits, 0x40ec7326) << "Hex literal bits should be preserved";
}

/**
 * Test serialization and deserialization
 */
TEST(HexLiteralTest, Serialization) {
    // Define a test intrinsic with hex literal
    std::string testDef = R"(
intrinsic SerializeTest {
  arguments {
    dest: vgpr
    src: vgpr
  }

  body {
    dest = v_mul_f32(src, 0x3f800000)
  }

  comment \"Test serialization 0x3f800000 equals 1.0f\"
  python_binding false
}
)";

    // Parse
    IRLexer lexer(testDef);
    lexer.lex();
    PatternParser parser(lexer);
    auto patterns = parser.parsePatterns();

    ASSERT_FALSE(parser.hasErrors());
    ASSERT_EQ(patterns.size(), 1);

    // Convert to IR and back to get clean patterns
    auto irModules = IntrinsicPatternConverter::patternsToIR(patterns);
    auto cleanPatterns = IntrinsicPatternConverter::irToPatterns(irModules);

    // Serialize to file and deserialize back
    auto tempPath = std::filesystem::temp_directory_path() / "test_hex_literal.st.bc";
    std::string tempFile = tempPath.string();
    bool serializeOk = IRSerializer::serializeToFile(cleanPatterns, tempFile);
    ASSERT_TRUE(serializeOk) << "Serialization should succeed";

    // Deserialize
    auto deserialized = IRSerializer::deserializeFromFile(tempFile);
    std::filesystem::remove(tempPath);

    ASSERT_EQ(deserialized.size(), 1) << "Should deserialize 1 intrinsic";

    const auto& deserPattern = deserialized[0];
    EXPECT_EQ(deserPattern.name, "SerializeTest");
    ASSERT_EQ(deserPattern.body.size(), 1);

    const auto& deserInst = deserPattern.body[0];
    ASSERT_EQ(deserInst.operands.size(), 2);

    // Verify hex literal was preserved
    EXPECT_EQ(deserInst.operands[1].type, IntrinsicOperand::HexLiteral)
        << "Operand should be hex literal after deserialization";

    // Verify bits: 0x3f800000 = 1.0f
    float floatVal = static_cast<float>(deserInst.operands[1].floatValue);
    uint32_t bits = std::bit_cast<uint32_t>(floatVal);
    EXPECT_EQ(bits, 0x3f800000) << "Hex literal bits should be 0x3f800000";
    EXPECT_FLOAT_EQ(floatVal, 1.0f) << "Float value should be 1.0";
}

/**
 * Test mixed operand types (register, int, float, hex)
 */
TEST(HexLiteralTest, MixedOperands) {
    std::string testDef = R"(
intrinsic MixedTest {
  arguments {
    dest: vgpr
    src: vgpr
    temp: vgpr
  }

  body {
    temp = v_add_f32(src, 3.14)
    dest = v_mul_f32(temp, 0x40490fdb)
  }

  comment \"Mixed operands\"
  python_binding false
}
)";

    // Parse
    IRLexer lexer(testDef);
    lexer.lex();
    PatternParser parser(lexer);
    auto patterns = parser.parsePatterns();

    ASSERT_FALSE(parser.hasErrors());
    ASSERT_EQ(patterns.size(), 1);

    const auto& pattern = patterns[0];
    ASSERT_EQ(pattern.body.size(), 2);

    // First instruction: temp = v_add_f32(src, 3.14)
    const auto& inst1 = pattern.body[0];
    EXPECT_EQ(inst1.operands[0].type, IntrinsicOperand::Register);
    EXPECT_EQ(inst1.operands[0].registerName, "src");
    EXPECT_EQ(inst1.operands[1].type, IntrinsicOperand::FloatLiteral);
    EXPECT_NEAR(inst1.operands[1].floatValue, 3.14, 0.001);

    // Second instruction: dest = v_mul_f32(temp, 0x40490fdb)
    const auto& inst2 = pattern.body[1];
    EXPECT_EQ(inst2.operands[0].type, IntrinsicOperand::Register);
    EXPECT_EQ(inst2.operands[0].registerName, "temp");
    EXPECT_EQ(inst2.operands[1].type, IntrinsicOperand::HexLiteral);

    // 0x40490fdb = pi (3.14159274...)
    float piFloat = static_cast<float>(inst2.operands[1].floatValue);
    uint32_t piBits = std::bit_cast<uint32_t>(piFloat);
    EXPECT_EQ(piBits, 0x40490fdb);
}

/**
 * Test GenericIRInstruction dump with hex literal
 */
TEST(HexLiteralTest, GenericIRInstructionDump) {
    // Create operands
    std::vector<IntrinsicOperand> operands;
    operands.push_back(IntrinsicOperand("src"));                  // Register
    operands.push_back(IntrinsicOperand::hexLiteral(7.3890562));  // 0x40ec7326

    // Create instruction via standard API
    GenericIRInstruction* inst =
        IRBase::createIR<GenericIRInstruction>("dest", "v_mul_f32", operands);

    // Dump to string
    std::ostringstream oss;
    inst->dump(oss);

    // Verify output contains hex literal
    std::string output = oss.str();
    EXPECT_TRUE(output.find("dest = v_mul_f32(src, 0x40ec7326)") != std::string::npos)
        << "Output should contain hex literal format. Got: " << output;

    inst->safeErase();
}
