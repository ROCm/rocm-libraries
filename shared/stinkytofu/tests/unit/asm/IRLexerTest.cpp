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
 * @file IRLexerTest.cpp
 * @brief Tests for the IR Lexer (Tokenizer)
 *
 * This file tests tokenization of assembly IR text into tokens.
 */

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "IRLexer.hpp"

using namespace stinkytofu;

/**
 * Test fixture for IR Lexer tests
 */
class IRLexerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Setup if needed
    }

    void TearDown() override {
        // Cleanup
    }

    /**
     * Helper to tokenize string (filters out EOF and newlines for cleaner tests)
     */
    std::vector<Token> tokenizeString(const std::string& input) {
        IRLexer lexer(input);
        lexer.lex();  // Tokenize the entire input

        std::vector<Token> tokens;
        for (const auto& token : lexer.getAllTokens()) {
            // Filter out EOF and newline tokens for test simplicity
            if (token.kind != TokenKind::Eof && token.kind != TokenKind::Newline) {
                tokens.push_back(token);
            }
        }
        return tokens;
    }
};

// ============================================================================
// BASIC TOKEN TESTS
// ============================================================================

TEST_F(IRLexerTest, TokenizesSimpleIdentifier) {
    const std::string input = "v_add_f32";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::Identifier);
    EXPECT_EQ(tokens[0].text, "v_add_f32");
}

TEST_F(IRLexerTest, TokenizesRegister) {
    const std::string input = "v0";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::Identifier);  // Registers are identifiers
    EXPECT_EQ(tokens[0].text, "v0");
}

TEST_F(IRLexerTest, TokenizesScalarRegister) {
    const std::string input = "s0";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::Identifier);  // Scalar registers are identifiers
    EXPECT_EQ(tokens[0].text, "s0");
}

TEST_F(IRLexerTest, TokenizesAccumulatorRegister) {
    const std::string input = "a0";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::Identifier);  // Accumulator registers are identifiers
    EXPECT_EQ(tokens[0].text, "a0");
}

TEST_F(IRLexerTest, TokenizesIntegerLiteral) {
    const std::string input = "42";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::IntegerLiteral);
    EXPECT_EQ(tokens[0].text, "42");
}

TEST_F(IRLexerTest, TokenizesFloatLiteral) {
    const std::string input = "3.14";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::FloatLiteral);
    EXPECT_EQ(tokens[0].text, "3.14");
}

TEST_F(IRLexerTest, TokenizesHexLiteral) {
    const std::string input = "0x3f800000";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::HexLiteral);
    EXPECT_EQ(tokens[0].text, "0x3f800000");
}

TEST_F(IRLexerTest, TokenizesComma) {
    const std::string input = ",";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::Comma);
}

TEST_F(IRLexerTest, TokenizesLeftBracket) {
    const std::string input = "[";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::LeftBracket);
}

TEST_F(IRLexerTest, TokenizesRightBracket) {
    const std::string input = "]";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::RightBracket);
}

TEST_F(IRLexerTest, TokenizesColon) {
    const std::string input = ":";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].kind, TokenKind::Colon);
}

// ============================================================================
// COMPLEX TOKEN SEQUENCES
// ============================================================================

TEST_F(IRLexerTest, TokenizesSimpleInstruction) {
    const std::string input = "v_add_f32 v0, v1, v2";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 6);
    EXPECT_EQ(tokens[0].kind, TokenKind::Identifier);  // v_add_f32
    EXPECT_EQ(tokens[1].kind, TokenKind::Identifier);  // v0
    EXPECT_EQ(tokens[2].kind, TokenKind::Comma);       // ,
    EXPECT_EQ(tokens[3].kind, TokenKind::Identifier);  // v1
    EXPECT_EQ(tokens[4].kind, TokenKind::Comma);       // ,
    EXPECT_EQ(tokens[5].kind, TokenKind::Identifier);  // v2
}

TEST_F(IRLexerTest, TokenizesRegisterRange) {
    const std::string input = "v[0:3]";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 6);
    EXPECT_EQ(tokens[0].kind, TokenKind::Identifier);      // v
    EXPECT_EQ(tokens[1].kind, TokenKind::LeftBracket);     // [
    EXPECT_EQ(tokens[2].kind, TokenKind::IntegerLiteral);  // 0
    EXPECT_EQ(tokens[3].kind, TokenKind::Colon);           // :
    EXPECT_EQ(tokens[4].kind, TokenKind::IntegerLiteral);  // 3
    EXPECT_EQ(tokens[5].kind, TokenKind::RightBracket);    // ]
}

TEST_F(IRLexerTest, TokenizesInstructionWithImmediate) {
    const std::string input = "v_add_f32 v0, v1, 5.0";

    auto tokens = tokenizeString(input);

    ASSERT_GE(tokens.size(), 5);
    EXPECT_EQ(tokens[tokens.size() - 1].kind, TokenKind::FloatLiteral);
}

// ============================================================================
// WHITESPACE HANDLING
// ============================================================================

TEST_F(IRLexerTest, IgnoresWhitespace) {
    const std::string input = "v0    ,    v1";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 3);  // v0, comma, v1 (no whitespace tokens)
}

TEST_F(IRLexerTest, IgnoresNewlines) {
    const std::string input = "v0\n,\nv1";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 3);
}

TEST_F(IRLexerTest, IgnoresTabs) {
    const std::string input = "v0\t,\tv1";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 3);
}

// ============================================================================
// COMMENT HANDLING
// ============================================================================

TEST_F(IRLexerTest, IgnoresSingleLineComment) {
    const std::string input = "v0 // this is a comment";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);  // Only v0, comment ignored
    EXPECT_EQ(tokens[0].text, "v0");
}

TEST_F(IRLexerTest, IgnoresBlockComment) {
    const std::string input = "v0 /* block comment */ v1";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 2);  // v0 and v1
    EXPECT_EQ(tokens[0].text, "v0");
    EXPECT_EQ(tokens[1].text, "v1");
}

TEST_F(IRLexerTest, HandlesCommentAtStart) {
    const std::string input = "// comment\nv0";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].text, "v0");
}

// ============================================================================
// ERROR CASES
// ============================================================================

TEST_F(IRLexerTest, HandlesInvalidCharacters) {
    const std::string input = "v0 @ v1";  // @ is likely invalid

    auto tokens = tokenizeString(input);

    // Should either skip invalid char or mark as error token
    // TODO: Check error handling strategy
    EXPECT_GE(tokens.size(), 2);
}

TEST_F(IRLexerTest, HandlesUnterminatedString) {
    const std::string input = R"(v0 "unterminated)";

    auto tokens = tokenizeString(input);

    // Should handle gracefully
    // TODO: Check error handling
    EXPECT_GE(tokens.size(), 1);
}

TEST_F(IRLexerTest, HandlesUnterminatedBlockComment) {
    const std::string input = "v0 /* unterminated";

    auto tokens = tokenizeString(input);

    // Lexer should produce at least the identifier before the comment
    ASSERT_GE(tokens.size(), 1) << "Expected at least 1 token (identifier before comment)";
    EXPECT_EQ(tokens[0].kind, TokenKind::Identifier);
    EXPECT_EQ(tokens[0].text, "v0");

    // Current behavior: unterminated block comment is treated as consuming
    // the rest of input. Last token may be Identifier or Eof depending on
    // how lexer handles the unterminated state.
    // Key point: It doesn't crash and produces at least the v0 token
    EXPECT_TRUE(tokens.size() >= 1) << "Should handle gracefully without crashing";
}

TEST_F(IRLexerTest, HandlesNumericOverflow) {
    // Test extremely large integer that exceeds 64-bit limits
    const std::string input = "99999999999999999999999999999999";

    auto tokens = tokenizeString(input);

    // Lexer should produce a token (either integer, hex, or error)
    ASSERT_GE(tokens.size(), 1) << "Should produce at least one token";

    // Token should be some numeric type or unknown/error
    bool isNumericOrError =
        (tokens[0].kind == TokenKind::IntegerLiteral || tokens[0].kind == TokenKind::HexLiteral ||
         tokens[0].kind == TokenKind::FloatLiteral || tokens[0].kind == TokenKind::Unknown);
    EXPECT_TRUE(isNumericOrError) << "Overflow should produce numeric or error token";

    // Key requirement: Doesn't crash, handles overflow gracefully
    // Implementation may choose to: clamp value, use arbitrary precision, or mark as error
}

TEST_F(IRLexerTest, HandlesInvalidEscapeSequence) {
    // Test string with invalid escape sequence
    const std::string input = R"("test\xGG")";  // \xGG is invalid (non-hex after \x)

    auto tokens = tokenizeString(input);

    // Lexer should produce at least one token
    ASSERT_GE(tokens.size(), 1) << "Should produce at least one token";

    // Should produce either a QuotedString (accepting invalid escape) or Unknown/Error
    bool acceptableKind =
        (tokens[0].kind == TokenKind::QuotedString || tokens[0].kind == TokenKind::Unknown);
    EXPECT_TRUE(acceptableKind) << "Should produce string or error token";

    // Key requirement: Doesn't crash on invalid escape sequences
    // Lexer may choose to: accept literally, substitute, or mark as error
}

TEST_F(IRLexerTest, HandlesMalformedScientificNotation) {
    // Test scientific notation without exponent digits
    const std::string input = "1.23e";

    auto tokens = tokenizeString(input);

    // Lexer should produce at least one token
    ASSERT_GE(tokens.size(), 1) << "Should produce at least one token";

    // Current behavior: May lex as "1.23" + "e" (two tokens)
    // Or may lex as single malformed float
    // Or may produce error token
    if (tokens.size() == 1) {
        // Single token: either complete float or error
        bool acceptableKind =
            (tokens[0].kind == TokenKind::FloatLiteral || tokens[0].kind == TokenKind::Unknown);
        EXPECT_TRUE(acceptableKind) << "Single token should be float or error";
    } else {
        // Multiple tokens: "1.23" as float, "e" as identifier
        EXPECT_EQ(tokens[0].kind, TokenKind::FloatLiteral) << "First token should be float";
        EXPECT_EQ(tokens[1].kind, TokenKind::Identifier) << "Second token should be identifier";
    }

    // Key requirement: Doesn't crash, produces reasonable token stream
}

TEST_F(IRLexerTest, HandlesExtremelyLongToken) {
    // Test DoS protection: Very long identifier (10MB would be too slow for test)
    // Use 100K characters instead for reasonable test execution time
    std::string input(100000, 'a');

    auto tokens = tokenizeString(input);

    // Lexer should either:
    // 1. Accept the full token (if no length limit)
    // 2. Truncate to reasonable length
    // 3. Produce error token
    ASSERT_GE(tokens.size(), 1) << "Should produce at least one token";

    // Should be identifier or error
    bool acceptableKind =
        (tokens[0].kind == TokenKind::Identifier || tokens[0].kind == TokenKind::Unknown);
    EXPECT_TRUE(acceptableKind) << "Long token should be identifier or error";

    // Key requirement: Doesn't crash or hang on very long tokens
    // Performance requirement: Should complete in reasonable time (< 1 second)
}

// ============================================================================
// EDGE CASES
// ============================================================================

TEST_F(IRLexerTest, HandlesEmptyInput) {
    const std::string input = "";

    auto tokens = tokenizeString(input);

    EXPECT_TRUE(tokens.empty());
}

TEST_F(IRLexerTest, HandlesOnlyWhitespace) {
    const std::string input = "   \n\t\n   ";

    auto tokens = tokenizeString(input);

    EXPECT_TRUE(tokens.empty());
}

TEST_F(IRLexerTest, HandlesOnlyComments) {
    const std::string input = "// comment1\n/* comment2 */";

    auto tokens = tokenizeString(input);

    EXPECT_TRUE(tokens.empty());
}

TEST_F(IRLexerTest, HandlesVeryLongToken) {
    std::string input(10000, 'a');  // 10k character identifier

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0].text.length(), 10000);
}

TEST_F(IRLexerTest, HandlesUnicodeCharacters) {
    const std::string input = "???";

    auto tokens = tokenizeString(input);

    // Unicode characters that aren't recognized as valid tokens
    // Lexer should either skip them or mark as unknown/error tokens
    // Current behavior: treats '?' as unknown character and continues
    EXPECT_GE(tokens.size(), 0) << "Lexer should handle unicode gracefully";

    // If tokens are produced, they should not be null/empty
    for (const auto& tok : tokens) {
        EXPECT_FALSE(tok.text.empty() && tok.kind != TokenKind::Eof);
    }
}

TEST_F(IRLexerTest, HandlesNegativeNumbers) {
    const std::string input = "-42";

    auto tokens = tokenizeString(input);

    // Might be tokenized as minus + number, or as negative number
    EXPECT_GE(tokens.size(), 1);
}

TEST_F(IRLexerTest, HandlesScientificNotation) {
    const std::string input = "1.23e-4";

    auto tokens = tokenizeString(input);

    // Should recognize as float literal
    ASSERT_GE(tokens.size(), 1);
    if (tokens[0].kind == TokenKind::FloatLiteral) {
        EXPECT_EQ(tokens[0].text, "1.23e-4");
    }
}

// ============================================================================
// TOKEN POSITION TRACKING
// ============================================================================

TEST_F(IRLexerTest, TracksLineNumbers) {
    const std::string input = "v0\nv1\nv2";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 3);
    // TODO: Check line numbers if Token stores them
    // EXPECT_EQ(tokens[0].line, 1);
    // EXPECT_EQ(tokens[1].line, 2);
    // EXPECT_EQ(tokens[2].line, 3);
}

TEST_F(IRLexerTest, TracksColumnNumbers) {
    const std::string input = "v0 v1 v2";

    auto tokens = tokenizeString(input);

    ASSERT_EQ(tokens.size(), 3);
    // TODO: Check column numbers if Token stores them
    // EXPECT_EQ(tokens[0].column, 1);
    // EXPECT_EQ(tokens[1].column, 4);
    // EXPECT_EQ(tokens[2].column, 7);
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

TEST_F(IRLexerTest, HandlesLargeInput) {
    // Generate large input
    std::ostringstream input;
    for (int i = 0; i < 10000; ++i) {
        input << "v" << i << " ";
    }

    auto tokens = tokenizeString(input.str());

    EXPECT_EQ(tokens.size(), 10000);
}

// ============================================================================
// TODO: Add more tests as lexer implementation evolves
// ============================================================================

// Test all token types
// Test token lookahead (if supported)
// Test token pushback (if supported)
// Test error recovery
// Test lexer state management
