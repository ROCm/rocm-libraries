/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 ************************************************************************ */

/**
 * Fuzzing-style tests for parsers
 *
 * These tests throw edge-case inputs at parsers to ensure they never crash.
 * Think of this as "manual fuzzing" - we test patterns a fuzzer would find.
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/serialization/asm/PatternParser.hpp"

using namespace stinkytofu;

class FuzzTest : public ::testing::Test {
   protected:
    // Helper to parse pattern from string
    PatternParseResult parsePatternString(const std::string& input) {
        auto tmpPath = std::filesystem::temp_directory_path() / "fuzz_test.pattern";
        std::ofstream out(tmpPath);
        out << input;
        out.close();

        auto result = parsePatternFileWithDiagnostics(tmpPath.string().c_str());
        std::filesystem::remove(tmpPath);
        return result;
    }
};

// ============================================================================
// IRParser Fuzz Tests
// ============================================================================

TEST_F(FuzzTest, IRParserHandlesEmptyInput) {
    auto result = parseSourceStringWithDiagnostics("");
    // Should not crash, diagnostics may or may not be present
    EXPECT_TRUE(true);  // Survived!
}

TEST_F(FuzzTest, IRParserHandlesVeryLongInput) {
    // FIXED! Previously crashed with assertion failure
    // Now stringToRegType() returns UNKNOWN instead of asserting
    std::string input(100000, 'v');
    auto result = parseSourceStringWithDiagnostics(input);

    // Should not crash, even on 100KB of 'v's
    // Parser should handle gracefully (may produce errors or parse what it can)
    EXPECT_TRUE(true);  // Main goal: no crash
}

TEST_F(FuzzTest, IRParserHandlesRepeatedSymbols) {
    std::string input = std::string(1000, '[') + std::string(1000, ']');
    auto result = parseSourceStringWithDiagnostics(input);
    EXPECT_TRUE(true);
}

TEST_F(FuzzTest, IRParserHandlesNullBytes) {
    std::string input = "v[0]\x00\x00\x00";
    auto result = parseSourceStringWithDiagnostics(input);
    EXPECT_TRUE(true);
}

TEST_F(FuzzTest, IRParserHandlesUnicode) {
    std::string input = "v[0] = ???";
    auto result = parseSourceStringWithDiagnostics(input);
    // Should produce error but not crash
    EXPECT_TRUE(result.hasErrors() || result.getInstructions().empty());
}

TEST_F(FuzzTest, IRParserHandlesDeepNesting) {
    std::string input = "v[0] = \"st.v_add_f32\"(";
    for (int i = 0; i < 100; i++) {
        input += "v[" + std::to_string(i) + "], ";
    }
    input += "v[100])";

    auto result = parseSourceStringWithDiagnostics(input);
    EXPECT_TRUE(true);  // Didn't crash
}

TEST_F(FuzzTest, IRParserHandlesLargeNumbers) {
    // FIXED! Previously threw std::out_of_range exception
    // Now uses safeStoi() which catches exceptions and returns error
    auto result = parseSourceStringWithDiagnostics("v[999999999999999999]");

    // Should produce error, not crash or throw
    EXPECT_TRUE(result.hasErrors());
    EXPECT_GT(result.diagnostics.size(), 0);
    EXPECT_TRUE(result.getInstructions().empty());
}

TEST_F(FuzzTest, IRParserHandlesManyQuotes) {
    std::string input = "v[0] = " + std::string(1000, '"');
    auto result = parseSourceStringWithDiagnostics(input);
    EXPECT_TRUE(true);
}

TEST_F(FuzzTest, IRParserValidatesDiagnosticInvariants) {
    // Test various bad inputs
    std::vector<std::string> badInputs = {
        "v[0] = ", "v[0] = (", "v[0] = \"st.v_add_f32\"(", "[[[[", "v[0] v[1] v[2]",
    };

    for (const auto& input : badInputs) {
        auto result = parseSourceStringWithDiagnostics(input);

        // Validate diagnostic invariants
        for (const auto& diag : result.diagnostics) {
            // Line and column must be positive
            EXPECT_GT(diag.getLine(), 0) << "For input: " << input;
            EXPECT_GT(diag.getColumn(), 0) << "For input: " << input;

            // Message must not be empty
            EXPECT_FALSE(diag.getMessage().empty()) << "For input: " << input;
        }
    }
}

// ============================================================================
// PatternParser Fuzz Tests
// ============================================================================

TEST_F(FuzzTest, PatternParserHandlesEmptyInput) {
    auto result = parsePatternString("");
    EXPECT_TRUE(true);  // Didn't crash
}

TEST_F(FuzzTest, PatternParserHandlesVeryLongInput) {
    std::string input = "peephole pattern " + std::string(10000, 'A') + " { }";
    auto result = parsePatternString(input);
    EXPECT_TRUE(true);
}

TEST_F(FuzzTest, PatternParserHandlesUnbalancedBraces) {
    std::vector<std::string> inputs = {
        "peephole pattern Test { { { { ",
        "} } } } }",
        "peephole pattern Test { match { } } } }",
    };

    for (const auto& input : inputs) {
        auto result = parsePatternString(input);
        EXPECT_TRUE(true);  // Didn't crash
    }
}

TEST_F(FuzzTest, PatternParserHandlesInvalidKeywords) {
    std::string input = R"(
        invalid_keyword pattern Test {
            invalid_block { $x = v_add_f32 $a, $b }
            another_invalid { }
        }
    )";

    auto result = parsePatternString(input);
    EXPECT_TRUE(true);
}

TEST_F(FuzzTest, PatternParserHandlesManyPatterns) {
    std::string input;
    for (int i = 0; i < 1000; i++) {
        input += "peephole pattern Pattern" + std::to_string(i) + " { }\n";
    }

    auto result = parsePatternString(input);
    EXPECT_TRUE(true);  // Didn't crash on 1000 patterns
}

TEST_F(FuzzTest, PatternParserValidatesDiagnosticInvariants) {
    std::vector<std::string> badInputs = {
        "peephole pattern { }",
        "peephole pattern Test {",
        "intrinsic Test { arguments { } }",
        "{ } { } { }",
    };

    for (const auto& input : badInputs) {
        auto result = parsePatternString(input);

        // Validate diagnostic invariants
        for (const auto& diag : result.diagnostics) {
            EXPECT_GT(diag.getLine(), 0) << "For input: " << input;
            EXPECT_GT(diag.getColumn(), 0) << "For input: " << input;
            EXPECT_FALSE(diag.getMessage().empty()) << "For input: " << input;
        }
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(FuzzTest, IRParserStressTest) {
    // Generate many random-ish inputs
    std::vector<std::string> inputs;

    // Valid-looking but with errors
    for (int i = 0; i < 100; i++) {
        inputs.push_back("v[" + std::to_string(i) + "] = \"st.v_add_f32\"(");
        inputs.push_back("v[" + std::to_string(i) + "]");
        inputs.push_back("= \"st.v_add_f32\"()");
    }

    // Parse all of them - parsers should never crash
    for (const auto& input : inputs) {
        auto result = parseSourceStringWithDiagnostics(input);
        // If we get here, parser didn't crash - good!
    }

    EXPECT_TRUE(true) << "Parser survived all " << inputs.size() << " inputs";
}

TEST_F(FuzzTest, PatternParserStressTest) {
    std::vector<std::string> inputs;

    // Generate many malformed patterns
    for (int i = 0; i < 100; i++) {
        inputs.push_back("peephole pattern Test" + std::to_string(i) + " {");
        inputs.push_back("intrinsic Test" + std::to_string(i) + " { arguments");
        inputs.push_back("{ } pattern { } intrinsic { }");
    }

    // Parse all of them - parsers should never crash
    for (const auto& input : inputs) {
        auto result = parsePatternString(input);
        // If we get here, parser didn't crash - good!
    }

    EXPECT_TRUE(true) << "Parser survived all " << inputs.size() << " inputs";
}
