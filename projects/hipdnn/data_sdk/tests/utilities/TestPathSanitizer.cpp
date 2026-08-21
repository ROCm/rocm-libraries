// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cctype>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PathSanitizer.hpp>
#include <set>
#include <string>
#include <vector>

using namespace hipdnn_data_sdk::utilities;

class TestPathSanitizer : public ::testing::Test
{
};

TEST_F(TestPathSanitizer, ScopedEngineNameHasNoColon)
{
    const auto result = sanitizeForPath("hipkernel:Pointwise");

    EXPECT_EQ(result.find(':'), std::string::npos);
}

TEST_F(TestPathSanitizer, ScopedEngineNameStaysHumanReadable)
{
    const auto result = sanitizeForPath("hipkernel:Pointwise");

    // The sanitized stem portion is still recognizable, even with the colon replaced and
    // the hash suffix appended.
    EXPECT_NE(result.find("hipkernel"), std::string::npos);
    EXPECT_NE(result.find("Pointwise"), std::string::npos);
}

TEST_F(TestPathSanitizer, ResultAlwaysCarriesTheUnconditionalHashSuffix)
{
    // The suffix is unconditional -- present on every result, not only when a collision is
    // detected (there is no collision-detection registry to detect one).
    const auto result = sanitizeForPath("plain_name");

    const auto dashPos = result.rfind('-');
    ASSERT_NE(dashPos, std::string::npos);
    const auto suffix = result.substr(dashPos + 1);
    EXPECT_EQ(suffix.size(), 16u); // 64-bit hash rendered as fixed-width hex
    for(const char c : suffix)
    {
        EXPECT_TRUE(std::isxdigit(static_cast<unsigned char>(c)));
    }
}

class TestPathSanitizerReservedStems : public ::testing::TestWithParam<std::string>
{
};

TEST_P(TestPathSanitizerReservedStems, SanitizesDistinctlyFromTheLiteralStem)
{
    const std::string& stem = GetParam();

    const auto result = sanitizeForPath(stem);

    // A Windows-reserved stem must never survive sanitization unchanged, in any casing.
    EXPECT_NE(result, stem);
}

INSTANTIATE_TEST_SUITE_P(
    ReservedNamesAndCaseVariants,
    TestPathSanitizerReservedStems,
    ::testing::Values(
        "CON", "con", "Con", "PRN", "AUX", "NUL", "COM1", "com1", "COM9", "LPT1", "lpt1", "LPT9"));

TEST_F(TestPathSanitizer, InjectivityAcrossDistinctInputs)
{
    // Includes pairs that would collide under a naive, non-suffixed sanitization scheme
    // (a colon replaced by '_' makes "a:b" and "a_b" collide without the hash suffix; two
    // differently-cased reserved names collide once both are altered the same way).
    const std::vector<std::string> inputs = {
        "a:b",
        "a_b",
        "hipkernel:Pointwise",
        "hipkernel_Pointwise",
        "CON",
        "con",
        "",
        "...",
        ".hidden.",
        "plain",
    };

    std::set<std::string> results;
    for(const auto& input : inputs)
    {
        const auto result = sanitizeForPath(input);
        EXPECT_TRUE(results.insert(result).second)
            << "collision for input \"" << input << "\" -> \"" << result << "\"";
    }

    EXPECT_EQ(results.size(), inputs.size());
}

TEST_F(TestPathSanitizer, EmptyInputProducesNonEmptyResult)
{
    const auto result = sanitizeForPath("");

    EXPECT_FALSE(result.empty());
}

TEST_F(TestPathSanitizer, LongInputIsCapped)
{
    const std::string longInput(1024, 'z');

    const auto result = sanitizeForPath(longInput);

    // The stem is capped well under a typical 255-byte filesystem component limit, even
    // after the "-<16 hex digits>" suffix is appended.
    EXPECT_LT(result.size(), 255u);
}

TEST_F(TestPathSanitizer, LeadingAndTrailingDotsAreStripped)
{
    const auto result = sanitizeForPath("...engine...");

    const auto dashPos = result.rfind('-');
    ASSERT_NE(dashPos, std::string::npos);
    const auto stem = result.substr(0, dashPos);
    EXPECT_EQ(stem.front(), 'e');
    EXPECT_EQ(stem.back(), 'e');
}

TEST_F(TestPathSanitizer, DifferentInputsSharingASanitizedStemStillDiffer)
{
    // "a:b" and "a/b" both sanitize their illegal character to '_', producing the same
    // stem ("a_b") -- the hash suffix, computed over the untouched raw input, is what
    // keeps the two final results distinct.
    const auto resultColon = sanitizeForPath("a:b");
    const auto resultSlash = sanitizeForPath("a/b");

    EXPECT_NE(resultColon, resultSlash);
}
