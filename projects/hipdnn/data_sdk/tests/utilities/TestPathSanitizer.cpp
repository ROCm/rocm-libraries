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

    EXPECT_NE(result.find("hipkernel"), std::string::npos);
    EXPECT_NE(result.find("Pointwise"), std::string::npos);
}

TEST_F(TestPathSanitizer, ResultAlwaysCarriesTheUnconditionalHashSuffix)
{
    // Unconditional: present on every result, not only when a collision is detected --
    // there is no collision-detection registry to detect one.
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

    // EXPECT_NE(result, stem) alone would hold for every input, reserved or not: the
    // unconditional hash suffix already guarantees result != stem regardless of
    // whether the reserved-name guard does anything at all. The guard specifically
    // must APPEND "_" to a reserved stem (see PathSanitizer.hpp's
    // `stem.push_back('_')`), so strip the "-<16 hex digits>" suffix (already covered
    // by ResultAlwaysCarriesTheUnconditionalHashSuffix above) and check the stem
    // portion gained exactly that suffix over the literal input.
    const auto dashPos = result.rfind('-');
    ASSERT_NE(dashPos, std::string::npos);
    const auto resultStem = result.substr(0, dashPos);
    EXPECT_EQ(resultStem, stem + "_")
        << "a reserved stem \"" << stem
        << "\" must have its stem portion gain a trailing \"_\" from the reserved-name "
           "guard, not pass through unchanged as \""
        << resultStem << "\"";
}

INSTANTIATE_TEST_SUITE_P(
    ReservedNamesAndCaseVariants,
    TestPathSanitizerReservedStems,
    ::testing::Values(
        "CON", "con", "Con", "PRN", "AUX", "NUL", "COM1", "com1", "COM9", "LPT1", "lpt1", "LPT9"));

TEST_F(TestPathSanitizer, InjectivityAcrossDistinctInputs)
{
    // Includes pairs that would collide without the hash suffix under a naive
    // non-suffixed scheme (e.g. "a:b" and "a_b" both sanitize to "a_b").
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

/// EXPECT_FALSE(result.empty()) alone is unfalsifiable: the unconditional 16-hex-digit
/// hash suffix (see ResultAlwaysCarriesTheUnconditionalHashSuffix) already guarantees a
/// non-empty result no matter what the stem-building logic does, even if it produced no
/// stem characters at all. The actual claim -- that an empty stem is turned into "_"
/// rather than left empty -- is only checked by examining the stem portion.
TEST_F(TestPathSanitizer, EmptyInputProducesNonEmptyResult)
{
    const auto result = sanitizeForPath("");

    const auto dashPos = result.rfind('-');
    ASSERT_NE(dashPos, std::string::npos);
    const auto stem = result.substr(0, dashPos);
    EXPECT_EQ(stem, "_") << "an empty input's stem portion must become \"_\", not stay "
                            "empty (result: \""
                         << result << "\")";
}

TEST_F(TestPathSanitizer, LongInputIsCapped)
{
    const std::string longInput(1024, 'z');

    const auto result = sanitizeForPath(longInput);

    // Capped well under a typical 255-byte filesystem component limit, even with the
    // "-<16 hex digits>" suffix appended.
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
    // "a:b" and "a/b" both sanitize to the same stem ("a_b"); the hash suffix over the
    // raw input keeps the results distinct.
    const auto resultColon = sanitizeForPath("a:b");
    const auto resultSlash = sanitizeForPath("a/b");

    EXPECT_NE(resultColon, resultSlash);
}
