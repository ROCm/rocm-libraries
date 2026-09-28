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

/// Windows resolves a device alias from the name before the FIRST dot, so "CON.txt" is
/// the CON device just as "CON" is. Matching the whole stem misses it, and the hash
/// suffix -- appended after the extension -- does not break the alias.
TEST_F(TestPathSanitizer, AReservedStemWithAnExtensionIsStillBroken)
{
    const auto result = sanitizeForPath("CON.txt");

    const auto dashPos = result.rfind('-');
    ASSERT_NE(dashPos, std::string::npos);
    EXPECT_EQ(result.substr(0, dashPos), "CON_.txt")
        << "the guard must break the alias before the first dot, got \"" << result << "\"";
}

/// A control byte is illegal in a path component on both platforms, and an embedded NUL
/// is worse than illegal: CreateFileW and c_str() end the name there, dropping the hash
/// suffix that keeps distinct inputs apart.
TEST_F(TestPathSanitizer, ControlBytesAreReplaced)
{
    const std::string raw("a\1b\x1f"
                          "c\x7f"
                          "d",
                          7);

    const auto result = sanitizeForPath(raw);

    const auto dashPos = result.rfind('-');
    ASSERT_NE(dashPos, std::string::npos);
    EXPECT_EQ(result.substr(0, dashPos), "a_b_c_d");
}

TEST_F(TestPathSanitizer, AnEmbeddedNulDoesNotTruncateTheComponent)
{
    const std::string_view raw("engine\0name", 11);

    const auto result = sanitizeForPath(raw);

    EXPECT_EQ(result.find('\0'), std::string::npos)
        << "a NUL survived into the component; the name would end there and lose the "
           "disambiguating hash suffix";
    EXPECT_EQ(result.rfind("engine_name-", 0), 0U);
}

/// The stem is capped by BYTE count and MSVC reads a narrow path as UTF-8, so a multibyte
/// sequence surviving into the stem could be cut in half at the cap and reach path
/// conversion malformed. Non-ASCII bytes are replaced instead, which deletes the boundary
/// question rather than answering it.
///
/// Falsifying mutation: narrow isIllegal()'s test back to `byte < 0x20 || byte == 0x7f`.
/// The sequences below then survive into the component.
TEST_F(TestPathSanitizer, NonAsciiBytesAreReplaced)
{
    const auto result = sanitizeForPath("engine\xc3\xa9name");

    const auto dashPos = result.rfind('-');
    ASSERT_NE(dashPos, std::string::npos);
    EXPECT_EQ(result.substr(0, dashPos), "engine__name")
        << "both bytes of the two-byte sequence are replaced individually";
}

TEST_F(TestPathSanitizer, ALongNonAsciiInputCannotEndOnAPartialCodepoint)
{
    // Long enough that the 96-byte stem cap lands partway through a three-byte sequence.
    std::string raw;
    for(int i = 0; i < 200; ++i)
    {
        raw += "\xe4\xb8\xad";
    }

    const auto result = sanitizeForPath(raw);

    for(const char c : result)
    {
        ASSERT_LT(static_cast<unsigned char>(c), 0x80U)
            << "a non-ASCII byte reached the path component, so the byte cap can still "
               "split a UTF-8 sequence and emit half a codepoint";
    }
}

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
