// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cctype>

#include "common/PlatformUtils.hpp"

using hipdnn_integration_tests::currentPlatform;
using hipdnn_integration_tests::globMatch;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

// ---------------------------------------------------------------------------
// Exact match
// ---------------------------------------------------------------------------

TEST(TestGlobMatch, ExactMatchReturnsTrue)
{
    EXPECT_TRUE(globMatch("hello", "hello"));
}

TEST(TestGlobMatch, ExactMismatchReturnsFalse)
{
    EXPECT_FALSE(globMatch("hello", "world"));
}

// ---------------------------------------------------------------------------
// Star wildcard
// ---------------------------------------------------------------------------

TEST(TestGlobMatch, StarMatchesEverything)
{
    EXPECT_TRUE(globMatch("*", "anything"));
}

TEST(TestGlobMatch, StarMatchesEmptyString)
{
    EXPECT_TRUE(globMatch("*", ""));
}

TEST(TestGlobMatch, LeadingStar)
{
    EXPECT_TRUE(globMatch("*world", "hello world"));
    EXPECT_FALSE(globMatch("*world", "hello world!"));
}

TEST(TestGlobMatch, TrailingStar)
{
    EXPECT_TRUE(globMatch("hello*", "hello world"));
    EXPECT_FALSE(globMatch("hello*", "hi world"));
}

TEST(TestGlobMatch, MiddleStar)
{
    EXPECT_TRUE(globMatch("he*ld", "hello world"));
    EXPECT_FALSE(globMatch("he*ld", "hello worlds"));
}

TEST(TestGlobMatch, MultipleStars)
{
    EXPECT_TRUE(globMatch("*Conv*Fp16*", "IntegrationGpuConvFwd2dFp16/Smoke"));
    EXPECT_FALSE(globMatch("*Conv*Fp16*", "IntegrationGpuConvFwd2dFp32/Smoke"));
}

// ---------------------------------------------------------------------------
// Question-mark wildcard
// ---------------------------------------------------------------------------

TEST(TestGlobMatch, QuestionMarkMatchesSingleChar)
{
    EXPECT_TRUE(globMatch("h?llo", "hello"));
    EXPECT_TRUE(globMatch("h?llo", "hallo"));
}

TEST(TestGlobMatch, QuestionMarkDoesNotMatchEmpty)
{
    EXPECT_FALSE(globMatch("h?llo", "hllo"));
}

TEST(TestGlobMatch, QuestionMarkDoesNotMatchMultiple)
{
    EXPECT_FALSE(globMatch("h?o", "hello"));
}

// ---------------------------------------------------------------------------
// Empty inputs
// ---------------------------------------------------------------------------

TEST(TestGlobMatch, BothEmptyMatches)
{
    EXPECT_TRUE(globMatch("", ""));
}

TEST(TestGlobMatch, StarMatchesEmptyText)
{
    EXPECT_TRUE(globMatch("*", ""));
}

TEST(TestGlobMatch, LiteralDoesNotMatchEmptyText)
{
    EXPECT_FALSE(globMatch("a", ""));
}

// ---------------------------------------------------------------------------
// Practical patterns from integration tests
// ---------------------------------------------------------------------------

TEST(TestGlobMatch, ConvFwdStylePattern)
{
    EXPECT_TRUE(
        globMatch("*ConvFwd*Fp16*", "IntegrationGpuConvFwd2dFp16/Smoke.Correctness/NCHW_params"));
    EXPECT_FALSE(
        globMatch("*ConvFwd*Fp16*", "IntegrationGpuConvFwd2dFp32/Smoke.Correctness/NCHW_params"));
}

TEST(TestGlobMatch, BroadWildcardPattern)
{
    EXPECT_TRUE(globMatch("*convolution*fp16*", "test_convolution_fwd_fp16_nchw"));
    EXPECT_FALSE(globMatch("*convolution*fp16*", "test_convolution_fwd_fp32_nchw"));
}

// ---------------------------------------------------------------------------
// currentPlatform
// ---------------------------------------------------------------------------

TEST(TestCurrentPlatform, ReturnsExpectedValueForBuildOs)
{
#ifdef _WIN32
    EXPECT_EQ(currentPlatform(), "windows");
#elif defined(__linux__)
    EXPECT_EQ(currentPlatform(), "linux");
#endif
}

TEST(TestCurrentPlatform, ReturnsLowercase)
{
    // Schema contract: values match against TOML 'platforms' entries which
    // we document as lowercase. Verify the helper itself returns lowercase.
    const auto p = currentPlatform();
    for(const char c : p)
    {
        EXPECT_FALSE(std::isupper(static_cast<unsigned char>(c))) << "non-lowercase char: " << c;
    }
}

// NOLINTEND(readability-identifier-naming)
