// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "harness/TestConfig.hpp"

using hipdnn_integration_tests::TestConfig;

// Tests are organized into two suites to guarantee execution order.
// The singleton starts uninitialized, so the "Uninitialized" suite must
// run before the "Initialized" suite which calls initialize().
// Gtest runs suites in definition order within a translation unit.

// ---------------------------------------------------------------------------
// Suite 1 – uninitialized singleton (plain TEST, no fixture needed)
// ---------------------------------------------------------------------------

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

TEST(TestConfigUninitialized, GetReturnsInstance)
{
    // get() should return a reference without crashing, even before initialize()
    const TestConfig& instance = TestConfig::get();
    EXPECT_EQ(&instance, &TestConfig::get());
}

TEST(TestConfigUninitialized, GetArticlePathThrowsWhenUninitialized)
{
    EXPECT_THROW(TestConfig::get().getArticlePath(), std::runtime_error);
}

TEST(TestConfigUninitialized, GetEngineNameThrowsWhenUninitialized)
{
    EXPECT_THROW(TestConfig::get().getEngineName(), std::runtime_error);
}

TEST(TestConfigUninitialized, GetToleranceModeThrowsWhenUninitialized)
{
    EXPECT_THROW(TestConfig::get().getToleranceMode(), std::runtime_error);
}

TEST(TestConfigUninitialized, HasEngineNameThrowsWhenUninitialized)
{
    EXPECT_THROW(TestConfig::get().hasEngineName(), std::runtime_error);
}

TEST(TestConfigUninitialized, HasArticlePathThrowsWhenUninitialized)
{
    EXPECT_THROW(TestConfig::get().hasArticlePath(), std::runtime_error);
}

TEST(TestConfigUninitialized, FailOnUnsupportedThrowsWhenUninitialized)
{
    EXPECT_THROW(TestConfig::get().failOnUnsupported(), std::runtime_error);
}

TEST(TestConfigUninitialized, SkipGraphValidationThrowsWhenUninitialized)
{
    EXPECT_THROW(TestConfig::get().skipGraphValidation(), std::runtime_error);
}

TEST(TestConfigUninitialized, GetReferenceExecutorTypeThrowsWhenUninitialized)
{
    EXPECT_THROW(TestConfig::get().getReferenceExecutorType(), std::runtime_error);
}

// parseVerificationMode is a free function (no singleton state), so it can be
// exercised regardless of initialization.
TEST(TestParseVerificationMode, AcceptsAllValidValuesCaseInsensitive)
{
    using hipdnn_integration_tests::parseVerificationMode;
    using hipdnn_integration_tests::VerificationMode;

    EXPECT_EQ(parseVerificationMode("auto"), VerificationMode::AUTO);
    EXPECT_EQ(parseVerificationMode("golden"), VerificationMode::GOLDEN);
    EXPECT_EQ(parseVerificationMode("gpu"), VerificationMode::GPU);
    EXPECT_EQ(parseVerificationMode("cpu"), VerificationMode::CPU);

    EXPECT_EQ(parseVerificationMode("AUTO"), VerificationMode::AUTO);
    EXPECT_EQ(parseVerificationMode("Golden"), VerificationMode::GOLDEN);
    EXPECT_EQ(parseVerificationMode("GPU"), VerificationMode::GPU);
}

TEST(TestParseVerificationMode, ThrowsOnInvalidValue)
{
    EXPECT_THROW(hipdnn_integration_tests::parseVerificationMode("bogus"), std::runtime_error);
    EXPECT_THROW(hipdnn_integration_tests::parseVerificationMode(""), std::runtime_error);
}

// resolveVerificationMode / resolveGoldenDataDir are free functions that
// implement the "CLI wins, then env, then nullopt" precedence chain.
// They don't touch the singleton so they can be tested freely.

TEST(TestResolveVerificationMode, CliValueWinsOverEnv)
{
    using hipdnn_integration_tests::resolveVerificationMode;
    using hipdnn_integration_tests::VerificationMode;

    // Even if the env var were set, the CLI value takes precedence.
    const auto result = resolveVerificationMode(VerificationMode::GPU);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, VerificationMode::GPU);
}

TEST(TestResolveVerificationMode, NulloptCliWithoutEnvReturnsNullopt)
{
    using hipdnn_integration_tests::resolveVerificationMode;

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        "HIPDNN_TEST_VERIFICATION_MODE");
    const auto result = resolveVerificationMode(std::nullopt);
    EXPECT_FALSE(result.has_value());
}

TEST(TestResolveGoldenDataDir, CliValueWinsOverEnv)
{
    using hipdnn_integration_tests::resolveGoldenDataDir;

    const std::filesystem::path cliPath = "/explicit/golden/dir";
    const auto result = resolveGoldenDataDir(cliPath);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, cliPath);
}

TEST(TestResolveGoldenDataDir, NulloptCliWithoutEnvReturnsNullopt)
{
    using hipdnn_integration_tests::resolveGoldenDataDir;

    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter guard(
        "HIPDNN_TEST_GOLDEN_DATA_DIR");
    const auto result = resolveGoldenDataDir(std::nullopt);
    EXPECT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// Suite 2 – initialized singleton (all args provided)
// ---------------------------------------------------------------------------

namespace
{
const std::string TEST_ENGINE_NAME = "MIOPEN_ENGINE"; // NOLINT(readability-identifier-naming)
const std::string TEST_ARTICLE_PATH
    = "/tmp/test_article.so"; // NOLINT(readability-identifier-naming)
} // namespace

class TestConfigInitialized : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        hipdnn_integration_tests::TestConfigOptions opts;
        opts.articlePath = TEST_ARTICLE_PATH;
        opts.engineName = TEST_ENGINE_NAME;
        opts.failOnUnsupported = true;
        TestConfig::initialize(std::move(opts));
    }
};

TEST_F(TestConfigInitialized, HasArticlePathReturnsTrue)
{
    EXPECT_TRUE(TestConfig::get().hasArticlePath());
}

TEST_F(TestConfigInitialized, HasEngineNameReturnsTrue)
{
    EXPECT_TRUE(TestConfig::get().hasEngineName());
}

TEST_F(TestConfigInitialized, FailOnUnsupportedReturnsTrue)
{
    EXPECT_TRUE(TestConfig::get().failOnUnsupported());
}

TEST_F(TestConfigInitialized, SkipGraphValidationReturnsFalse)
{
    EXPECT_FALSE(TestConfig::get().skipGraphValidation());
}

TEST_F(TestConfigInitialized, InitializeSetsArticlePath)
{
    EXPECT_EQ(TestConfig::get().getArticlePath(), TEST_ARTICLE_PATH);
}

TEST_F(TestConfigInitialized, InitializeSetsEngineName)
{
    EXPECT_EQ(TestConfig::get().getEngineName(), TEST_ENGINE_NAME);
}

TEST_F(TestConfigInitialized, GetEngineIdReturnsConsistentHash)
{
    const auto expected = hipdnn_data_sdk::utilities::engineNameToId(TEST_ENGINE_NAME);
    EXPECT_EQ(TestConfig::get().getEngineId(), expected);
}

TEST_F(TestConfigInitialized, GetReferenceExecutorTypeDefaultsToCpu)
{
    EXPECT_EQ(TestConfig::get().getReferenceExecutorType(),
              hipdnn_integration_tests::ReferenceExecutorType::CPU);
}

TEST_F(TestConfigInitialized, GetVerificationModeDefaultsToAuto)
{
    // No CLI flag and (assuming) no env var -> AUTO.
    EXPECT_EQ(TestConfig::get().getVerificationMode(),
              hipdnn_integration_tests::VerificationMode::AUTO);
}

TEST_F(TestConfigInitialized, HasGoldenDataDirReturnsFalseWhenNotProvided)
{
    EXPECT_FALSE(TestConfig::get().hasGoldenDataDir());
}

TEST_F(TestConfigInitialized, GetGoldenDataDirThrowsWhenNotProvided)
{
    EXPECT_THROW(TestConfig::get().getGoldenDataDir(), std::runtime_error);
}

TEST_F(TestConfigInitialized, DoubleInitializeThrows)
{
    EXPECT_THROW(TestConfig::initialize(hipdnn_integration_tests::TestConfigOptions{}),
                 std::runtime_error);
}

// NOLINTEND(readability-identifier-naming)
