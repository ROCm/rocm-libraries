// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

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

// ---------------------------------------------------------------------------
// Suite 2 – initialized singleton
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
        TestConfig::initialize(TEST_ARTICLE_PATH, TEST_ENGINE_NAME);
    }
};

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

TEST_F(TestConfigInitialized, DoubleInitializeThrows)
{
    EXPECT_THROW(TestConfig::initialize("/other/path", "OTHER_ENGINE"), std::runtime_error);
}

// NOLINTEND(readability-identifier-naming)
