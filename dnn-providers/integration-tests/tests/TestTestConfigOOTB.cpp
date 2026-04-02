// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "harness/TestConfig.hpp"

using hipdnn_integration_tests::TestConfig;

// Separate binary from TestTestConfig.cpp because TestConfig is a singleton
// that can only be initialized once per process.

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

class TestConfigOOTBInitialized : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        TestConfig::initializeOOTB();
    }
};

TEST_F(TestConfigOOTBInitialized, IsOOTBModeReturnsTrue)
{
    EXPECT_TRUE(TestConfig::get().isOOTBMode());
}

TEST_F(TestConfigOOTBInitialized, GetArticlePathThrowsInOOTBMode)
{
    EXPECT_THROW(TestConfig::get().getArticlePath(), std::runtime_error);
}

TEST_F(TestConfigOOTBInitialized, GetEngineNameThrowsInOOTBMode)
{
    EXPECT_THROW(TestConfig::get().getEngineName(), std::runtime_error);
}

TEST_F(TestConfigOOTBInitialized, GetEngineIdThrowsInOOTBMode)
{
    EXPECT_THROW(TestConfig::get().getEngineId(), std::runtime_error);
}

TEST_F(TestConfigOOTBInitialized, GetToleranceModeWorksInOOTBMode)
{
    EXPECT_EQ(TestConfig::get().getToleranceMode(),
              hipdnn_integration_tests::ToleranceMode::DEFAULT);
}

TEST_F(TestConfigOOTBInitialized, DoubleInitializeThrows)
{
    EXPECT_THROW(TestConfig::initializeOOTB(), std::runtime_error);
}

TEST_F(TestConfigOOTBInitialized, EngineInitializeAfterOOTBThrows)
{
    EXPECT_THROW(TestConfig::initialize("/other/path", "OTHER_ENGINE"), std::runtime_error);
}

// NOLINTEND(readability-identifier-naming)
