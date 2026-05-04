// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "harness/ReferenceGraphExecutorFactory.hpp"

using hipdnn_integration_tests::ReferenceExecutorType;
using hipdnn_integration_tests::ReferenceGraphExecutorFactory;

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

TEST(TestReferenceGraphExecutorFactory, CreateCpuExecutor)
{
    auto executor = ReferenceGraphExecutorFactory::create(ReferenceExecutorType::CPU);
    ASSERT_NE(executor, nullptr);
    EXPECT_FALSE(executor->requiresDeviceMemory());
}

TEST(TestReferenceGraphExecutorFactory, CreateDeviceExecutor)
{
    auto executor = ReferenceGraphExecutorFactory::create(ReferenceExecutorType::GPU);
    ASSERT_NE(executor, nullptr);
    EXPECT_TRUE(executor->requiresDeviceMemory());
}

TEST(TestReferenceGraphExecutorFactory, DefaultConfigReturnsCpu)
{
    // TestConfig is initialized by TestConfigInitialized (TestTestConfig.cpp) without a
    // reference executor type. When run with --gtest_filter that excludes TestConfigInitialized,
    // the singleton may not be initialized — skip in that case.
    try
    {
        static_cast<void>(hipdnn_integration_tests::TestConfig::get().getReferenceExecutorType());
    }
    catch(const std::runtime_error&)
    {
        GTEST_SKIP() << "TestConfig not initialized (requires TestConfigInitialized suite)";
    }

    auto executor = ReferenceGraphExecutorFactory::createFromConfig();
    ASSERT_NE(executor, nullptr);
    EXPECT_FALSE(executor->requiresDeviceMemory());
}

// NOLINTEND(readability-identifier-naming)
