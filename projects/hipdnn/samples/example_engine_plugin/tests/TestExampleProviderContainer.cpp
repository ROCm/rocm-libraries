// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ExampleProviderContainer.hpp"

using namespace example_provider;

class ExampleProviderContainerTest : public ::testing::Test
{
protected:
    ExampleProviderContainer _container;
};

TEST_F(ExampleProviderContainerTest, CopyEngineIds_QueryCountOnly)
{
    uint32_t numEngines = 0;
    const auto total = ExampleProviderContainer::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(total, 2u);
    EXPECT_EQ(numEngines, 2u);
}

TEST_F(ExampleProviderContainerTest, CopyEngineIds_CopyAll)
{
    std::vector<int64_t> ids(2, 0);
    uint32_t numEngines = 0;
    const auto total = ExampleProviderContainer::copyEngineIds(ids.data(), 2, numEngines);

    EXPECT_EQ(total, 2u);
    EXPECT_EQ(numEngines, 2u);
    // Engine IDs should be non-zero (they are hashed from engine names)
    EXPECT_NE(ids[0], 0);
    EXPECT_NE(ids[1], 0);
    // Both engine IDs should be distinct
    EXPECT_NE(ids[0], ids[1]);
}

TEST_F(ExampleProviderContainerTest, CopyEngineIds_CopyPartial)
{
    std::vector<int64_t> ids(1, 0);
    uint32_t numEngines = 0;
    const auto total = ExampleProviderContainer::copyEngineIds(ids.data(), 1, numEngines);

    // Total is always the full count, but only 1 was copied
    EXPECT_EQ(total, 2u);
    EXPECT_EQ(numEngines, 1u);
    EXPECT_NE(ids[0], 0);
}

TEST_F(ExampleProviderContainerTest, GetEngineManager_HasAllEngines)
{
    auto& manager = _container.getEngineManager();
    const auto ids = manager.getAllEngineIds();
    // The EngineManager should have 2 engines registered
    EXPECT_EQ(ids.size(), 2u);
}
