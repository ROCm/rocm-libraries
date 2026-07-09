// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <set>

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

    EXPECT_EQ(total, 5u);
    EXPECT_EQ(numEngines, 5u);
}

TEST_F(ExampleProviderContainerTest, CopyEngineIds_CopyAll)
{
    std::vector<int64_t> ids(5, 0);
    uint32_t numEngines = 0;
    const auto total = ExampleProviderContainer::copyEngineIds(ids.data(), 5, numEngines);

    EXPECT_EQ(total, 5u);
    EXPECT_EQ(numEngines, 5u);
    // Engine IDs should be non-zero (they are hashed from engine names) and distinct.
    for(const auto id : ids)
    {
        EXPECT_NE(id, 0);
    }
    EXPECT_EQ(std::set<int64_t>(ids.begin(), ids.end()).size(), ids.size());
}

TEST_F(ExampleProviderContainerTest, CopyEngineIds_CopyPartial)
{
    std::vector<int64_t> ids(1, 0);
    uint32_t numEngines = 0;
    const auto total = ExampleProviderContainer::copyEngineIds(ids.data(), 1, numEngines);

    // Total is always the full count, but only 1 was copied
    EXPECT_EQ(total, 5u);
    EXPECT_EQ(numEngines, 1u);
    EXPECT_NE(ids[0], 0);
}

TEST_F(ExampleProviderContainerTest, GetEngineManager_HasAllEngines)
{
    auto& manager = _container.getEngineManager();
    const auto ids = manager.getAllEngineIds();
    EXPECT_EQ(ids.size(), 5u);
}
