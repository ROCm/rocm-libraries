// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ExamplePluginContainer.hpp"

using namespace example_plugin;

class ExamplePluginContainerTest : public ::testing::Test
{
protected:
    ExamplePluginContainer container;
};

TEST_F(ExamplePluginContainerTest, CopyEngineIds_QueryCountOnly)
{
    uint32_t numEngines = 0;
    auto total = ExamplePluginContainer::copyEngineIds(nullptr, 0, numEngines);

    EXPECT_EQ(total, 3u);
    EXPECT_EQ(numEngines, 3u);
}

TEST_F(ExamplePluginContainerTest, CopyEngineIds_CopyAll)
{
    std::vector<int64_t> ids(3, 0);
    uint32_t numEngines = 0;
    auto total = ExamplePluginContainer::copyEngineIds(ids.data(), 3, numEngines);

    EXPECT_EQ(total, 3u);
    EXPECT_EQ(numEngines, 3u);
    // Engine IDs should be non-zero (they are hashed from engine names)
    EXPECT_NE(ids[0], 0);
    EXPECT_NE(ids[1], 0);
    EXPECT_NE(ids[2], 0);
    // All three engine IDs should be distinct
    EXPECT_NE(ids[0], ids[1]);
    EXPECT_NE(ids[0], ids[2]);
    EXPECT_NE(ids[1], ids[2]);
}

TEST_F(ExamplePluginContainerTest, CopyEngineIds_CopyPartial)
{
    std::vector<int64_t> ids(1, 0);
    uint32_t numEngines = 0;
    auto total = ExamplePluginContainer::copyEngineIds(ids.data(), 1, numEngines);

    // Total is always the full count, but only 1 was copied
    EXPECT_EQ(total, 3u);
    EXPECT_EQ(numEngines, 1u);
    EXPECT_NE(ids[0], 0);
}

TEST_F(ExamplePluginContainerTest, GetEngineManager_HasAllEngines)
{
    auto& manager = container.getEngineManager();
    auto ids = manager.getAllEngineIds();
    // The EngineManager should have 3 engines registered
    EXPECT_EQ(ids.size(), 3u);
}
