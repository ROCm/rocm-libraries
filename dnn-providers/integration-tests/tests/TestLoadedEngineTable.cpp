// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "harness/bundle/LoadedEngineTable.hpp"

using hipdnn_integration_tests::bundle::LoadedEngine;
using hipdnn_integration_tests::bundle::LoadedEngineTable;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

class TestLoadedEngineTable : public ::testing::Test
{
protected:
    void SetUp() override
    {
        LoadedEngineTable::get().reset();
    }
    void TearDown() override
    {
        LoadedEngineTable::get().reset();
    }
};

} // namespace

TEST_F(TestLoadedEngineTable, IsLoadedReturnsTrueForKnownEngines)
{
    LoadedEngineTable::get().setForTesting({{42, "ENGINE_A"}, {7, "ENGINE_B"}});

    EXPECT_TRUE(LoadedEngineTable::get().isLoaded("ENGINE_A"));
    EXPECT_TRUE(LoadedEngineTable::get().isLoaded("ENGINE_B"));
    EXPECT_FALSE(LoadedEngineTable::get().isLoaded("ENGINE_C"));
}

TEST_F(TestLoadedEngineTable, AllReturnsAllEngines)
{
    LoadedEngineTable::get().setForTesting({{42, "ENGINE_A"}, {7, "ENGINE_B"}, {99, "ENGINE_C"}});

    const auto& all = LoadedEngineTable::get().all();
    ASSERT_EQ(all.size(), 3u);
    EXPECT_EQ(all[0].id, 42);
    EXPECT_EQ(all[0].name, "ENGINE_A");
    EXPECT_EQ(all[1].id, 7);
    EXPECT_EQ(all[1].name, "ENGINE_B");
    EXPECT_EQ(all[2].id, 99);
    EXPECT_EQ(all[2].name, "ENGINE_C");
}

TEST_F(TestLoadedEngineTable, ResetClearsBuiltFlag)
{
    LoadedEngineTable::get().setForTesting({{42, "ENGINE_A"}});
    ASSERT_TRUE(LoadedEngineTable::get().isBuilt());

    LoadedEngineTable::get().reset();

    EXPECT_FALSE(LoadedEngineTable::get().isBuilt());
}

TEST_F(TestLoadedEngineTable, AllThrowsBeforeBuild)
{
    EXPECT_THROW(LoadedEngineTable::get().all(), std::runtime_error);
}

TEST_F(TestLoadedEngineTable, IsLoadedThrowsBeforeBuild)
{
    EXPECT_THROW(LoadedEngineTable::get().isLoaded("anything"), std::runtime_error);
}

TEST_F(TestLoadedEngineTable, FindReturnsTheMatchingEngine)
{
    LoadedEngineTable::get().setForTesting({{42, "ENGINE_A"}, {7, "ENGINE_B"}});

    const LoadedEngine* found = LoadedEngineTable::get().find("ENGINE_B");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->id, 7);
    EXPECT_EQ(found->name, "ENGINE_B");
}

TEST_F(TestLoadedEngineTable, FindReturnsNullptrForAnUnknownName)
{
    LoadedEngineTable::get().setForTesting({{42, "ENGINE_A"}});

    EXPECT_EQ(LoadedEngineTable::get().find("ENGINE_C"), nullptr);
}

TEST_F(TestLoadedEngineTable, FindReturnsNullptrOnAnEmptyTable)
{
    LoadedEngineTable::get().setForTesting({});

    EXPECT_EQ(LoadedEngineTable::get().find("ANYTHING"), nullptr);
}

// NOLINTEND(readability-identifier-naming)
