// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ExamplePluginHandle.hpp"
#include "TestHelpers.hpp"
#include "engines/AdvancedEngineScaffold.hpp"

#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>

using namespace example_plugin;

class AdvancedEngineScaffoldTest : public ::testing::Test
{
protected:
    static constexpr int64_t kEngineId = 42;

    ExamplePluginHandle handle;
    AdvancedEngineScaffold scaffold{kEngineId};
};

TEST_F(AdvancedEngineScaffoldTest, Id_ReturnsAssignedId)
{
    EXPECT_EQ(scaffold.id(), kEngineId);
}

TEST_F(AdvancedEngineScaffoldTest, IsApplicable_ReturnsFalse_ForReluGraph)
{
    auto builder = test_helpers::createReluFwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    EXPECT_FALSE(scaffold.isApplicable(handle, graph));
}

TEST_F(AdvancedEngineScaffoldTest, IsApplicable_ReturnsFalse_ForNonReluGraph)
{
    auto builder = test_helpers::createNonReluPointwiseGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    EXPECT_FALSE(scaffold.isApplicable(handle, graph));
}

TEST_F(AdvancedEngineScaffoldTest, IsApplicable_ReturnsFalse_ForMultiNodeGraph)
{
    auto builder = test_helpers::createMultiNodeReluGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    EXPECT_FALSE(scaffold.isApplicable(handle, graph));
}
