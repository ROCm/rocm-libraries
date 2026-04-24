// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/layernorm/LayernormFwdPlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider::layernorm;

TEST(TestLayernormFwdParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid layernorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    // Get the layernorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormAttributes();
    ASSERT_NE(attrs, nullptr);

    // Expect that params construction doesn't throw
    EXPECT_NO_THROW(LayernormFwdParams(*attrs, graph.getTensorMap()));
}
