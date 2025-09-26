// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenBatchnormBwdPlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_legacy_plugin;

TEST(TestBatchnormBwdParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid batchnorm graph
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormBwdGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    BatchnormBwdParams params(*attrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.dy());
    EXPECT_NO_THROW(params.dx());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.dscale());
    EXPECT_NO_THROW(params.dbias());

    // Optional tensors should be present
    auto& meanOpt = params.optMean();
    auto& varOpt = params.optInvVariance();
    EXPECT_TRUE(meanOpt.has_value());
    EXPECT_TRUE(varOpt.has_value());
}

TEST(TestBatchnormBwdParams, HandlesMissingOptionalTensors)
{
    // Create a valid batchnorm graph and remove mean/variance from tensor map
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormBwdGraph(
        {1, 1, 1, 1}, {1, 1, 1, 1}, false // Set has_optional_attributes to false
    );
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const auto& tensorMap = graph.getTensorMap();
    BatchnormBwdParams params(*attrs, tensorMap);

    // Optional tensors should not be present
    EXPECT_FALSE(params.optMean().has_value());
    EXPECT_FALSE(params.optInvVariance().has_value());
}
