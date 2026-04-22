// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenBatchnormFwdInferenceWithVariancePlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_plugin;

TEST(TestMiopenBatchnormFwdInferenceWithVarianceParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid batchnorm graph with variance
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    ASSERT_NE(attrs, nullptr);

    // Expect that params construction doesn't throw
    EXPECT_NO_THROW(BatchnormFwdInferenceWithVarianceParams(*attrs, graph.getTensorMap()));

    BatchnormFwdInferenceWithVarianceParams params(*attrs, graph.getTensorMap());
    // verify activation optional params are null when no activation is specified
    EXPECT_EQ(params.optActivation(), std::nullopt);
    EXPECT_EQ(params.activationOut(), std::nullopt);

    // Verify required tensors are initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.bias());
    EXPECT_NO_THROW(params.estMean());
    EXPECT_NO_THROW(params.variance());
    EXPECT_NO_THROW(params.epsilonValue());
    EXPECT_NEAR(params.epsilonValue(), 1e-5, 1e-10);
}

TEST(TestMiopenBatchnormFwdInferenceWithVarianceParams,
     InitializesAllTensorsFromValidGraphWithActivation)
{
    // Create a valid batchnorm graph with variance and activation
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceActivGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    ASSERT_NE(attrs, nullptr);

    // Get the activation node and attributes
    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    // Expect that params construction doesn't throw
    EXPECT_NO_THROW(
        BatchnormFwdInferenceWithVarianceParams(*attrs, *activAttrs, graph.getTensorMap()));

    BatchnormFwdInferenceWithVarianceParams params(*attrs, *activAttrs, graph.getTensorMap());

    // verify activation optional params are present
    EXPECT_NE(params.optActivation(), std::nullopt);
    EXPECT_NE(params.activationOut(), std::nullopt);

    // Verify required tensors are initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.bias());
    EXPECT_NO_THROW(params.estMean());
    EXPECT_NO_THROW(params.variance());
    EXPECT_NO_THROW(params.epsilonValue());
    EXPECT_NEAR(params.epsilonValue(), 1e-5, 1e-10);
}
