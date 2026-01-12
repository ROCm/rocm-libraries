// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenBatchnormFwdInferenceWithVariancePlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_legacy_plugin;

TEST(TestMiopenBatchnormFwdInferenceWithVarianceParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid batchnorm graph with variance
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

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
