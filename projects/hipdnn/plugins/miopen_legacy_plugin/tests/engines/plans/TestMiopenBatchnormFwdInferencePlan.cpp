// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenBatchnormFwdInferencePlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_legacy_plugin;

TEST(TestMiopenBatchnormFwdInferenceParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid batchnorm graph
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    // Get the batchnorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    // Construct params
    BatchnormFwdInferenceParams params(*attrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.bias());
    EXPECT_NO_THROW(params.estMean());
    EXPECT_NO_THROW(params.estVariance());
}
