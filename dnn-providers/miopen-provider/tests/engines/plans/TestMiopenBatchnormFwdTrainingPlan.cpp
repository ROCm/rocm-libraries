// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenBatchnormFwdTrainingPlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_plugin;

TEST(TestBatchnormFwdTrainingParams, InitializesRequiredTensorsFromValidGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.bias());
}

TEST(TestBatchnormFwdTrainingParams, ExtractsEpsilonValueCorrectly)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    // Epsilon should be extracted as double
    EXPECT_NEAR(params.epsilonValue(nullptr, 0), 1e-5, 1e-10);
}

TEST(TestBatchnormFwdTrainingParams, HandlesMeanVariancePresent)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {1, 3, 14, 14}, {1, 3, 14, 14}, true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_TRUE(params.hasSaveMeanVariance());
    EXPECT_NO_THROW(params.mean());
    EXPECT_NO_THROW(params.invVariance());
}

TEST(TestBatchnormFwdTrainingParams, HandlesMeanVarianceMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(
        {1, 3, 14, 14}, {1, 3, 14, 14}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasSaveMeanVariance());
}

// A virtual saved pair that nothing consumes is "saved statistics not requested".
TEST(TestBatchnormFwdTrainingParams, LeavesSavedStatisticsUnsetWhenBothVirtual)
{
    auto savedBuilder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper savedGraph(
        savedBuilder.GetBufferPointer(), savedBuilder.GetSize());
    const auto* savedAttrs = savedGraph.getNode(0).attributes_as_BatchnormAttributes();
    ASSERT_NE(savedAttrs, nullptr);
    const BatchnormFwdTrainingParams savedParams(*savedAttrs, savedGraph.getTensorMap());
    EXPECT_TRUE(savedParams.hasSaveMeanVariance());

    auto virtualBuilder = hipdnn_test_sdk::utilities::mutateGraph(
        hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph(),
        [](hipdnn_flatbuffers_sdk::data_objects::GraphT& graph) {
            for(auto& tensor : graph.tensors)
            {
                if(tensor->name == "mean" || tensor->name == "inv_variance")
                {
                    tensor->virtual_ = true;
                }
            }
        });
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper virtualGraph(
        virtualBuilder.GetBufferPointer(), virtualBuilder.GetSize());
    const auto* virtualAttrs = virtualGraph.getNode(0).attributes_as_BatchnormAttributes();
    ASSERT_NE(virtualAttrs, nullptr);
    const BatchnormFwdTrainingParams virtualParams(*virtualAttrs, virtualGraph.getTensorMap());
    EXPECT_FALSE(virtualParams.hasSaveMeanVariance());
}

TEST(TestBatchnormFwdTrainingParams, HasRunningStatsReturnsFalseWhenNotProvided)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormFwdTrainingParams params(*attrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasRunningStats());
}
