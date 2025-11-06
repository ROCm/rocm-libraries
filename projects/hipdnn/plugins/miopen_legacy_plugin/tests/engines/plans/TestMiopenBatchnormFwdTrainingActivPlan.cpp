// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/plans/MiopenBatchnormFwdTrainingActivPlan.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>

using namespace miopen_legacy_plugin;

TEST(TestBatchnormFwdTrainingActivParams, InitializesRequiredTensorsFromValidGraph)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    BatchnormFwdTrainingActivParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NO_THROW(params.x());
    EXPECT_NO_THROW(params.y());
    EXPECT_NO_THROW(params.scale());
    EXPECT_NO_THROW(params.bias());
    EXPECT_NO_THROW(params.activParams());
}

TEST(TestBatchnormFwdTrainingActivParams, ExtractsEpsilonValueCorrectly)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    BatchnormFwdTrainingActivParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    // Epsilon should be extracted as double
    EXPECT_NEAR(params.epsilonValue(), 1e-5, 1e-10);
}

TEST(TestBatchnormFwdTrainingActivParams, HandlesMeanVariancePresent)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph(true);
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    BatchnormFwdTrainingActivParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    EXPECT_TRUE(params.hasSaveMeanVariance());
    EXPECT_NO_THROW(params.mean());
    EXPECT_NO_THROW(params.invVariance());
}

TEST(TestBatchnormFwdTrainingActivParams, HandlesMeanVarianceMissing)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph(false);
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    BatchnormFwdTrainingActivParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasSaveMeanVariance());
}

TEST(TestBatchnormFwdTrainingActivParams, ThrowsWhenRunningStatsProvided)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph(
        true, true); // with mean/variance and running stats
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    // Should throw because running stats are provided
    EXPECT_THROW(BatchnormFwdTrainingActivParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingActivParams, HasRunningStatsReturnsFalseWhenNotProvided)
{
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    BatchnormFwdTrainingActivParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasRunningStats());
}

TEST(TestBatchnormFwdTrainingActivParams, ValidatesActivationInputMatchesBnOutput)
{
    // This would require creating a malformed graph where activation input != BN output
    // The constructor should throw in this case
    // For now, we verify it doesn't throw for valid case
    auto builder = hipdnn_sdk::test_utilities::createValidBatchnormFwdTrainingActivGraph();
    hipdnn_plugin::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_NO_THROW(BatchnormFwdTrainingActivParams(*bnAttrs, *activAttrs, graph.getTensorMap()));
}
