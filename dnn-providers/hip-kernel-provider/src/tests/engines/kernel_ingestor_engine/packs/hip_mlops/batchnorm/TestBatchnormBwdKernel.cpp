// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/hip_mlops/batchnorm/BatchnormBwdKernel.hpp"

#include "../TestPlanCommon.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

namespace hip_kernel_provider::batchnorm
{

TEST(TestBatchnormBwdParams, InitializesAllTensorsFromValidBwdGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormBwdParams params(*attrs, graph.getTensorMap());
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_TRUE(params.hasSavedStats());
    EXPECT_NE(params.savedMean(), nullptr);
    EXPECT_NE(params.savedInvVariance(), nullptr);
    EXPECT_FALSE(params.optActivation().has_value());
    EXPECT_EQ(params.bias(), nullptr);
}

TEST(TestBatchnormBwdParams, HandlesMissingOptionalSavedStats)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph(
        {1, 1, 1, 1}, {1, 1, 1, 1}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_BatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    const BatchnormBwdParams params(*attrs, graph.getTensorMap());
    EXPECT_FALSE(params.hasSavedStats());
    EXPECT_EQ(params.savedMean(), nullptr);
    EXPECT_EQ(params.savedInvVariance(), nullptr);
}

TEST(TestBatchnormBwdParams, InitializesFusedActivationWithAllTensors)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    ASSERT_EQ(graph.nodeCount(), 3u);

    const auto& bnInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    const BatchnormBwdParams params(
        *bnBwdAttrs, *pointwiseAttrs, *bnInfAttrs, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_TRUE(params.hasSavedStats());
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_EQ(params.dy()->uid(), pointwiseAttrs->in_0_tensor_uid());
}

TEST(TestBatchnormBwdParams, FusedModeHasActivationAndBias)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    ASSERT_NE(bnInfAttrs, nullptr);
    ASSERT_NE(pointwiseAttrs, nullptr);
    ASSERT_NE(bnBwdAttrs, nullptr);

    const BatchnormBwdParams params(
        *bnBwdAttrs, *pointwiseAttrs, *bnInfAttrs, graph.getTensorMap());
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.bias(), nullptr);
}

TEST(TestBatchnormBwdKernel, BuildDefaultSetsCorrectDefines)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& pointwiseNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    auto* bnInfAttrs = bnInfNode.attributes_as_BatchnormInferenceAttributes();
    auto* pointwiseAttrs = pointwiseNode.attributes_as_PointwiseAttributes();
    auto* bnBwdAttrs = bnBwdNode.attributes_as_BatchnormBackwardAttributes();
    const BatchnormBwdParams params(
        *bnBwdAttrs, *pointwiseAttrs, *bnInfAttrs, graph.getTensorMap());

    const auto deviceProps = createTestDeviceProps();
    const std::vector<std::string> capturedOptions
        = buildBatchnormBwdKernel(
              params, std::string(deviceProps.gcnArchName), defaultBatchnormBwdConfig(params))
              .options;

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FPUPCAST=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFPUPCAST=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SAVE_MEAN_VARIANCE=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_RUNNING_RESULT=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_BATCHES=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_CHANNELS=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SPATIAL_SIZE=50176"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP0_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP1_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP2_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP0=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP1=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP2=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRPS0=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRPS1=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRPS2=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_BATCH_ELEMENTS=HIP_PLUGIN_BN_BATCHES"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAX_BATCH=768"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDS_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VEC_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAX_SPATIAL=2500"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NODPP=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDSGCN_SIZE=16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_USESAVED=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_STASH_METHOD=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VARIANT=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NRN_OP_ID=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MEAN_VAR_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SCALE_TYPE=float"));
}

TEST(TestBatchnormBwdKernel, BuildWithUnsupportedDimensionThrows)
{
    // 2D tensor is not supported
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph({4, 1}, {1, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormBackwardAttributes();

    const BatchnormBwdParams params(attr, graph.getTensorMap());

    EXPECT_THROW(static_cast<void>(
                     buildBatchnormBwdKernel(params, "gfx942", defaultBatchnormBwdConfig(params))),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// buildBatchnormBwdKernel - explicit performance configuration
// ============================================================================

namespace
{

// A backward graph and the params of its batchnorm node, which point into the graph's buffer.
struct BwdGraph
{
    flatbuffers::FlatBufferBuilder builder;
    BatchnormBwdParams params;
};

BwdGraph createBwdGraph()
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    BatchnormBwdParams params(*graph.getNode(0).attributes_as_BatchnormBackwardAttributes(),
                              graph.getTensorMap());
    return {std::move(builder), std::move(params)};
}

bool hasBuildOption(const BatchnormKernelBuild& build, const std::string& option)
{
    const std::vector<std::string> options = build.options;
    return std::find(options.begin(), options.end(), option) != options.end();
}

} // namespace

TEST(TestBatchnormBwdKernelConfig, DefaultConfigIsTheHeuristicChoice)
{
    // NCHW 1x3x224x224: spatial single variant 1 of 1024 threads
    const auto graph = createBwdGraph();
    EXPECT_EQ(defaultBatchnormBwdConfig(graph.params), (BatchnormSpatialConfig{1, 1024, 1}));
}

TEST(TestBatchnormBwdKernelConfig, BuildTakesTheWorkgroupSizeOfTheConfig)
{
    const auto graph = createBwdGraph();

    const auto build = buildBatchnormBwdKernel(graph.params, "gfx942", {1, 256, 1});

    EXPECT_EQ(build.variant, 1);
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_VARIANT=1"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_GRP0=256"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_LDS_SIZE=256"));
    ASSERT_EQ(build.launches.size(), 1u);
    EXPECT_STREQ(build.launches[0].entryPoint, "BatchNormBwdSpatial");
    EXPECT_EQ(build.launches[0].blockSize, (std::array<unsigned int, 3>{256, 1, 1}));
    EXPECT_EQ(build.launches[0].gridSize, (std::array<unsigned int, 3>{3, 1, 1}));
}

TEST(TestBatchnormBwdKernelConfig, BuildTakesTheVectorSizeOfTheConfig)
{
    const auto graph = createBwdGraph();

    const auto build = buildBatchnormBwdKernel(graph.params, "gfx942", {2, 0, 4});

    // NCHW vectorizes the spatial dimension: 50176 / 4 values over workgroups of 1024
    EXPECT_EQ(build.variant, 2);
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_VARIANT=2"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_VEC_SIZE=4"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_GRP1=1024"));
    EXPECT_TRUE(hasBuildOption(build, "-DHIP_PLUGIN_BN_GRPS1=13"));
    // A run with saved statistics makes launches 0, 1 and 4
    ASSERT_EQ(build.launches.size(), 5u);
    EXPECT_STREQ(build.launches[0].entryPoint, "BatchNormBwdSpatialDScaleDBias");
    EXPECT_STREQ(build.launches[1].entryPoint, "BatchNormBwdSpatialFinalDScaleDBias");
    EXPECT_STREQ(build.launches[4].entryPoint, "BatchNormBwdSpatialDX");
    EXPECT_EQ(build.launches[0].gridSize, (std::array<unsigned int, 3>{3, 13, 1}));
}

TEST(TestBatchnormBwdKernelConfig, ApplicabilityFollowsTheProblem)
{
    const auto graph = createBwdGraph();

    EXPECT_TRUE(isBatchnormBwdConfigApplicable(graph.params, {1, 1024, 1}));
    EXPECT_TRUE(isBatchnormBwdConfigApplicable(graph.params, {2, 0, 4}));
    // Variants 0 and 3 give each thread one spatial position: 50176 do not fit 1024 threads
    EXPECT_FALSE(isBatchnormBwdConfigApplicable(graph.params, {0, 1024, 1}));
    EXPECT_FALSE(isBatchnormBwdConfigApplicable(graph.params, {3, 1024, 1}));
}

} // namespace hip_kernel_provider::batchnorm
