// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/hip_mlops/batchnorm/BatchnormFwdInferenceWithVarianceKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "../TestPlanCommon.hpp"

namespace hip_kernel_provider::batchnorm::test
{

// ============================================================================
// BatchnormFwdInferenceWithVarianceParams - construction from valid graph data
// ============================================================================

TEST(TestBatchnormFwdInferenceWithVarianceParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();

    EXPECT_NO_THROW(
        const BatchnormFwdInferenceWithVarianceParams params(attr, graph.getTensorMap()));
}

TEST(TestBatchnormFwdInferenceWithVarianceParams, HasCorrectTensorPointersForSingleNode)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();

    const BatchnormFwdInferenceWithVarianceParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_NE(params.estMean(), nullptr);
    EXPECT_NE(params.estVariance(), nullptr);
    EXPECT_NEAR(params.epsilonValue(nullptr, 0), 1e-5, 1e-10);

    // No activation in this graph, so these should be nullopt / nullptr
    EXPECT_EQ(params.optActivation(), std::nullopt);
    EXPECT_EQ(params.activationOut(), nullptr);
}

TEST(TestBatchnormFwdInferenceWithVarianceParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();

    const BatchnormFwdInferenceWithVarianceParams params(attr, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attr.x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attr.y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attr.scale_tensor_uid());
    EXPECT_EQ(params.bias()->uid(), attr.bias_tensor_uid());
    EXPECT_EQ(params.estMean()->uid(), attr.mean_tensor_uid());
    EXPECT_EQ(params.estVariance()->uid(), attr.variance_tensor_uid());
}

TEST(TestBatchnormFwdInferenceWithVarianceParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();

    BatchnormFwdInferenceWithVarianceParams params(attr, graph.getTensorMap());
    const BatchnormFwdInferenceWithVarianceParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestBatchnormFwdInferenceWithVarianceParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<BatchnormFwdInferenceWithVarianceParams>);
}

TEST(TestBatchnormFwdInferenceWithVarianceParams, ConstructGraphWithActivation)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceActivGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    EXPECT_NO_THROW(const BatchnormFwdInferenceWithVarianceParams params(
        attr, activAttrs, graph.getTensorMap()));
}

TEST(TestBatchnormFwdInferenceWithVarianceParams, HasCorrectTensorPointersForGraphWithActivation)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceActivGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    const BatchnormFwdInferenceWithVarianceParams params(attr, activAttrs, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_NE(params.estMean(), nullptr);
    EXPECT_NE(params.estVariance(), nullptr);
    EXPECT_NE(params.optActivation(), std::nullopt);
    EXPECT_NE(params.activationOut(), nullptr);
}

TEST(TestBatchnormFwdInferenceWithVarianceParams,
     TensorPointersMatchExpectedUidsForGraphWithActivation)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceActivGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    const BatchnormFwdInferenceWithVarianceParams params(attr, activAttrs, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attr.x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attr.y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attr.scale_tensor_uid());
    EXPECT_EQ(params.bias()->uid(), attr.bias_tensor_uid());
    EXPECT_EQ(params.estMean()->uid(), attr.mean_tensor_uid());
    EXPECT_EQ(params.estVariance()->uid(), attr.variance_tensor_uid());
    EXPECT_EQ(params.activationOut()->uid(), activAttrs.out_0_tensor_uid());
}

// ============================================================================
// buildBatchnormFwdInferenceWithVarianceKernel - helpers
// ============================================================================

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, BatchnormFwdInferenceWithVarianceParams>
    createParamsFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                          const std::vector<int64_t>& dims = {1, 3, 224, 224},
                          hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                          = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph(
        strides, dims, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();

    BatchnormFwdInferenceWithVarianceParams params(attr, graph.getTensorMap());
    return {std::move(builder), std::move(params)};
}

// The kernel build for @p params on a device of @p deviceProps with the default configuration.
BatchnormInferenceKernelBuild defaultBuild(const BatchnormFwdInferenceWithVarianceParams& params,
                                           const hipDeviceProp_t& deviceProps)
{
    return buildBatchnormFwdInferenceWithVarianceKernel(
        params,
        std::string(deviceProps.gcnArchName),
        deviceProps.multiProcessorCount,
        deviceProps.warpSize,
        defaultBatchnormFwdInferenceWithVarianceConfig(params));
}

} // namespace

// ============================================================================
// buildBatchnormFwdInferenceWithVarianceKernel - default configuration
// ============================================================================

TEST(TestBatchnormFwdInferenceWithVarianceKernel, BuildNamesTheEstEntryPoint)
{
    auto [fbb, params] = createParamsFromGraph();
    EXPECT_STREQ(defaultBuild(params, createTestDeviceProps()).entryPoint,
                 "BatchNormFwdInferSpatialEst");
}

TEST(TestBatchnormFwdInferenceWithVarianceKernel, BuildIncludesOffloadArchOption)
{
    auto [fbb, params] = createParamsFromGraph();
    const std::vector<std::string> options
        = defaultBuild(params, createTestDeviceProps("gfx942")).options;
    EXPECT_NE(std::find(options.begin(), options.end(), std::string("--offload-arch=gfx942")),
              options.end());
}

TEST(TestBatchnormFwdInferenceWithVarianceKernelFp32, BuildDefaultSetsCorrectDefines)
{
    auto [fbb, params] = createParamsFromGraph();
    const std::vector<std::string> capturedOptions
        = defaultBuild(params, createTestDeviceProps()).options;

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

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
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP0=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP1=256"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP2=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRPS0=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRPS1=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRPS2=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_BATCH_ELEMENTS=HIP_PLUGIN_BN_BATCHES"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAX_BATCH=768"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDS_SIZE=256"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VEC_SIZE=4"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAX_SPATIAL=2500"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NODPP=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDSGCN_SIZE=16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_USESAVED=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_STASH_METHOD=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VARIANT=255"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NRN_OP_ID=0"));
}

TEST(TestBatchnormFwdInferenceWithVarianceKernelFp32, BuildSetsCorrectDefines)
{
    auto [fbb, params] = createParamsFromGraph();
    const std::vector<std::string> capturedOptions
        = defaultBuild(params, createTestDeviceProps()).options;

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MEAN_VAR_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_BATCHES=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_CHANNELS=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SPATIAL_SIZE=50176"));
}

TEST(TestBatchnormFwdInferenceWithVarianceKernelFp16, BuildSetsCorrectDefines)
{
    auto [fbb, params]
        = createParamsFromGraph({150528, 50176, 224, 1},
                                {1, 3, 224, 224},
                                hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const std::vector<std::string> capturedOptions
        = defaultBuild(params, createTestDeviceProps()).options;

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INPUT_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_OUTPUT_TYPE=half"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MEAN_VAR_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FPMIX=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_BATCHES=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_CHANNELS=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SPATIAL_SIZE=50176"));
}

TEST(TestBatchnormFwdInferenceWithVarianceKernelBfp16, BuildSetsCorrectDefines)
{
    auto [fbb, params]
        = createParamsFromGraph({150528, 50176, 224, 1},
                                {1, 3, 224, 224},
                                hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    const std::vector<std::string> capturedOptions
        = defaultBuild(params, createTestDeviceProps()).options;

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_OUTPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MEAN_VAR_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFPMIX=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_BATCHES=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_CHANNELS=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SPATIAL_SIZE=50176"));
}

TEST(TestBatchnormFwdInferenceWithVarianceKernel, BuildNchwLayoutSetsCorrectDefine)
{
    // NCHW strides: N=C*H*W, C=H*W, H=W, W=1
    auto [fbb, params] = createParamsFromGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});
    const std::vector<std::string> capturedOptions
        = defaultBuild(params, createTestDeviceProps()).options;

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYOUT_CHANNEL_LAST=0"));
}

TEST(TestBatchnormFwdInferenceWithVarianceKernel, BuildNhwcLayoutSetsCorrectDefine)
{
    // NHWC strides: N=H*W*C, H=W*C, W=C, C=1
    auto [fbb, params] = createParamsFromGraph({150528, 1, 672, 3}, {1, 3, 224, 224});
    const std::vector<std::string> capturedOptions
        = defaultBuild(params, createTestDeviceProps()).options;

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_LAYOUT_CHANNEL_LAST=1"));
}

TEST(TestBatchnormFwdInferenceWithVarianceKernel, BuildWithUnsupportedDimensionThrows)
{
    // 2D tensor is not supported
    auto [fbb, params] = createParamsFromGraph(
        {4, 1}, {1, 4}, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    EXPECT_THROW(static_cast<void>(defaultBuild(params, createTestDeviceProps())),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ============================================================================
// buildBatchnormFwdInferenceWithVarianceKernel - explicit performance configuration
// ============================================================================

namespace
{

// An inference graph and the params of its batchnorm node, which point into the graph's buffer.
struct InferenceGraph
{
    flatbuffers::FlatBufferBuilder builder;
    BatchnormFwdInferenceWithVarianceParams params;
};

InferenceGraph createInferenceGraph(const std::vector<int64_t>& strides,
                                    const std::vector<int64_t>& dims)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph(
        strides, dims, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    BatchnormFwdInferenceWithVarianceParams params(
        *graph.getNode(0).attributes_as_BatchnormInferenceAttributesVarianceExt(),
        graph.getTensorMap());
    return {std::move(builder), std::move(params)};
}

bool hasBuildOption(const BatchnormInferenceKernelBuild& build, const std::string& option)
{
    const std::vector<std::string> options = build.options;
    return std::find(options.begin(), options.end(), option) != options.end();
}

} // namespace

TEST(TestBatchnormFwdInferenceWithVarianceKernelConfig, DefaultConfigIsTheLargestDividingVectorSize)
{
    // NCHW vectorizes the spatial size 50176, NHWC the 3 channels
    const auto nchw = createInferenceGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});
    EXPECT_EQ(defaultBatchnormFwdInferenceWithVarianceConfig(nchw.params),
              (BatchnormInferenceConfig{4}));
    const auto nhwc = createInferenceGraph({150528, 1, 672, 3}, {1, 3, 224, 224});
    EXPECT_EQ(defaultBatchnormFwdInferenceWithVarianceConfig(nhwc.params),
              (BatchnormInferenceConfig{1}));
}

TEST(TestBatchnormFwdInferenceWithVarianceKernelConfig, BuildTakesTheVectorSizeOfTheConfig)
{
    const auto graph = createInferenceGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});

    const auto vectorized
        = buildBatchnormFwdInferenceWithVarianceKernel(graph.params, "gfx942", 60, 64, {4});
    EXPECT_TRUE(hasBuildOption(vectorized, "-DHIP_PLUGIN_BN_VEC_SIZE=4"));
    EXPECT_EQ(vectorized.blockSize, (std::array<unsigned int, 3>{1, 256, 1}));
    EXPECT_EQ(vectorized.gridSize, (std::array<unsigned int, 3>{3, 49, 1}));

    const auto scalar
        = buildBatchnormFwdInferenceWithVarianceKernel(graph.params, "gfx942", 60, 64, {1});
    EXPECT_TRUE(hasBuildOption(scalar, "-DHIP_PLUGIN_BN_VEC_SIZE=1"));
    EXPECT_EQ(scalar.blockSize, (std::array<unsigned int, 3>{1, 256, 1}));
    EXPECT_EQ(scalar.gridSize, (std::array<unsigned int, 3>{3, 196, 1}));
}

TEST(TestBatchnormFwdInferenceWithVarianceKernelConfig, ApplicabilityNeedsADividingVectorSize)
{
    const auto nchw = createInferenceGraph({150528, 50176, 224, 1}, {1, 3, 224, 224});
    EXPECT_TRUE(isBatchnormFwdInferenceWithVarianceConfigApplicable(nchw.params, {4}));
    EXPECT_FALSE(isBatchnormFwdInferenceWithVarianceConfigApplicable(nchw.params, {8}));

    const auto nhwc = createInferenceGraph({150528, 1, 672, 3}, {1, 3, 224, 224});
    EXPECT_TRUE(isBatchnormFwdInferenceWithVarianceConfigApplicable(nhwc.params, {1}));
    EXPECT_FALSE(isBatchnormFwdInferenceWithVarianceConfigApplicable(nhwc.params, {2}));
}

} // namespace hip_kernel_provider::batchnorm::test
