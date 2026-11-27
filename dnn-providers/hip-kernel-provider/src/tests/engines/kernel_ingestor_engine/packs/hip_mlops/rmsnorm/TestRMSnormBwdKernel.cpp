// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/hip_mlops/rmsnorm/RMSnormBwdKernel.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/rmsnorm/RMSnormCommon.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::rmsnorm;

// ============================================================================
// RMSnormBwdParams - construction from valid graph data
// ============================================================================

TEST(TestRMSnormBwdParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    EXPECT_NO_THROW(const RMSnormBwdParams params(attr, graph.getTensorMap()));
}

TEST(TestRMSnormBwdParams, ConstructsFromDoubleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(0);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(1);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    EXPECT_NO_THROW(const RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap()));
}

TEST(TestRMSnormBwdParams, ConstructsFromUnaryIdentityGraph)
{
    auto original = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph();
    auto nativeGraph
        = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(original.GetBufferPointer());
    auto* activation = nativeGraph->nodes[0]->attributes.AsPointwiseAttributes();
    activation->operation = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY;
    activation->in_1_tensor_uid = flatbuffers::nullopt;
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::CreateGraph(builder, nativeGraph.get()));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const RMSnormBwdParams params(*graph.getNode(1).attributes_as_RMSNormBackwardAttributes(),
                                  *graph.getNode(0).attributes_as_PointwiseAttributes(),
                                  graph.getTensorMap());

    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_EQ(params.y(), params.x());
    EXPECT_EQ(params.dy()->uid(),
              graph.getNode(0).attributes_as_PointwiseAttributes()->in_0_tensor_uid());
}

TEST(TestRMSnormBwdParams, HasCorrectTensorPointersWithOptionalAttributesForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.invRMS(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_FALSE(params.optActivation().has_value()); // Not set by single node graph
    EXPECT_EQ(params.y(), nullptr); // Not set by single node graph
}

TEST(TestRMSnormBwdParams, HasCorrectTensorPointersWithOptionalAttributesForDoubleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(0);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(1);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.invRMS(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_NE(params.dbias(), nullptr);
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.y(), nullptr);
}

TEST(TestRMSnormBwdParams, TensorPointersMatchExpectedUidsForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attr, graph.getTensorMap());

    EXPECT_EQ(params.dy()->uid(), attr.dy_tensor_uid());
    EXPECT_EQ(params.x()->uid(), attr.x_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attr.scale_tensor_uid());
    EXPECT_EQ(params.invRMS()->uid(), attr.inv_rms_tensor_uid());
    EXPECT_EQ(params.dx()->uid(), attr.dx_tensor_uid());
    EXPECT_EQ(params.dscale()->uid(), attr.dscale_tensor_uid());
    EXPECT_EQ(params.dbias()->uid(), attr.dbias_tensor_uid().value());
}

TEST(TestRMSnormBwdParams, TensorPointersMatchExpectedUidsForDoubleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(0);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(1);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());

    EXPECT_EQ(params.dy()->uid(), attrActivation.in_0_tensor_uid());
    EXPECT_EQ(params.x()->uid(), attrBwd.x_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attrBwd.scale_tensor_uid());
    EXPECT_EQ(params.invRMS()->uid(), attrBwd.inv_rms_tensor_uid());
    EXPECT_EQ(params.dx()->uid(), attrBwd.dx_tensor_uid());
    EXPECT_EQ(params.dscale()->uid(), attrBwd.dscale_tensor_uid());
    EXPECT_EQ(params.dbias()->uid(), attrBwd.dbias_tensor_uid().value());
    EXPECT_EQ(params.y()->uid(), attrActivation.in_1_tensor_uid().value());
}

TEST(TestRMSnormBwdParams, OptionalTensorsAreNullWhenNotProvidedForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.invRMS(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_EQ(params.dbias(), nullptr);
    EXPECT_FALSE(params.optActivation().has_value());
    EXPECT_EQ(params.y(), nullptr);
}

TEST(TestRMSnormBwdParams, OptionalTensorsAreNullWhenNotProvidedForDoubleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(0);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(1);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.invRMS(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_NE(params.dscale(), nullptr);
    EXPECT_EQ(params.dbias(), nullptr);
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.y(), nullptr);
}

TEST(TestRMSnormBwdParams, IsMoveConstructibleForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    RMSnormBwdParams params(attr, graph.getTensorMap());
    const RMSnormBwdParams moved(std::move(params));

    EXPECT_NE(moved.dy(), nullptr);
    EXPECT_NE(moved.x(), nullptr);
}

TEST(TestRMSnormBwdParams, IsMoveConstructibleForDoubleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeActivation = graph.getNode(0);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();
    const auto& nodeBwd = graph.getNode(1);
    const auto& attrBwd = *nodeBwd.attributes_as_RMSNormBackwardAttributes();

    RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());
    const RMSnormBwdParams moved(std::move(params));

    EXPECT_NE(moved.dy(), nullptr);
    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestRMSnormBwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<RMSnormBwdParams>);
}

// ============================================================================
// buildRMSnormBwdKernel - helpers
// ============================================================================

namespace
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;

RMSnormKernelBuild buildFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                                  const std::vector<int64_t>& dims = {1, 3, 224, 224},
                                  bool hasOptionalAttributes = true,
                                  DataType inputDataType = DataType::FLOAT)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        strides, dims, hasOptionalAttributes, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& attr = *graph.getNode(0).attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attr, graph.getTensorMap());
    return buildRMSnormBwdKernel(params, "gfx942", RMSnormConfig{});
}

RMSnormKernelBuild
    buildFromFusedActivationGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                                  const std::vector<int64_t>& dims = {1, 3, 224, 224},
                                  bool hasOptionalAttributes = true,
                                  DataType inputDataType = DataType::FLOAT)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        strides, dims, hasOptionalAttributes, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& attrActivation = *graph.getNode(0).attributes_as_PointwiseAttributes();
    const auto& attrBwd = *graph.getNode(1).attributes_as_RMSNormBackwardAttributes();

    const RMSnormBwdParams params(attrBwd, attrActivation, graph.getTensorMap());
    return buildRMSnormBwdKernel(params, "gfx942", RMSnormConfig{});
}

std::vector<std::string> optionsOf(const RMSnormKernelBuild& build)
{
    return std::vector<std::string>(build.options);
}

bool hasOption(const std::vector<std::string>& options, const std::string& option)
{
    return std::find(options.begin(), options.end(), option) != options.end();
}

} // namespace

// ============================================================================
// buildRMSnormBwdKernel - launches and compile options
// ============================================================================

TEST(TestRMSnormBwdKernel, LaunchesTheDataThenTheScaleBiasEntryPointForSingleNodeGraph)
{
    const auto build = buildFromGraph();

    ASSERT_EQ(build.launches.size(), 2u);
    EXPECT_STREQ(build.launches[0].entryPoint, "RMSnormBwdData");
    EXPECT_EQ(build.launches[0].blockSize, 256u);
    EXPECT_EQ(build.launches[0].gridSize, 1u);
    EXPECT_STREQ(build.launches[1].entryPoint, "RMSnormBwdScaleBias");
    EXPECT_EQ(build.launches[1].blockSize, 256u);
    EXPECT_EQ(build.launches[1].gridSize, 588u);
}

TEST(TestRMSnormBwdKernel, LaunchesTheDataThenTheScaleBiasEntryPointForDoubleNodeGraph)
{
    const auto build = buildFromFusedActivationGraph();

    ASSERT_EQ(build.launches.size(), 2u);
    EXPECT_STREQ(build.launches[0].entryPoint, "RMSnormBwdData");
    EXPECT_EQ(build.launches[0].blockSize, 256u);
    EXPECT_EQ(build.launches[0].gridSize, 1u);
    EXPECT_STREQ(build.launches[1].entryPoint, "RMSnormBwdScaleBias");
    EXPECT_EQ(build.launches[1].blockSize, 256u);
    EXPECT_EQ(build.launches[1].gridSize, 588u);
}

TEST(TestRMSnormBwdKernel, LaunchShapesNameTheDataThenTheScaleBiasKernel)
{
    const auto shapes = rmsnormBwdLaunchShapes(3, 2, 1000, 256);

    ASSERT_EQ(shapes.size(), 2u);
    EXPECT_STREQ(shapes[0].kernelName, "RMSnormBwdData");
    EXPECT_EQ(shapes[0].blocks, 6);
    EXPECT_EQ(shapes[0].threadsPerBlock, 256);
    EXPECT_STREQ(shapes[1].kernelName, "RMSnormBwdScaleBias");
    EXPECT_EQ(shapes[1].blocks, 4);
    EXPECT_EQ(shapes[1].threadsPerBlock, 256);
}

TEST(TestRMSnormBwdKernelFp32, BuildSetsCorrectDefinesForSingleNodeGraph)
{
    const auto options = optionsOf(buildFromGraph());

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_X_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DY_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DX_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormBwdKernelFp32, BuildSetsCorrectDefinesForDoubleNodeGraph)
{
    const auto options = optionsOf(buildFromFusedActivationGraph());

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_X_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DY_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DX_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormBwdKernelFp16, BuildSetsCorrectDefinesForSingleNodeGraph)
{
    const auto options = optionsOf(
        buildFromGraph({150528, 50176, 224, 1}, {1, 3, 224, 224}, true, DataType::HALF));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_X_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DY_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DX_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormBwdKernelFp16, BuildSetsCorrectDefinesForDoubleNodeGraph)
{
    const auto options = optionsOf(buildFromFusedActivationGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true, DataType::HALF));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_X_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DY_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DX_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormBwdKernelBfp16, BuildSetsCorrectDefinesForSingleNodeGraph)
{
    const auto options = optionsOf(
        buildFromGraph({150528, 50176, 224, 1}, {1, 3, 224, 224}, true, DataType::BFLOAT16));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_X_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DY_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DX_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormBwdKernelBfp16, BuildSetsCorrectDefinesForDoubleNodeGraph)
{
    const auto options = optionsOf(buildFromFusedActivationGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, true, DataType::BFLOAT16));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_X_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DY_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DX_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormBwdKernel, BuildWithChannelLastInputCorrectlySetsDefinesForSingleNodeGraph)
{
    const auto options = optionsOf(buildFromGraph({150528, 1, 672, 3}));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_X_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DY_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DX_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormBwdKernel, BuildWithChannelLastInputCorrectlySetsDefinesForDoubleNodeGraph)
{
    const auto options = optionsOf(buildFromFusedActivationGraph({150528, 1, 672, 3}));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_X_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DY_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_DX_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormBwdKernel, BuildWithUnsupportedWorkgroupsThrowsForSingleNodeGraph)
{
    // Number of workgroups exeeds UINT32_MAX
    EXPECT_THROW(buildFromGraph({4, 4, 2, 1}, {UINT32_MAX, 1, 2, 2}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormBwdKernel, BuildWithUnsupportedWorkgroupsThrowsForDoubleNodeGraph)
{
    // Number of workgroups exeeds UINT32_MAX
    EXPECT_THROW(buildFromFusedActivationGraph({4, 4, 2, 1}, {UINT32_MAX, 1, 2, 2}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormBwdKernel, BuildWithBlockCountReachingThreadLimitThrows)
{
    // 2^24 workgroups of 256 threads is 2^32 threads
    EXPECT_THROW(buildFromGraph({2, 1, 1, 1}, {16777216, 2, 1, 1}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormBwdKernel, BuildWithScaleBiasBlockCountReachingThreadLimitThrows)
{
    // One row keeps the data kernel at one workgroup. The scale/bias kernel needs
    // 2^24 workgroups of 256 threads, which is 2^32 threads.
    EXPECT_THROW(buildFromGraph({4294967296, 256, 1, 1}, {1, 16777216, 256, 1}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormBwdKernel, BuildWithUnsupportedDimensionThrowsForSingleNodeGraph)
{
    // 3D tensor is not supported
    EXPECT_THROW(buildFromGraph({12, 4, 1}, {1, 3, 4}, false),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormBwdKernel, BuildWithUnsupportedDimensionThrowsForDoubleNodeGraph)
{
    // 3D tensor is not supported
    EXPECT_THROW(buildFromFusedActivationGraph({12, 4, 1}, {1, 3, 4}, false),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
