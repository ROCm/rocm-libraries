// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/hip_mlops/rmsnorm/RMSnormCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/rmsnorm/RMSnormFwdKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::rmsnorm;

// ============================================================================
// RMSnormFwdParams - construction from valid graph data
// ============================================================================
TEST(TestRMSnormFwdParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    EXPECT_NO_THROW(const RMSnormFwdParams params(attr, graph.getTensorMap()));
}
TEST(TestRMSnormFwdParams, ConstructsFromDoubleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeFwd = graph.getNode(0);
    const auto& attrFwd = *nodeFwd.attributes_as_RMSNormAttributes();
    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();

    EXPECT_NO_THROW(const RMSnormFwdParams params(attrFwd, attrActivation, graph.getTensorMap()));
}

TEST(TestRMSnormFwdParams, HasCorrectTensorPointersForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    const RMSnormFwdParams params(attr, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_EQ(params.bias(), nullptr); // Optional and not set in `createValidRMSNormGraph`
    EXPECT_EQ(params.invRMS(), nullptr); // Optional and not set in `createValidRMSNormGraph`
    EXPECT_FALSE(params.optActivation().has_value()); // Not set by single node graph
    EXPECT_EQ(params.activationOut(), nullptr); // Not set by single node graph
}

TEST(TestRMSnormFwdParams, HasCorrectTensorPointersForDoubleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeFwd = graph.getNode(0);
    const auto& attrFwd = *nodeFwd.attributes_as_RMSNormAttributes();
    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();

    const RMSnormFwdParams params(attrFwd, attrActivation, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_EQ(params.bias(), nullptr); // Optional and not set in `createValidRMSNormGraph`
    EXPECT_EQ(params.invRMS(), nullptr); // Optional and not set in `createValidRMSNormGraph`
    EXPECT_TRUE(params.optActivation().has_value());
    EXPECT_NE(params.activationOut(), nullptr);
}

TEST(TestRMSnormFwdParams, TensorPointersMatchExpectedUidsForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    const RMSnormFwdParams params(attr, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attr.x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attr.y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attr.scale_tensor_uid());
}

TEST(TestRMSnormFwdParams, TensorPointersMatchExpectedUidsForDoubleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeFwd = graph.getNode(0);
    const auto& attrFwd = *nodeFwd.attributes_as_RMSNormAttributes();
    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();

    const RMSnormFwdParams params(attrFwd, attrActivation, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attrFwd.x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attrFwd.y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attrFwd.scale_tensor_uid());
    EXPECT_EQ(params.activationOut()->uid(), attrActivation.out_0_tensor_uid());
}

TEST(TestRMSnormFwdParams, IsMoveConstructibleForSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    RMSnormFwdParams params(attr, graph.getTensorMap());
    const RMSnormFwdParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestRMSnormFwdParams, IsMoveConstructibleForDoubleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& nodeFwd = graph.getNode(0);
    const auto& attrFwd = *nodeFwd.attributes_as_RMSNormAttributes();
    const auto& nodeActivation = graph.getNode(1);
    const auto& attrActivation = *nodeActivation.attributes_as_PointwiseAttributes();

    RMSnormFwdParams params(attrFwd, attrActivation, graph.getTensorMap());
    const RMSnormFwdParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
    EXPECT_NE(moved.activationOut(), nullptr);
}

TEST(TestRMSnormFwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<RMSnormFwdParams>);
}

TEST(TestRMSnormProblemDescription, ComputesExecutionGeometry)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attr = *graph.getNode(0).attributes_as_RMSNormAttributes();
    const RMSnormFwdParams params(attr, graph.getTensorMap());

    const ProblemDescription problem(params.x(), params.scale(), Direction::FORWARD);

    EXPECT_EQ(problem.direction(), Direction::FORWARD);
    EXPECT_EQ(problem.normalizeDim(), 1);
    EXPECT_EQ(problem.outerSize(), 1);
    EXPECT_EQ(problem.innerSize(), 150528);
    EXPECT_EQ(problem.stride(), 1);
}
// ============================================================================
// buildRMSnormFwdKernel - helpers
// ============================================================================

namespace
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;

RMSnormKernelBuild buildFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                                  const std::vector<int64_t>& dims = {1, 3, 224, 224},
                                  DataType inputDataType = DataType::FLOAT,
                                  const std::string& gcnArchName = "gfx942")
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidRMSNormGraph(strides, dims, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& attr = *graph.getNode(0).attributes_as_RMSNormAttributes();

    const RMSnormFwdParams params(attr, graph.getTensorMap());
    return buildRMSnormFwdKernel(params, gcnArchName, RMSnormConfig{});
}

RMSnormKernelBuild
    buildFromFusedActivationGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                                  const std::vector<int64_t>& dims = {1, 3, 224, 224},
                                  DataType inputDataType = DataType::FLOAT,
                                  const std::string& gcnArchName = "gfx942")
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph(
        strides, dims, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& attrFwd = *graph.getNode(0).attributes_as_RMSNormAttributes();
    const auto& attrActivation = *graph.getNode(1).attributes_as_PointwiseAttributes();

    const RMSnormFwdParams params(attrFwd, attrActivation, graph.getTensorMap());
    return buildRMSnormFwdKernel(params, gcnArchName, RMSnormConfig{});
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
// buildRMSnormFwdKernel - launches and compile options
// ============================================================================

TEST(TestRMSnormFwdKernel, LaunchesTheRMSnormFwdEntryPointForSingleNodeGraph)
{
    const auto build = buildFromGraph();

    ASSERT_EQ(build.launches.size(), 1u);
    EXPECT_STREQ(build.launches[0].entryPoint, "RMSnormFwd");
    EXPECT_EQ(build.launches[0].blockSize, 256u);
    EXPECT_EQ(build.launches[0].gridSize, 1u);
}

TEST(TestRMSnormFwdKernel, LaunchesTheRMSnormFwdEntryPointForDoubleNodeGraph)
{
    const auto build = buildFromFusedActivationGraph();

    ASSERT_EQ(build.launches.size(), 1u);
    EXPECT_STREQ(build.launches[0].entryPoint, "RMSnormFwd");
    EXPECT_EQ(build.launches[0].blockSize, 256u);
    EXPECT_EQ(build.launches[0].gridSize, 1u);
}

TEST(TestRMSnormFwdKernel, LaunchShapesNameTheRMSnormFwdKernel)
{
    const auto shapes = rmsnormFwdLaunchShapes(3, 2, 256);

    ASSERT_EQ(shapes.size(), 1u);
    EXPECT_STREQ(shapes[0].kernelName, "RMSnormFwd");
    EXPECT_EQ(shapes[0].blocks, 6);
    EXPECT_EQ(shapes[0].threadsPerBlock, 256);
}

TEST(TestRMSnormFwdKernel, BuildIncludesOffloadArchOptionForSingleNodeGraph)
{
    const auto options = optionsOf(buildFromGraph());

    EXPECT_TRUE(hasOption(options, "--offload-arch=gfx942"));
}

TEST(TestRMSnormFwdKernel, BuildIncludesOffloadArchOptionForDoubleNodeGraph)
{
    const auto options = optionsOf(buildFromFusedActivationGraph());

    EXPECT_TRUE(hasOption(options, "--offload-arch=gfx942"));
}

TEST(TestRMSnormFwdKernelFp32, BuildSetsCorrectDefinesForSingleNodeGraph)
{
    const auto options = optionsOf(buildFromGraph());

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormFwdKernelFp32, BuildSetsCorrectDefinesForDoubleNodeGraph)
{
    const auto options = optionsOf(buildFromFusedActivationGraph());

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormFwdKernelFp16, BuildSetsCorrectDefinesForSingleNodeGraph)
{
    const auto options
        = optionsOf(buildFromGraph({150528, 50176, 224, 1}, {1, 3, 224, 224}, DataType::HALF));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormFwdKernelFp16, BuildSetsCorrectDefinesForDoubleNodeGraph)
{
    const auto options = optionsOf(
        buildFromFusedActivationGraph({150528, 50176, 224, 1}, {1, 3, 224, 224}, DataType::HALF));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormFwdKernelBfp16, BuildSetsCorrectDefinesForSingleNodeGraph)
{
    const auto options
        = optionsOf(buildFromGraph({150528, 50176, 224, 1}, {1, 3, 224, 224}, DataType::BFLOAT16));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormFwdKernelBfp16, BuildSetsCorrectDefinesForDoubleNodeGraph)
{
    const auto options = optionsOf(buildFromFusedActivationGraph(
        {150528, 50176, 224, 1}, {1, 3, 224, 224}, DataType::BFLOAT16));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}

TEST(TestRMSnormFwdKernel, BuildWithUnsupportedDimensionThrowsForSingleNodeGraph)
{
    // 3D tensor is not supported
    EXPECT_THROW(buildFromGraph({12, 4, 1}, {1, 3, 4}, DataType::FLOAT),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormFwdKernel, BuildWithUnsupportedDimensionThrowsForDoubleNodeGraph)
{
    // 3D tensor is not supported
    EXPECT_THROW(buildFromFusedActivationGraph({12, 4, 1}, {1, 3, 4}, DataType::FLOAT),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormFwdKernel, BuildWithUnsupportedWorkgroupsThrowsForSingleNodeGraph)
{
    // Number of workgroups exeeds UINT32_MAX
    EXPECT_THROW(buildFromGraph({4, 4, 2, 1}, {UINT32_MAX, 1, 2, 2}, DataType::FLOAT),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormFwdKernel, BuildWithUnsupportedWorkgroupsThrowsForDoubleNodeGraph)
{
    // Number of workgroups exeeds UINT32_MAX
    EXPECT_THROW(
        buildFromFusedActivationGraph({4, 4, 2, 1}, {UINT32_MAX, 1, 2, 2}, DataType::FLOAT),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormFwdKernel, BuildWithBlockCountReachingThreadLimitThrows)
{
    // 2^24 workgroups of 256 threads is 2^32 threads
    EXPECT_THROW(buildFromGraph({2, 1, 1, 1}, {16777216, 2, 1, 1}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormFwdKernel, BuildWithChannelLastInputCorrectlySetsDefinesForSingleNodeGraph)
{
    const auto options = optionsOf(buildFromGraph({150528, 1, 672, 3}, {1, 3, 224, 224}));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=0"));
}

TEST(TestRMSnormFwdKernel, BuildWithChannelLastInputCorrectlySetsDefinesForDoubleNodeGraph)
{
    const auto options
        = optionsOf(buildFromFusedActivationGraph({150528, 1, 672, 3}, {1, 3, 224, 224}));

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_STRIDE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_SCALE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RMSNORM_NRN_OP_ID=10"));
}
