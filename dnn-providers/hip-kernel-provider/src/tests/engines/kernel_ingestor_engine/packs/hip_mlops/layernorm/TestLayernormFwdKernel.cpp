// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormFwdKernel.hpp"
#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormUtilities.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include <algorithm>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::layernorm;

TEST(TestLayernormFwdParams, InitializesAllTensorsFromValidGraph)
{
    // Create a valid layernorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the layernorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormAttributes();
    ASSERT_NE(attrs, nullptr);

    // Expect that params construction doesn't throw
    EXPECT_NO_THROW(LayernormFwdParams(*attrs, graph.getTensorMap()));
}

TEST(TestLayernormFwdParams, HasCorrectTensorPointersForSingleNode)
{
    // Create a valid layernorm graph
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    // Get the layernorm node and attributes
    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormAttributes();
    ASSERT_NE(attrs, nullptr);

    const LayernormFwdParams params(*attrs, graph.getTensorMap());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_EQ(params.mean(), nullptr);
    EXPECT_NEAR(params.epsilonValue(nullptr, 0), 1e-5, 1e-10);
    EXPECT_EQ(params.invVariance(), nullptr);
}

TEST(TestLayernormFwdParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    auto* attrs = node.attributes_as_LayernormAttributes();
    ASSERT_NE(attrs, nullptr);

    const LayernormFwdParams params(*attrs, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), attrs->x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), attrs->y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), attrs->scale_tensor_uid());
    EXPECT_EQ(params.bias()->uid(), attrs->bias_tensor_uid());
    EXPECT_NEAR(params.epsilonValue(nullptr, 0), 1e-5, 1e-10);
}

TEST(TestLayernormFwdParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormFwdParams params(attr, graph.getTensorMap());
    const LayernormFwdParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestLayernormFwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<LayernormFwdParams>);
}

TEST(TestLayernormProblemDescription, ComputesExecutionGeometry)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attr = *graph.getNode(0).attributes_as_LayernormAttributes();
    const LayernormFwdParams params(attr, graph.getTensorMap());

    const ProblemDescription problem(params.x(), params.scale(), params.mean(), Direction::FORWARD);

    EXPECT_EQ(problem.direction(), Direction::FORWARD);
    EXPECT_EQ(problem.normalizedDim(), 1);
    EXPECT_EQ(problem.outerSize(), 1);
    EXPECT_EQ(problem.innerSize(), 588);
    EXPECT_EQ(problem.stride(), 1);
}

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, LayernormFwdParams>
    createParamsFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                          const std::vector<int64_t>& dims = {1, 3, 224, 224},
                          hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                          = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph(strides, dims, inputDataType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormFwdParams params(attr, graph.getTensorMap());
    return {std::move(builder), std::move(params)};
}

/// The compile options buildLayernormFwdKernel() bakes for @p params on gfx942, with the
/// default configuration.
std::vector<std::string> defaultBuildOptions(const LayernormFwdParams& params)
{
    return buildLayernormFwdKernel(params, "gfx942", defaultLayernormFwdConfig()).options;
}

bool hasOption(const std::vector<std::string>& options, const std::string& option)
{
    return std::find(options.begin(), options.end(), option) != options.end();
}

} // namespace

TEST(TestLayernormFwdKernel, BuildLaunchesTheLayernormFwdEntryPoint)
{
    auto [fbb, params] = createParamsFromGraph();

    const auto build = buildLayernormFwdKernel(params, "gfx942", defaultLayernormFwdConfig());
    ASSERT_EQ(build.launches.size(), 1u);
    EXPECT_EQ(std::string(build.launches[0].entryPoint), "LayernormFwd");
    EXPECT_EQ(build.launches[0].blockSize, 1024u);
    EXPECT_EQ(build.launches[0].gridSize, 1u);

    const auto shapes = layernormFwdLaunchShapes(params, defaultLayernormFwdConfig());
    ASSERT_EQ(shapes.size(), 1u);
    EXPECT_EQ(std::string(shapes[0].kernelName), "LayernormFwd");
    EXPECT_EQ(shapes[0].blocks, 1);
    EXPECT_EQ(shapes[0].threadsPerBlock, 1024);
}

TEST(TestLayernormFwdKernel, BuildIncludesOffloadArchOption)
{
    auto [fbb, params] = createParamsFromGraph();

    EXPECT_TRUE(hasOption(defaultBuildOptions(params), "--offload-arch=gfx942"));
}

TEST(TestLayernormFwdKernelFp32, BuildSetsCorrectDefines)
{
    auto [fbb, params] = createParamsFromGraph();
    const auto options = defaultBuildOptions(params);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
}

TEST(TestLayernormFwdKernelFp16, BuildSetsCorrectDefines)
{
    auto [fbb, params]
        = createParamsFromGraph({150528, 50176, 224, 1},
                                {1, 3, 224, 224},
                                hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const auto options = defaultBuildOptions(params);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=half"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
}

TEST(TestLayernormFwdKernelBfp16, BuildSetsCorrectDefines)
{
    auto [fbb, params]
        = createParamsFromGraph({150528, 50176, 224, 1},
                                {1, 3, 224, 224},
                                hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16);
    const auto options = defaultBuildOptions(params);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTPUT_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE=__bf16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_INNER_SIZE=150528"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_OUTER_SIZE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_LOCAL_SIZE=1024"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_LAYERNORM_STRIDE=1"));
}

TEST(TestLayernormFwdKernel, BuildWithUnsupportedDimensionThrows)
{
    // 3D tensor is not supported
    auto [fbb, params] = createParamsFromGraph(
        {12, 4, 1}, {1, 3, 4}, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    EXPECT_THROW(buildLayernormFwdKernel(params, "gfx942", defaultLayernormFwdConfig()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormFwdKernel, BuildWithThreadCountAtLimitThrows)
{
    // 4194304 blocks of 1024 threads reach 2^32 threads
    auto [fbb, params] = createParamsFromGraph({4, 1, 1, 1}, {4194304, 4, 1, 1});

    EXPECT_THROW(buildLayernormFwdKernel(params, "gfx942", defaultLayernormFwdConfig()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
