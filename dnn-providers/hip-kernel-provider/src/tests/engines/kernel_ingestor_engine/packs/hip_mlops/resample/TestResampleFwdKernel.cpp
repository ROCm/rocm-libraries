// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/hip_mlops/resample/ResampleFwdKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::resample;

namespace
{

flatbuffers::FlatBufferBuilder createCustomResampleFwdGraph(
    const std::vector<int64_t>& xDims,
    const std::vector<int64_t>& xStrides,
    const std::vector<int64_t>& yDims,
    const std::vector<int64_t>& yStrides,
    const std::vector<int64_t>& prePadding,
    const std::vector<int64_t>& postPadding,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& window,
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> indexDataType = std::nullopt,
    std::optional<bool> generateIndex = std::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &yStrides, &yDims));
    if(indexDataType.has_value())
    {
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 3, "index", *indexDataType, &yStrides, &yDims));
    }

    ::flatbuffers::Optional<bool> flatbufferGenerateIndex = ::flatbuffers::nullopt;
    if(generateIndex.has_value())
    {
        const bool generateIndexValue = generateIndex.value_or(false);
        flatbufferGenerateIndex = ::flatbuffers::Optional<bool>(generateIndexValue);
    }

    auto resampleAttr = hipdnn_flatbuffers_sdk::data_objects::CreateResampleFwdAttributesDirect(
        builder,
        1,
        2,
        indexDataType.has_value() ? ::flatbuffers::Optional<int64_t>(3) : ::flatbuffers::nullopt,
        &prePadding,
        &postPadding,
        &stride,
        &window,
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
        hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD,
        flatbufferGenerateIndex);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "resample_fwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleFwdAttributes,
        resampleAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

std::pair<flatbuffers::FlatBufferBuilder, ResampleFwdParams>
    createParamsFromGraph(std::optional<bool> generateIndex = std::nullopt)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(generateIndex);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    return {std::move(builder), std::move(params)};
}

std::pair<flatbuffers::FlatBufferBuilder, ResampleFwdParams>
    createParamsFromCustomGraph(flatbuffers::FlatBufferBuilder&& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    return {std::move(builder), std::move(params)};
}

// The compile options buildResampleFwdKernel() gives @p params on gfx942 at the default block.
std::vector<std::string> buildOptions(const ResampleFwdParams& params)
{
    return buildResampleFwdKernel(params, "gfx942", ResampleConfig{}).options;
}

bool hasOption(const std::vector<std::string>& options, const std::string& option)
{
    return std::find(options.begin(), options.end(), option) != options.end();
}

} // namespace

TEST(TestResampleFwdParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    EXPECT_NO_THROW(
        const ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type()));
}

TEST(TestResampleFwdParams, CapturesTensorPointersAndAttributes)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    const ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());

    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_EQ(params.index(), nullptr);
    EXPECT_EQ(params.prePadding(), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(params.postPadding(), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(params.stride(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(params.window(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(params.mode(), hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL);
    EXPECT_EQ(params.paddingMode(), hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD);
}

TEST(TestResampleFwdParams, CapturesOptionalIndexTensor)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    const ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());

    ASSERT_NE(params.index(), nullptr);
    EXPECT_EQ(params.index()->uid(), attr.index_tensor_uid().value());
    EXPECT_TRUE(params.generateIndex());
}

TEST(TestResampleFwdParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    ResampleFwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    const ResampleFwdParams moved(std::move(params));

    EXPECT_NE(moved.x(), nullptr);
    EXPECT_NE(moved.y(), nullptr);
}

TEST(TestResampleFwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<ResampleFwdParams>);
}

TEST(TestResampleFwdKernel, LaunchesOneResampleFwdKernelOverTheOutput)
{
    const auto shapes = resampleFwdLaunchShapes(1000, 256);
    ASSERT_EQ(shapes.size(), 1u);
    EXPECT_STREQ(shapes[0].kernelName, "ResampleFwd");
    EXPECT_EQ(shapes[0].blocks, 4);
    EXPECT_EQ(shapes[0].threadsPerBlock, 256);
}

TEST(TestResampleFwdKernel, BuildTakesTheBlockOfTheConfigAndCoversTheOutput)
{
    auto [fbb, params] = createParamsFromGraph();

    const auto build = buildResampleFwdKernel(params, "gfx942", ResampleConfig{});
    EXPECT_EQ(build.blockSize, 256u);
    EXPECT_EQ(build.gridSize, 1u);
    EXPECT_TRUE(hasOption(build.options, "--offload-arch=gfx942"));
}

TEST(TestResampleFwdKernel, BuildSetsExpectedDefines)
{
    auto [fbb, params] = createParamsFromGraph();
    const auto options = buildOptions(params);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_INPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_OUTPUT_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_SPATIAL_DIMS=2"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_MODE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_PADDING_MODE=2"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_OUTPUT_ELEMENT_COUNT=4"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_WINDOW_H=2"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_WINDOW_W=2"));
}

TEST(TestResampleFwdKernel, BuildSetsIndexDefines)
{
    auto [fbb, params] = createParamsFromGraph(true);
    const auto options = buildOptions(params);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_INDEX_TYPE=int32_t"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_HAS_INDEX=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_GENERATE_INDEX=1"));
}

TEST(TestResampleFwdKernel, BuildSetsChannelLastStrideDefines)
{
    auto [fbb, params] = createParamsFromCustomGraph(createCustomResampleFwdGraph(
        {1, 3, 4, 4}, {48, 1, 12, 3}, {1, 3, 2, 2}, {12, 1, 6, 3}, {0, 0}, {0, 0}, {2, 2}, {2, 2}));
    const auto options = buildOptions(params);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_X_STRIDE_N=48"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_X_STRIDE_C=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_X_STRIDE_H=12"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_X_STRIDE_W=3"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_Y_STRIDE_N=12"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_Y_STRIDE_C=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_Y_STRIDE_H=6"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_Y_STRIDE_W=3"));
}

TEST(TestResampleFwdKernel, BuildRejectsUnsupportedTensorDimensions)
{
    auto [fbb, params] = createParamsFromCustomGraph(createCustomResampleFwdGraph(
        {1, 1, 4}, {4, 4, 1}, {1, 1, 2}, {2, 2, 1}, {0}, {0}, {2}, {2}));

    EXPECT_THROW(buildOptions(params), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleFwdKernel, BuildRejectsUnsupportedWorkgroups)
{
    auto [fbb, params] = createParamsFromCustomGraph(
        createCustomResampleFwdGraph({1, 1, 1048576, 1048576},
                                     {1099511627776, 1099511627776, 1048576, 1},
                                     {1, 1, 1048576, 1048576},
                                     {1099511627776, 1099511627776, 1048576, 1},
                                     {0, 0},
                                     {0, 0},
                                     {1, 1},
                                     {1, 1}));

    EXPECT_THROW(buildOptions(params), hipdnn_plugin_sdk::HipdnnPluginException);
}

// 2^32 output elements is 2^24 blocks: below UINT32_MAX blocks, but 2^32 threads.
TEST(TestResampleFwdKernel, BuildRejectsBlockCountReachingThreadLimit)
{
    auto [fbb, params] = createParamsFromCustomGraph(
        createCustomResampleFwdGraph({1, 1, 65536, 65536},
                                     {4294967296, 4294967296, 65536, 1},
                                     {1, 1, 65536, 65536},
                                     {4294967296, 4294967296, 65536, 1},
                                     {0, 0},
                                     {0, 0},
                                     {1, 1},
                                     {1, 1}));

    EXPECT_THROW(buildOptions(params), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleFwdKernel, BuildRejectsUnsupportedIndexDataType)
{
    auto [fbb, params] = createParamsFromCustomGraph(
        createCustomResampleFwdGraph({1, 1, 4, 4},
                                     {16, 16, 4, 1},
                                     {1, 1, 2, 2},
                                     {4, 4, 2, 1},
                                     {0, 0},
                                     {0, 0},
                                     {2, 2},
                                     {2, 2},
                                     hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                                     true));

    EXPECT_THROW(buildOptions(params), hipdnn_plugin_sdk::HipdnnPluginException);
}
