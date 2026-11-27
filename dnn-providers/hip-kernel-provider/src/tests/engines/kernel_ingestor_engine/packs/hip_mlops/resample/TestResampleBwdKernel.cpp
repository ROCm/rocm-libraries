// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/hip_mlops/resample/ResampleBwdKernel.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::resample;

namespace
{

flatbuffers::FlatBufferBuilder createCustomResampleBwdGraph(
    const std::vector<int64_t>& dyDims,
    const std::vector<int64_t>& dyStrides,
    const std::vector<int64_t>& dxDims,
    const std::vector<int64_t>& dxStrides,
    const std::vector<int64_t>& prePadding,
    const std::vector<int64_t>& postPadding,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& window,
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> indexDataType = std::nullopt)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "dy",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dyStrides,
        &dyDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "dx",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dxStrides,
        &dxDims));
    if(indexDataType.has_value())
    {
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 3, "index", *indexDataType, &dyStrides, &dyDims));
    }

    auto resampleAttr = hipdnn_flatbuffers_sdk::data_objects::CreateResampleBwdAttributesDirect(
        builder,
        1,
        2,
        indexDataType.has_value() ? ::flatbuffers::Optional<int64_t>(3) : ::flatbuffers::nullopt,
        &prePadding,
        &postPadding,
        &stride,
        &window,
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
        hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "resample_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleBwdAttributes,
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

std::pair<flatbuffers::FlatBufferBuilder, ResampleBwdParams> createParamsFromGraph()
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    return {std::move(builder), std::move(params)};
}

std::pair<flatbuffers::FlatBufferBuilder, ResampleBwdParams>
    createParamsFromCustomGraph(flatbuffers::FlatBufferBuilder&& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    return {std::move(builder), std::move(params)};
}

// The compile options buildResampleBwdKernel() gives @p params on gfx942 at the default block.
std::vector<std::string> buildOptions(const ResampleBwdParams& params)
{
    return buildResampleBwdKernel(params, "gfx942", ResampleConfig{}).options;
}

bool hasOption(const std::vector<std::string>& options, const std::string& option)
{
    return std::find(options.begin(), options.end(), option) != options.end();
}

} // namespace

// ============================================================================
// ResampleBwdParams - construction from valid graph data
// ============================================================================

TEST(TestResampleBwdParams, ConstructsFromSingleNodeGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    EXPECT_NO_THROW(
        const ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type()));
}

TEST(TestResampleBwdParams, HasCorrectTensorPointers)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    const ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());

    EXPECT_NE(params.dy(), nullptr);
    EXPECT_NE(params.dx(), nullptr);
    EXPECT_EQ(params.index(), nullptr);
    EXPECT_EQ(params.prePadding(), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(params.postPadding(), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(params.stride(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(params.window(), std::vector<int64_t>({2, 2}));
    EXPECT_EQ(params.resampleMode(), hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL);
    EXPECT_EQ(params.paddingMode(), hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD);
}

TEST(TestResampleBwdParams, TensorPointersMatchExpectedUids)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    const ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());

    EXPECT_EQ(params.dy()->uid(), attr.dy_tensor_uid());
    EXPECT_EQ(params.dx()->uid(), attr.dx_tensor_uid());
}

TEST(TestResampleBwdParams, HasOptionalIndexTensor)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    const ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());

    ASSERT_NE(params.index(), nullptr);
    EXPECT_EQ(params.index()->uid(), attr.index_tensor_uid().value());
}

TEST(TestResampleBwdParams, IsMoveConstructible)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    ResampleBwdParams params(attr, graph.getTensorMap(), node.compute_data_type());
    const ResampleBwdParams moved(std::move(params));

    EXPECT_NE(moved.dy(), nullptr);
    EXPECT_NE(moved.dx(), nullptr);
}

TEST(TestResampleBwdParams, IsNotCopyConstructible)
{
    EXPECT_FALSE(std::is_copy_constructible_v<ResampleBwdParams>);
}

// ============================================================================
// buildResampleBwdKernel
// ============================================================================

TEST(TestResampleBwdKernel, LaunchesOneResampleBwdKernelOverDx)
{
    const auto shapes = resampleBwdLaunchShapes(1000, 256);
    ASSERT_EQ(shapes.size(), 1u);
    EXPECT_STREQ(shapes[0].kernelName, "ResampleBwd");
    EXPECT_EQ(shapes[0].blocks, 4);
    EXPECT_EQ(shapes[0].threadsPerBlock, 256);
}

TEST(TestResampleBwdKernel, BuildTakesTheBlockOfTheConfigAndCoversDx)
{
    auto [fbb, params] = createParamsFromGraph();

    const auto build = buildResampleBwdKernel(params, "gfx942", ResampleConfig{});
    EXPECT_EQ(build.blockSize, 256u);
    EXPECT_EQ(build.gridSize, 1u);
    EXPECT_TRUE(hasOption(build.options, "--offload-arch=gfx942"));
}

TEST(TestResampleBwdKernel, BuildSetsExpectedDefines)
{
    auto [fbb, params] = createParamsFromGraph();
    const auto options = buildOptions(params);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DY_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DX_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_COMPUTE_TYPE=float"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_SPATIAL_DIMS=2"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_MODE=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DX_ELEMENT_COUNT=16"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_WINDOW_H=2"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_WINDOW_W=2"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_INDEX_TYPE=int32_t"));
}

TEST(TestResampleBwdKernel, BuildSetsChannelLastStrideDefines)
{
    auto [fbb, params] = createParamsFromCustomGraph(
        createCustomResampleBwdGraph({1, 3, 2, 2},
                                     {12, 1, 6, 3},
                                     {1, 3, 4, 4},
                                     {48, 1, 12, 3},
                                     {0, 0},
                                     {0, 0},
                                     {2, 2},
                                     {2, 2},
                                     hipdnn_flatbuffers_sdk::data_objects::DataType::INT32));
    const auto options = buildOptions(params);

    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DX_STRIDE_N=48"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DX_STRIDE_C=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DX_STRIDE_H=12"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DX_STRIDE_W=3"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DY_STRIDE_N=12"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DY_STRIDE_C=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DY_STRIDE_H=6"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_DY_STRIDE_W=3"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_INDEX_STRIDE_N=12"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_INDEX_STRIDE_C=1"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_INDEX_STRIDE_H=6"));
    EXPECT_TRUE(hasOption(options, "-DHIP_PLUGIN_RESAMPLE_INDEX_STRIDE_W=3"));
}

TEST(TestResampleBwdKernel, BuildRejectsUnsupportedTensorDimensions)
{
    auto [fbb, params] = createParamsFromCustomGraph(createCustomResampleBwdGraph(
        {1, 1, 2}, {2, 2, 1}, {1, 1, 4}, {4, 4, 1}, {0}, {0}, {2}, {2}));

    EXPECT_THROW(buildOptions(params), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleBwdKernel, BuildRejectsUnsupportedWorkgroups)
{
    auto [fbb, params] = createParamsFromCustomGraph(
        createCustomResampleBwdGraph({1, 1, 1048576, 1048576},
                                     {1099511627776, 1099511627776, 1048576, 1},
                                     {1, 1, 1048576, 1048576},
                                     {1099511627776, 1099511627776, 1048576, 1},
                                     {0, 0},
                                     {0, 0},
                                     {1, 1},
                                     {1, 1}));

    EXPECT_THROW(buildOptions(params), hipdnn_plugin_sdk::HipdnnPluginException);
}

// 2^32 dx elements is 2^24 blocks: below UINT32_MAX blocks, but 2^32 threads.
TEST(TestResampleBwdKernel, BuildRejectsBlockCountReachingThreadLimit)
{
    auto [fbb, params] = createParamsFromCustomGraph(
        createCustomResampleBwdGraph({1, 1, 65536, 65536},
                                     {4294967296, 4294967296, 65536, 1},
                                     {1, 1, 65536, 65536},
                                     {4294967296, 4294967296, 65536, 1},
                                     {0, 0},
                                     {0, 0},
                                     {1, 1},
                                     {1, 1}));

    EXPECT_THROW(buildOptions(params), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleBwdKernel, BuildRejectsUnsupportedIndexDataType)
{
    auto [fbb, params] = createParamsFromCustomGraph(
        createCustomResampleBwdGraph({1, 1, 2, 2},
                                     {4, 4, 2, 1},
                                     {1, 1, 4, 4},
                                     {16, 16, 4, 1},
                                     {0, 0},
                                     {0, 0},
                                     {2, 2},
                                     {2, 2},
                                     hipdnn_flatbuffers_sdk::data_objects::DataType::INT64));

    EXPECT_THROW(buildOptions(params), hipdnn_plugin_sdk::HipdnnPluginException);
}
