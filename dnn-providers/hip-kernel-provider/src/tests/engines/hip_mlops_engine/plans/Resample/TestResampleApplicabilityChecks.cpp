// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/resample/ResampleApplicabilityChecks.hpp"

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include <optional>
#include <vector>

using namespace hip_kernel_provider::resample;

namespace
{
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

struct ResampleGraphConfig
{
    std::vector<int64_t> xDims{1, 1, 4, 4};
    std::vector<int64_t> xStrides{16, 16, 4, 1};
    std::vector<int64_t> yDims{1, 1, 2, 2};
    std::vector<int64_t> yStrides{4, 4, 2, 1};
    std::vector<int64_t> indexDims{1, 1, 2, 2};
    std::vector<int64_t> indexStrides{4, 4, 2, 1};
    std::vector<int64_t> prePadding{0, 0};
    std::vector<int64_t> postPadding{0, 0};
    std::vector<int64_t> stride{2, 2};
    std::vector<int64_t> window{2, 2};
    data_objects::DataType xType{data_objects::DataType::FLOAT};
    data_objects::DataType yType{data_objects::DataType::FLOAT};
    data_objects::DataType indexType{data_objects::DataType::INT32};
    data_objects::ResampleMode mode{data_objects::ResampleMode::MAXPOOL};
    data_objects::PaddingMode paddingMode{data_objects::PaddingMode::ZERO_PAD};
    bool includeIndexTensor{false};
    bool setIndexUid{false};
    std::optional<bool> generateIndex{std::nullopt};
};

struct ResampleBwdGraphConfig
{
    std::vector<int64_t> dyDims{1, 1, 2, 2};
    std::vector<int64_t> dyStrides{4, 4, 2, 1};
    std::vector<int64_t> dxDims{1, 1, 4, 4};
    std::vector<int64_t> dxStrides{16, 16, 4, 1};
    std::vector<int64_t> indexDims{1, 1, 2, 2};
    std::vector<int64_t> indexStrides{4, 4, 2, 1};
    std::vector<int64_t> prePadding{0, 0};
    std::vector<int64_t> postPadding{0, 0};
    std::vector<int64_t> stride{2, 2};
    std::vector<int64_t> window{2, 2};
    data_objects::DataType dyType{data_objects::DataType::FLOAT};
    data_objects::DataType dxType{data_objects::DataType::FLOAT};
    data_objects::DataType indexType{data_objects::DataType::INT32};
    data_objects::ResampleMode mode{data_objects::ResampleMode::MAXPOOL};
    data_objects::PaddingMode paddingMode{data_objects::PaddingMode::ZERO_PAD};
    bool includeIndexTensor{false};
    bool setIndexUid{false};
};

flatbuffers::FlatBufferBuilder createResampleFwdGraph(const ResampleGraphConfig& config)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<data_objects::TensorAttributes>> tensorAttributes;

    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", config.xType, &config.xStrides, &config.xDims));
    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", config.yType, &config.yStrides, &config.yDims));
    if(config.includeIndexTensor)
    {
        tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
            builder, 3, "index", config.indexType, &config.indexStrides, &config.indexDims));
    }

    ::flatbuffers::Optional<bool> flatbufferGenerateIndex = ::flatbuffers::nullopt;
    if(config.generateIndex.has_value())
    {
        const bool generateIndex = config.generateIndex.value_or(false);
        flatbufferGenerateIndex = ::flatbuffers::Optional<bool>(generateIndex);
    }

    auto resampleAttr = data_objects::CreateResampleFwdAttributesDirect(
        builder,
        1,
        2,
        config.setIndexUid ? ::flatbuffers::Optional<int64_t>(3) : ::flatbuffers::nullopt,
        &config.prePadding,
        &config.postPadding,
        &config.stride,
        &config.window,
        config.mode,
        config.paddingMode,
        flatbufferGenerateIndex);

    std::vector<::flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "resample_fwd",
                                       data_objects::DataType::FLOAT,
                                       data_objects::NodeAttributes::ResampleFwdAttributes,
                                       resampleAttr.Union()));

    auto graphOffset = data_objects::CreateGraphDirect(builder,
                                                       "test",
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::DataType::FLOAT,
                                                       &tensorAttributes,
                                                       &nodes);
    builder.Finish(graphOffset);
    return builder;
}

void validateResampleGraph(flatbuffers::FlatBufferBuilder& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleFwdAttributes();

    ResampleValidator validator(graph.getTensorMap());
    validator.checkTensorConfigSupported(attr);
}

flatbuffers::FlatBufferBuilder createResampleBwdGraph(const ResampleBwdGraphConfig& config)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<data_objects::TensorAttributes>> tensorAttributes;

    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, 1, "dy", config.dyType, &config.dyStrides, &config.dyDims));

    tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
        builder, 2, "dx", config.dxType, &config.dxStrides, &config.dxDims));

    if(config.includeIndexTensor)
    {
        tensorAttributes.push_back(data_objects::CreateTensorAttributesDirect(
            builder, 3, "index", config.indexType, &config.indexStrides, &config.indexDims));
    }

    auto resampleAttr = data_objects::CreateResampleBwdAttributesDirect(
        builder,
        1,
        2,
        config.setIndexUid ? ::flatbuffers::Optional<int64_t>(3) : ::flatbuffers::nullopt,
        &config.prePadding,
        &config.postPadding,
        &config.stride,
        &config.window,
        config.mode,
        config.paddingMode);

    std::vector<::flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "resample_bwd",
                                       data_objects::DataType::FLOAT,
                                       data_objects::NodeAttributes::ResampleBwdAttributes,
                                       resampleAttr.Union()));

    auto graphOffset = data_objects::CreateGraphDirect(builder,
                                                       "test",
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::DataType::FLOAT,
                                                       &tensorAttributes,
                                                       &nodes);
    builder.Finish(graphOffset);
    return builder;
}

void validateResampleBwdGraph(flatbuffers::FlatBufferBuilder& builder)
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_ResampleBwdAttributes();

    ResampleValidator validator(graph.getTensorMap());
    validator.checkBwdTensorConfigSupported(attr);
}

} // namespace

// ============================================================================
// Forward Resample - applicability checks
// ============================================================================

TEST(TestResampleValidator, Valid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph();

    EXPECT_NO_THROW(validateResampleGraph(builder));
}

TEST(TestResampleValidator, ValidWithIndex)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(true);

    EXPECT_NO_THROW(validateResampleGraph(builder));
}

TEST(TestResampleValidator, UnsupportedDim)
{
    ResampleGraphConfig config;
    config.xDims = {1, 1, 4};
    config.xStrides = {4, 4, 1};
    config.yDims = {1, 1, 2};
    config.yStrides = {2, 2, 1};
    config.prePadding = {0};
    config.postPadding = {0};
    config.stride = {2};
    config.window = {2};
    auto builder = createResampleFwdGraph(config);

    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, MismatchIOTypes)
{
    ResampleGraphConfig config;
    config.yType = data_objects::DataType::HALF;
    auto builder = createResampleFwdGraph(config);

    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, InvalidOutputShape)
{
    ResampleGraphConfig config;
    config.yDims = {1, 1, 3, 3};
    config.yStrides = {9, 9, 3, 1};
    auto builder = createResampleFwdGraph(config);

    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, GenerateIndexRequiresIndexTensor)
{
    ResampleGraphConfig config;
    config.generateIndex = true;
    auto builder = createResampleFwdGraph(config);

    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, IndexRequiresMaxPoolMode)
{
    ResampleGraphConfig config;
    config.includeIndexTensor = true;
    config.setIndexUid = true;
    config.mode = data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING;
    auto builder = createResampleFwdGraph(config);

    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, UnsupportedIndexType)
{
    ResampleGraphConfig config;
    config.includeIndexTensor = true;
    config.setIndexUid = true;
    config.indexType = data_objects::DataType::INT64;
    auto builder = createResampleFwdGraph(config);

    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, InvalidIndexShape)
{
    ResampleGraphConfig config;
    config.includeIndexTensor = true;
    config.setIndexUid = true;
    config.indexDims = {1, 1, 1, 2};
    config.indexStrides = {2, 2, 2, 1};
    auto builder = createResampleFwdGraph(config);

    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, MismatchedIOLayouts)
{
    ResampleGraphConfig config;
    config.xStrides = {16, 16, 4, 1}; // NCHW
    config.yStrides = {4, 1, 2, 1}; // NHWC

    auto builder = createResampleFwdGraph(config);
    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BatchSizeMismatch)
{
    ResampleGraphConfig config;
    config.xDims = {2, 1, 4, 4}; // 2 batches
    config.xStrides = {16, 16, 4, 1};
    config.yDims = {1, 1, 2, 2}; // 1 batch
    config.yStrides = {4, 4, 2, 1};

    auto builder = createResampleFwdGraph(config);
    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, ChannelCountMismatch)
{
    ResampleGraphConfig config;
    config.xDims = {1, 3, 4, 4}; // 3 channels
    config.xStrides = {48, 16, 4, 1};
    config.yDims = {1, 1, 2, 2}; // 1 channel
    config.yStrides = {4, 4, 2, 1};

    auto builder = createResampleFwdGraph(config);
    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, SpatialParameterRankMismatch)
{
    ResampleGraphConfig config;
    config.prePadding = {0, 0, 0}; // 3D padding for 2D spatial input

    auto builder = createResampleFwdGraph(config);
    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, UnsupportedIODataType)
{
    ResampleGraphConfig config;
    config.xType = data_objects::DataType::INT8;
    config.yType = data_objects::DataType::INT8;

    auto builder = createResampleFwdGraph(config);
    EXPECT_THROW(validateResampleGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

// ===========================================================================
// Backward Resample - applicability checks
// ===========================================================================

TEST(TestResampleValidator, BwdValid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph();

    EXPECT_NO_THROW(validateResampleBwdGraph(builder));
}

TEST(TestResampleValidator, BwdValidWithoutIndex)
{
    auto builder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(false);

    EXPECT_NO_THROW(validateResampleBwdGraph(builder));
}

TEST(TestResampleValidator, BwdUnsupportedDim)
{
    ResampleBwdGraphConfig config;
    config.dyDims = {1, 1, 2};
    config.dyStrides = {2, 2, 1};
    config.dxDims = {1, 1, 4};
    config.dxStrides = {4, 4, 1};
    config.prePadding = {0};
    config.postPadding = {0};
    config.stride = {2};
    config.window = {2};
    auto builder = createResampleBwdGraph(config);

    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BwdMismatchIOTypes)
{
    ResampleBwdGraphConfig config;
    config.dxType = data_objects::DataType::HALF;
    auto builder = createResampleBwdGraph(config);

    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BwdInvalidOutputShape)
{
    ResampleBwdGraphConfig config;
    config.dxDims = {1, 1, 3, 3};
    config.dxStrides = {9, 9, 3, 1};
    auto builder = createResampleBwdGraph(config);

    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BwdIndexRequiresMaxPoolMode)
{
    ResampleBwdGraphConfig config;
    config.includeIndexTensor = true;
    config.setIndexUid = true;
    config.mode = data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING;
    auto builder = createResampleBwdGraph(config);

    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BwdUnsupportedIndexType)
{
    ResampleBwdGraphConfig config;
    config.includeIndexTensor = true;
    config.setIndexUid = true;
    config.indexType = data_objects::DataType::INT64;
    auto builder = createResampleBwdGraph(config);

    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BwdInvalidIndexShape)
{
    ResampleBwdGraphConfig config;
    config.includeIndexTensor = true;
    config.setIndexUid = true;
    config.indexDims = {1, 1, 1, 2};
    config.indexStrides = {2, 2, 2, 1};
    auto builder = createResampleBwdGraph(config);

    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BwdMismatchedIOLayouts)
{
    ResampleBwdGraphConfig config;
    config.dyStrides = {4, 4, 2, 1}; // NCHW
    config.dxStrides = {16, 1, 4, 1}; // NHWC

    auto builder = createResampleBwdGraph(config);
    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BwdBatchSizeMismatch)
{
    ResampleBwdGraphConfig config;
    config.dyDims = {1, 1, 2, 2};
    config.dxDims = {2, 1, 4, 4}; // 2 batches instead of 1
    config.dxStrides = {16, 16, 4, 1};
    auto builder = createResampleBwdGraph(config);

    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BwdChannelCountMismatch)
{
    ResampleBwdGraphConfig config;
    config.dyDims = {1, 1, 2, 2};
    config.dxDims = {1, 3, 4, 4}; // 3 channels instead of 1
    config.dxStrides = {48, 16, 4, 1};
    auto builder = createResampleBwdGraph(config);

    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BwdSpatialParameterRankMismatch)
{
    ResampleBwdGraphConfig config;
    config.prePadding = {0, 0, 0}; // 3D padding for 2D input
    auto builder = createResampleBwdGraph(config);

    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestResampleValidator, BwdUnsupportedIODataType)
{
    ResampleBwdGraphConfig config;
    config.dyType = data_objects::DataType::INT8;
    config.dxType = data_objects::DataType::INT8;
    auto builder = createResampleBwdGraph(config);

    EXPECT_THROW(validateResampleBwdGraph(builder), hipdnn_plugin_sdk::HipdnnPluginException);
}
