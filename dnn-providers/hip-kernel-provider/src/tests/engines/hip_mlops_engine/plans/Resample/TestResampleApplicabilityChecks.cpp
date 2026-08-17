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

} // namespace

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
