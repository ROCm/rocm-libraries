// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/layernorm/LayernormApplicabilityChecks.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider::layernorm;

TEST(TestLayernormValidator, ValidFprop)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
}

TEST(TestLayernormValidator, UnsupportedDim)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph({12, 4, 1}, {1, 3, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

namespace
{

flatbuffers::FlatBufferBuilder createInvalidTypeLayernormFpropGraph(
    hipdnn_flatbuffers_sdk::data_objects::DataType xType,
    hipdnn_flatbuffers_sdk::data_objects::DataType yType,
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleType,
    hipdnn_flatbuffers_sdk::data_objects::DataType biasType,
    hipdnn_flatbuffers_sdk::data_objects::DataType epsilonType,
    hipdnn_flatbuffers_sdk::data_objects::DataType meanType,
    hipdnn_flatbuffers_sdk::data_objects::DataType invVarianceType)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> ioStrides = {588, 196, 14, 1};
    const std::vector<int64_t> ioDims = {1, 3, 14, 14};

    const std::vector<int64_t> affineDims(ioDims.begin() + 1, ioDims.end());
    const std::vector<int64_t> affineStrides
        = hipdnn_data_sdk::utilities::generateStrides(affineDims);

    const std::vector<int64_t> statsDims = {ioDims[0], 1, 1, 1};
    const std::vector<int64_t> statsStrides
        = hipdnn_data_sdk::utilities::generateStrides(statsDims);

    // Required tensors
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", xType, &ioStrides, &ioDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", yType, &ioStrides, &ioDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", scaleType, &affineStrides, &affineDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "bias", biasType, &affineStrides, &affineDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> epsilonDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "epsilon",
        epsilonType,
        &epsilonDims,
        &epsilonDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "mean", meanType, &statsDims, &statsDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "inv_variance", invVarianceType, &statsDims, &statsDims));

    auto layernormAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateLayernormAttributes(
        builder,
        1, // x tensor uid
        3, // scale tensor uid
        4, // bias tensor uid
        5, // epsilon tensor uid
        2, // y tensor uid
        0, // normalized dim count
        6, // mean tensor uid
        7 // inv variance tensor uid
    );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "layernorm",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes,
        layernormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}
} // anonymous namespace

TEST(TestLayernormValidator, MismatchIOTypes)
{
    auto builder = createInvalidTypeLayernormFpropGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormValidator, InvalidEpsilonType)
{
    auto builder = createInvalidTypeLayernormFpropGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

namespace
{

flatbuffers::FlatBufferBuilder
    createInvalidShapeLayernormActivGraph(const std::vector<int64_t>& xDims,
                                          const std::vector<int64_t>& xStrides,
                                          const std::vector<int64_t>& yDims,
                                          const std::vector<int64_t>& yStrides,
                                          const std::vector<int64_t>& scaleDims,
                                          const std::vector<int64_t>& scaleStrides,
                                          const std::vector<int64_t>& biasDims,
                                          const std::vector<int64_t>& biasStrides,
                                          const std::vector<int64_t>& epsilonDims,
                                          const std::vector<int64_t>& epsilonStrides,
                                          const std::vector<int64_t>& meanDims,
                                          const std::vector<int64_t>& meanStrides,
                                          const std::vector<int64_t>& invVarianceDims,
                                          const std::vector<int64_t>& invVarianceStrides)
{

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // Required tensors
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &yStrides, &yDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &scaleStrides,
        &scaleDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &biasStrides,
        &biasDims));

    // Epsilon (pass-by-value)
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "epsilon",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &epsilonStrides,
        &epsilonDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "mean",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &meanStrides,
        &meanDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "inv_variance",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &invVarianceStrides,
        &invVarianceDims));

    auto layernormAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateLayernormAttributes(
        builder,
        1, // x tensor uid
        3, // scale tensor uid
        4, // bias tensor uid
        5, // epsilon tensor uid
        2, // y tensor uid
        0, // normalized dim count
        6, // mean tensor uid
        7 // inv variance tensor uid
    );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "layernorm",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes,
        layernormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}
} // anonymous namespace

TEST(TestLayernormValidator, MismatchIOShapes)
{

    const std::vector<int64_t> xDims{1, 3, 224, 224};
    const std::vector<int64_t> xStrides{150528, 50176, 224, 1};

    const std::vector<int64_t> ioDims{1, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    // For LayerNorm, scale and bias match the normalized dimensions (all dims except batch).
    // E.g., for input [N, C, H, W], normalized dims are [C, H, W].
    const std::vector<int64_t> affineDims(ioDims.begin() + 1, ioDims.end());
    const std::vector<int64_t> affineStrides
        = hipdnn_data_sdk::utilities::generateStrides(affineDims);

    const std::vector<int64_t> epsilonDims = {1};

    const std::vector<int64_t> statsDims = {1, 1, 1, 1};
    const std::vector<int64_t> statsStrides
        = hipdnn_data_sdk::utilities::generateStrides(statsDims);

    // NOLINTBEGIN(readability-suspicious-call-argument)
    auto builder = createInvalidShapeLayernormActivGraph(xDims,
                                                         xStrides,
                                                         ioDims,
                                                         ioStrides,
                                                         affineDims,
                                                         affineStrides,
                                                         affineDims,
                                                         affineStrides,
                                                         epsilonDims,
                                                         epsilonDims,
                                                         statsDims,
                                                         statsStrides,
                                                         statsDims,
                                                         statsStrides);
    // NOLINTEND(readability-suspicious-call-argument)
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormValidator, MismatchAffineShapes)
{
    const std::vector<int64_t> ioDims{1, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> affineDims(ioDims.begin() + 1, ioDims.end());
    const std::vector<int64_t> affineStrides
        = hipdnn_data_sdk::utilities::generateStrides(affineDims);

    const std::vector<int64_t> biasDims{1, 3, 224, 224};
    const std::vector<int64_t> biasStrides{150528, 50176, 224, 1};

    const std::vector<int64_t> epsilonDims = {1};

    const std::vector<int64_t> statsDims = {1, 1, 1, 1};
    const std::vector<int64_t> statsStrides
        = hipdnn_data_sdk::utilities::generateStrides(statsDims);

    // NOLINTBEGIN(readability-suspicious-call-argument)
    auto builder = createInvalidShapeLayernormActivGraph(ioDims,
                                                         ioStrides,
                                                         ioDims,
                                                         ioStrides,
                                                         affineDims,
                                                         affineStrides,
                                                         biasDims,
                                                         biasStrides,
                                                         epsilonDims,
                                                         epsilonDims,
                                                         statsDims,
                                                         statsStrides,
                                                         statsDims,
                                                         statsStrides);
    // NOLINTEND(readability-suspicious-call-argument)
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormValidator, MismatchStatsShapes)
{
    const std::vector<int64_t> ioDims{1, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> affineDims(ioDims.begin() + 1, ioDims.end());
    const std::vector<int64_t> affineStrides
        = hipdnn_data_sdk::utilities::generateStrides(affineDims);

    const std::vector<int64_t> epsilonDims = {1};

    const std::vector<int64_t> statsDims = {1, 1, 1, 1};
    const std::vector<int64_t> statsStrides
        = hipdnn_data_sdk::utilities::generateStrides(statsDims);

    const std::vector<int64_t> invVarDims{1, 3, 224, 224};
    const std::vector<int64_t> invVarStrides{150528, 50176, 224, 1};

    // NOLINTBEGIN(readability-suspicious-call-argument)
    auto builder = createInvalidShapeLayernormActivGraph(ioDims,
                                                         ioStrides,
                                                         ioDims,
                                                         ioStrides,
                                                         affineDims,
                                                         affineStrides,
                                                         affineDims,
                                                         affineStrides,
                                                         epsilonDims,
                                                         epsilonDims,
                                                         statsDims,
                                                         statsStrides,
                                                         invVarDims,
                                                         invVarStrides);
    // NOLINTEND(readability-suspicious-call-argument)
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormValidator, InvalidNormalization)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> affineDims(ioDims.begin() + 1, ioDims.end());
    const std::vector<int64_t> affineStrides
        = hipdnn_data_sdk::utilities::generateStrides(affineDims);
    const std::vector<int64_t> epsilonDims = {1};

    const std::vector<int64_t> statsDims = {2, 1, 1, 1};
    const std::vector<int64_t> statsStrides
        = hipdnn_data_sdk::utilities::generateStrides(statsDims);

    // NOLINTBEGIN(readability-suspicious-call-argument)
    auto builder = createInvalidShapeLayernormActivGraph(ioDims,
                                                         ioStrides,
                                                         ioDims,
                                                         ioStrides,
                                                         affineDims,
                                                         affineStrides,
                                                         affineDims,
                                                         affineStrides,
                                                         epsilonDims,
                                                         epsilonDims,
                                                         statsDims,
                                                         statsStrides,
                                                         statsDims,
                                                         statsStrides);
    // NOLINTEND(readability-suspicious-call-argument)
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/*
TEST(TestBatchnormValidator, MismatchAffineShapes)
{
    std::vector<int64_t> dims{1, 3, 224, 224};
    std::vector<int64_t> strides{150528, 50176, 224, 1};

    std::vector<int64_t> scaleDims{1, 3, 4, 4};
    std::vector<int64_t> scaleStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    auto builder = createInvalidShapeBatchnormActivGraph(dims,
                                                         strides,
                                                         scaleDims,
                                                         scaleStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         dims,
                                                         strides,
                                                         dims,
                                                         strides);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    // Data type of io tensors should match, expect exception when this isn't the case
    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormValidator, MismatchStatShapes)
{
    std::vector<int64_t> meanDims{1, 3, 4, 4};
    std::vector<int64_t> meanStrides{48, 16, 4, 1};

    std::vector<int64_t> dims{1, 3, 224, 224};
    std::vector<int64_t> strides{150528, 50176, 224, 1};

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    auto builder = createInvalidShapeBatchnormActivGraph(dims,
                                                         strides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         meanDims,
                                                         meanStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         dims,
                                                         strides,
                                                         dims,
                                                         strides);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    // Data type of io tensors should match, expect exception when this isn't the case
    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}*/
