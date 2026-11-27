// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/hip_mlops/layernorm/LayernormApplicabilityChecks.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <string>
#include <unordered_set>

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

TEST(TestLayernormValidator, ValidBprop)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
}

TEST(TestLayernormValidator, UnsupportedDimFprop)
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

TEST(TestLayernormValidator, UnsupportedDimBprop)
{
    auto builder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph({12, 4, 1}, {1, 3, 4});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

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

inline flatbuffers::FlatBufferBuilder createInvalidTypeLayernormBwdGraph(
    hipdnn_flatbuffers_sdk::data_objects::DataType dyDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dxDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dscaleDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dbiasDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType epsilonDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType meanDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType invVarianceDataType)
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
        builder, 1, "dy", dyDataType, &ioStrides, &ioDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "x", xDataType, &ioStrides, &ioDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "dx", dxDataType, &ioStrides, &ioDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "scale", scaleDataType, &affineStrides, &affineDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "dscale", dscaleDataType, &affineStrides, &affineDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 8, "dbias", dbiasDataType, &affineStrides, &affineDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "mean", meanDataType, &statsStrides, &statsDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "inv_variance", invVarianceDataType, &statsStrides, &statsDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> epsilonDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        9,
        "epsilon",
        epsilonDataType,
        &epsilonDims,
        &epsilonDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto layernormBackwardAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateLayernormBackwardAttributes(
            builder,
            1, // dy tensor uid
            2, // x tensor uid
            4, // scale tensor uid
            flatbuffers::Optional<int64_t>(5), // mean tensor uid
            flatbuffers::Optional<int64_t>(6), // rstd tensor uid
            flatbuffers::Optional<int64_t>(9), // epsilon tensor uid
            3, // dx tensor uid
            7, // dscale tensor uid
            8, // dbias tensor uid
            3 // normalizedDimCount
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "layernorm_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormBackwardAttributes,
        layernormBackwardAttributes.Union());
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

TEST(TestLayernormValidator, ValidDifferentIOTypesFprop)
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
    EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
}

TEST(TestLayernormValidator, ValidDifferentIOTypesBprop)
{
    auto builder
        = createInvalidTypeLayernormBwdGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
}

TEST(TestLayernormValidator, MismatchAffineTypesFprop)
{
    auto builder = createInvalidTypeLayernormFpropGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
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

TEST(TestLayernormValidator, MismatchAffineTypesBprop)
{
    auto builder
        = createInvalidTypeLayernormBwdGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormValidator, MismatchStatsTypesFprop)
{
    auto builder = createInvalidTypeLayernormFpropGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormValidator, MismatchStatsTypesBprop)
{
    auto builder
        = createInvalidTypeLayernormBwdGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormValidator, InvalidEpsilonTypeFprop)
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

TEST(TestLayernormValidator, InvalidEpsilonTypeBprop)
{
    auto builder
        = createInvalidTypeLayernormBwdGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
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
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

namespace
{

// An operand named in passByValue carries a host value in place of a device buffer.
flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>
    createFloatOperand(flatbuffers::FlatBufferBuilder& builder,
                       int64_t uid,
                       const char* name,
                       const std::vector<int64_t>& strides,
                       const std::vector<int64_t>& dims,
                       const std::unordered_set<std::string>& passByValue)
{
    if(passByValue.count(name) == 0)
    {
        return hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            uid,
            name,
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            &strides,
            &dims);
    }
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value value(1.0f);
    return hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        uid,
        name,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(value).Union());
}

flatbuffers::FlatBufferBuilder
    createInvalidShapeLayernormFpropGraph(const std::vector<int64_t>& xDims,
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
                                          const std::vector<int64_t>& invVarianceStrides,
                                          const std::unordered_set<std::string>& passByValue = {})
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // Required tensors
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &yStrides, &yDims));

    tensorAttributes.push_back(
        createFloatOperand(builder, 3, "scale", scaleStrides, scaleDims, passByValue));

    tensorAttributes.push_back(
        createFloatOperand(builder, 4, "bias", biasStrides, biasDims, passByValue));

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

    tensorAttributes.push_back(
        createFloatOperand(builder, 6, "mean", meanStrides, meanDims, passByValue));

    tensorAttributes.push_back(createFloatOperand(
        builder, 7, "inv_variance", invVarianceStrides, invVarianceDims, passByValue));

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

flatbuffers::FlatBufferBuilder
    createInvalidShapeLayernormBwdGraph(const std::vector<int64_t>& dyDims,
                                        const std::vector<int64_t>& dyStrides,
                                        const std::vector<int64_t>& xDims,
                                        const std::vector<int64_t>& xStrides,
                                        const std::vector<int64_t>& dxDims,
                                        const std::vector<int64_t>& dxStrides,
                                        const std::vector<int64_t>& scaleDims,
                                        const std::vector<int64_t>& scaleStrides,
                                        const std::vector<int64_t>& dscaleDims,
                                        const std::vector<int64_t>& dscaleStrides,
                                        const std::vector<int64_t>& dbiasDims,
                                        const std::vector<int64_t>& dbiasStrides,
                                        const std::vector<int64_t>& epsilonDims,
                                        const std::vector<int64_t>& epsilonStrides,
                                        const std::vector<int64_t>& meanDims,
                                        const std::vector<int64_t>& meanStrides,
                                        const std::vector<int64_t>& invVarianceDims,
                                        const std::vector<int64_t>& invVarianceStrides,
                                        const std::unordered_set<std::string>& passByValue = {})
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // Required tensors
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "dy",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dyStrides,
        &dyDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "dx",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dxStrides,
        &dxDims));

    tensorAttributes.push_back(
        createFloatOperand(builder, 4, "scale", scaleStrides, scaleDims, passByValue));

    tensorAttributes.push_back(
        createFloatOperand(builder, 7, "dscale", dscaleStrides, dscaleDims, passByValue));

    tensorAttributes.push_back(
        createFloatOperand(builder, 8, "dbias", dbiasStrides, dbiasDims, passByValue));

    tensorAttributes.push_back(
        createFloatOperand(builder, 5, "mean", meanStrides, meanDims, passByValue));

    tensorAttributes.push_back(createFloatOperand(
        builder, 6, "inv_variance", invVarianceStrides, invVarianceDims, passByValue));

    // Epsilon (pass-by-value)
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        9,
        "epsilon",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &epsilonStrides,
        &epsilonDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto layernormBackwardAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateLayernormBackwardAttributes(
            builder,
            1, // dy tensor uid
            2, // x tensor uid
            4, // scale tensor uid
            flatbuffers::Optional<int64_t>(5), // mean tensor uid
            flatbuffers::Optional<int64_t>(6), // rstd tensor uid
            flatbuffers::Optional<int64_t>(9), // epsilon tensor uid
            3, // dx tensor uid
            7, // dscale tensor uid
            8, // dbias tensor uid
            3 // normalizedDimCount
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "layernorm_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormBackwardAttributes,
        layernormBackwardAttributes.Union());
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

TEST(TestLayernormValidator, MismatchIOShapesFprop)
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
    auto builder = createInvalidShapeLayernormFpropGraph(xDims,
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

TEST(TestLayernormValidator, MismatchIOShapesBprop)
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
    auto builder = createInvalidShapeLayernormBwdGraph(ioDims,
                                                       ioStrides,
                                                       xDims,
                                                       xStrides,
                                                       ioDims,
                                                       ioStrides,
                                                       affineDims,
                                                       affineStrides,
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
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormValidator, MismatchAffineShapesFprop)
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
    auto builder = createInvalidShapeLayernormFpropGraph(ioDims,
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

TEST(TestLayernormValidator, MismatchAffineShapesBprop)
{
    const std::vector<int64_t> ioDims{1, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> affineDims(ioDims.begin() + 1, ioDims.end());
    const std::vector<int64_t> affineStrides
        = hipdnn_data_sdk::utilities::generateStrides(affineDims);

    const std::vector<int64_t> dbiasDims{1, 3, 224, 224};
    const std::vector<int64_t> dbiasStrides{150528, 50176, 224, 1};

    const std::vector<int64_t> epsilonDims = {1};

    const std::vector<int64_t> statsDims = {1, 1, 1, 1};
    const std::vector<int64_t> statsStrides
        = hipdnn_data_sdk::utilities::generateStrides(statsDims);

    // NOLINTBEGIN(readability-suspicious-call-argument)
    auto builder = createInvalidShapeLayernormBwdGraph(ioDims,
                                                       ioStrides,
                                                       ioDims,
                                                       ioStrides,
                                                       ioDims,
                                                       ioStrides,
                                                       affineDims,
                                                       affineStrides,
                                                       affineDims,
                                                       affineStrides,
                                                       dbiasDims,
                                                       dbiasStrides,
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
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLayernormValidator, MismatchStatsShapesFprop)
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
    auto builder = createInvalidShapeLayernormFpropGraph(ioDims,
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

TEST(TestLayernormValidator, MismatchStatsShapesBprop)
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
    auto builder = createInvalidShapeLayernormBwdGraph(ioDims,
                                                       ioStrides,
                                                       ioDims,
                                                       ioStrides,
                                                       ioDims,
                                                       ioStrides,
                                                       affineDims,
                                                       affineStrides,
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
    const auto& attr = *node.attributes_as_LayernormBackwardAttributes();

    LayernormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

namespace
{

using DT = hipdnn_flatbuffers_sdk::data_objects::DataType;

std::vector<int64_t> nchwStrides()
{
    return {128, 16, 4, 1};
}
std::vector<int64_t> nhwcStrides()
{
    return {128, 1, 32, 8};
}

// Strides are per tensor so a test can mutate exactly one tensor. Fprop ignores dscale and dbias.
struct LayoutCase
{
    std::vector<int64_t> xDims = {2, 8, 4, 4};
    std::vector<int64_t> xStrides = nchwStrides();
    std::vector<int64_t> affineDims = {1, 8, 4, 4};
    std::vector<int64_t> scaleStrides = nchwStrides();
    std::vector<int64_t> biasStrides = nchwStrides();
    std::vector<int64_t> dscaleStrides = nchwStrides();
    std::vector<int64_t> dbiasStrides = nchwStrides();
    std::vector<int64_t> statDims = {2, 1, 1, 1};
    std::vector<int64_t> meanStrides = {1, 1, 1, 1};
    std::vector<int64_t> invVarianceStrides = {1, 1, 1, 1};
    // Operands, by name, that carry a host value in place of a device buffer.
    std::unordered_set<std::string> passByValue;
};

LayoutCase makeNhwcCase()
{
    LayoutCase layout;
    layout.xStrides = nhwcStrides();
    layout.scaleStrides = nhwcStrides();
    layout.biasStrides = nhwcStrides();
    layout.dscaleStrides = nhwcStrides();
    layout.dbiasStrides = nhwcStrides();
    return layout;
}

void expectFprop(const LayoutCase& layout, bool declined)
{
    const std::vector<int64_t> epsilonDims = {1};
    auto builder = createInvalidShapeLayernormFpropGraph(layout.xDims,
                                                         layout.xStrides,
                                                         layout.xDims,
                                                         layout.xStrides,
                                                         layout.affineDims,
                                                         layout.scaleStrides,
                                                         layout.affineDims,
                                                         layout.biasStrides,
                                                         epsilonDims,
                                                         epsilonDims,
                                                         layout.statDims,
                                                         layout.meanStrides,
                                                         layout.statDims,
                                                         layout.invVarianceStrides,
                                                         layout.passByValue);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attr = *graph.getNode(0).attributes_as_LayernormAttributes();

    LayernormValidator validator(graph.getTensorMap());
    if(declined)
    {
        EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
    else
    {
        EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
    }
}

void expectBprop(const LayoutCase& layout, bool declined)
{
    const std::vector<int64_t> epsilonDims = {1};
    auto builder = createInvalidShapeLayernormBwdGraph(layout.xDims,
                                                       layout.xStrides,
                                                       layout.xDims,
                                                       layout.xStrides,
                                                       layout.xDims,
                                                       layout.xStrides,
                                                       layout.affineDims,
                                                       layout.scaleStrides,
                                                       layout.affineDims,
                                                       layout.dscaleStrides,
                                                       layout.affineDims,
                                                       layout.dbiasStrides,
                                                       epsilonDims,
                                                       epsilonDims,
                                                       layout.statDims,
                                                       layout.meanStrides,
                                                       layout.statDims,
                                                       layout.invVarianceStrides,
                                                       layout.passByValue);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attr = *graph.getNode(0).attributes_as_LayernormBackwardAttributes();

    LayernormValidator validator(graph.getTensorMap());
    if(declined)
    {
        EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
    else
    {
        EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
    }
}

void expectBpropDxType(DT xType, DT dxType, bool declined)
{
    auto builder = createInvalidTypeLayernormBwdGraph(
        xType, xType, dxType, xType, xType, xType, DT::FLOAT, xType, xType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& attr = *graph.getNode(0).attributes_as_LayernormBackwardAttributes();

    LayernormValidator validator(graph.getTensorMap());
    if(declined)
    {
        EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                     hipdnn_plugin_sdk::HipdnnPluginException);
    }
    else
    {
        EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
    }
}

} // anonymous namespace

TEST(TestLayernormValidator, MatchingAffineLayoutAcceptedFprop)
{
    expectFprop(LayoutCase{}, false);
    expectFprop(makeNhwcCase(), false);
}

TEST(TestLayernormValidator, AffineLayoutDifferentFromXDeclinedFprop)
{
    {
        SCOPED_TRACE("NCHW x, NHWC scale");
        LayoutCase layout;
        layout.scaleStrides = nhwcStrides();
        expectFprop(layout, true);
    }
    {
        SCOPED_TRACE("NCHW x, NHWC bias");
        LayoutCase layout;
        layout.biasStrides = nhwcStrides();
        expectFprop(layout, true);
    }
    {
        SCOPED_TRACE("NHWC x, NCHW scale");
        auto layout = makeNhwcCase();
        layout.scaleStrides = nchwStrides();
        expectFprop(layout, true);
    }
    {
        SCOPED_TRACE("NHWC x, NCHW bias");
        auto layout = makeNhwcCase();
        layout.biasStrides = nchwStrides();
        expectFprop(layout, true);
    }
}

TEST(TestLayernormValidator, LowerRankAffineAlignsToTrailingDimsOfXFprop)
{
    LayoutCase layout;
    layout.affineDims = {8, 4, 4};
    layout.scaleStrides = {16, 4, 1};
    layout.biasStrides = {16, 4, 1};
    expectFprop(layout, false);

    // Row-major [C,H,W] against an NHWC x reads C with the wrong stride.
    layout.xStrides = nhwcStrides();
    expectFprop(layout, true);
}

TEST(TestLayernormValidator, HigherRankAffineDeclinedFprop)
{
    // Declined before the normalized-dim checks, which index x's dims by the operand's rank.
    LayoutCase layout;
    layout.affineDims = {1, 1, 8, 4, 4};
    layout.scaleStrides = {128, 128, 16, 4, 1};
    layout.biasStrides = {128, 128, 16, 4, 1};
    expectFprop(layout, true);
}

TEST(TestLayernormValidator, MatchingAffineLayoutAcceptedBprop)
{
    expectBprop(LayoutCase{}, false);
    expectBprop(makeNhwcCase(), false);
}

TEST(TestLayernormValidator, AffineLayoutDifferentFromXDeclinedBprop)
{
    {
        SCOPED_TRACE("NHWC x, row-major scale");
        auto layout = makeNhwcCase();
        layout.scaleStrides = nchwStrides();
        expectBprop(layout, true);
    }
    {
        SCOPED_TRACE("NHWC x, row-major dscale");
        auto layout = makeNhwcCase();
        layout.dscaleStrides = nchwStrides();
        expectBprop(layout, true);
    }
    {
        SCOPED_TRACE("NHWC x, row-major dbias");
        auto layout = makeNhwcCase();
        layout.dbiasStrides = nchwStrides();
        expectBprop(layout, true);
    }
    {
        SCOPED_TRACE("NCHW x, NHWC scale");
        LayoutCase layout;
        layout.scaleStrides = nhwcStrides();
        expectBprop(layout, true);
    }
}

TEST(TestLayernormValidator, PaddedOrTransposedDscaleDbiasDeclinedBprop)
{
    const std::vector<int64_t> padded = {168, 56, 8, 1};
    const std::vector<int64_t> nhwc = {96, 1, 24, 3};

    LayoutCase layout;
    layout.xDims = {2, 3, 4, 8};
    layout.xStrides = {96, 32, 8, 1};
    layout.affineDims = {1, 3, 4, 8};
    layout.scaleStrides = {96, 32, 8, 1};
    layout.dscaleStrides = {96, 32, 8, 1};
    layout.dbiasStrides = {96, 32, 8, 1};
    expectBprop(layout, false);

    for(const auto* strides : {&padded, &nhwc})
    {
        {
            SCOPED_TRACE("dscale");
            auto mutated = layout;
            mutated.dscaleStrides = *strides;
            expectBprop(mutated, true);
        }
        {
            SCOPED_TRACE("dbias");
            auto mutated = layout;
            mutated.dbiasStrides = *strides;
            expectBprop(mutated, true);
        }
    }

    // The same NHWC pitch is consistent with an NHWC x.
    layout.xStrides = nhwc;
    layout.scaleStrides = nhwc;
    layout.dscaleStrides = nhwc;
    layout.dbiasStrides = nhwc;
    expectBprop(layout, false);
}

TEST(TestLayernormValidator, StatOrderDifferentFromXDeclinedBprop)
{
    // Normalizing over H and W leaves stats [N,C,1,1] with two non-unit dims.
    auto layout = makeNhwcCase();
    layout.affineDims = {1, 1, 4, 4};
    layout.scaleStrides = {16, 16, 4, 1};
    layout.dscaleStrides = {16, 16, 4, 1};
    layout.dbiasStrides = {16, 16, 4, 1};
    layout.statDims = {2, 8, 1, 1};
    layout.meanStrides = {8, 1, 1, 1};
    layout.invVarianceStrides = {8, 1, 1, 1};
    // Strides {C,1,1,1} order N before C exactly as NHWC x does, despite the unit-dim ties.
    expectBprop(layout, false);

    {
        SCOPED_TRACE("mean");
        auto mutated = layout;
        mutated.meanStrides = {1, 2, 1, 1};
        expectBprop(mutated, true);
    }
    {
        SCOPED_TRACE("inv_variance");
        auto mutated = layout;
        mutated.invVarianceStrides = {1, 2, 1, 1};
        expectBprop(mutated, true);
    }
}

TEST(TestLayernormValidator, HigherRankAffineOrStatDeclinedBprop)
{
    // Declined before the normalized-dim checks, which index x's dims by the operand's rank.
    {
        SCOPED_TRACE("affine");
        LayoutCase layout;
        layout.affineDims = {1, 1, 8, 4, 4};
        layout.scaleStrides = {128, 128, 16, 4, 1};
        layout.dscaleStrides = {128, 128, 16, 4, 1};
        layout.dbiasStrides = {128, 128, 16, 4, 1};
        expectBprop(layout, true);
    }
    {
        SCOPED_TRACE("stats");
        LayoutCase layout;
        layout.statDims = {2, 1, 1, 1, 1};
        layout.meanStrides = {1, 1, 1, 1, 1};
        layout.invVarianceStrides = {1, 1, 1, 1, 1};
        expectBprop(layout, true);
    }
}

TEST(TestLayernormValidator, DxDataTypeDifferentFromXDeclinedBprop)
{
    expectBpropDxType(DT::FLOAT, DT::FLOAT, false);
    expectBpropDxType(DT::FLOAT, DT::HALF, true);
    expectBpropDxType(DT::FLOAT, DT::BFLOAT16, true);
    expectBpropDxType(DT::HALF, DT::FLOAT, true);
}

TEST(TestLayernormValidator, PassByValueOperandDeclinedFprop)
{
    // The kernels read every affine and stat operand as a device buffer.
    expectFprop(LayoutCase{}, false);
    for(const char* name : {"scale", "mean"})
    {
        SCOPED_TRACE(name);
        LayoutCase layout;
        layout.passByValue = {name};
        expectFprop(layout, true);
    }
}

TEST(TestLayernormValidator, PassByValueMeanDeclinedBprop)
{
    expectBprop(LayoutCase{}, false);

    LayoutCase layout;
    layout.passByValue = {"mean"};
    expectBprop(layout, true);
}

TEST(TestLayernormValidator, HigherRankPassByValueAffineDeclinedFprop)
{
    // Declined before the normalized-dim checks, which index x's dims by the operand's rank.
    // Bias is pass-by-value as well, so no rank-5 device operand reaches the rank check, and
    // [1,8,4,4,2] read against x [2,8,4,4] gives no normalized-dim conflict with the stats.
    expectFprop(LayoutCase{}, false);

    LayoutCase layout;
    layout.affineDims = {1, 8, 4, 4, 2};
    layout.scaleStrides = {256, 32, 8, 2, 1};
    layout.biasStrides = {256, 32, 8, 2, 1};
    layout.passByValue = {"scale", "bias"};
    expectFprop(layout, true);
}

namespace
{

// [1,8,4,4] with C, H, W strides 16, 3, 2: the span is 128 for 128 elements and the order
// matches NCHW x, yet (h, w) = (0, 3) and (2, 0) share an element.
std::vector<int64_t> overlappingAffineStrides()
{
    return {128, 16, 3, 2};
}

// Normalized over W only: affine [1,1,1,4], statistics [2,8,4,1].
LayoutCase makeNchStatCase()
{
    LayoutCase layout;
    layout.affineDims = {1, 1, 1, 4};
    layout.scaleStrides = {4, 4, 4, 1};
    layout.biasStrides = {4, 4, 4, 1};
    layout.dscaleStrides = {4, 4, 4, 1};
    layout.dbiasStrides = {4, 4, 4, 1};
    layout.statDims = {2, 8, 4, 1};
    layout.meanStrides = {32, 4, 1, 1};
    layout.invVarianceStrides = {32, 4, 1, 1};
    return layout;
}

// N, C, H strides 36, 3, 2: the span is 64 for 64 elements and the order matches NCHW x,
// yet (c, h) = (0, 3) and (2, 0) share an element.
std::vector<int64_t> overlappingStatStrides()
{
    return {36, 3, 2, 1};
}

} // anonymous namespace

TEST(TestLayernormValidator, OverlappingOperandStridesDeclinedFprop)
{
    {
        SCOPED_TRACE("scale");
        LayoutCase layout;
        layout.scaleStrides = overlappingAffineStrides();
        expectFprop(layout, true);
    }
    {
        SCOPED_TRACE("bias");
        LayoutCase layout;
        layout.biasStrides = overlappingAffineStrides();
        expectFprop(layout, true);
    }
    {
        SCOPED_TRACE("dense statistics");
        expectFprop(makeNchStatCase(), false);
    }
    {
        SCOPED_TRACE("mean");
        auto layout = makeNchStatCase();
        layout.meanStrides = overlappingStatStrides();
        expectFprop(layout, true);
    }
    {
        SCOPED_TRACE("inv_variance");
        auto layout = makeNchStatCase();
        layout.invVarianceStrides = overlappingStatStrides();
        expectFprop(layout, true);
    }
}

TEST(TestLayernormValidator, OverlappingOperandStridesDeclinedBprop)
{
    {
        SCOPED_TRACE("scale");
        LayoutCase layout;
        layout.scaleStrides = overlappingAffineStrides();
        expectBprop(layout, true);
    }
    {
        SCOPED_TRACE("dscale");
        LayoutCase layout;
        layout.dscaleStrides = overlappingAffineStrides();
        expectBprop(layout, true);
    }
    {
        SCOPED_TRACE("dbias");
        LayoutCase layout;
        layout.dbiasStrides = overlappingAffineStrides();
        expectBprop(layout, true);
    }
    {
        SCOPED_TRACE("dense statistics");
        expectBprop(makeNchStatCase(), false);
    }
    {
        SCOPED_TRACE("mean");
        auto layout = makeNchStatCase();
        layout.meanStrides = overlappingStatStrides();
        expectBprop(layout, true);
    }
    {
        SCOPED_TRACE("inv_variance");
        auto layout = makeNchStatCase();
        layout.invVarianceStrides = overlappingStatStrides();
        expectBprop(layout, true);
    }
}

namespace
{

// x [4,6,8,8] NCHW normalized over C, H and W: affine [1,6,8,8], statistics [4,1,1,1].
LayoutCase makeNormalizedOverChwCase()
{
    const std::vector<int64_t> strides = {384, 64, 8, 1};
    LayoutCase layout;
    layout.xDims = {4, 6, 8, 8};
    layout.xStrides = strides;
    layout.affineDims = {1, 6, 8, 8};
    layout.scaleStrides = strides;
    layout.biasStrides = strides;
    layout.dscaleStrides = strides;
    layout.dbiasStrides = strides;
    layout.statDims = {4, 1, 1, 1};
    return layout;
}

void setAffine(LayoutCase& layout,
               const std::vector<int64_t>& dims,
               const std::vector<int64_t>& strides)
{
    layout.affineDims = dims;
    layout.scaleStrides = strides;
    layout.biasStrides = strides;
    layout.dscaleStrides = strides;
    layout.dbiasStrides = strides;
}

using NamedLayoutCase = std::pair<const char*, LayoutCase>;

// Operand dims that match x's normalized dims, for several normalizedDim values and ranks.
std::vector<NamedLayoutCase> operandDimsMatchingXCases()
{
    std::vector<NamedLayoutCase> cases;
    cases.emplace_back("rank 4, normalized over C, H and W", makeNormalizedOverChwCase());

    auto overHw = makeNormalizedOverChwCase();
    setAffine(overHw, {1, 1, 8, 8}, {64, 64, 8, 1});
    overHw.statDims = {4, 6, 1, 1};
    overHw.meanStrides = {6, 1, 1, 1};
    overHw.invVarianceStrides = {6, 1, 1, 1};
    cases.emplace_back("rank 4, normalized over H and W", overHw);

    const std::vector<int64_t> strides5 = {192, 64, 16, 4, 1};
    LayoutCase rank5;
    rank5.xDims = {2, 3, 4, 4, 4};
    rank5.xStrides = strides5;
    setAffine(rank5, {1, 3, 4, 4, 4}, strides5);
    rank5.statDims = {2, 1, 1, 1, 1};
    rank5.meanStrides = {1, 1, 1, 1, 1};
    rank5.invVarianceStrides = {1, 1, 1, 1, 1};
    cases.emplace_back("rank 5, normalized over C, D, H and W", rank5);
    return cases;
}

// Operand dims that differ from x's, smaller and larger, each passing validateNormalizedDim.
std::vector<NamedLayoutCase> operandDimsDifferentFromXCases()
{
    std::vector<NamedLayoutCase> cases;
    for(const auto& [name, rows] : {std::pair<const char*, int64_t>{"statistics rows 2 of 4", 2},
                                    std::pair<const char*, int64_t>{"statistics rows 3 of 4", 3},
                                    std::pair<const char*, int64_t>{"statistics rows 5 of 4", 5}})
    {
        auto layout = makeNormalizedOverChwCase();
        layout.statDims = {rows, 1, 1, 1};
        cases.emplace_back(name, layout);
    }
    for(const auto& [name, width] : {std::pair<const char*, int64_t>{"affine W 4 of 8", 4},
                                     std::pair<const char*, int64_t>{"affine W 7 of 8", 7},
                                     std::pair<const char*, int64_t>{"affine W 9 of 8", 9}})
    {
        auto layout = makeNormalizedOverChwCase();
        setAffine(layout, {1, 6, 8, width}, {48 * width, 8 * width, width, 1});
        cases.emplace_back(name, layout);
    }
    auto middleOne = makeNormalizedOverChwCase();
    setAffine(middleOne, {1, 6, 1, 8}, {48, 8, 8, 1});
    cases.emplace_back("affine H 1 of 8", middleOne);
    return cases;
}

} // anonymous namespace

TEST(TestLayernormValidator, OperandDimsMatchingXAcceptedFprop)
{
    for(const auto& [name, layout] : operandDimsMatchingXCases())
    {
        SCOPED_TRACE(name);
        expectFprop(layout, false);
    }
}

TEST(TestLayernormValidator, OperandDimsDifferentFromXDeclinedFprop)
{
    for(const auto& [name, layout] : operandDimsDifferentFromXCases())
    {
        SCOPED_TRACE(name);
        expectFprop(layout, true);
    }
}

TEST(TestLayernormValidator, OperandDimsMatchingXAcceptedBprop)
{
    for(const auto& [name, layout] : operandDimsMatchingXCases())
    {
        SCOPED_TRACE(name);
        expectBprop(layout, false);
    }
}

TEST(TestLayernormValidator, OperandDimsDifferentFromXDeclinedBprop)
{
    for(const auto& [name, layout] : operandDimsDifferentFromXCases())
    {
        SCOPED_TRACE(name);
        expectBprop(layout, true);
    }
}
