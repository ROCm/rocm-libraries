// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <gtest/gtest.h>

#include "engines/plans/RMSnorm/RMSnormApplicabilityChecks.hpp"
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider::rmsnorm;

TEST(TestRMSnormValidator, Valid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ValidBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, UnsupportedDim)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph(
        {12, 4, 1}, {1, 3, 4}, hipdnn_data_sdk::data_objects::DataType::FLOAT);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    // 3D tensor is not supported
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

namespace
{
flatbuffers::FlatBufferBuilder
    createInvalidTypeRMSNormGraph(hipdnn_data_sdk::data_objects::DataType xType,
                                  hipdnn_data_sdk::data_objects::DataType yType,
                                  hipdnn_data_sdk::data_objects::DataType scaleType,
                                  hipdnn_data_sdk::data_objects::DataType biasType,
                                  hipdnn_data_sdk::data_objects::DataType invRMSType)
{
    std::vector<int64_t> strides{48, 16, 4, 1};
    std::vector<int64_t> dims{1, 3, 4, 4};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", xType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", yType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", scaleType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "bias", biasType, &derivedStrides, &derivedDims));

    // inv_rms should get norm stats shape [N, 1, H, W]
    std::vector<int64_t> invRMSDims = dims;
    invRMSDims[1] = 1;
    std::vector<int64_t> invRMSStrides = strides;
    invRMSStrides[0] = invRMSStrides[1];

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "inv_rms", invRMSType, &invRMSStrides, &invRMSDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_data_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "epsilon",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_data_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto rmsnormAttributes
        = hipdnn_data_sdk::data_objects::CreateRMSNormAttributes(builder,
                                                                 1, // x uid
                                                                 3, // scale uid
                                                                 6, // epsilon uid
                                                                 2, // y uid
                                                                 4, // bias uid
                                                                 5 // invRMS
        );

    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormAttributes,
        rmsnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}
} // anonymous namespace

TEST(TestRMSnormValidator, MismatchIOTypes)
{
    auto builder = createInvalidTypeRMSNormGraph(hipdnn_data_sdk::data_objects::DataType::HALF,
                                                 hipdnn_data_sdk::data_objects::DataType::FLOAT,
                                                 hipdnn_data_sdk::data_objects::DataType::FLOAT,
                                                 hipdnn_data_sdk::data_objects::DataType::FLOAT,
                                                 hipdnn_data_sdk::data_objects::DataType::FLOAT);

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // Data type of x and y tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedScaleType)
{
    auto builder = createInvalidTypeRMSNormGraph(hipdnn_data_sdk::data_objects::DataType::FLOAT,
                                                 hipdnn_data_sdk::data_objects::DataType::FLOAT,
                                                 hipdnn_data_sdk::data_objects::DataType::HALF,
                                                 hipdnn_data_sdk::data_objects::DataType::FLOAT,
                                                 hipdnn_data_sdk::data_objects::DataType::FLOAT);

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // Data type of scale should be the same as bias, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedInvRMSType)
{
    auto builder = createInvalidTypeRMSNormGraph(hipdnn_data_sdk::data_objects::DataType::FLOAT,
                                                 hipdnn_data_sdk::data_objects::DataType::FLOAT,
                                                 hipdnn_data_sdk::data_objects::DataType::FLOAT,
                                                 hipdnn_data_sdk::data_objects::DataType::FLOAT,
                                                 hipdnn_data_sdk::data_objects::DataType::HALF);

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // only FLOAT inv_rms type is supported at the moment, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

namespace
{
flatbuffers::FlatBufferBuilder
    createInvalidShapeRMSNormGraph(const std::vector<int64_t>& xDims,
                                   const std::vector<int64_t>& xStrides,
                                   const std::vector<int64_t>& yDims,
                                   const std::vector<int64_t>& yStrides,
                                   const std::vector<int64_t>& scaleDims,
                                   const std::vector<int64_t>& scaleStrides,
                                   const std::vector<int64_t>& biasDims,
                                   const std::vector<int64_t>& biasStrides,
                                   const std::vector<int64_t>& invRMSDims,
                                   const std::vector<int64_t>& invRMSStrides)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_data_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_data_sdk::data_objects::DataType::FLOAT, &yStrides, &yDims));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &scaleStrides,
        &scaleDims));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &biasStrides,
        &biasDims));

    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "inv_rms",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &invRMSStrides,
        &invRMSDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_data_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "epsilon",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_data_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto rmsnormAttributes
        = hipdnn_data_sdk::data_objects::CreateRMSNormAttributes(builder,
                                                                 1, // x uid
                                                                 3, // scale uid
                                                                 6, // epsilon uid
                                                                 2, // y uid
                                                                 4, // bias uid
                                                                 5 // invRMS
        );

    std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_data_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormAttributes,
        rmsnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}
} // anonymous namespace

TEST(TestRMSnormValidator, MismatchIOShapes)
{
    std::vector<int64_t> xDims{1, 3, 4, 4};
    std::vector<int64_t> xStrides{48, 16, 4, 1};

    std::vector<int64_t> yDims{1, 3, 2, 2};
    std::vector<int64_t> yStrides{12, 4, 2, 1};

    // inv_rms should get norm stats shape [N, 1, H, W]
    std::vector<int64_t> invRMSDims = xDims;
    invRMSDims[1] = 1;
    std::vector<int64_t> invRMSStrides = xStrides;
    invRMSStrides[0] = invRMSStrides[1];

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(xDims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    auto builder = createInvalidShapeRMSNormGraph(xDims,
                                                  xStrides,
                                                  yDims,
                                                  yStrides,
                                                  derivedDims,
                                                  derivedStrides,
                                                  derivedDims,
                                                  derivedStrides,
                                                  invRMSDims,
                                                  invRMSStrides);

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // Shape of x and y tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedScaleShape)
{
    std::vector<int64_t> ioDims{1, 3, 4, 4};
    std::vector<int64_t> ioStrides{48, 16, 4, 1};

    // inv_rms should get norm stats shape [N, 1, H, W]
    std::vector<int64_t> invRMSDims = ioDims;
    invRMSDims[1] = 1;
    std::vector<int64_t> invRMSStrides = ioStrides;
    invRMSStrides[0] = invRMSStrides[1];

    auto builder = createInvalidShapeRMSNormGraph(ioDims,
                                                  ioStrides,
                                                  ioDims,
                                                  ioStrides,
                                                  invRMSDims,
                                                  invRMSDims,
                                                  invRMSDims,
                                                  invRMSDims,
                                                  invRMSDims,
                                                  invRMSStrides);

    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // Shape of scale and bias should be channel-only, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedInvRMShape)
{
    std::vector<int64_t> ioDims{1, 3, 4, 4};
    std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(ioDims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    // Shape of x and y tensors should match
    auto builder = createInvalidShapeRMSNormGraph(ioDims,
                                                  ioStrides,
                                                  ioDims,
                                                  ioStrides,
                                                  derivedDims,
                                                  derivedStrides,
                                                  derivedDims,
                                                  derivedStrides,
                                                  derivedDims,
                                                  derivedStrides);
    hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                              builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // inv_rms should get norm stats shape [N, 1, H, W], expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
