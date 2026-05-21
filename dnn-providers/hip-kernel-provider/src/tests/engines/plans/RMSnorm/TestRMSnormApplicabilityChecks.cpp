// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <gtest/gtest.h>

#include "engines/plans/RMSnorm/RMSnormApplicabilityChecks.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider::rmsnorm;

TEST(TestRMSnormValidator, Valid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ValidBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, UnsupportedDim)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph(
        {12, 4, 1}, {1, 3, 4}, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

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
    createExplicitTypeRMSNormGraph(hipdnn_flatbuffers_sdk::data_objects::DataType xType,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType yType,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType scaleType,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType biasType,
                                   hipdnn_flatbuffers_sdk::data_objects::DataType invRMSType)
{
    const std::vector<int64_t> strides{48, 16, 4, 1};
    const std::vector<int64_t> dims{1, 3, 4, 4};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", xType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", yType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", scaleType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "bias", biasType, &derivedStrides, &derivedDims));

    // inv_rms should get norm stats shape [N, 1, H, W]
    std::vector<int64_t> invRMSDims = dims;
    invRMSDims[1] = 1;
    std::vector<int64_t> invRMSStrides = strides;
    invRMSStrides[0] = invRMSStrides[1];

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "inv_rms", invRMSType, &invRMSStrides, &invRMSDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "epsilon",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto rmsnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormAttributes(builder,
                                                                        1, // x uid
                                                                        3, // scale uid
                                                                        6, // epsilon uid
                                                                        2, // y uid
                                                                        4, // bias uid
                                                                        5 // invRMS
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes,
        rmsnormAttributes.Union());
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

TEST(TestRMSnormValidator, MismatchIOTypes)
{
    auto builder
        = createExplicitTypeRMSNormGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // Data type of x and y tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedScaleType)
{
    auto builder
        = createExplicitTypeRMSNormGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // Data type of scale should be the same as bias, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedInvRMSType)
{
    auto builder
        = createExplicitTypeRMSNormGraph(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

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
    createExplicitShapeRMSNormGraph(const std::vector<int64_t>& xDims,
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
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

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

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "inv_rms",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &invRMSStrides,
        &invRMSDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "epsilon",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto rmsnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormAttributes(builder,
                                                                        1, // x uid
                                                                        3, // scale uid
                                                                        6, // epsilon uid
                                                                        2, // y uid
                                                                        4, // bias uid
                                                                        5 // invRMS
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes,
        rmsnormAttributes.Union());
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

TEST(TestRMSnormValidator, MismatchIOShapes)
{
    const std::vector<int64_t> xDims{2, 3, 4, 4};
    const std::vector<int64_t> xStrides{48, 16, 4, 1};

    const std::vector<int64_t> yDims{2, 3, 2, 2};
    const std::vector<int64_t> yStrides{12, 4, 2, 1};

    const std::vector<int64_t> derivedDims{1, 1, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    // inv_rms should be infered from IO and derived dims
    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    auto builder = createExplicitShapeRMSNormGraph(xDims,
                                                   xStrides,
                                                   yDims,
                                                   yStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   invRMSDims,
                                                   invRMSStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // Shape of x and y tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchAffineDims)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    // Scale and bias both normalized correctly, but don't match
    const std::vector<int64_t> scaleDims{1, 3, 4, 4};
    const std::vector<int64_t> scaleStrides = hipdnn_data_sdk::utilities::generateStrides(
        scaleDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> biasDims{1, 1, 4, 4};
    const std::vector<int64_t> biasStrides = hipdnn_data_sdk::utilities::generateStrides(
        biasDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    // inv_rms should be infered from IO and derived dims
    const std::vector<int64_t> invRMSDims{2, 3, 4, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormGraph(ioDims,
                                                   ioStrides,
                                                   ioDims,
                                                   ioStrides,
                                                   scaleDims,
                                                   scaleStrides,
                                                   biasDims,
                                                   biasStrides,
                                                   invRMSDims,
                                                   invRMSStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // Shape of scale and bias tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedScaleShape)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 3, 1, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    // inv_rms should be infered from IO and derived dims
    const std::vector<int64_t> invRMSDims{2, 3, 4, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormGraph(ioDims,
                                                   ioStrides,
                                                   ioDims,
                                                   ioStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   invRMSDims,
                                                   invRMSStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // Scale not normalized correctly, throw if this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedInvRMShape)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormGraph(ioDims,
                                                   ioStrides,
                                                   ioDims,
                                                   ioStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   invRMSDims,
                                                   invRMSStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis1)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormGraph(ioDims,
                                                   ioStrides,
                                                   ioDims,
                                                   ioStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   invRMSDims,
                                                   invRMSStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis2)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 1, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormGraph(ioDims,
                                                   ioStrides,
                                                   ioDims,
                                                   ioStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   invRMSDims,
                                                   invRMSStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis3)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 1, 1, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 4, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormGraph(ioDims,
                                                   ioStrides,
                                                   ioDims,
                                                   ioStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   derivedDims,
                                                   derivedStrides,
                                                   invRMSDims,
                                                   invRMSStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkTensorConfigSupported(attr));
}
