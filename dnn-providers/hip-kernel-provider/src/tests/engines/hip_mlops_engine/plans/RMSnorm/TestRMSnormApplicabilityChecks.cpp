// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdint>
#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormApplicabilityChecks.hpp"
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
    EXPECT_NO_THROW(validator.checkFwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ValidActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node0 = graph.getNode(0);
    const auto& node1 = graph.getNode(1);
    const auto& fwdAttr = *node0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *node1.attributes_as_PointwiseAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr));
}

TEST(TestRMSnormValidator, ValidBackward)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ValidBackwardActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto& node1 = graph.getNode(1);
    const auto& node2 = graph.getNode(2);
    const auto& activationAttr = *node1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *node2.attributes_as_RMSNormBackwardAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr));
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
    EXPECT_THROW(validator.checkFwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedDimActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormActivationGraph(
        {12, 4, 1}, {1, 3, 4}, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node0 = graph.getNode(0);
    const auto& node1 = graph.getNode(1);
    const auto& fwdAttr = *node0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *node1.attributes_as_PointwiseAttributes();

    // 3D tensor is not supported
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedDimBackward)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph(
        {12, 4, 1}, {1, 3, 4}, true, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_RMSNormBackwardAttributes();

    // 3D tensor is not supported
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedDimBackwardActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdActivationGraph(
        {12, 4, 1}, {1, 3, 4}, true, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& node1 = graph.getNode(1);
    const auto& node2 = graph.getNode(2);
    const auto& activationAttr = *node1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *node2.attributes_as_RMSNormBackwardAttributes();

    // 3D tensor is not supported
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr),
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
    const std::vector<int64_t> dims{2, 3, 4, 4};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // Normalize bias/scale on first axis
    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides{48, 16, 4, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", xType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", yType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", scaleType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "bias", biasType, &derivedStrides, &derivedDims));

    // inv_rms stat shape is [N, 1, 1, 1, ...] when scale is [1, C, H, W ..]
    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides{1, 1, 1, 1};

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
                                                                        5 // invRMS uid
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

flatbuffers::FlatBufferBuilder createExplicitTypeRMSNormActivationGraph(
    hipdnn_flatbuffers_sdk::data_objects::DataType xType,
    hipdnn_flatbuffers_sdk::data_objects::DataType yType,
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleType,
    hipdnn_flatbuffers_sdk::data_objects::DataType biasType,
    hipdnn_flatbuffers_sdk::data_objects::DataType invRMSType,
    hipdnn_flatbuffers_sdk::data_objects::DataType yActivType)
{
    const std::vector<int64_t> strides{48, 16, 4, 1};
    const std::vector<int64_t> dims{2, 3, 4, 4};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // Normalize bias/scale on first axis
    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides{48, 16, 4, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", xType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", yType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", scaleType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "bias", biasType, &derivedStrides, &derivedDims));

    // inv_rms stat shape is [N, 1, 1, 1, ...] when scale is [1, C, H, W ..]
    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides{1, 1, 1, 1};

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

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "yActiv", yActivType, &strides, &dims));

    auto rmsnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormAttributes(builder,
                                                                        1, // x uid
                                                                        3, // scale uid
                                                                        6, // epsilon uid
                                                                        2, // y uid
                                                                        4, // bias uid
                                                                        5 // invRMS uid
        );

    auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        0.1f,
        0.5f,
        std::nullopt,
        std::nullopt,
        2, // y uid
        std::nullopt,
        std::nullopt,
        7, // yActiv uid
        std::nullopt,
        std::nullopt,
        std::nullopt);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes,
        rmsnormAttributes.Union());
    nodes.push_back(node);

    auto nodeActivation = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttributes.Union());
    nodes.push_back(nodeActivation);

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

flatbuffers::FlatBufferBuilder createExplicitTypeRMSNormBackwardGraph(
    hipdnn_flatbuffers_sdk::data_objects::DataType dyType,
    hipdnn_flatbuffers_sdk::data_objects::DataType xType,
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dxType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dscaleType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dbiasType,
    hipdnn_flatbuffers_sdk::data_objects::DataType invRMSType)
{
    const std::vector<int64_t> strides{48, 16, 4, 1};
    const std::vector<int64_t> dims{2, 3, 4, 4};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // Normalize bias/scale on first axis
    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides{48, 16, 4, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "dy", dyType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "x", xType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", scaleType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "dx", dxType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "dscale", dscaleType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "dbias", dbiasType, &derivedStrides, &derivedDims));

    // inv_rms stat shape is [N, 1, 1, 1, ...] when scale is [1, C, H, W ..]
    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides{1, 1, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "inv_rms", invRMSType, &invRMSStrides, &invRMSDims));

    auto rmsnormBwdAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormBackwardAttributes(
            builder,
            1, // dy uid
            2, // x uid
            3, // scale uid
            7, // invRMS uid
            4, // dx uid
            5, // dscale uid
            flatbuffers::Optional<int64_t>(6) // dbias uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes,
        rmsnormBwdAttributes.Union());
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

flatbuffers::FlatBufferBuilder createExplicitTypeRMSNormBackwardActivationGraph(
    hipdnn_flatbuffers_sdk::data_objects::DataType dyType,
    hipdnn_flatbuffers_sdk::data_objects::DataType xType,
    hipdnn_flatbuffers_sdk::data_objects::DataType yType,
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dxType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dscaleType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dbiasType,
    hipdnn_flatbuffers_sdk::data_objects::DataType invRMSType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dyActivType)
{
    const std::vector<int64_t> strides{48, 16, 4, 1};
    const std::vector<int64_t> dims{2, 3, 4, 4};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // Normalize bias/scale on first axis
    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides{48, 16, 4, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "dy", dyType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "x", xType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", scaleType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "dx", dxType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "dscale", dscaleType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "dbias", dbiasType, &derivedStrides, &derivedDims));

    // inv_rms stat shape is [N, 1, 1, 1, ...] when scale is [1, C, H, W ..]
    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides{1, 1, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "inv_rms", invRMSType, &invRMSStrides, &invRMSDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        8,
        "epsilon",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 9, "y", yType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 10, "dyActiv", dyActivType, &strides, &dims));

    auto rmsnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormAttributes(builder,
                                                                        1, // x uid
                                                                        3, // scale uid
                                                                        8, // epsilon uid
                                                                        9 // y uid
        );

    auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        0.1f,
        0.5f,
        std::nullopt,
        std::nullopt,
        1, // dy uid
        9, // y uid
        std::nullopt,
        10, // dyActiv uid
        std::nullopt,
        std::nullopt,
        std::nullopt);

    auto rmsnormBwdAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormBackwardAttributes(
            builder,
            10, // dyActiv uid
            2, // x uid
            3, // scale uid
            7, // invRMS uid
            4, // dx uid
            5, // dscale uid
            flatbuffers::Optional<int64_t>(6) // dbias uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto nodeFwd = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes,
        rmsnormAttributes.Union());
    nodes.push_back(nodeFwd);

    auto nodePointwise = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttributes.Union());
    nodes.push_back(nodePointwise);

    auto nodeBwd = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes,
        rmsnormBwdAttributes.Union());
    nodes.push_back(nodeBwd);

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

    // Data type of x and y tensors don't need to match
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkFwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, MismatchIOTypesActivation)
{
    auto builder = createExplicitTypeRMSNormActivationGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    // Data type of x and y tensors don't need to match
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr));
}

TEST(TestRMSnormValidator, MismatchIOTypesBackward)
{
    auto builder = createExplicitTypeRMSNormBackwardGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    // Data type of x and y tensors don't need to match
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, MismatchIOTypesBackwardActivation)
{
    auto builder = createExplicitTypeRMSNormBackwardActivationGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    // Data type of x and y tensors don't need to match
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr));
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
    EXPECT_THROW(validator.checkFwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedScaleTypeActivation)
{
    auto builder = createExplicitTypeRMSNormActivationGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    // Data type of scale should be the same as bias, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedScaleTypeBackward)
{
    auto builder = createExplicitTypeRMSNormBackwardGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    // Data type of scale should be the same as bias, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedScaleTypeBackwardActivation)
{
    auto builder = createExplicitTypeRMSNormBackwardActivationGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    // Data type of scale should be the same as bias, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr),
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
    EXPECT_THROW(validator.checkFwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedInvRMSTypeActivation)
{
    auto builder = createExplicitTypeRMSNormActivationGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    // only FLOAT inv_rms type is supported at the moment, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedInvRMSTypeBackward)
{
    auto builder = createExplicitTypeRMSNormBackwardGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    // only FLOAT inv_rms type is supported at the moment, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedInvRMSTypeBackwardActivation)
{
    auto builder = createExplicitTypeRMSNormBackwardActivationGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
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

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    // only FLOAT inv_rms type is supported at the moment, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr),
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
                                                                        5 // invRMS uid
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

flatbuffers::FlatBufferBuilder
    createExplicitShapeRMSNormActivationGraph(const std::vector<int64_t>& xDims,
                                              const std::vector<int64_t>& xStrides,
                                              const std::vector<int64_t>& yDims,
                                              const std::vector<int64_t>& yStrides,
                                              const std::vector<int64_t>& scaleDims,
                                              const std::vector<int64_t>& scaleStrides,
                                              const std::vector<int64_t>& biasDims,
                                              const std::vector<int64_t>& biasStrides,
                                              const std::vector<int64_t>& invRMSDims,
                                              const std::vector<int64_t>& invRMSStrides,
                                              const std::vector<int64_t>& yActivDims,
                                              const std::vector<int64_t>& yActivStrides)
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

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "yActiv",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &yActivStrides,
        &yActivDims));

    auto rmsnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormAttributes(builder,
                                                                        1, // x uid
                                                                        3, // scale uid
                                                                        6, // epsilon uid
                                                                        2, // y uid
                                                                        4, // bias uid
                                                                        5 // invRMS uid
        );

    auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        0.1f,
        0.5f,
        std::nullopt,
        std::nullopt,
        2, // y uid
        std::nullopt,
        std::nullopt,
        7, // yActiv uid
        std::nullopt,
        std::nullopt,
        std::nullopt);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes,
        rmsnormAttributes.Union());
    nodes.push_back(node);

    auto nodeActivation = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttributes.Union());
    nodes.push_back(nodeActivation);

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
    createExplicitShapeRMSNormBackwardGraph(const std::vector<int64_t>& dyDims,
                                            const std::vector<int64_t>& dyStrides,
                                            const std::vector<int64_t>& xDims,
                                            const std::vector<int64_t>& xStrides,
                                            const std::vector<int64_t>& scaleDims,
                                            const std::vector<int64_t>& scaleStrides,
                                            const std::vector<int64_t>& dxDims,
                                            const std::vector<int64_t>& dxStrides,
                                            const std::vector<int64_t>& dscaleDims,
                                            const std::vector<int64_t>& dscaleStrides,
                                            const std::vector<int64_t>& dbiasDims,
                                            const std::vector<int64_t>& dbiasStrides,
                                            const std::vector<int64_t>& invRMSDims,
                                            const std::vector<int64_t>& invRMSStrides)
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
        builder, 2, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));

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
        "dx",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dxStrides,
        &dxDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "dscale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dscaleStrides,
        &dscaleDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "dbias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dbiasStrides,
        &dbiasDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "inv_rms",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &invRMSStrides,
        &invRMSDims));

    auto rmsnormBwdAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormBackwardAttributes(
            builder,
            1, // dy uid
            2, // x uid
            3, // scale uid
            7, // invRMS uid
            4, // dx uid
            5, // dscale uid
            flatbuffers::Optional<int64_t>(6) // dbias uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes,
        rmsnormBwdAttributes.Union());
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
    createExplicitShapeRMSNormBackwardActivationGraph(const std::vector<int64_t>& dyDims,
                                                      const std::vector<int64_t>& dyStrides,
                                                      const std::vector<int64_t>& xDims,
                                                      const std::vector<int64_t>& xStrides,
                                                      const std::vector<int64_t>& yDims,
                                                      const std::vector<int64_t>& yStrides,
                                                      const std::vector<int64_t>& scaleDims,
                                                      const std::vector<int64_t>& scaleStrides,
                                                      const std::vector<int64_t>& dxDims,
                                                      const std::vector<int64_t>& dxStrides,
                                                      const std::vector<int64_t>& dscaleDims,
                                                      const std::vector<int64_t>& dscaleStrides,
                                                      const std::vector<int64_t>& dbiasDims,
                                                      const std::vector<int64_t>& dbiasStrides,
                                                      const std::vector<int64_t>& invRMSDims,
                                                      const std::vector<int64_t>& invRMSStrides,
                                                      const std::vector<int64_t>& dyActivDims,
                                                      const std::vector<int64_t>& dyActivStrides)
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
        builder, 2, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));

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
        "dx",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dxStrides,
        &dxDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "dscale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dscaleStrides,
        &dscaleDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "dbias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dbiasStrides,
        &dbiasDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "inv_rms",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &invRMSStrides,
        &invRMSDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        8,
        "epsilon",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 9, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &yStrides, &yDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        10,
        "dyActiv",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dyActivStrides,
        &dyActivDims));

    auto rmsnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormAttributes(builder,
                                                                        1, // x uid
                                                                        3, // scale uid
                                                                        8, // epsilon uid
                                                                        9 // y uid
        );

    auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        0.1f,
        0.5f,
        std::nullopt,
        std::nullopt,
        1, // dy uid
        9, // y uid
        std::nullopt,
        10, // dyActiv uid
        std::nullopt,
        std::nullopt,
        std::nullopt);

    auto rmsnormBwdAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormBackwardAttributes(
            builder,
            10, // dyActiv uid
            2, // x uid
            3, // scale uid
            7, // invRMS uid
            4, // dx uid
            5, // dscale uid
            flatbuffers::Optional<int64_t>(6) // dbias uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto nodeFwd = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes,
        rmsnormAttributes.Union());
    nodes.push_back(nodeFwd);

    auto nodePointwise = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttributes.Union());
    nodes.push_back(nodePointwise);

    auto nodeBwd = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes,
        rmsnormBwdAttributes.Union());
    nodes.push_back(nodeBwd);

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
    EXPECT_THROW(validator.checkFwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchIOShapesActivation)
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

    auto builder = createExplicitShapeRMSNormActivationGraph(xDims,
                                                             xStrides,
                                                             yDims,
                                                             yStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             invRMSDims,
                                                             invRMSStrides,
                                                             yDims,
                                                             yStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    // Shape of x and y tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchIOShapesBackward)
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

    auto builder = createExplicitShapeRMSNormBackwardGraph(yDims,
                                                           yStrides,
                                                           xDims,
                                                           xStrides,
                                                           derivedDims,
                                                           derivedStrides,
                                                           xDims,
                                                           xStrides,
                                                           derivedDims,
                                                           derivedStrides,
                                                           derivedDims,
                                                           derivedStrides,
                                                           invRMSDims,
                                                           invRMSStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    // Shape of x and y tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchIOShapesBackwardActivation)
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

    // NOLINTBEGIN(readability-suspicious-call-argument)
    auto builder = createExplicitShapeRMSNormBackwardActivationGraph(yDims,
                                                                     yStrides,
                                                                     xDims,
                                                                     xStrides,
                                                                     yDims,
                                                                     yStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     xDims,
                                                                     xStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     invRMSDims,
                                                                     invRMSStrides,
                                                                     yDims,
                                                                     yStrides);
    // NOLINTEND(readability-suspicious-call-argument)

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    // Shape of x and y tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr),
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
    EXPECT_THROW(validator.checkFwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchAffineDimsActivation)
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

    auto builder = createExplicitShapeRMSNormActivationGraph(ioDims,
                                                             ioStrides,
                                                             ioDims,
                                                             ioStrides,
                                                             scaleDims,
                                                             scaleStrides,
                                                             biasDims,
                                                             biasStrides,
                                                             invRMSDims,
                                                             invRMSStrides,
                                                             ioDims,
                                                             ioStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    // Shape of scale and bias tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchAffineDimsBackward)
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

    auto builder = createExplicitShapeRMSNormBackwardGraph(ioDims,
                                                           ioStrides,
                                                           ioDims,
                                                           ioStrides,
                                                           scaleDims,
                                                           scaleStrides,
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
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    // Shape of scale and bias tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchAffineDimsBackwardActivation)
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

    auto builder = createExplicitShapeRMSNormBackwardActivationGraph(ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     scaleDims,
                                                                     scaleStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     scaleDims,
                                                                     scaleStrides,
                                                                     biasDims,
                                                                     biasStrides,
                                                                     invRMSDims,
                                                                     invRMSStrides,
                                                                     ioDims,
                                                                     ioStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    // Shape of scale and bias tensors should match, expect exception when this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr),
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
    EXPECT_THROW(validator.checkFwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedScaleShapeActivation)
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

    auto builder = createExplicitShapeRMSNormActivationGraph(ioDims,
                                                             ioStrides,
                                                             ioDims,
                                                             ioStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             invRMSDims,
                                                             invRMSStrides,
                                                             ioDims,
                                                             ioStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    // Scale not normalized correctly, throw if this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedScaleShapeBackward)
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

    auto builder = createExplicitShapeRMSNormBackwardGraph(ioDims,
                                                           ioStrides,
                                                           ioDims,
                                                           ioStrides,
                                                           derivedDims,
                                                           derivedStrides,
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
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    // Scale not normalized correctly, throw if this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedScaleShapeBackwardActivation)
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

    auto builder = createExplicitShapeRMSNormBackwardActivationGraph(ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     invRMSDims,
                                                                     invRMSStrides,
                                                                     ioDims,
                                                                     ioStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    // Scale not normalized correctly, throw if this isn't the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr),
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
    EXPECT_THROW(validator.checkFwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedInvRMShapeActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormActivationGraph(ioDims,
                                                             ioStrides,
                                                             ioDims,
                                                             ioStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             invRMSDims,
                                                             invRMSStrides,
                                                             ioDims,
                                                             ioStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedInvRMShapeBackward)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormBackwardGraph(ioDims,
                                                           ioStrides,
                                                           ioDims,
                                                           ioStrides,
                                                           derivedDims,
                                                           derivedStrides,
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
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, UnsupportedInvRMShapeBackwardActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormBackwardActivationGraph(ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     invRMSDims,
                                                                     invRMSStrides,
                                                                     ioDims,
                                                                     ioStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr),
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
    EXPECT_NO_THROW(validator.checkFwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis1Activation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormActivationGraph(ioDims,
                                                             ioStrides,
                                                             ioDims,
                                                             ioStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             invRMSDims,
                                                             invRMSStrides,
                                                             ioDims,
                                                             ioStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis1Backward)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormBackwardGraph(ioDims,
                                                           ioStrides,
                                                           ioDims,
                                                           ioStrides,
                                                           derivedDims,
                                                           derivedStrides,
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
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis1BackwardActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormBackwardActivationGraph(ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     invRMSDims,
                                                                     invRMSStrides,
                                                                     ioDims,
                                                                     ioStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr));
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
    EXPECT_NO_THROW(validator.checkFwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis2Activation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 1, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormActivationGraph(ioDims,
                                                             ioStrides,
                                                             ioDims,
                                                             ioStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             invRMSDims,
                                                             invRMSStrides,
                                                             ioDims,
                                                             ioStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis2Backward)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 1, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormBackwardGraph(ioDims,
                                                           ioStrides,
                                                           ioDims,
                                                           ioStrides,
                                                           derivedDims,
                                                           derivedStrides,
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
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis2BackwardActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 1, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormBackwardActivationGraph(ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     invRMSDims,
                                                                     invRMSStrides,
                                                                     ioDims,
                                                                     ioStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr));
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
    EXPECT_NO_THROW(validator.checkFwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis3Activation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 1, 1, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 4, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormActivationGraph(ioDims,
                                                             ioStrides,
                                                             ioDims,
                                                             ioStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             invRMSDims,
                                                             invRMSStrides,
                                                             ioDims,
                                                             ioStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis3Backward)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 1, 1, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 4, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormBackwardGraph(ioDims,
                                                           ioStrides,
                                                           ioDims,
                                                           ioStrides,
                                                           derivedDims,
                                                           derivedStrides,
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
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, ScaleNormalizeAxis3BackwardActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims{1, 1, 1, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 4, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormBackwardActivationGraph(ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     invRMSDims,
                                                                     invRMSStrides,
                                                                     ioDims,
                                                                     ioStrides);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    // inv_rms should be infered from IO and derived dims, throw if not the case
    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr));
}

TEST(TestRMSnormValidator, Valid4DChannelLast)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 1, 12, 3}; // Channel last format

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

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkFwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, Valid4DChannelLastActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 1, 12, 3}; // Channel last format

    const std::vector<int64_t> derivedDims{1, 1, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormActivationGraph(ioDims,
                                                             ioStrides,
                                                             ioDims,
                                                             ioStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             invRMSDims,
                                                             invRMSStrides,
                                                             ioDims,
                                                             ioStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr));
}

TEST(TestRMSnormValidator, Valid4DChannelLastBackward)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 1, 12, 3}; // Channel last format

    const std::vector<int64_t> derivedDims{1, 1, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormBackwardGraph(ioDims,
                                                           ioStrides,
                                                           ioDims,
                                                           ioStrides,
                                                           derivedDims,
                                                           derivedStrides,
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
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdTensorConfigSupported(attr));
}

TEST(TestRMSnormValidator, Valid4DChannelLastBackwardActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 1, 12, 3}; // Channel last format

    const std::vector<int64_t> derivedDims{1, 1, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    const std::vector<int64_t> invRMSDims{2, 3, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormBackwardActivationGraph(ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     invRMSDims,
                                                                     invRMSStrides,
                                                                     ioDims,
                                                                     ioStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr));
}

TEST(TestRMSnormValidator, MismatchInputOutputLayouts)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};

    const std::vector<int64_t> xStrides{48, 16, 4, 1}; // NCHW
    const std::vector<int64_t> yStrides{48, 1, 12, 3}; // NHWC

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    auto builder = createExplicitShapeRMSNormGraph(ioDims,
                                                   xStrides,
                                                   ioDims,
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

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchInputOutputLayoutsActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};

    const std::vector<int64_t> xStrides{48, 16, 4, 1}; // NCHW
    const std::vector<int64_t> yStrides{48, 1, 12, 3}; // NHWC

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    auto builder = createExplicitShapeRMSNormActivationGraph(ioDims,
                                                             xStrides,
                                                             ioDims,
                                                             yStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             derivedDims,
                                                             derivedStrides,
                                                             invRMSDims,
                                                             invRMSStrides,
                                                             ioDims,
                                                             yStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchInputOutputLayoutsBackward)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};

    const std::vector<int64_t> xStrides{48, 16, 4, 1}; // NCHW
    const std::vector<int64_t> yStrides{48, 1, 12, 3}; // NHWC

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    auto builder = createExplicitShapeRMSNormBackwardGraph(ioDims,
                                                           yStrides,
                                                           ioDims,
                                                           xStrides,
                                                           derivedDims,
                                                           derivedStrides,
                                                           ioDims,
                                                           xStrides,
                                                           derivedDims,
                                                           derivedStrides,
                                                           derivedDims,
                                                           derivedStrides,
                                                           invRMSDims,
                                                           invRMSStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchInputOutputLayoutsBackwardActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};

    const std::vector<int64_t> xStrides{48, 16, 4, 1}; // NCHW
    const std::vector<int64_t> yStrides{48, 1, 12, 3}; // NHWC

    const std::vector<int64_t> derivedDims{1, 3, 4, 4};
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(xStrides));

    // NOLINTBEGIN(readability-suspicious-call-argument)
    auto builder = createExplicitShapeRMSNormBackwardActivationGraph(ioDims,
                                                                     yStrides,
                                                                     ioDims,
                                                                     xStrides,
                                                                     ioDims,
                                                                     yStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     ioDims,
                                                                     xStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     derivedDims,
                                                                     derivedStrides,
                                                                     invRMSDims,
                                                                     invRMSStrides,
                                                                     ioDims,
                                                                     yStrides);
    // NOLINTEND(readability-suspicious-call-argument)

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchAffineTensorLayout)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 1, 12, 3}; // Channel last format

    const std::vector<int64_t> scaleDims{1, 3, 4, 4};
    // Scale strides follow NCHW format, which doesn't match IO layout
    const std::vector<int64_t> incorrectScaleStrides{48, 16, 4, 1};

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormGraph(ioDims,
                                                   ioStrides,
                                                   ioDims,
                                                   ioStrides,
                                                   scaleDims,
                                                   incorrectScaleStrides,
                                                   scaleDims,
                                                   incorrectScaleStrides,
                                                   invRMSDims,
                                                   invRMSStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchAffineTensorLayoutActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 1, 12, 3}; // Channel last format

    const std::vector<int64_t> scaleDims{1, 3, 4, 4};
    const std::vector<int64_t> incorrectScaleStrides{48, 16, 4, 1};

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    auto builder = createExplicitShapeRMSNormActivationGraph(ioDims,
                                                             ioStrides,
                                                             ioDims,
                                                             ioStrides,
                                                             scaleDims,
                                                             incorrectScaleStrides,
                                                             scaleDims,
                                                             incorrectScaleStrides,
                                                             invRMSDims,
                                                             invRMSStrides,
                                                             ioDims,
                                                             ioStrides);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode0 = graph.getNode(0);
    const auto& graphNode1 = graph.getNode(1);
    const auto& fwdAttr = *graphNode0.attributes_as_RMSNormAttributes();
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkFwdActivationTensorConfigSupported(fwdAttr, activationAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchAffineTensorLayoutBackward)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 1, 12, 3}; // Channel last format

    const std::vector<int64_t> scaleDims{1, 3, 4, 4};
    const std::vector<int64_t> incorrectScaleStrides{48, 16, 4, 1};

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    // NOLINTBEGIN(readability-suspicious-call-argument)
    auto builder = createExplicitShapeRMSNormBackwardGraph(ioDims,
                                                           ioStrides,
                                                           ioDims,
                                                           ioStrides,
                                                           scaleDims,
                                                           incorrectScaleStrides,
                                                           ioDims,
                                                           ioStrides,
                                                           scaleDims,
                                                           incorrectScaleStrides,
                                                           scaleDims,
                                                           incorrectScaleStrides,
                                                           invRMSDims,
                                                           invRMSStrides);
    // NOLINTEND(readability-suspicious-call-argument)

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode = graph.getNode(0);
    const auto& attr = *graphNode.attributes_as_RMSNormBackwardAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdTensorConfigSupported(attr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestRMSnormValidator, MismatchAffineTensorLayoutBackwardActivation)
{
    const std::vector<int64_t> ioDims{2, 3, 4, 4};
    const std::vector<int64_t> ioStrides{48, 1, 12, 3}; // Channel last format

    const std::vector<int64_t> scaleDims{1, 3, 4, 4};
    const std::vector<int64_t> incorrectScaleStrides{48, 16, 4, 1};

    const std::vector<int64_t> invRMSDims{2, 1, 1, 1};
    const std::vector<int64_t> invRMSStrides = hipdnn_data_sdk::utilities::generateStrides(
        invRMSDims, hipdnn_data_sdk::utilities::extractStrideOrder(ioStrides));

    // NOLINTBEGIN(readability-suspicious-call-argument)
    auto builder = createExplicitShapeRMSNormBackwardActivationGraph(ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     scaleDims,
                                                                     incorrectScaleStrides,
                                                                     ioDims,
                                                                     ioStrides,
                                                                     scaleDims,
                                                                     incorrectScaleStrides,
                                                                     scaleDims,
                                                                     incorrectScaleStrides,
                                                                     invRMSDims,
                                                                     invRMSStrides,
                                                                     ioDims,
                                                                     ioStrides);
    // NOLINTEND(readability-suspicious-call-argument)
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& graphNode1 = graph.getNode(1);
    const auto& graphNode2 = graph.getNode(2);
    const auto& activationAttr = *graphNode1.attributes_as_PointwiseAttributes();
    const auto& bwdAttr = *graphNode2.attributes_as_RMSNormBackwardAttributes();

    RMSnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkBwdActivationTensorConfigSupported(activationAttr, bwdAttr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
