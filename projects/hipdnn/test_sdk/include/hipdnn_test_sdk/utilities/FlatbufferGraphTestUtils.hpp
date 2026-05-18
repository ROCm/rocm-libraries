// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>

namespace hipdnn_test_sdk::utilities
{

inline flatbuffers::FlatBufferBuilder createEmptyValidGraph()
{
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    flatbuffers::FlatBufferBuilder builder;
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

inline flatbuffers::FlatBufferBuilder createValidBatchnormInferenceGraph(
    const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
    const std::vector<int64_t>& dims = {1, 3, 224, 224},
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "est_mean",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "est_variance",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    auto bnormAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder,
        1, // x uid
        5, // mean uid
        6, // inv_variance uid
        3, // scale uid
        4, // bias uid
        2 // y uid
    );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnormAttributes.Union());
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

inline flatbuffers::FlatBufferBuilder createValidBatchnormWithVarianceInferenceGraph(
    const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
    const std::vector<int64_t>& dims = {1, 3, 224, 224},
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "est_mean",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "variance",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "epsilon",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributesVarianceExt(
            builder,
            1, // x uid
            5, // mean uid
            6, // variance uid
            3, // scale uid
            4, // bias uid
            2, // y uid
            7 // epsilon uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnormWithVariance",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
            BatchnormInferenceAttributesVarianceExt,
        bnormAttributes.Union());
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

inline flatbuffers::FlatBufferBuilder createValidBatchnormWithVarianceInferenceActivGraph(
    const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
    const std::vector<int64_t>& dims = {1, 3, 224, 224},
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    int64_t uid = 1;

    const auto xUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, xUid, "x", inputDataType, &strides, &dims));

    const auto yBnUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yBnUid, "y_bn", inputDataType, &strides, &dims, true)); // virtual

    const auto scaleUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        scaleUid,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const auto biasUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        biasUid,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const auto meanUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        meanUid,
        "est_mean",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const auto varUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        varUid,
        "variance",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    const auto epsUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        epsUid,
        "epsilon",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    const auto yUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yUid, "y", inputDataType, &strides, &dims));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributesVarianceExt(
            builder,
            xUid,
            meanUid,
            varUid,
            scaleUid,
            biasUid,
            yBnUid, // Virtual output
            epsUid);

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnormWithVariance",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
            BatchnormInferenceAttributesVarianceExt,
        bnormAttributes.Union()));

    auto activAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt, // relu_lower_clip
        flatbuffers::nullopt, // relu_upper_clip
        flatbuffers::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        yBnUid, // in_0_tensor_uid (virtual)
        flatbuffers::nullopt, // in_1_tensor_uid
        flatbuffers::nullopt, // in_2_tensor_uid
        yUid); // out_0_tensor_uid

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "activation",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        activAttributes.Union()));

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

inline flatbuffers::FlatBufferBuilder
    createValidBatchnormBwdGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                                 const std::vector<int64_t>& dims = {1, 3, 224, 224},
                                 bool hasOptionalAttributes = true,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                                 = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType
                                 = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType meanVarianceDataType
                                 = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "dy", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "dx", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "scale", scaleBiasDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "dscale", scaleBiasDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "dbias", scaleBiasDataType, &derivedStrides, &derivedDims));

    if(hasOptionalAttributes)
    {
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 7, "mean", meanVarianceDataType, &derivedStrides, &derivedDims));

        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 8, "inv_variance", meanVarianceDataType, &derivedStrides, &derivedDims));
    }

    auto bnormAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        2, // dy_tensor_uid
        1, // x_tensor_uid
        hasOptionalAttributes ? flatbuffers::Optional<int64_t>(7)
                              : flatbuffers::nullopt, // mean_tensor_uid
        hasOptionalAttributes ? flatbuffers::Optional<int64_t>(8)
                              : flatbuffers::nullopt, // inv_variance_tensor_uid
        4, // scale_tensor_uid
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(), // peer_stats_tensor_uid
        3, // dx_tensor_uid
        5, // dscale_tensor_uid
        6 // dbias_tensor_uid
    );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnormAttributes.Union());
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

inline flatbuffers::FlatBufferBuilder createValidBatchnormFwdInferActGraph(
    const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
    const std::vector<int64_t>& dims = {1, 3, 224, 224},
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    hipdnn_flatbuffers_sdk::data_objects::DataType intermediateDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    // inputs
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "scale", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "bias", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "mean", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "inv_variance", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "y", inputDataType, &strides, &dims, true));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "Dy", inputDataType, &strides, &dims, false)); // is_virtual = true

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm Inference
    auto bnInfAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder,
        1, // x_tensor_uid
        4, // mean_tensor_uid
        5, // inv_variance_tensor_uid
        2, // scale_tensor_uid
        3, // bias_tensor_uid
        6 // y_tensor_uid (BN_Y - virtual)
    );

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference",
        intermediateDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttributes.Union()));

    // Node 1: Pointwise (RELU_FWD)
    auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        std::nullopt, // relu_lower_clip
        std::nullopt, // relu_upper_clip
        std::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        6, // in_0_tensor_uid (BN_Y)
        flatbuffers::nullopt, // in_1_tensor_uid
        flatbuffers::nullopt, // in_2_tensor_uid
        7 // out_0_tensor_uid (Dy - not virtual)
    );

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "relu_fwd",
        intermediateDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttributes.Union()));

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

inline flatbuffers::FlatBufferBuilder createValidBatchnormInferActBwdGraph(
    const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
    const std::vector<int64_t>& dims = {1, 3, 224, 224},
    bool hasOptionalAttributes = true,
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    hipdnn_flatbuffers_sdk::data_objects::DataType intermediateDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    // inputs
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "scale", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "bias", intermediateDataType, &derivedStrides, &derivedDims));

    if(hasOptionalAttributes)
    {
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 4, "mean", intermediateDataType, &derivedStrides, &derivedDims));

        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 5, "inv_variance", intermediateDataType, &derivedStrides, &derivedDims));
    }

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "dy", inputDataType, &strides, &dims));

    // output tensors
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "dx", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 8, "dscale", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 9, "dbias", intermediateDataType, &derivedStrides, &derivedDims));

    // virtual tensors
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 10, "BN_Y", inputDataType, &strides, &dims, true)); // is_virtual = true

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 11, "DX_drelu", inputDataType, &strides, &dims, true)); // is_virtual = true

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm Inference
    auto bnInfAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder,
        1, // x_tensor_uid
        hasOptionalAttributes ? 4 : 0, // mean_tensor_uid
        hasOptionalAttributes ? 5 : 0, // inv_variance_tensor_uid
        2, // scale_tensor_uid
        3, // bias_tensor_uid
        10 // y_tensor_uid (BN_Y - virtual)
    );

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference",
        intermediateDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttributes.Union()));

    // Node 1: Pointwise (RELU_BWD)
    auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
        std::nullopt, // relu_lower_clip
        std::nullopt, // relu_upper_clip
        std::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        10, // in_0_tensor_uid (BN_Y)
        flatbuffers::Optional<int64_t>(6), // in_1_tensor_uid (dy)
        flatbuffers::nullopt, // in_2_tensor_uid
        11 // out_0_tensor_uid (DX_drelu - virtual)
    );

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "relu_bwd",
        intermediateDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttributes.Union()));

    // Node 2: Batchnorm Backward
    auto bnBwdAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        11, // dy_tensor_uid (DX_drelu)
        1, // x_tensor_uid
        hasOptionalAttributes ? flatbuffers::Optional<int64_t>(4)
                              : flatbuffers::nullopt, // mean_tensor_uid
        hasOptionalAttributes ? flatbuffers::Optional<int64_t>(5)
                              : flatbuffers::nullopt, // inv_variance_tensor_uid
        2, // scale_tensor_uid
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(), // peer_stats_tensor_uid
        7, // dx_tensor_uid
        8, // dscale_tensor_uid
        9 // dbias_tensor_uid
    );

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_backward",
        intermediateDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttributes.Union()));

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

inline flatbuffers::FlatBufferBuilder
    createValidBatchnormFwdTrainingGraph(const std::vector<int64_t>& strides = {588, 196, 14, 1},
                                         const std::vector<int64_t>& dims = {1, 3, 14, 14},
                                         bool withMeanVariance = true)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    // Required tensors
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "epsilon",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    flatbuffers::Optional<int64_t> meanUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> invVarUid = flatbuffers::nullopt;

    if(withMeanVariance)
    {
        // Optional mean/variance output tensors
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                6,
                "mean",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                7,
                "inv_variance",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));
        meanUid = flatbuffers::Optional<int64_t>(6);
        invVarUid = flatbuffers::Optional<int64_t>(7);
    }

    auto bnormAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(
        builder,
        1, // x_tensor_uid
        3, // scale_tensor_uid
        4, // bias_tensor_uid
        5, // epsilon_tensor_uid
        0, // peer_stats_tensor_uid
        flatbuffers::nullopt, // prev_running_mean_tensor_uid
        flatbuffers::nullopt, // prev_running_variance_tensor_uid
        flatbuffers::nullopt, // momentum_tensor_uid
        2, // y_tensor_uid
        meanUid, // mean_tensor_uid
        invVarUid, // inv_variance_tensor_uid
        flatbuffers::nullopt, // next_running_mean_tensor_uid
        flatbuffers::nullopt // next_running_variance_tensor_uid
    );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
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

inline flatbuffers::FlatBufferBuilder createValidBatchnormFwdTrainingActivGraph(
    bool withMeanVariance = true,
    bool withRunningStats = false,
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode activMode
    = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
    const std::vector<int64_t>& strides = {588, 196, 14, 1},
    const std::vector<int64_t>& dims = {1, 3, 14, 14})
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    int64_t uid = 1;

    // Required tensors
    const auto xTensorUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        xTensorUid,
        "x",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    const auto scaleTensorUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        scaleTensorUid,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const auto biasTensorUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        biasTensorUid,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Epsilon (pass-by-value)
    const auto epsilonTensorUid = uid++;
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        epsilonTensorUid,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    // BN output (virtual - intermediate between BN and activation)
    const auto yBnTensorUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        yBnTensorUid,
        "y_bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true)); // virtual

    // Final activation output (non-virtual)
    const auto yTensorUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        yTensorUid,
        "y",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    flatbuffers::Optional<int64_t> meanUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> invVarUid = flatbuffers::nullopt;

    if(withMeanVariance)
    {
        const auto meanTensorUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                meanTensorUid,
                "mean",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));
        meanUid = flatbuffers::Optional<int64_t>(meanTensorUid);

        const auto invVarTensorUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                invVarTensorUid,
                "inv_variance",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));
        invVarUid = flatbuffers::Optional<int64_t>(invVarTensorUid);
    }

    flatbuffers::Optional<int64_t> prevRunningMeanUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> prevRunningVarUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> momentumUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> nextRunningMeanUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> nextRunningVarUid = flatbuffers::nullopt;

    if(withRunningStats)
    {
        const auto prevMeanUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                prevMeanUid,
                "prev_running_mean",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));
        prevRunningMeanUid = flatbuffers::Optional<int64_t>(prevMeanUid);

        const auto prevVarUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                prevVarUid,
                "prev_running_variance",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));
        prevRunningVarUid = flatbuffers::Optional<int64_t>(prevVarUid);

        const auto momUid = uid++;
        const hipdnn_flatbuffers_sdk::data_objects::Float32Value momentumVal(0.1f);
        tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
            builder,
            momUid,
            builder.CreateString("momentum"),
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            0,
            0,
            false,
            hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
            builder.CreateStruct(momentumVal).Union()));
        momentumUid = flatbuffers::Optional<int64_t>(momUid);

        const auto nextMeanUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                nextMeanUid,
                "next_running_mean",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));
        nextRunningMeanUid = flatbuffers::Optional<int64_t>(nextMeanUid);

        const auto nextVarUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                nextVarUid,
                "next_running_variance",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));
        nextRunningVarUid = flatbuffers::Optional<int64_t>(nextVarUid);
    }

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm forward training
    auto bnormAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(
        builder,
        xTensorUid,
        scaleTensorUid,
        biasTensorUid,
        epsilonTensorUid,
        0, // peer_stats_tensor_uid
        prevRunningMeanUid,
        prevRunningVarUid,
        momentumUid,
        yBnTensorUid, // Virtual output
        meanUid,
        invVarUid,
        nextRunningMeanUid,
        nextRunningVarUid);

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    // Node 1: Activation
    auto activAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        activMode,
        flatbuffers::nullopt, // relu_lower_clip
        flatbuffers::nullopt, // relu_upper_clip
        flatbuffers::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        yBnTensorUid, // in_0_tensor_uid (virtual)
        flatbuffers::nullopt, // in_1_tensor_uid
        flatbuffers::nullopt, // in_2_tensor_uid
        yTensorUid); // out_0_tensor_uid

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "activation",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        activAttributes.Union()));

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

inline flatbuffers::FlatBufferBuilder
    createValidConvFwdGraph(const std::vector<int64_t>& xDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& xStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& wDims = {4, 4, 1, 1},
                            const std::vector<int64_t>& wStrides = {4, 1, 1, 1},
                            const std::vector<int64_t>& yDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& yStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& convPrePadding = {0, 0},
                            const std::vector<int64_t>& convPostPadding = {0, 0},
                            const std::vector<int64_t>& convStrides = {1, 1},
                            const std::vector<int64_t>& convDilation = {1, 1},
                            hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", dataType, &xStrides, &xDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "w", dataType, &wStrides, &wDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "y", dataType, &yStrides, &yDims));

    auto convAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
            builder,
            1, // x tensor uid
            2, // w tensor uid
            3, // y tensor uid
            &convPrePadding,
            &convPostPadding,
            &convStrides,
            &convDilation,
            hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_fwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes,
        convAttributes.Union());
    nodes.push_back(node);

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

inline flatbuffers::FlatBufferBuilder
    createValidConvBwdGraph(const std::vector<int64_t>& dxDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& dxStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& wDims = {4, 4, 1, 1},
                            const std::vector<int64_t>& wStrides = {4, 1, 1, 1},
                            const std::vector<int64_t>& dyDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& dyStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& convPrePadding = {0, 0},
                            const std::vector<int64_t>& convPostPadding = {0, 0},
                            const std::vector<int64_t>& convStrides = {1, 1},
                            const std::vector<int64_t>& convDilation = {1, 1},
                            hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "dy", dataType, &dyStrides, &dyDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "w", dataType, &wStrides, &wDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "dx", dataType, &dxStrides, &dxDims));

    auto convAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
            builder,
            1, // dy tensor uid
            2, // w tensor uid
            3, // dx tensor uid
            &convPrePadding,
            &convPostPadding,
            &convStrides,
            &convDilation,
            hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_bwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes,
        convAttributes.Union());
    nodes.push_back(node);

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

inline flatbuffers::FlatBufferBuilder
    createValidConvWrwGraph(const std::vector<int64_t>& xDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& xStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& dwDims = {4, 4, 1, 1},
                            const std::vector<int64_t>& dwStrides = {4, 1, 1, 1},
                            const std::vector<int64_t>& dyDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& dyStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& convPrePadding = {0, 0},
                            const std::vector<int64_t>& convPostPadding = {0, 0},
                            const std::vector<int64_t>& convStrides = {1, 1},
                            const std::vector<int64_t>& convDilation = {1, 1},
                            hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", dataType, &xStrides, &xDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "dy", dataType, &dyStrides, &dyDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "dw", dataType, &dwStrides, &dwDims));

    auto convAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
            builder,
            1, // x tensor uid
            2, // dy tensor uid
            3, // w tensor uid
            &convPrePadding,
            &convPostPadding,
            &convStrides,
            &convDilation,
            hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_wrw",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes,
        convAttributes.Union());
    nodes.push_back(node);

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

// TODO: Replace with a createValidPointwiseGraph function once one is made and tested
// This may be useful to keep in general though, as it has distinct and non-null values for all fields
inline flatbuffers::FlatBufferBuilder createPointwiseGraph()
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto pointwiseNode = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::DIV, // operation
        1.f, // relu_lower_clip
        2.f, // relu_upper_clip
        3.f, // relu_lower_clip_slope
        0, // axis_tensor_uid
        1, // in_0_tensor_uid
        2, // in_1_tensor_uid
        3, // in_2_tensor_uid
        4, // out_0_tensor_uid
        4.f, // swish_beta
        5.f, // elu_alpha
        6.f); // softplus_beta

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "hipdnn_flatbuffers_sdk::data_objects::Node",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseNode.Union()));

    const std::array tensorNames = {"axis", "in_0", "in_1", "in_2", "out_0"};
    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensors;
    tensors.reserve(tensorNames.size());
    int64_t tensorUid = 0;
    const std::vector<int64_t> dims = {1, 2, 3, 4};
    const std::vector<int64_t> strides = {5, 6, 7, 8};
    for(auto name : tensorNames)
    {
        tensors.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            tensorUid++,
            name,
            hipdnn_flatbuffers_sdk::data_objects::DataType::UINT8,
            &strides,
            &dims,
            false));
    }

    auto graph = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "PointwiseGraph",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensors,
        &nodes);

    builder.Finish(graph);

    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidConvFwdBiasActivGraph(const std::vector<int64_t>& xDims,
                                     const std::vector<int64_t>& xStrides,
                                     const std::vector<int64_t>& wDims,
                                     const std::vector<int64_t>& wStrides,
                                     const std::vector<int64_t>& yDims,
                                     const std::vector<int64_t>& yStrides,
                                     const std::vector<int64_t>& convPrePadding,
                                     const std::vector<int64_t>& convPostPadding,
                                     const std::vector<int64_t>& convStrides,
                                     const std::vector<int64_t>& convDilation,
                                     bool doBias,
                                     hipdnn_flatbuffers_sdk::data_objects::PointwiseMode activMode,
                                     std::optional<float> reluLowerClip,
                                     std::optional<float> reluUpperClip,
                                     std::optional<float> reluLowerClipSlope,
                                     std::optional<float> swishBeta,
                                     std::optional<float> eluAlpha,
                                     std::optional<float> softplusBeta,
                                     hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    int64_t tensorUid = 1;

    const auto xTensorUid = tensorUid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, xTensorUid, "x", dataType, &xStrides, &xDims));

    const auto wTensorUid = tensorUid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, wTensorUid, "w", dataType, &wStrides, &wDims));

    // Virtual y_conv tensor
    const auto yConvTensorUid = tensorUid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yConvTensorUid, "y_conv", dataType, &yStrides, &yDims, true));

    int64_t biasTensorUid;
    int64_t yBiasTensorUid;
    if(doBias)
    {
        const auto biasDims = hipdnn_data_sdk::utilities::getDerivedShape(yDims);
        const auto biasStrides = hipdnn_data_sdk::utilities::generateStrides(
            biasDims, hipdnn_data_sdk::utilities::extractStrideOrder(yDims));

        biasTensorUid = tensorUid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, biasTensorUid, "bias", dataType, &biasStrides, &biasDims));
        // Virtual y_bias tensor
        yBiasTensorUid = tensorUid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, yBiasTensorUid, "y_bias", dataType, &yStrides, &yDims, true));
    }

    const auto yTensorUid = tensorUid;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yTensorUid, "y", dataType, &yStrides, &yDims));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    auto convAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
            builder,
            xTensorUid,
            wTensorUid,
            yConvTensorUid,
            &convPrePadding,
            &convPostPadding,
            &convStrides,
            &convDilation,
            hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_fwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes,
        convAttributes.Union()));

    if(doBias)
    {
        auto biasAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            yConvTensorUid,
            biasTensorUid,
            flatbuffers::nullopt,
            yBiasTensorUid);
        nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
            builder,
            "bias",
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
            biasAttributes.Union()));
    }

    auto activAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        activMode,
        reluLowerClip,
        reluUpperClip,
        reluLowerClipSlope,
        flatbuffers::nullopt,
        doBias ? yBiasTensorUid : yConvTensorUid,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        yTensorUid,
        swishBeta,
        eluAlpha,
        softplusBeta);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "activ",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        activAttributes.Union()));

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

inline flatbuffers::FlatBufferBuilder
    createValidConvFwdActivGraph(const std::vector<int64_t>& xDims = {4, 4, 4, 4},
                                 const std::vector<int64_t>& xStrides = {64, 16, 4, 1},
                                 const std::vector<int64_t>& wDims = {4, 4, 1, 1},
                                 const std::vector<int64_t>& wStrides = {4, 1, 1, 1},
                                 const std::vector<int64_t>& yDims = {4, 4, 4, 4},
                                 const std::vector<int64_t>& yStrides = {64, 16, 4, 1},
                                 const std::vector<int64_t>& convPrePadding = {0, 0},
                                 const std::vector<int64_t>& convPostPadding = {0, 0},
                                 const std::vector<int64_t>& convStrides = {1, 1},
                                 const std::vector<int64_t>& convDilation = {1, 1},
                                 hipdnn_flatbuffers_sdk::data_objects::PointwiseMode activMode
                                 = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
                                 std::optional<float> reluLowerClip = std::nullopt,
                                 std::optional<float> reluUpperClip = std::nullopt,
                                 std::optional<float> reluLowerClipSlope = std::nullopt,
                                 std::optional<float> swishBeta = std::nullopt,
                                 std::optional<float> eluAlpha = std::nullopt,
                                 std::optional<float> softplusBeta = std::nullopt,
                                 hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                                 = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    return createValidConvFwdBiasActivGraph(xDims,
                                            xStrides,
                                            wDims,
                                            wStrides,
                                            yDims,
                                            yStrides,
                                            convPrePadding,
                                            convPostPadding,
                                            convStrides,
                                            convDilation,
                                            false,
                                            activMode,
                                            reluLowerClip,
                                            reluUpperClip,
                                            reluLowerClipSlope,
                                            swishBeta,
                                            eluAlpha,
                                            softplusBeta,
                                            dataType);
}

inline flatbuffers::FlatBufferBuilder createValidConvFwdBiasActivGraph(
    const std::vector<int64_t>& xDims = {4, 4, 4, 4},
    const std::vector<int64_t>& xStrides = {64, 16, 4, 1},
    const std::vector<int64_t>& wDims = {4, 4, 1, 1},
    const std::vector<int64_t>& wStrides = {4, 1, 1, 1},
    const std::vector<int64_t>& yDims = {4, 4, 4, 4},
    const std::vector<int64_t>& yStrides = {64, 16, 4, 1},
    const std::vector<int64_t>& convPrePadding = {0, 0},
    const std::vector<int64_t>& convPostPadding = {0, 0},
    const std::vector<int64_t>& convStrides = {1, 1},
    const std::vector<int64_t>& convDilation = {1, 1},
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode activMode
    = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
    std::optional<float> reluLowerClip = std::nullopt,
    std::optional<float> reluUpperClip = std::nullopt,
    std::optional<float> reluLowerClipSlope = std::nullopt,
    std::optional<float> swishBeta = std::nullopt,
    std::optional<float> eluAlpha = std::nullopt,
    std::optional<float> softplusBeta = std::nullopt,
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    return createValidConvFwdBiasActivGraph(xDims,
                                            xStrides,
                                            wDims,
                                            wStrides,
                                            yDims,
                                            yStrides,
                                            convPrePadding,
                                            convPostPadding,
                                            convStrides,
                                            convDilation,
                                            true,
                                            activMode,
                                            reluLowerClip,
                                            reluUpperClip,
                                            reluLowerClipSlope,
                                            swishBeta,
                                            eluAlpha,
                                            softplusBeta,
                                            dataType);
}

inline flatbuffers::FlatBufferBuilder
    createValidMatmulGraph(const std::vector<int64_t>& aDims = {4, 8},
                           const std::vector<int64_t>& aStrides = {8, 1},
                           const std::vector<int64_t>& bDims = {8, 5},
                           const std::vector<int64_t>& bStrides = {5, 1},
                           const std::vector<int64_t>& cDims = {4, 5},
                           const std::vector<int64_t>& cStrides = {5, 1},
                           hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                           = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "A", dataType, &aStrides, &aDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "B", dataType, &bStrides, &bDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "C", dataType, &cStrides, &cDims));

    auto matmulAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateMatmulAttributes(builder,
                                                                       1, // A tensor uid
                                                                       2, // B tensor uid
                                                                       3 // C tensor uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "matmul",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes,
        matmulAttributes.Union()));

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

inline flatbuffers::FlatBufferBuilder
    createValidLayernormFpropGraph(const std::vector<int64_t>& strides = {588, 196, 14, 1},
                                   const std::vector<int64_t>& dims = {1, 3, 14, 14},
                                   hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                                   = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // For LayerNorm, scale and bias match the normalized dimensions (all dims except batch).
    // E.g., for input [N, C, H, W], normalized dims are [C, H, W].
    const std::vector<int64_t> normalizedDims(dims.begin() + 1, dims.end());
    const std::vector<int64_t> normalizedStrides
        = hipdnn_data_sdk::utilities::generateStrides(normalizedDims);

    // Required tensors
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &normalizedStrides,
        &normalizedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &normalizedStrides,
        &normalizedDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "epsilon",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    auto layernormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateLayernormAttributes(builder,
                                                                          1, // x tensor uid
                                                                          3, // scale tensor uid
                                                                          4, // bias tensor uid
                                                                          5, // epsilon tensor uid
                                                                          2 // y tensor uid
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

inline flatbuffers::FlatBufferBuilder
    createValidRMSNormGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                            const std::vector<int64_t>& dims = {1, 3, 224, 224},
                            hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                            hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
                            = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> derivedDims(dims);
    derivedDims[0] = 1; // Normalize bias/scale on first axis

    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Epsilon (pass-by-value)
    const std::vector<int64_t> passByValueDims = {1};
    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
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
                                                                        4, // epsilon uid
                                                                        2 // y uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm",
        computeDataType,
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

inline flatbuffers::FlatBufferBuilder
    createValidRMSNormBwdGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                               const std::vector<int64_t>& dims = {2, 3, 224, 224},
                               bool hasOptionalAttributes = true,
                               hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
                               = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                               hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
                               = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> derivedDims(dims);
    derivedDims[0] = 1; // Normalize bias/scale on first axis
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    // inv_rms stat shape is [N, 1, 1, 1, ...] when scale is [1, C, H, W ..]
    const std::vector<int64_t> statDims = {dims[0], 1, 1, 1};
    const std::vector<int64_t> statStrides = hipdnn_data_sdk::utilities::generateStrides(
        statDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    // dy (gradient of output)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "dy", inputDataType, &strides, &dims));

    // x (original input)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "x", inputDataType, &strides, &dims));

    // scale
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // dx (gradient of input)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "dx", inputDataType, &strides, &dims));

    // dscale (gradient of scale)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "dscale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // inv_rms (inverse RMS from forward pass)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "inv_rms",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &statStrides,
        &statDims));

    if(hasOptionalAttributes)
    {
        // dbias (gradient of bias)
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                7,
                "dbias",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &derivedStrides,
                &derivedDims));
    }

    auto rmsnormBwdAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateRMSNormBackwardAttributes(
            builder,
            1, // dy uid
            2, // x uid
            3, // scale uid
            6, // inv_rms uid
            4, // dx uid
            5, // dscale uid
            hasOptionalAttributes ? flatbuffers::Optional<int64_t>(7)
                                  : flatbuffers::nullopt // dbias uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "rmsnorm_bwd",
        computeDataType,
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

inline flatbuffers::FlatBufferBuilder
    createValidMatmulBiasActivGraph(const std::vector<int64_t>& aDims = {4, 8},
                                    const std::vector<int64_t>& aStrides = {8, 1},
                                    const std::vector<int64_t>& bDims = {8, 5},
                                    const std::vector<int64_t>& bStrides = {5, 1},
                                    const std::vector<int64_t>& cDims = {4, 5},
                                    const std::vector<int64_t>& cStrides = {5, 1},
                                    bool doBias = true,
                                    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode activMode
                                    = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
                                    std::optional<float> reluLowerClip = std::nullopt,
                                    std::optional<float> reluUpperClip = std::nullopt,
                                    std::optional<float> swishBeta = std::nullopt,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                                    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    int64_t tensorUid = 1;

    const auto aTensorUid = tensorUid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, aTensorUid, "A", dataType, &aStrides, &aDims));

    const auto bTensorUid = tensorUid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, bTensorUid, "B", dataType, &bStrides, &bDims));

    const auto cMatmulTensorUid = tensorUid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, cMatmulTensorUid, "C_matmul", dataType, &cStrides, &cDims));

    int64_t biasTensorUid;
    int64_t cBiasTensorUid;
    if(doBias)
    {
        const auto biasDims = hipdnn_data_sdk::utilities::getDerivedShape(cDims);
        const auto biasStrides = hipdnn_data_sdk::utilities::generateStrides(
            biasDims, hipdnn_data_sdk::utilities::extractStrideOrder(cDims));

        biasTensorUid = tensorUid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, biasTensorUid, "bias", dataType, &biasStrides, &biasDims));
        cBiasTensorUid = tensorUid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, cBiasTensorUid, "C_bias", dataType, &cStrides, &cDims));
    }

    const auto cTensorUid = tensorUid;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, cTensorUid, "C", dataType, &cStrides, &cDims));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Node 0: MatMul
    auto matmulAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateMatmulAttributes(
        builder, aTensorUid, bTensorUid, cMatmulTensorUid);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "matmul",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes,
        matmulAttributes.Union()));

    // Node 1: Bias (optional)
    if(doBias)
    {
        auto biasAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            cMatmulTensorUid,
            biasTensorUid,
            flatbuffers::nullopt,
            cBiasTensorUid);
        nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
            builder,
            "bias",
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
            biasAttributes.Union()));
    }

    // Node 2: Activation (supports RELU_FWD, GELU_FWD, SWISH_FWD, etc.)
    if(activMode != hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::UNSET)
    {
        auto activAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            activMode,
            reluLowerClip,
            reluUpperClip,
            flatbuffers::nullopt, // relu_lower_clip_slope
            flatbuffers::nullopt, // axis_tensor_uid
            doBias ? cBiasTensorUid : cMatmulTensorUid,
            flatbuffers::nullopt, // in_1_tensor_uid
            flatbuffers::nullopt, // in_2_tensor_uid
            cTensorUid,
            swishBeta); // swish_beta
        nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
            builder,
            "activation",
            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
            hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
            activAttributes.Union()));
    }

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

inline flatbuffers::FlatBufferBuilder
    createValidSdpaFwdGraph(const std::vector<int64_t>& qDims = {2, 8, 16, 64},
                            const std::vector<int64_t>& qStrides = {8192, 1024, 64, 1},
                            const std::vector<int64_t>& kDims = {2, 8, 16, 64},
                            const std::vector<int64_t>& kStrides = {8192, 1024, 64, 1},
                            const std::vector<int64_t>& vDims = {2, 8, 16, 64},
                            const std::vector<int64_t>& vStrides = {8192, 1024, 64, 1},
                            const std::vector<int64_t>& oDims = {2, 8, 16, 64},
                            const std::vector<int64_t>& oStrides = {8192, 1024, 64, 1},
                            hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                            = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                            bool withAttnMask = false,
                            bool withScale = false,
                            bool withStats = false,
                            bool alibiMask = false,
                            bool paddingMask = false,
                            bool causalMask = false)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    int64_t uid = 1;

    const auto qUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, qUid, "q", dataType, &qStrides, &qDims));

    const auto kUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, kUid, "k", dataType, &kStrides, &kDims));

    const auto vUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, vUid, "v", dataType, &vStrides, &vDims));

    const auto oUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, oUid, "o", dataType, &oStrides, &oDims));

    flatbuffers::Optional<int64_t> attnMaskUid = flatbuffers::nullopt;
    if(withAttnMask)
    {
        // attn_mask: [batch, num_heads, seq_q, seq_kv]
        const std::vector<int64_t> attnMaskDims = {qDims[0], qDims[1], qDims[2], kDims[2]};
        const std::vector<int64_t> attnMaskStrides
            = {qDims[1] * qDims[2] * kDims[2], qDims[2] * kDims[2], kDims[2], 1};
        const auto maskUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, maskUid, "attn_mask", dataType, &attnMaskStrides, &attnMaskDims));
        attnMaskUid = flatbuffers::Optional<int64_t>(maskUid);
    }

    flatbuffers::Optional<int64_t> scaleUid = flatbuffers::nullopt;
    if(withScale)
    {
        const std::vector<int64_t> passByValueDims = {1};
        const hipdnn_flatbuffers_sdk::data_objects::Float32Value scaleVal(1.0f);
        const auto sUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                sUid,
                "scale",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &passByValueDims,
                &passByValueDims,
                false,
                hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
                builder.CreateStruct(scaleVal).Union()));
        scaleUid = flatbuffers::Optional<int64_t>(sUid);
    }

    flatbuffers::Optional<int64_t> statsUid = flatbuffers::nullopt;
    if(withStats)
    {
        // stats: [batch, num_heads, seq_q, 1]
        const std::vector<int64_t> statsDims = {qDims[0], qDims[1], qDims[2], 1};
        const std::vector<int64_t> statsStrides = {qDims[1] * qDims[2], qDims[2], 1, 1};
        const auto stUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                stUid,
                "stats",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &statsStrides,
                &statsDims));
        statsUid = flatbuffers::Optional<int64_t>(stUid);
    }

    auto sdpaAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateSdpaAttributes(
        builder,
        qUid,
        kUid,
        vUid,
        oUid,
        attnMaskUid,
        scaleUid,
        flatbuffers::nullopt, // seq_len_q_tensor_uid
        flatbuffers::nullopt, // seq_len_kv_tensor_uid
        flatbuffers::nullopt, // seed_tensor_uid
        flatbuffers::nullopt, // offset_tensor_uid
        flatbuffers::nullopt, // dropout_mask_tensor_uid
        flatbuffers::nullopt, // dropout_scale_tensor_uid
        flatbuffers::nullopt, // page_table_k_tensor_uid
        flatbuffers::nullopt, // page_table_v_tensor_uid
        flatbuffers::nullopt, // block_mask_tensor_uid
        flatbuffers::nullopt, // sink_token_tensor_uid
        flatbuffers::nullopt, // descale_q_tensor_uid
        flatbuffers::nullopt, // descale_k_tensor_uid
        flatbuffers::nullopt, // descale_v_tensor_uid
        flatbuffers::nullopt, // descale_s_tensor_uid
        flatbuffers::nullopt, // scale_s_tensor_uid
        flatbuffers::nullopt, // scale_o_tensor_uid
        statsUid,
        flatbuffers::nullopt, // max_tensor_uid
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        alibiMask,
        paddingMask,
        causalMask);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "sdpa_fwd",
        dataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaAttributes,
        sdpaAttributes.Union()));

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

inline flatbuffers::FlatBufferBuilder createValidBlockScaleQuantizeGraph(
    const std::vector<int64_t>& strides = {65536, 1024, 32, 1},
    const std::vector<int64_t>& dims = {2, 64, 32, 32},
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    flatbuffers::Optional<int64_t> axis = 1)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> scaleDims = {2, 2, 32, 32};
    const std::vector<int64_t> scaleStrides = {2048, 1024, 32, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", inputDataType, &scaleStrides, &scaleDims));

    auto blockScaleQuantizeAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBlockScaleQuantizeAttributes(
            builder,
            1, // x uid
            2, // y uid
            3, // scale uid
            32, // block_size
            axis,
            false // transpose
        );

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "block_scale_quantize",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleQuantizeAttributes,
        blockScaleQuantizeAttributes.Union());
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

inline flatbuffers::FlatBufferBuilder createValidEngineDetails(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineDetailsOffset
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, engineId);
    builder.Finish(engineDetailsOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder createValidBlockScaleDequantizeGraph(
    const std::vector<int64_t>& strides = {65536, 2048, 64, 1},
    const std::vector<int64_t>& dims = {2, 32, 32, 64},
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    bool isNegativeScale = false,
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> scaleDims = {2, 32, 32, 2};
    const std::vector<int64_t> scaleStrides = {2048, 64, 2, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "scale", scaleDataType, &scaleStrides, &scaleDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "y", outputDataType, &strides, &dims));

    const std::vector<int32_t> blockSize = {32};
    auto blockSizeVector = builder.CreateVector(blockSize);

    auto blockScaleDequantizeAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBlockScaleDequantizeAttributes(
            builder,
            1, // x uid
            2, // scale uid
            3, // y uid
            blockSizeVector,
            isNegativeScale);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "block_scale_dequantize",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes,
        blockScaleDequantizeAttributes.Union());
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

inline flatbuffers::FlatBufferBuilder createValidBlockScaleDequantizeMxGraph(
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> dims = {1, 4};
    const std::vector<int64_t> strides = {4, 1};
    const std::vector<int64_t> scaleDims = {1, 2};
    const std::vector<int64_t> scaleStrides = {2, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "scale", scaleDataType, &scaleStrides, &scaleDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "y", outputDataType, &strides, &dims));

    const std::vector<int32_t> blockSize = {2};
    auto blockSizeVector = builder.CreateVector(blockSize);

    auto blockScaleDequantizeAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBlockScaleDequantizeAttributes(
            builder, 1, 2, 3, blockSizeVector, false);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "block_scale_dequantize",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes,
        blockScaleDequantizeAttributes.Union());
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

inline flatbuffers::FlatBufferBuilder
    createValidSdpaBwdGraph(const std::vector<int64_t>& qDims = {2, 8, 16, 64},
                            const std::vector<int64_t>& qStrides = {8192, 1024, 64, 1},
                            const std::vector<int64_t>& kDims = {2, 8, 16, 64},
                            const std::vector<int64_t>& kStrides = {8192, 1024, 64, 1},
                            const std::vector<int64_t>& vDims = {2, 8, 16, 64},
                            const std::vector<int64_t>& vStrides = {8192, 1024, 64, 1},
                            const std::vector<int64_t>& oDims = {2, 8, 16, 64},
                            const std::vector<int64_t>& oStrides = {8192, 1024, 64, 1},
                            hipdnn_flatbuffers_sdk::data_objects::DataType dataType
                            = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                            bool withScale = false,
                            bool alibiMask = false,
                            bool paddingMask = false,
                            bool causalMask = false)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    int64_t uid = 1;

    const auto qUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, qUid, "q", dataType, &qStrides, &qDims));

    const auto kUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, kUid, "k", dataType, &kStrides, &kDims));

    const auto vUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, vUid, "v", dataType, &vStrides, &vDims));

    const auto oUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, oUid, "o", dataType, &oStrides, &oDims));

    // dO has same shape as O
    const auto doUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, doUid, "do", dataType, &oStrides, &oDims));

    // Stats: [batch, num_heads, seq_q, 1]
    const std::vector<int64_t> statsDims = {qDims[0], qDims[1], qDims[2], 1};
    const std::vector<int64_t> statsStrides = {qDims[1] * qDims[2], qDims[2], 1, 1};
    const auto statsUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        statsUid,
        "stats",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &statsStrides,
        &statsDims));

    // Output gradient tensors (same shapes as Q, K, V)
    const auto dqUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, dqUid, "dq", dataType, &qStrides, &qDims));

    const auto dkUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, dkUid, "dk", dataType, &kStrides, &kDims));

    const auto dvUid = uid++;
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, dvUid, "dv", dataType, &vStrides, &vDims));

    flatbuffers::Optional<int64_t> scaleUid = flatbuffers::nullopt;
    if(withScale)
    {
        const std::vector<int64_t> passByValueDims = {1};
        const hipdnn_flatbuffers_sdk::data_objects::Float32Value scaleVal(1.0f);
        const auto sUid = uid++;
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                sUid,
                "scale",
                hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                &passByValueDims,
                &passByValueDims,
                false,
                hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
                builder.CreateStruct(scaleVal).Union()));
        scaleUid = flatbuffers::Optional<int64_t>(sUid);
    }

    auto sdpaBwdAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateSdpaBackwardAttributes(
        builder,
        qUid,
        kUid,
        vUid,
        oUid,
        doUid,
        statsUid,
        dqUid,
        dkUid,
        dvUid,
        scaleUid,
        flatbuffers::nullopt, // attn_mask_tensor_uid
        flatbuffers::nullopt, // seq_len_q_tensor_uid
        flatbuffers::nullopt, // seq_len_kv_tensor_uid
        flatbuffers::nullopt, // seed_tensor_uid
        flatbuffers::nullopt, // offset_tensor_uid
        flatbuffers::nullopt, // dropout_mask_tensor_uid
        flatbuffers::nullopt, // dropout_scale_tensor_uid
        flatbuffers::nullopt, // dropout_scale_inv_tensor_uid
        flatbuffers::nullopt, // dbias_tensor_uid
        alibiMask,
        paddingMask,
        causalMask);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "sdpa_bwd",
        dataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaBackwardAttributes,
        sdpaBwdAttributes.Union()));

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

inline flatbuffers::FlatBufferBuilder createValidCustomOpGraph()
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> dims = {4, 8};
    const std::vector<int64_t> strides = {8, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "input_0",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "output_0",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    const std::vector<int64_t> inputUids = {1};
    const std::vector<int64_t> outputUids = {2};
    const std::vector<uint8_t> data = {0x01, 0x02};

    auto customOpAttr = hipdnn_flatbuffers_sdk::data_objects::CreateCustomOpAttributesDirect(
        builder, "my_plugin_op", &inputUids, &outputUids, &data);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "custom_op",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::CustomOpAttributes,
        customOpAttr.Union()));

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

inline flatbuffers::FlatBufferBuilder createValidReductionGraph()
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> inDims = {4, 8};
    const std::vector<int64_t> inStrides = {8, 1};
    const std::vector<int64_t> outDims = {1, 8};
    const std::vector<int64_t> outStrides = {8, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "input",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &inStrides,
        &inDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "output",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &outStrides,
        &outDims));

    auto reductionAttr = hipdnn_flatbuffers_sdk::data_objects::CreateReductionAttributes(
        builder, hipdnn_flatbuffers_sdk::data_objects::ReductionMode::ADD, 1, 2);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "reduction",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ReductionAttributes,
        reductionAttr.Union()));

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

inline flatbuffers::FlatBufferBuilder createValidResampleFwdGraph()
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> xDims = {1, 1, 4, 4};
    const std::vector<int64_t> xStrides = {16, 16, 4, 1};
    const std::vector<int64_t> yDims = {1, 1, 2, 2};
    const std::vector<int64_t> yStrides = {4, 4, 2, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &yStrides, &yDims));

    const std::vector<int64_t> prePadding = {0, 0};
    const std::vector<int64_t> postPadding = {0, 0};
    const std::vector<int64_t> stride = {2, 2};
    const std::vector<int64_t> window = {2, 2};

    auto resampleAttr = hipdnn_flatbuffers_sdk::data_objects::CreateResampleFwdAttributesDirect(
        builder,
        1,
        2,
        ::flatbuffers::nullopt,
        &prePadding,
        &postPadding,
        &stride,
        &window,
        hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
        hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD);

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

inline flatbuffers::FlatBufferBuilder createValidEngineConfig(int64_t configId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineConfigOffset
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(builder, configId);
    builder.Finish(engineConfigOffset);
    return builder;
}

}
