// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

namespace hipdnn_test_sdk::utilities
{

inline flatbuffers::FlatBufferBuilder createEmptyValidGraph()
{
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder createValidBatchnormInferenceGraph(
    const std::vector<int64_t>& strides = {1, 3, 224, 224},
    const std::vector<int64_t>& dims = {1, 3, 224, 224},
    hipdnn_sdk::data_objects::DataType inputDataType = hipdnn_sdk::data_objects::DataType::FLOAT,
    hipdnn_sdk::data_objects::DataType computeDataType = hipdnn_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedStrides = hipdnn_sdk::utilities::getDerivedShape(strides);
    std::vector<int64_t> derivedDims = hipdnn_sdk::utilities::getDerivedShape(dims);

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "est_mean",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "est_variance",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    auto bnormAttributes
        = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(builder,
                                                                       1, // x uid
                                                                       5, // mean uid
                                                                       6, // inv_variance uid
                                                                       3, // scale uid
                                                                       4, // bias uid
                                                                       2 // y uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm",
        computeDataType,
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder createValidBatchnormWithVarianceInferenceGraph(
    const std::vector<int64_t>& strides = {1, 3, 224, 224},
    const std::vector<int64_t>& dims = {1, 3, 224, 224},
    hipdnn_sdk::data_objects::DataType inputDataType = hipdnn_sdk::data_objects::DataType::FLOAT,
    hipdnn_sdk::data_objects::DataType computeDataType = hipdnn_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedStrides = hipdnn_sdk::utilities::getDerivedShape(strides);
    std::vector<int64_t> derivedDims = hipdnn_sdk::utilities::getDerivedShape(dims);

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "est_mean",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "variance",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    auto bnormAttributes
        = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributesVarianceExt(builder,
                                                                                  1, // x uid
                                                                                  5, // mean uid
                                                                                  6, // variance uid
                                                                                  3, // scale uid
                                                                                  4, // bias uid
                                                                                  2 // y uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnormWithVariance",
        computeDataType,
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidBatchnormBwdGraph(const std::vector<int64_t>& strides = {1, 3, 224, 224},
                                 const std::vector<int64_t>& dims = {1, 3, 224, 224},
                                 bool hasOptionalAttributes = true,
                                 hipdnn_sdk::data_objects::DataType inputDataType
                                 = hipdnn_sdk::data_objects::DataType::FLOAT,
                                 hipdnn_sdk::data_objects::DataType scaleBiasDataType
                                 = hipdnn_sdk::data_objects::DataType::FLOAT,
                                 hipdnn_sdk::data_objects::DataType meanVarianceDataType
                                 = hipdnn_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedDims = hipdnn_sdk::utilities::getDerivedShape(dims);
    std::vector<int64_t> derivedStrides = hipdnn_sdk::utilities::generateStrides(derivedDims);

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "dy", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "dx", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "scale", scaleBiasDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "dscale", scaleBiasDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "dbias", scaleBiasDataType, &derivedStrides, &derivedDims));

    if(hasOptionalAttributes)
    {
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 7, "mean", meanVarianceDataType, &derivedStrides, &derivedDims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 8, "inv_variance", meanVarianceDataType, &derivedStrides, &derivedDims));
    }

    auto bnormAttributes = hipdnn_sdk::data_objects::CreateBatchnormBackwardAttributes(
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

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_bwd",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidBatchnormFwdInferActGraph(const std::vector<int64_t>& strides = {1, 3, 224, 224},
                                         const std::vector<int64_t>& dims = {1, 3, 224, 224},
                                         hipdnn_sdk::data_objects::DataType inputDataType
                                         = hipdnn_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_sdk::data_objects::DataType intermediateDataType
                                         = hipdnn_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedDims = hipdnn_sdk::utilities::getDerivedShape(dims);
    std::vector<int64_t> derivedStrides = hipdnn_sdk::utilities::generateStrides(derivedDims);

    // inputs
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "scale", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "bias", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "mean", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "inv_variance", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "y", inputDataType, &strides, &dims, true));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "Dy", inputDataType, &strides, &dims, false)); // is_virtual = true

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm Inference
    auto bnInfAttributes = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder,
        1, // x_tensor_uid
        4, // mean_tensor_uid
        5, // inv_variance_tensor_uid
        2, // scale_tensor_uid
        3, // bias_tensor_uid
        6 // y_tensor_uid (BN_Y - virtual)
    );

    nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference",
        intermediateDataType,
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttributes.Union()));

    // Node 1: Pointwise (RELU_FWD)
    auto pointwiseAttributes = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD,
        std::nullopt, // relu_lower_clip
        std::nullopt, // relu_upper_clip
        std::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        6, // in_0_tensor_uid (BN_Y)
        flatbuffers::nullopt, // in_1_tensor_uid
        flatbuffers::nullopt, // in_2_tensor_uid
        7 // out_0_tensor_uid (Dy - not virtual)
    );

    nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "relu_fwd",
        intermediateDataType,
        hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttributes.Union()));

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidBatchnormInferActBwdGraph(const std::vector<int64_t>& strides = {1, 3, 224, 224},
                                         const std::vector<int64_t>& dims = {1, 3, 224, 224},
                                         bool hasOptionalAttributes = true,
                                         hipdnn_sdk::data_objects::DataType inputDataType
                                         = hipdnn_sdk::data_objects::DataType::FLOAT,
                                         hipdnn_sdk::data_objects::DataType intermediateDataType
                                         = hipdnn_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedDims = hipdnn_sdk::utilities::getDerivedShape(dims);
    std::vector<int64_t> derivedStrides = hipdnn_sdk::utilities::generateStrides(derivedDims);

    // inputs
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "scale", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "bias", intermediateDataType, &derivedStrides, &derivedDims));

    if(hasOptionalAttributes)
    {
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 4, "mean", intermediateDataType, &derivedStrides, &derivedDims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 5, "inv_variance", intermediateDataType, &derivedStrides, &derivedDims));
    }

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "dy", inputDataType, &strides, &dims));

    // output tensors
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "dx", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 8, "dscale", intermediateDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 9, "dbias", intermediateDataType, &derivedStrides, &derivedDims));

    // virtual tensors
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 10, "BN_Y", inputDataType, &strides, &dims, true)); // is_virtual = true

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 11, "DX_drelu", inputDataType, &strides, &dims, true)); // is_virtual = true

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm Inference
    auto bnInfAttributes = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder,
        1, // x_tensor_uid
        hasOptionalAttributes ? 4 : 0, // mean_tensor_uid
        hasOptionalAttributes ? 5 : 0, // inv_variance_tensor_uid
        2, // scale_tensor_uid
        3, // bias_tensor_uid
        10 // y_tensor_uid (BN_Y - virtual)
    );

    nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference",
        intermediateDataType,
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttributes.Union()));

    // Node 1: Pointwise (RELU_BWD)
    auto pointwiseAttributes = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD,
        std::nullopt, // relu_lower_clip
        std::nullopt, // relu_upper_clip
        std::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        10, // in_0_tensor_uid (BN_Y)
        flatbuffers::Optional<int64_t>(6), // in_1_tensor_uid (dy)
        flatbuffers::nullopt, // in_2_tensor_uid
        11 // out_0_tensor_uid (DX_drelu - virtual)
    );

    nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "relu_bwd",
        intermediateDataType,
        hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttributes.Union()));

    // Node 2: Batchnorm Backward
    auto bnBwdAttributes = hipdnn_sdk::data_objects::CreateBatchnormBackwardAttributes(
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

    nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_backward",
        intermediateDataType,
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnBwdAttributes.Union()));

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidBatchnormFwdTrainingGraph(const std::vector<int64_t>& strides = {1, 3, 14, 14},
                                         const std::vector<int64_t>& dims = {1, 3, 14, 14},
                                         bool withMeanVariance = true)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedStrides = hipdnn_sdk::utilities::getDerivedShape(strides);
    std::vector<int64_t> derivedDims = hipdnn_sdk::utilities::getDerivedShape(dims);

    // Required tensors
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", hipdnn_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Epsilon (pass-by-value)
    std::vector<int64_t> passByValueDims = {1};
    hipdnn_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "epsilon",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &passByValueDims,
        &passByValueDims,
        false,
        hipdnn_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    flatbuffers::Optional<int64_t> meanUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> invVarUid = flatbuffers::nullopt;

    if(withMeanVariance)
    {
        // Optional mean/variance output tensors
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            6,
            "mean",
            hipdnn_sdk::data_objects::DataType::FLOAT,
            &derivedStrides,
            &derivedDims));
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            7,
            "inv_variance",
            hipdnn_sdk::data_objects::DataType::FLOAT,
            &derivedStrides,
            &derivedDims));
        meanUid = flatbuffers::Optional<int64_t>(6);
        invVarUid = flatbuffers::Optional<int64_t>(7);
    }

    auto bnormAttributes = hipdnn_sdk::data_objects::CreateBatchnormAttributes(
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

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
                                                      &tensorAttributes,
                                                      &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidBatchnormFwdTrainingActivGraph(bool withMeanVariance = true,
                                              bool withRunningStats = false,
                                              hipdnn_sdk::data_objects::PointwiseMode activMode
                                              = hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD,
                                              const std::vector<int64_t>& strides = {1, 3, 14, 14},
                                              const std::vector<int64_t>& dims = {1, 3, 14, 14})
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedStrides = hipdnn_sdk::utilities::getDerivedShape(strides);
    std::vector<int64_t> derivedDims = hipdnn_sdk::utilities::getDerivedShape(dims);

    int64_t uid = 1;

    // Required tensors
    const auto xTensorUid = uid++;
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, xTensorUid, "x", hipdnn_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    const auto scaleTensorUid = uid++;
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        scaleTensorUid,
        "scale",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const auto biasTensorUid = uid++;
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        biasTensorUid,
        "bias",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Epsilon (pass-by-value)
    const auto epsilonTensorUid = uid++;
    hipdnn_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributes(
        builder,
        epsilonTensorUid,
        builder.CreateString("epsilon"),
        hipdnn_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    // BN output (virtual - intermediate between BN and activation)
    const auto yBnTensorUid = uid++;
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        yBnTensorUid,
        "y_bn",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true)); // virtual

    // Final activation output (non-virtual)
    const auto yTensorUid = uid++;
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yTensorUid, "y", hipdnn_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    flatbuffers::Optional<int64_t> meanUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> invVarUid = flatbuffers::nullopt;

    if(withMeanVariance)
    {
        const auto meanTensorUid = uid++;
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            meanTensorUid,
            "mean",
            hipdnn_sdk::data_objects::DataType::FLOAT,
            &derivedStrides,
            &derivedDims));
        meanUid = flatbuffers::Optional<int64_t>(meanTensorUid);

        const auto invVarTensorUid = uid++;
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            invVarTensorUid,
            "inv_variance",
            hipdnn_sdk::data_objects::DataType::FLOAT,
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
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            prevMeanUid,
            "prev_running_mean",
            hipdnn_sdk::data_objects::DataType::FLOAT,
            &derivedStrides,
            &derivedDims));
        prevRunningMeanUid = flatbuffers::Optional<int64_t>(prevMeanUid);

        const auto prevVarUid = uid++;
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            prevVarUid,
            "prev_running_variance",
            hipdnn_sdk::data_objects::DataType::FLOAT,
            &derivedStrides,
            &derivedDims));
        prevRunningVarUid = flatbuffers::Optional<int64_t>(prevVarUid);

        const auto momUid = uid++;
        hipdnn_sdk::data_objects::Float32Value momentumVal(0.1f);
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributes(
            builder,
            momUid,
            builder.CreateString("momentum"),
            hipdnn_sdk::data_objects::DataType::FLOAT,
            0,
            0,
            false,
            hipdnn_sdk::data_objects::TensorValue::Float32Value,
            builder.CreateStruct(momentumVal).Union()));
        momentumUid = flatbuffers::Optional<int64_t>(momUid);

        const auto nextMeanUid = uid++;
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            nextMeanUid,
            "next_running_mean",
            hipdnn_sdk::data_objects::DataType::FLOAT,
            &derivedStrides,
            &derivedDims));
        nextRunningMeanUid = flatbuffers::Optional<int64_t>(nextMeanUid);

        const auto nextVarUid = uid++;
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            nextVarUid,
            "next_running_variance",
            hipdnn_sdk::data_objects::DataType::FLOAT,
            &derivedStrides,
            &derivedDims));
        nextRunningVarUid = flatbuffers::Optional<int64_t>(nextVarUid);
    }

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm forward training
    auto bnormAttributes
        = hipdnn_sdk::data_objects::CreateBatchnormAttributes(builder,
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

    nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    // Node 1: Activation
    auto activAttributes = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
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

    nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "activation",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        activAttributes.Union()));

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
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
                            hipdnn_sdk::data_objects::DataType dataType
                            = hipdnn_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", dataType, &xStrides, &xDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "w", dataType, &wStrides, &wDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "y", dataType, &yStrides, &yDims));

    auto convAttributes = hipdnn_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        1, // x tensor uid
        2, // w tensor uid
        3, // y tensor uid
        &convPrePadding,
        &convPostPadding,
        &convStrides,
        &convDilation,
        hipdnn_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_fwd",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes,
        convAttributes.Union());
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
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
                            hipdnn_sdk::data_objects::DataType dataType
                            = hipdnn_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "dy", dataType, &dyStrides, &dyDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "w", dataType, &wStrides, &wDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "dx", dataType, &dxStrides, &dxDims));

    auto convAttributes = hipdnn_sdk::data_objects::CreateConvolutionBwdAttributesDirect(
        builder,
        1, // dy tensor uid
        2, // w tensor uid
        3, // dx tensor uid
        &convPrePadding,
        &convPostPadding,
        &convStrides,
        &convDilation,
        hipdnn_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_bwd",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes,
        convAttributes.Union());
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
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
                            hipdnn_sdk::data_objects::DataType dataType
                            = hipdnn_sdk::data_objects::DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", dataType, &xStrides, &xDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "dy", dataType, &dyStrides, &dyDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "dw", dataType, &dwStrides, &dwDims));

    auto convAttributes = hipdnn_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
        builder,
        1, // x tensor uid
        2, // dy tensor uid
        3, // w tensor uid
        &convPrePadding,
        &convPostPadding,
        &convStrides,
        &convDilation,
        hipdnn_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_wrw",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        hipdnn_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes,
        convAttributes.Union());
    nodes.push_back(node);

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
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

    std::vector<flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto pointwiseNode = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_sdk::data_objects::PointwiseMode::DIV, // operation
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

    nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "hipdnn_sdk::data_objects::Node",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseNode.Union()));

    std::array tensorNames = {"axis", "in_0", "in_1", "in_2", "out_0"};
    std::vector<flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensors;
    tensors.reserve(tensorNames.size());
    int64_t tensorUid = 0;
    std::vector<int64_t> dims = {1, 2, 3, 4};
    std::vector<int64_t> strides = {5, 6, 7, 8};
    for(auto name : tensorNames)
    {
        tensors.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder,
            tensorUid++,
            name,
            hipdnn_sdk::data_objects::DataType::UINT8,
            &strides,
            &dims,
            false));
    }

    auto graph
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "PointwiseGraph",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::HALF,
                                                      hipdnn_sdk::data_objects::DataType::BFLOAT16,
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
                                     hipdnn_sdk::data_objects::PointwiseMode activMode,
                                     std::optional<float> reluLowerClip,
                                     std::optional<float> reluUpperClip,
                                     std::optional<float> reluLowerClipSlope,
                                     std::optional<float> swishBeta,
                                     std::optional<float> eluAlpha,
                                     std::optional<float> softplusBeta,
                                     hipdnn_sdk::data_objects::DataType dataType)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    int64_t tensorUid = 1;

    const auto xTensorUid = tensorUid++;
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, xTensorUid, "x", dataType, &xStrides, &xDims));

    const auto wTensorUid = tensorUid++;
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, wTensorUid, "w", dataType, &wStrides, &wDims));

    // Virtual y_conv tensor
    const auto yConvTensorUid = tensorUid++;
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yConvTensorUid, "y_conv", dataType, &yStrides, &yDims, true));

    int64_t biasTensorUid;
    int64_t yBiasTensorUid;
    if(doBias)
    {
        const auto biasDims = hipdnn_sdk::utilities::getDerivedShape(yDims);
        const auto biasStrides = hipdnn_sdk::utilities::generateStrides(
            biasDims, hipdnn_sdk::utilities::extractStrideOrder(yDims));

        biasTensorUid = tensorUid++;
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, biasTensorUid, "bias", dataType, &biasStrides, &biasDims));
        // Virtual y_bias tensor
        yBiasTensorUid = tensorUid++;
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, yBiasTensorUid, "y_bias", dataType, &yStrides, &yDims, true));
    }

    const auto yTensorUid = tensorUid;
    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, yTensorUid, "y", dataType, &yStrides, &yDims));

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;

    auto convAttributes = hipdnn_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        xTensorUid,
        wTensorUid,
        yConvTensorUid,
        &convPrePadding,
        &convPostPadding,
        &convStrides,
        &convDilation,
        hipdnn_sdk::data_objects::ConvMode::CROSS_CORRELATION);
    nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_fwd",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes,
        convAttributes.Union()));

    if(doBias)
    {
        auto biasAttributes = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            hipdnn_sdk::data_objects::PointwiseMode::ADD,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            flatbuffers::nullopt,
            yConvTensorUid,
            biasTensorUid,
            flatbuffers::nullopt,
            yBiasTensorUid);
        nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
            builder,
            "bias",
            hipdnn_sdk::data_objects::DataType::FLOAT,
            hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes,
            biasAttributes.Union()));
    }

    auto activAttributes = hipdnn_sdk::data_objects::CreatePointwiseAttributes(
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
    nodes.push_back(hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "activ",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        activAttributes.Union()));

    auto graphOffset
        = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                      "test",
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
                                                      hipdnn_sdk::data_objects::DataType::FLOAT,
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
                                 hipdnn_sdk::data_objects::PointwiseMode activMode
                                 = hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD,
                                 std::optional<float> reluLowerClip = std::nullopt,
                                 std::optional<float> reluUpperClip = std::nullopt,
                                 std::optional<float> reluLowerClipSlope = std::nullopt,
                                 std::optional<float> swishBeta = std::nullopt,
                                 std::optional<float> eluAlpha = std::nullopt,
                                 std::optional<float> softplusBeta = std::nullopt,
                                 hipdnn_sdk::data_objects::DataType dataType
                                 = hipdnn_sdk::data_objects::DataType::FLOAT)
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

inline flatbuffers::FlatBufferBuilder
    createValidConvFwdBiasActivGraph(const std::vector<int64_t>& xDims = {4, 4, 4, 4},
                                     const std::vector<int64_t>& xStrides = {64, 16, 4, 1},
                                     const std::vector<int64_t>& wDims = {4, 4, 1, 1},
                                     const std::vector<int64_t>& wStrides = {4, 1, 1, 1},
                                     const std::vector<int64_t>& yDims = {4, 4, 4, 4},
                                     const std::vector<int64_t>& yStrides = {64, 16, 4, 1},
                                     const std::vector<int64_t>& convPrePadding = {0, 0},
                                     const std::vector<int64_t>& convPostPadding = {0, 0},
                                     const std::vector<int64_t>& convStrides = {1, 1},
                                     const std::vector<int64_t>& convDilation = {1, 1},
                                     hipdnn_sdk::data_objects::PointwiseMode activMode
                                     = hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD,
                                     std::optional<float> reluLowerClip = std::nullopt,
                                     std::optional<float> reluUpperClip = std::nullopt,
                                     std::optional<float> reluLowerClipSlope = std::nullopt,
                                     std::optional<float> swishBeta = std::nullopt,
                                     std::optional<float> eluAlpha = std::nullopt,
                                     std::optional<float> softplusBeta = std::nullopt,
                                     hipdnn_sdk::data_objects::DataType dataType
                                     = hipdnn_sdk::data_objects::DataType::FLOAT)
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

inline flatbuffers::FlatBufferBuilder createValidEngineDetails(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineDetailsOffset = hipdnn_sdk::data_objects::CreateEngineDetails(builder, engineId);
    builder.Finish(engineDetailsOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder createValidEngineConfig(int64_t configId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineConfigOffset = hipdnn_sdk::data_objects::CreateEngineConfig(builder, configId);
    builder.Finish(engineConfigOffset);
    return builder;
}

}
