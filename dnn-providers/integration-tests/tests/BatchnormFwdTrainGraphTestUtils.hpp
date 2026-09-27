// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <vector>

#include "ScalarTestUtils.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;

inline flatbuffers::FlatBufferBuilder
    createBatchnormFwdTrainGraph(const std::vector<int64_t>& dims = {1, 3, 14, 14},
                                 const std::vector<int64_t>& strides = {588, 196, 14, 1},
                                 bool withMeanVariance = false,
                                 bool withRunningStats = false,
                                 const DataType inputDataType = DataType::FLOAT,
                                 const DataType scaleBiasDataType = DataType::FLOAT,
                                 const DataType meanVarianceDataType = DataType::FLOAT,
                                 const DataType outputDataType = DataType::FLOAT,
                                 const DataType computeDataType = DataType::FLOAT)
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
        builder, 2, "y", outputDataType, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "scale", scaleBiasDataType, &derivedStrides, &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "bias", scaleBiasDataType, &derivedStrides, &derivedDims));
    tensorAttributes.push_back(
        createScalarTensorAttributes(builder, 5, 1.0e-05, computeDataType, "epsilon"));
    tensorAttributes.push_back(
        createScalarTensorAttributes(builder, 6, 0.90, computeDataType, "momentum"));

    flatbuffers::Optional<int64_t> meanUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> invVarUid = flatbuffers::nullopt;
    if(withMeanVariance)
    {
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 7, "mean", meanVarianceDataType, &derivedStrides, &derivedDims));
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder, 8, "inv_variance", meanVarianceDataType, &derivedStrides, &derivedDims));
        meanUid = 7;
        invVarUid = 8;
    }

    flatbuffers::Optional<int64_t> prevMeanUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> prevVarianceUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> nextMeanUid = flatbuffers::nullopt;
    flatbuffers::Optional<int64_t> nextVarianceUid = flatbuffers::nullopt;
    if(withRunningStats)
    {
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                               9,
                                                                               "prev_running_mean",
                                                                               meanVarianceDataType,
                                                                               &derivedStrides,
                                                                               &derivedDims));
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                10,
                "prev_running_variance",
                meanVarianceDataType,
                &derivedStrides,
                &derivedDims));
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(builder,
                                                                               11,
                                                                               "next_running_mean",
                                                                               meanVarianceDataType,
                                                                               &derivedStrides,
                                                                               &derivedDims));
        tensorAttributes.push_back(
            hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
                builder,
                12,
                "next_running_variance",
                meanVarianceDataType,
                &derivedStrides,
                &derivedDims));
        prevMeanUid = 9;
        prevVarianceUid = 10;
        nextMeanUid = 11;
        nextVarianceUid = 12;
    }

    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          prevMeanUid,
                                                                          prevVarianceUid,
                                                                          6,
                                                                          2,
                                                                          meanUid,
                                                                          invVarUid,
                                                                          nextMeanUid,
                                                                          nextVarianceUid);

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_training",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(builder,
                                                                               "test",
                                                                               computeDataType,
                                                                               scaleBiasDataType,
                                                                               inputDataType,
                                                                               &tensorAttributes,
                                                                               &nodes,
                                                                               flatbuffers::nullopt,
                                                                               false);
    builder.Finish(graphOffset);
    return builder;
}

} // namespace hipdnn_integration_tests::test_utils
