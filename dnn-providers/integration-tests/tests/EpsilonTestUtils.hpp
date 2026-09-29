// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

using namespace hipdnn_flatbuffers_sdk::data_objects;

inline flatbuffers::Offset<TensorAttributes>
    createEpsilonTensorAttributes(flatbuffers::FlatBufferBuilder& builder,
                                  int64_t epsilonUid,
                                  double epsilon,
                                  DataType epsilonDataType)
{
    const std::vector<int64_t> epsilonDimsStrides = {1};
    switch(epsilonDataType)
    {
    case DataType::FLOAT:
        return CreateTensorAttributesDirect(
            builder,
            epsilonUid,
            "epsilon",
            DataType::FLOAT,
            &epsilonDimsStrides,
            &epsilonDimsStrides,
            false,
            TensorValue::Float32Value,
            builder.CreateStruct(Float32Value(static_cast<float>(epsilon))).Union());
    case DataType::HALF:
        return CreateTensorAttributesDirect(
            builder,
            epsilonUid,
            "epsilon",
            DataType::HALF,
            &epsilonDimsStrides,
            &epsilonDimsStrides,
            false,
            TensorValue::Float16Value,
            builder.CreateStruct(Float16Value(static_cast<float>(epsilon))).Union());
    case DataType::BFLOAT16:
        return CreateTensorAttributesDirect(
            builder,
            epsilonUid,
            "epsilon",
            DataType::BFLOAT16,
            &epsilonDimsStrides,
            &epsilonDimsStrides,
            false,
            TensorValue::BFloat16Value,
            builder.CreateStruct(BFloat16Value(static_cast<float>(epsilon))).Union());
    case DataType::DOUBLE:
        return CreateTensorAttributesDirect(builder,
                                            epsilonUid,
                                            "epsilon",
                                            DataType::DOUBLE,
                                            &epsilonDimsStrides,
                                            &epsilonDimsStrides,
                                            false,
                                            TensorValue::Float64Value,
                                            builder.CreateStruct(Float64Value(epsilon)).Union());
    default:
        throw std::runtime_error("Unsupported epsilon data type");
    }
}

} // namespace hipdnn_integration_tests::test_utils
