// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

using namespace hipdnn_flatbuffers_sdk::data_objects;

inline flatbuffers::Offset<TensorAttributes>
    createScalarTensorAttributes(flatbuffers::FlatBufferBuilder& builder,
                                 int64_t uid,
                                 double value,
                                 DataType dataType,
                                 const char* name)
{
    const std::vector<int64_t> dimsStrides = {1};
    switch(dataType)
    {
    case DataType::FLOAT:
        return CreateTensorAttributesDirect(
            builder,
            uid,
            name,
            DataType::FLOAT,
            &dimsStrides,
            &dimsStrides,
            false,
            TensorValue::Float32Value,
            builder.CreateStruct(Float32Value(static_cast<float>(value))).Union());
    case DataType::HALF:
        return CreateTensorAttributesDirect(
            builder,
            uid,
            name,
            DataType::HALF,
            &dimsStrides,
            &dimsStrides,
            false,
            TensorValue::Float16Value,
            builder.CreateStruct(Float16Value(static_cast<float>(value))).Union());
    case DataType::BFLOAT16:
        return CreateTensorAttributesDirect(
            builder,
            uid,
            name,
            DataType::BFLOAT16,
            &dimsStrides,
            &dimsStrides,
            false,
            TensorValue::BFloat16Value,
            builder.CreateStruct(BFloat16Value(static_cast<float>(value))).Union());
    case DataType::DOUBLE:
        return CreateTensorAttributesDirect(builder,
                                            uid,
                                            name,
                                            DataType::DOUBLE,
                                            &dimsStrides,
                                            &dimsStrides,
                                            false,
                                            TensorValue::Float64Value,
                                            builder.CreateStruct(Float64Value(value)).Union());
    default:
        throw std::runtime_error("Unsupported " + std::string(name) + " data type");
    }
}

} // namespace hipdnn_integration_tests::test_utils
