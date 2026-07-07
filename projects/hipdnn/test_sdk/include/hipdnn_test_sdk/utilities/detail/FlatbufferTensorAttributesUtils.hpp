// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>

namespace hipdnn_test_sdk::detail
{

inline hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT unpackTensorAttributes(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensorAttributes)
{
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT tensorAttributesT;
    tensorAttributes.UnPackTo(&tensorAttributesT);
    return tensorAttributesT;
}

template <typename T>
inline std::unique_ptr<hipdnn_data_sdk::utilities::ShallowTensor<T>> createShallowTensor(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensorDetails, void* ptr)
{
    return std::make_unique<hipdnn_data_sdk::utilities::ShallowTensor<T>>(
        ptr, tensorDetails.dims, tensorDetails.strides);
}

inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    createTensor(hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                 const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& strides)
{
    using namespace hipdnn_data_sdk::utilities;
    using namespace hipdnn_data_sdk::types;
    switch(dataType)
    {
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
        return std::make_unique<Tensor<float>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
        return std::make_unique<Tensor<half>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
        return std::make_unique<Tensor<bfloat16>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE:
        return std::make_unique<Tensor<double>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::UINT8:
        return std::make_unique<Tensor<uint8_t>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT32:
        return std::make_unique<Tensor<int32_t>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT8:
        return std::make_unique<Tensor<int8_t>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3:
        return std::make_unique<Tensor<fp8_e4m3>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2:
        return std::make_unique<Tensor<fp8_e5m2>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT64:
        return std::make_unique<Tensor<int64_t>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0:
        return std::make_unique<Tensor<fp8_e8m0>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP4_E2M1:
        return std::make_unique<Tensor<fp4_e2m1>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT4:
        return std::make_unique<Tensor<uint8_t>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E2M3:
        return std::make_unique<Tensor<fp6_e2m3>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP6_E3M2:
        return std::make_unique<Tensor<fp6_e3m2>>(dims, strides);
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BOOLEAN:
        return std::make_unique<Tensor<bool>>(dims, strides);
    default:
        throw std::runtime_error("Unsupported data type for tensor");
    }
}

inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor> createTensorFromAttribute(
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attribute)
{
    auto dims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attribute.dims());
    auto strides = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
        attribute.strides());

    return createTensor(attribute.data_type(), dims, strides);
}

} // namespace hipdnn_test_sdk::detail
