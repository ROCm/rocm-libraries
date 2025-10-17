// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>

namespace hipdnn_sdk::test_utilities
{

inline hipdnn_sdk::data_objects::TensorAttributesT
    unpackTensorAttributes(const hipdnn_sdk::data_objects::TensorAttributes& tensorAttributes)
{
    hipdnn_sdk::data_objects::TensorAttributesT tensorAttributesT;
    tensorAttributes.UnPackTo(&tensorAttributesT);
    return tensorAttributesT;
}

template <typename T>
inline std::unique_ptr<hipdnn_sdk::utilities::ShallowTensor<T>>
    createShallowTensor(const hipdnn_sdk::data_objects::TensorAttributesT& tensorDetails, void* ptr)
{
    return std::make_unique<hipdnn_sdk::utilities::ShallowTensor<T>>(
        ptr, tensorDetails.dims, tensorDetails.strides);
}

inline std::unique_ptr<hipdnn_sdk::utilities::ITensor>
    createTensorFromAttribute(const hipdnn_sdk::data_objects::TensorAttributes& attribute)
{
    auto dims = hipdnn_sdk::utilities::convertFlatBufferVectorToStdVector(attribute.dims());
    auto strides = hipdnn_sdk::utilities::convertFlatBufferVectorToStdVector(attribute.strides());

    switch(attribute.data_type())
    {
    case hipdnn_sdk::data_objects::DataType::FLOAT:
        return std::make_unique<utilities::Tensor<float>>(dims, strides);
    case hipdnn_sdk::data_objects::DataType::HALF:
        return std::make_unique<utilities::Tensor<half>>(dims, strides);
    case hipdnn_sdk::data_objects::DataType::BFLOAT16:
        return std::make_unique<utilities::Tensor<hip_bfloat16>>(dims, strides);
    case hipdnn_sdk::data_objects::DataType::DOUBLE:
        return std::make_unique<utilities::Tensor<double>>(dims, strides);
    case hipdnn_sdk::data_objects::DataType::UINT8:
        return std::make_unique<utilities::Tensor<uint8_t>>(dims, strides);
    case hipdnn_sdk::data_objects::DataType::INT32:
        return std::make_unique<utilities::Tensor<int32_t>>(dims, strides);
    default:
        throw std::runtime_error("Unsupported data type for tensor");
    }
}
}
