// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>

namespace hipdnn_test_sdk::utilities
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

    return hipdnn_sdk::utilities::createTensor(attribute.data_type(), dims, strides);
}

} // namespace hipdnn_test_sdk::utilities
