// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>

namespace hipdnn_sdk::test_utilities
{

inline data_objects::TensorAttributesT
    unpackTensorAttributes(const data_objects::TensorAttributes& tensorAttributes)
{
    data_objects::TensorAttributesT tensorAttributesT;
    tensorAttributes.UnPackTo(&tensorAttributesT);
    return tensorAttributesT;
}

template <typename T>
inline std::unique_ptr<utilities::ShallowTensor<T>>
    createShallowTensor(const data_objects::TensorAttributesT& tensorDetails, void* ptr)
{
    return std::make_unique<utilities::ShallowTensor<T>>(
        ptr, tensorDetails.dims, tensorDetails.strides);
}

inline std::unique_ptr<utilities::ITensor>
    createTensorFromAttribute(const data_objects::TensorAttributes& attribute)
{
    auto dims = utilities::convertFlatBufferVectorToStdVector(attribute.dims());
    auto strides = utilities::convertFlatBufferVectorToStdVector(attribute.strides());

    return utilities::createTensor(attribute.data_type(), dims, strides);
}

} // namespace hipdnn_sdk::test_utilities
