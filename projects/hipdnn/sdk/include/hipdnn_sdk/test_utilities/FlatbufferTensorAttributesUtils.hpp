// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>

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
    CreateShallowTensor(const hipdnn_sdk::data_objects::TensorAttributesT& tensorDetails, void* ptr)
{
    return std::make_unique<hipdnn_sdk::utilities::ShallowTensor<T>>(
        ptr, tensorDetails.dims, tensorDetails.strides);
}

}