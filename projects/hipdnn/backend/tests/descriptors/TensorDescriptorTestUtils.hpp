// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "DescriptorTestUtils.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <vector>

namespace hipdnn_backend::test_utilities
{

inline std::unique_ptr<HipdnnBackendDescriptor> createFinalizedTensor(
    int64_t uid,
    std::vector<int64_t> dims = hipdnn_tests::toVec(hipdnn_tests::constants::K_TENSOR_X_DIMS),
    std::vector<int64_t> strides = hipdnn_tests::toVec(hipdnn_tests::constants::K_TENSOR_X_STRIDES),
    hipdnnDataType_t dataType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<TensorDescriptor>();
    auto desc = wrapper->asDescriptor<TensorDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid);
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(dims.size()),
                       dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES,
                       HIPDNN_TYPE_INT64,
                       static_cast<int64_t>(strides.size()),
                       strides.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    desc->finalize();

    return wrapper;
}

} // namespace hipdnn_backend::test_utilities
