// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "DescriptorTestUtils.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace hipdnn_backend::test_utilities
{

inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedTensor(int64_t uid,
                          std::vector<int64_t> dims
                          = hipdnn_tests::toVec(hipdnn_tests::constants::K_FPROP_TENSOR_X_DIMS),
                          std::vector<int64_t> strides
                          = hipdnn_tests::toVec(hipdnn_tests::constants::K_FPROP_TENSOR_X_STRIDES),
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

// Verifies that a packed tensor descriptor (retrieved via getAttribute) has the
// expected UID, data_type, dimensions, and strides.
inline void verifyTensorDescriptor(hipdnnBackendDescriptor_t tensorDesc,
                                   int64_t expectedUid,
                                   hipdnnDataType_t expectedDataType,
                                   const std::vector<int64_t>& expectedDims,
                                   const std::vector<int64_t>& expectedStrides)
{
    int64_t uid = 0;
    int64_t uidCount = 0;
    tensorDesc->getAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uidCount, &uid);
    EXPECT_EQ(uid, expectedUid);

    hipdnnDataType_t dataType = {};
    int64_t dtCount = 0;
    tensorDesc->getAttribute(
        HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &dataType);
    EXPECT_EQ(dataType, expectedDataType);

    int64_t dimCount = 0;
    tensorDesc->getAttribute(
        HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 0, &dimCount, nullptr);
    ASSERT_EQ(dimCount, static_cast<int64_t>(expectedDims.size()));
    std::vector<int64_t> dims(static_cast<size_t>(dimCount));
    int64_t actualDimCount = 0;
    tensorDesc->getAttribute(
        HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, dimCount, &actualDimCount, dims.data());
    EXPECT_EQ(dims, expectedDims);

    int64_t strideCount = 0;
    tensorDesc->getAttribute(
        HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 0, &strideCount, nullptr);
    ASSERT_EQ(strideCount, static_cast<int64_t>(expectedStrides.size()));
    std::vector<int64_t> strides(static_cast<size_t>(strideCount));
    int64_t actualStrideCount = 0;
    tensorDesc->getAttribute(HIPDNN_ATTR_TENSOR_STRIDES,
                             HIPDNN_TYPE_INT64,
                             strideCount,
                             &actualStrideCount,
                             strides.data());
    EXPECT_EQ(strides, expectedStrides);
}

} // namespace hipdnn_backend::test_utilities
