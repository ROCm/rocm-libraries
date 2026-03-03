// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "hipdnn_backend.h"
#include <cstring>
#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>
#include <vector>

namespace backend_test
{

using hipdnn_tests::toVec;

/// Set all required tensor attributes on an already-created descriptor.
/// Does NOT create or finalize the descriptor.
/// Uses ASSERT macros (void return) — call from test body or ASSERT_NO_FATAL_FAILURE.
inline void setAllTensorAttributes(hipdnnBackendDescriptor_t desc,
                                   int64_t uid,
                                   const char* name,
                                   const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& strides,
                                   bool isVirtual = false,
                                   hipdnnDataType_t dataType = HIPDNN_DATA_FLOAT)
{
    ASSERT_EQ(
        hipdnnBackendSetAttribute(desc, HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_NAME_EXT,
                                        HIPDNN_TYPE_CHAR,
                                        static_cast<int64_t>(std::strlen(name)),
                                        name),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  desc, HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_DIMENSIONS,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(dims.size()),
                                        dims.data()),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(desc,
                                        HIPDNN_ATTR_TENSOR_STRIDES,
                                        HIPDNN_TYPE_INT64,
                                        static_cast<int64_t>(strides.size()),
                                        strides.data()),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnBackendSetAttribute(
                  desc, HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, &isVirtual),
              HIPDNN_STATUS_SUCCESS);
}

/// Create a tensor descriptor, set all attributes, and finalize it.
/// Uses EXPECT macros (non-void return). Wrap calls in ASSERT_NO_FATAL_FAILURE
/// if early abort on failure is desired.
/// Caller is responsible for calling hipdnnBackendDestroyDescriptor on the result.
inline hipdnnBackendDescriptor_t createAndFinalizeTensorDesc(int64_t uid,
                                                             const char* name,
                                                             const std::vector<int64_t>& dims,
                                                             const std::vector<int64_t>& strides,
                                                             bool isVirtual = false,
                                                             hipdnnDataType_t dataType
                                                             = HIPDNN_DATA_FLOAT)
{
    hipdnnBackendDescriptor_t desc = nullptr;
    EXPECT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_TENSOR_DESCRIPTOR, &desc),
              HIPDNN_STATUS_SUCCESS);

    setAllTensorAttributes(desc, uid, name, dims, strides, isVirtual, dataType);
    if(::testing::Test::HasFatalFailure())
    {
        return desc;
    }

    EXPECT_EQ(hipdnnBackendFinalize(desc), HIPDNN_STATUS_SUCCESS);
    return desc;
}

} // namespace backend_test
