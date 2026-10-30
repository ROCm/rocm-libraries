// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_shim.h
//     Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

// Legacy-corpus harness only; never installed under include/. The shim has no raw
// descriptor C API, so the raw-backend half of "Tensor creation comparison"
// (test_list.cpp:60-87) compiles against these stubs, including the only attribute
// names and types it uses, and fails at its first REQUIRE instead of failing the
// whole legacy TU.
#pragma once

#include "../../overlay/cudnn.h"

#include <cstdint>

// NOLINTBEGIN(readability-identifier-naming)
typedef enum
{
    CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
    CUDNN_ATTR_TENSOR_DATA_TYPE,
    CUDNN_ATTR_TENSOR_DIMENSIONS,
    CUDNN_ATTR_TENSOR_STRIDES,
    CUDNN_ATTR_TENSOR_UNIQUE_ID,
} cudnnBackendAttributeName_t;

typedef enum
{
    CUDNN_TYPE_DATA_TYPE,
    CUDNN_TYPE_INT64,
} cudnnBackendAttributeType_t;
// NOLINTEND(readability-identifier-naming)

inline cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t /*descriptorType*/,
                                                  cudnnBackendDescriptor_t* descriptor)
{
    if(descriptor != nullptr)
    {
        *descriptor = nullptr;
    }
    return CUDNN_STATUS_NOT_SUPPORTED;
}

inline cudnnStatus_t cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t /*descriptor*/)
{
    return CUDNN_STATUS_NOT_SUPPORTED;
}

inline cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t /*descriptor*/,
                                              cudnnBackendAttributeName_t /*attributeName*/,
                                              cudnnBackendAttributeType_t /*attributeType*/,
                                              int64_t /*elementCount*/,
                                              const void* /*arrayOfElements*/)
{
    return CUDNN_STATUS_NOT_SUPPORTED;
}

inline cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t /*descriptor*/)
{
    return CUDNN_STATUS_NOT_SUPPORTED;
}
