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
// (test_list.cpp:60-87) compiles against these stubs and fails at its first
// REQUIRE instead of failing the whole legacy TU.
#pragma once

#include "../../overlay/cudnn.h"

#include <cstdint>

inline cudnnStatus_t cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t,
                                                  cudnnBackendDescriptor_t* descriptor)
{
    if(descriptor != nullptr)
    {
        *descriptor = nullptr;
    }
    return CUDNN_STATUS_NOT_SUPPORTED;
}

inline cudnnStatus_t cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t)
{
    return CUDNN_STATUS_NOT_SUPPORTED;
}

inline cudnnStatus_t cudnnBackendSetAttribute(cudnnBackendDescriptor_t,
                                              cudnnBackendAttributeName_t,
                                              cudnnBackendAttributeType_t,
                                              int64_t,
                                              const void*)
{
    return CUDNN_STATUS_NOT_SUPPORTED;
}

inline cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t)
{
    return CUDNN_STATUS_NOT_SUPPORTED;
}
