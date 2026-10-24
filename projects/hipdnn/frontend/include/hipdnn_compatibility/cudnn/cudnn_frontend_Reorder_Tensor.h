// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_Reorder_Tensor.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_Reorder_Tensor.h
 * @brief cuDNN v8 `cudnnReorderFilterAndBiasInt8x32`; the shim validates the arguments and
 *        returns `CUDNN_STATUS_NOT_SUPPORTED`.
 */

#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_frontend_ConvDesc.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Tensor.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_Reorder_Tensor.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,readability-implicit-bool-conversion):
// upstream's signature and argument checks are the source-compatible surface.

inline cudnnStatus_t cudnnReorderFilterAndBiasInt8x32(cudnnHandle_t handle,
                                                      const Tensor_v8& tensor,
                                                      const ConvDesc_v8& conv_desc,
                                                      void* dev_filter_ptr,
                                                      void* reordered_filter_ptr,
                                                      void* dev_bias_ptr,
                                                      void* reordered_bias_ptr)
{
    CUDNN_FRONTEND_UNUSED(handle);

    if(dev_filter_ptr && reordered_filter_ptr == nullptr)
    {
        return CUDNN_STATUS_BAD_PARAM;
    }
    if(dev_bias_ptr && reordered_bias_ptr == nullptr)
    {
        return CUDNN_STATUS_BAD_PARAM;
    }

    auto conv_dims = int(conv_desc.getDimensionCount());
    auto tensor_dims = int(tensor.getDimensionCount());
    auto non_shape_dims = tensor_dims - conv_dims;

    if(non_shape_dims != 2 && non_shape_dims != 3)
    {
        return CUDNN_STATUS_BAD_PARAM;
    }

    if(conv_dims != 2 && conv_dims != 3)
    {
        return CUDNN_STATUS_BAD_PARAM;
    }

    CUDNN_FE_LOG_LABEL_ENDL("cudnnReorderFilterAndBiasInt8x32: INT8x32 filter and bias "
                            "reordering is not supported by the hipDNN cuDNN shim");
    return CUDNN_STATUS_NOT_SUPPORTED;
}

// NOLINTEND(readability-identifier-naming,readability-implicit-bool-conversion)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend
