// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend (include/cudnn_frontend_version.h),
// used under the MIT license.

/**
 * @file cudnn_frontend_version.h
 * @brief cuDNN frontend (FE) version the shim claims source compatibility with.
 *
 * The cuDNN *frontend* library version (a cudnn-frontend tag), distinct from
 * hipDNN's version and from the cuDNN *runtime* version in
 * `cudnn_runtime_version.h`. Consumers gate on `CUDNN_FRONTEND_VERSION` (e.g.
 * PyTorch's `MHA.cpp`), so it must match upstream. Pinned to cuDNN FE v1.24.0.
 */

#pragma once

// Must remain preprocessor macros, not an enum: consumers gate on these in `#if`
// directives (e.g. `CUDNN_FRONTEND_VERSION <= 11200`), which an enum cannot do.
// NOLINTBEGIN(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
#define CUDNN_FRONTEND_MAJOR_VERSION 1
#define CUDNN_FRONTEND_MINOR_VERSION 24
#define CUDNN_FRONTEND_PATCH_VERSION 0
#define CUDNN_FRONTEND_VERSION                                                     \
    ((CUDNN_FRONTEND_MAJOR_VERSION * 10000) + (CUDNN_FRONTEND_MINOR_VERSION * 100) \
     + CUDNN_FRONTEND_PATCH_VERSION)
// NOLINTEND(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
