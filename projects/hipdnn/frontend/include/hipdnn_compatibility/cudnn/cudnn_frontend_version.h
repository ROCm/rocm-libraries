// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend (include/cudnn_frontend_version.h),
// used under the MIT license.

/**
 * @file cudnn_frontend_version.h
 * @brief Version macros for the hipDNN cuDNN-compatibility shim.
 *
 * These mirror NVIDIA's `cudnn_frontend_version.h` and declare which cuDNN
 * frontend (FE) release this shim claims *source* compatibility with
 * (RFC 0012 §4.8). This is the cuDNN **frontend** library version (an
 * NVIDIA/cudnn-frontend tag), independent of both hipDNN's own version and the
 * cuDNN **runtime** version returned by `cudnnGetVersion()` (see `cudnn.h`).
 *
 * Consumers such as PyTorch's `MHA.cpp` gate on `CUDNN_FRONTEND_VERSION`
 * (e.g. `#if CUDNN_FRONTEND_VERSION <= 11200`), so matching upstream matters.
 *
 * Pinned to cuDNN FE v1.24.0 (RFC 0012 §2).
 */

#pragma once

// These must remain preprocessor macros (not an enum): consumers such as
// PyTorch's MHA.cpp gate on them in `#if CUDNN_FRONTEND_VERSION <= 11200`
// directives, which an enum cannot satisfy. Suppress modernize-macro-to-enum.
// NOLINTBEGIN(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
#define CUDNN_FRONTEND_MAJOR_VERSION 1
#define CUDNN_FRONTEND_MINOR_VERSION 24
#define CUDNN_FRONTEND_PATCH_VERSION 0
#define CUDNN_FRONTEND_VERSION                                                     \
    ((CUDNN_FRONTEND_MAJOR_VERSION * 10000) + (CUDNN_FRONTEND_MINOR_VERSION * 100) \
     + CUDNN_FRONTEND_PATCH_VERSION)
// NOLINTEND(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
