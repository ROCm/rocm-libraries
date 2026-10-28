// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Declares identifiers matching the NVIDIA cuDNN C API, for source compatibility.

/**
 * @file cudnn_runtime_version.h
 * @brief cuDNN runtime and CUDA runtime versions claimed by the shim.
 */

#pragma once

// TODO: revisit the exact claimed version before a PyTorch integration.

// Macros, not an enum: consumers test `CUDNN_VERSION` in `#if` directives.
// NOLINTBEGIN(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)

/// @brief cuDNN runtime version claimed by the shim (9.22.0; major, minor, patch, and combined).
#define CUDNN_MAJOR 9
#define CUDNN_MINOR 22
#define CUDNN_PATCHLEVEL 0
#define CUDNN_VERSION ((CUDNN_MAJOR * 10000) + (CUDNN_MINOR * 100) + CUDNN_PATCHLEVEL)

/// @brief CUDA runtime version claimed by the shim (12.0); no CUDA runtime is behind it.
/// @note A consumer that needs `CUDART_VERSION` should define it from this value.
#define CUDNN_CUDART_VERSION 12000
// NOLINTEND(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
