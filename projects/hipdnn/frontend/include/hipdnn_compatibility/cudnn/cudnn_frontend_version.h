// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_version.h
//     Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_version.h
 * @brief cuDNN FE version the shim claims source compatibility with (v1.24.0).
 */

#pragma once

// Macros, not an enum: consumers test these in `#if` directives.
// NOLINTBEGIN(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
// Bumping: also update the pinned tag and commit in THIRD_PARTY_LICENSES.md;
// detail/graph_wrapper.h enforces the node-signature re-diff.

/// @brief cuDNN FE version claimed by the shim (major, minor, patch, and combined).
#define CUDNN_FRONTEND_MAJOR_VERSION 1
#define CUDNN_FRONTEND_MINOR_VERSION 24
#define CUDNN_FRONTEND_PATCH_VERSION 0
#define CUDNN_FRONTEND_VERSION                                                     \
    ((CUDNN_FRONTEND_MAJOR_VERSION * 10000) + (CUDNN_FRONTEND_MINOR_VERSION * 100) \
     + CUDNN_FRONTEND_PATCH_VERSION)

/// @brief hipDNN extension: defined when the shim headers are included.
#define HIPDNN_COMPATIBILITY_CUDNN_FRONTEND 1
// NOLINTEND(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
