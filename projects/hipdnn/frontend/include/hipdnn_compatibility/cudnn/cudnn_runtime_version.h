// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN, used under the MIT license.

/**
 * @file cudnn_runtime_version.h
 * @brief cuDNN runtime-library version claimed by the shim.
 *
 * `cudnnGetVersion()` / `CUDNN_VERSION` report the cuDNN *runtime* version (e.g.
 * `90500` == 9.5.0), distinct from the *frontend* version in
 * `cudnn_frontend_version.h`. Upstream samples gate on it (`cudnnGetVersion() >=
 * 91400`, …), so the shim claims a 9.x runtime here; `cudnn.h`'s
 * `cudnnGetVersion()` returns `CUDNN_VERSION`. The claimed 9.22.0 matches what
 * cuDNN FE v1.24.0 recommends.
 *
 * @note TODO: revisit the exact claimed version before a PyTorch integration.
 */

#pragma once

// Must remain preprocessor macros, not an enum: consumers gate on `CUDNN_VERSION`
// in `#if` directives that an enum cannot satisfy.
// NOLINTBEGIN(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
#define CUDNN_MAJOR 9
#define CUDNN_MINOR 22
#define CUDNN_PATCHLEVEL 0
#define CUDNN_VERSION ((CUDNN_MAJOR * 10000) + (CUDNN_MINOR * 100) + CUDNN_PATCHLEVEL)
// NOLINTEND(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
