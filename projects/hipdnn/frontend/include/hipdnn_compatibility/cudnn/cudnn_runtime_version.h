// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN, used under the MIT license.

/**
 * @file cudnn_runtime_version.h
 * @brief cuDNN **runtime library** version claimed by the hipDNN shim.
 *
 * NVIDIA's `cudnnGetVersion()` (and the `CUDNN_VERSION` macro) report the cuDNN
 * *runtime library* version — e.g. `90500` == 9.5.0 — which is distinct from the
 * cuDNN *frontend* version (`CUDNN_FRONTEND_VERSION`, see
 * `cudnn_frontend_version.h`). Upstream samples gate on this as a runtime
 * version (`cudnnGetVersion() < 90500`, `>= 91400`, ...), so the shim claims a
 * cuDNN 9.x runtime version here. `cudnn.h`'s `cudnnGetVersion()` returns
 * `CUDNN_VERSION` from this header.
 *
 * The claimed version (9.22.0) matches the runtime the pinned cuDNN frontend
 * v1.24.0 release notes recommend (9.22.0 and later).
 *
 * @note TODO: the exact runtime version claimed (9.22.0) tracks
 *       the FE v1.24.0 recommendation for now; revisit before attempting a
 *       PyTorch integration.
 */

#pragma once

// These must remain preprocessor macros (not an enum) to mirror NVIDIA's
// `cudnn.h`/`cudnn_version.h`, where consumers gate on `CUDNN_VERSION` in `#if`
// directives that an enum cannot satisfy. Suppress modernize-macro-to-enum.
// NOLINTBEGIN(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
#define CUDNN_MAJOR 9
#define CUDNN_MINOR 22
#define CUDNN_PATCHLEVEL 0
#define CUDNN_VERSION ((CUDNN_MAJOR * 10000) + (CUDNN_MINOR * 100) + CUDNN_PATCHLEVEL)
// NOLINTEND(modernize-macro-to-enum,cppcoreguidelines-macro-to-enum)
