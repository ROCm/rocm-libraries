// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file cudnn_frontend.h
 * @brief Umbrella header for the hipDNN cuDNN-compatibility shim (v9-only).
 *
 * This is the entry point for the cuDNN frontend compatibility shim described in
 * RFC 0012 ("cuDNN shim for hipDNN"). It mirrors the filename of NVIDIA's
 * upstream `cudnn_frontend.h` so that consumer source can be hipified by
 * swapping the include path and (optionally) aliasing the namespace:
 *
 * @code{.cpp}
 * // Textual hipification (RFC §4.2 workflow 1):
 * #include <hipdnn_compatibility/cudnn/cudnn_frontend.h>
 * namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;
 * @endcode
 *
 * @note Scope: this shim targets the cuDNN frontend **v9 graph API only**
 *       (RFC §1, §4.7). It does not reconstruct the v0.x / v8 builder surface.
 *
 * @note This header must remain includable standalone with no extra `#define`s
 *       or CMake variables (RFC §4.2). It is installed by the `Development`
 *       CMake component and is gated, build-side, behind the
 *       `HIPDNN_ENABLE_CUDNN_COMPATIBILITY` option.
 *
 * @par Status
 * Skeleton (ALMIOPEN-2034): intentionally empty but installable. Translation
 * logic — the stub C-API `cudnn.h`, type/status mapping, error aliasing, and
 * the graph/attribute wrappers — lands in subsequent tickets.
 */

#pragma once
