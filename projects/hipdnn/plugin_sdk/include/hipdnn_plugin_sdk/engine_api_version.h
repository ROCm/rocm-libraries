// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file engine_api_version.h
 * @brief Version constants for the engine plugin C ABI (RFC 0008)
 *
 * The engine plugin API has its own versioning scheme, independent of the
 * backend library version. This allows the engine plugin interface to evolve
 * separately from the core hipDNN backend, mirroring the heuristic plugin
 * versioning introduced in RFC 0007.
 *
 * Version compatibility:
 * - Major version mismatch: Plugin will be rejected at load time
 * - Minor version differences: Backward compatible (plugins can use older minor versions)
 * - Patch version differences: Always compatible
 */

/**
 * @brief Major version of the engine plugin C ABI
 *
 * Incremented when backward-incompatible changes are made to the API.
 * Plugins must match this major version to be loaded.
 */
// NOLINTBEGIN(modernize-macro-to-enum)
#define HIPDNN_ENGINE_API_VERSION_MAJOR 1

/**
 * @brief Minor version of the engine plugin C ABI
 *
 * Incremented when backward-compatible features are added.
 * Plugins with older minor versions can still be loaded.
 */
#define HIPDNN_ENGINE_API_VERSION_MINOR 0

/**
 * @brief Patch version of the engine plugin C ABI
 *
 * Incremented for bug fixes and non-functional changes.
 * Does not affect compatibility.
 */
#define HIPDNN_ENGINE_API_VERSION_PATCH 0
// NOLINTEND(modernize-macro-to-enum)

/**
 * @brief Full version string in semantic versioning format
 */
#define HIPDNN_STRINGIFY_(x) #x
#define HIPDNN_STRINGIFY(x) HIPDNN_STRINGIFY_(x)
#define HIPDNN_ENGINE_API_VERSION                                               \
    HIPDNN_STRINGIFY(HIPDNN_ENGINE_API_VERSION_MAJOR)                           \
    "." HIPDNN_STRINGIFY(HIPDNN_ENGINE_API_VERSION_MINOR) "." HIPDNN_STRINGIFY( \
        HIPDNN_ENGINE_API_VERSION_PATCH)
