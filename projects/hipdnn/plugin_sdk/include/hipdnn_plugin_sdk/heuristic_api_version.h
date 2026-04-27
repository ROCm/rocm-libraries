// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file heuristic_api_version.h
 * @brief Version constants for the heuristic plugin C ABI (RFC 0007)
 *
 * The heuristic plugin API has its own versioning scheme, independent of the
 * backend library version. This allows the heuristic plugin interface to evolve
 * separately from the core hipDNN backend.
 *
 * Version compatibility:
 * - Major version mismatch: Plugin will be rejected at load time
 * - Minor version differences: Backward compatible (plugins can use older minor versions)
 * - Patch version differences: Always compatible
 */

/**
 * @brief Major version of the heuristic plugin C ABI
 *
 * Incremented when backward-incompatible changes are made to the API.
 * Plugins must match this major version to be loaded.
 */
// NOLINTBEGIN(modernize-macro-to-enum)
#define HIPDNN_HEURISTIC_API_VERSION_MAJOR 0

/**
 * @brief Minor version of the heuristic plugin C ABI
 *
 * Incremented when backward-compatible features are added.
 * Plugins with older minor versions can still be loaded.
 */
#define HIPDNN_HEURISTIC_API_VERSION_MINOR 0

/**
 * @brief Patch version of the heuristic plugin C ABI
 *
 * Incremented for bug fixes and non-functional changes.
 * Does not affect compatibility.
 */
#define HIPDNN_HEURISTIC_API_VERSION_PATCH 1
// NOLINTEND(modernize-macro-to-enum)

/**
 * @brief Full version string in semantic versioning format
 */
#define HIPDNN_HEURISTIC_API_VERSION "0.0.1"
