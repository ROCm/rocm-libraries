// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string_view>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Minimum plugin SDK API version required to serve graphs that opt
 *        into RFC 0008 Phase 1 overridable tensor shapes.
 *
 * The override-execute optional plugin symbol is added at this version.
 * This is the placeholder value per RFC 0008 §4.5; finalize before any
 * non-fake provider ships override execute. Centralizing the literal here
 * lets the version-comparison filter, fake test plugins, and tests all
 * reference the same constant rather than re-spelling the version string.
 */
// RFC 0008 Phase 1: minimum plugin SDK API version that supports the override execute entry point.
inline constexpr std::string_view K_PHASE1_OVERRIDE_MIN_VERSION = "1.1.0";

} // namespace hipdnn_plugin_sdk
