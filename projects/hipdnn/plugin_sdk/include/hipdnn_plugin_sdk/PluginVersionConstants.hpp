// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string_view>

namespace hipdnn_plugin_sdk
{

// Baseline engine plugin C ABI version for plugins that do not export
// `hipdnnPluginGetApiVersion`. This preserves compatibility with existing
// plugins across the first explicit engine-plugin API versioning rollout.
inline constexpr std::string_view K_ENGINE_PLUGIN_API_VERSION_BASELINE = "1.0.0";

// Minimum engine plugin C ABI version that advertises support for the
// override-execute entry point (RFC 0008 §4.5). Override-execute is the
// additive minor feature introduced in engine plugin API 1.1.0; see
// `engine_api_version.h` for the canonical MAJOR.MINOR.PATCH macros. The
// host's applicability filter rejects any plugin reporting an API version
// strictly less than this when the graph opts into overridable shapes.
inline constexpr std::string_view K_OVERRIDE_EXECUTE_MIN_API_VERSION = "1.1.0";

} // namespace hipdnn_plugin_sdk
