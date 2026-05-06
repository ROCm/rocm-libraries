// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string_view>

namespace hipdnn_plugin_sdk
{

// RFC 0008 Phase 1 (§4.5): minimum plugin SDK API version that supports the
// override-execute entry point. Placeholder value; finalize before any
// non-fake provider ships override execute.
inline constexpr std::string_view K_PHASE1_OVERRIDE_MIN_VERSION = "1.1.0";

} // namespace hipdnn_plugin_sdk
