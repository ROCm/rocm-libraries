// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

#include <string_view>

namespace hipdnn_tests::plugin_constants
{

// The engine plugin API version test_GoodPlugin reports. It is fixed at the override-execute
// floor, below K_PASS_BY_VALUE_MIN_API_VERSION, so the fake is a plugin that predates runtime
// pass-by-value scalars whatever the plugin SDK package version is. TestGoodPlugin.cpp returns it
// and the tests that rely on it compare it with the floors they need.
inline constexpr std::string_view K_TEST_GOOD_PLUGIN_API_VERSION
    = hipdnn_plugin_sdk::K_OVERRIDE_EXECUTE_MIN_API_VERSION;

} // namespace hipdnn_tests::plugin_constants
