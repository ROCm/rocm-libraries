// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Small [[noreturn]] helper so the catalog/launch code can fail closed with a
// plugin-status-carrying exception without repeating the throw boilerplate.
// Forked/renamed from PR #9207's rocke_client::throwPluginError.

#pragma once

#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace aot_catalog_engine
{

[[noreturn]] inline void throwPluginError(hipdnnPluginStatus_t status, const std::string& message)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(status, message);
}

} // namespace aot_catalog_engine
