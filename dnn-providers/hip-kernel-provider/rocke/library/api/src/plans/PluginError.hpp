// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace rocke_client
{

// Single throw site for rocke-client failure paths: raises a HipdnnPluginException
// carrying the given plugin status and message.
[[noreturn]] inline void throwPluginError(hipdnnPluginStatus_t status, const std::string& message)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(status, message);
}

} // namespace rocke_client
