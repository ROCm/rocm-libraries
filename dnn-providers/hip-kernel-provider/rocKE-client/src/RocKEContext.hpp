// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>

#include "RocKESettings.hpp"

namespace rocke_client
{

struct RocKEHandle;

struct RocKEContext : HipdnnEnginePluginExecutionContext,
                      hipdnn_plugin_sdk::ExecutionContextBase<RocKEHandle, RocKESettings>
{
};

} // namespace rocke_client
