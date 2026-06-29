// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>

#include "RockeClientSettings.hpp"

namespace rocke_client
{

struct RockeClientHandle;

struct RockeClientContext
    : HipdnnEnginePluginExecutionContext,
      hipdnn_plugin_sdk::ExecutionContextBase<RockeClientHandle, RockeClientSettings>
{
};

} // namespace rocke_client
