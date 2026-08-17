// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>

#include "HipdnnMiopenSettings.hpp"

// Forward declaration
struct HipdnnMiopenHandle;

/**
 * @brief MIOpen plugin execution context.
 *
 * Inherits from:
 * - HipdnnEnginePluginExecutionContext: For opaque pointer compatibility
 * - ExecutionContextBase: For plan and settings storage
 */
struct HipdnnMiopenContext
    : HipdnnEnginePluginExecutionContext,
      hipdnn_plugin_sdk::ExecutionContextBase<HipdnnMiopenHandle, HipdnnMiopenSettings>
{
};
