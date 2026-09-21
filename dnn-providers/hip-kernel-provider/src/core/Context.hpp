// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>

#include "Settings.hpp"

// Forward declaration
struct Handle;

/**
 * @brief HIP kernel provider plugin execution context.
 *
 * Inherits from:
 * - HipdnnEnginePluginExecutionContext: For opaque pointer compatibility
 * - ExecutionContextBase: For plan and settings storage
 */
struct Context : HipdnnEnginePluginExecutionContext,
                 hipdnn_plugin_sdk::ExecutionContextBase<Handle, Settings>
{
};
