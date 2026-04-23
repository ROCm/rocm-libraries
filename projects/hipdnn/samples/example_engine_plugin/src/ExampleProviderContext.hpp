// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>

#include "ExampleProviderSettings.hpp"

// TEMPLATE ADAPTATION: Copy and rename. This class inherits from the SDK's execution context
// base classes and needs no operation-specific changes. The dual inheritance
// (HipdnnEnginePluginExecutionContext + ExecutionContextBase) is required by the SDK.

// Forward declaration
struct ExampleProviderHandle;

/// Execution context for the plugin.
///
/// Inherits from:
/// - HipdnnEnginePluginExecutionContext: opaque pointer compatibility
/// - ExecutionContextBase: plan and settings storage
struct ExampleProviderContext
    : HipdnnEnginePluginExecutionContext,
      hipdnn_plugin_sdk::ExecutionContextBase<ExampleProviderHandle, ExampleProviderSettings>
{
};
