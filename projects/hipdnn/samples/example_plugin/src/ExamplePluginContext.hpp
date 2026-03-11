// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>

#include "ExamplePluginSettings.hpp"

// Forward declaration
struct ExamplePluginHandle;

/// Execution context for the example plugin.
///
/// Inherits from:
/// - HipdnnEnginePluginExecutionContext: opaque pointer compatibility
/// - ExecutionContextBase: plan and settings storage
struct ExamplePluginContext
    : HipdnnEnginePluginExecutionContext,
      hipdnn_plugin_sdk::ExecutionContextBase<ExamplePluginHandle, ExamplePluginSettings>
{
};
