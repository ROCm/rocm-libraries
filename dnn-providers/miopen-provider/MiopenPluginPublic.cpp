// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_plugin_sdk/EnginePluginMacros.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenContainer.hpp"

using namespace miopen_legacy_plugin;

DECLARE_ENGINE_PLUGIN_DEFAULT_IMPL("miopen_provider_plugin", "1.0.0", MiopenContainer)
