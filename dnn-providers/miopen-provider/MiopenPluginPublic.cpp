// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "MiopenContainer.hpp"

using namespace miopen_plugin;

#define HIPDNN_PLUGIN_NAME "miopen_provider_plugin"
#define HIPDNN_PLUGIN_VERSION "1.0.0"
#define HIPDNN_PLUGIN_CONTAINER_TYPE MiopenContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE HipdnnEnginePluginHandle

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
