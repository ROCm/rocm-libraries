// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnMiopenHandle.hpp"
#include "MiopenContainer.hpp"
#include "version.h"

// Defines are formatted as required for hipdnn_plugin_sdk/EnginePluginImpl.inl
#define HIPDNN_PLUGIN_NAME "miopen_provider_plugin"
#define HIPDNN_PLUGIN_VERSION MIOPEN_PROVIDER_VERSION_STRING
// HIPDNN_PLUGIN_API_VERSION is the version of the plugin API in hipdnn_plugin_sdk
// that has been fully implemented by the MIOpen provider. This must be updated as
// new APIs are implemented as Hipdnn will only attempt to use plugin API functions
// present at this version or earlier
#define HIPDNN_PLUGIN_API_VERSION "0.0.1"
#define HIPDNN_PLUGIN_CONTAINER_TYPE miopen_plugin::MiopenContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE HipdnnMiopenHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE HipdnnMiopenContext
