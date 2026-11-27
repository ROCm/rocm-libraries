// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnMiopenHandle.hpp"
#include "MiopenContainer.hpp"
#include "version.h"

// Defines are formatted as required for hipdnn_plugin_sdk/EnginePluginImpl.inl
#define HIPDNN_PLUGIN_NAME "miopen_provider_plugin"
#define HIPDNN_PLUGIN_VERSION MIOPEN_PROVIDER_VERSION_STRING
// HIPDNN_PLUGIN_API_VERSION is the engine plugin API version the MIOpen provider reports;
// hipDNN offers a graph only to plugins that report at least the version the graph requires
// (hipdnn_plugin_sdk/PluginVersionConstants.hpp). 1.2.0 covers runtime pass-by-value scalars.
// The 1.1.0 override-shape execute entry is exported by MiopenPluginPublic.cpp as a stub
// that rejects every call. Raise this only once the provider serves what the new version adds.
#define HIPDNN_PLUGIN_API_VERSION "1.2.0"
#define HIPDNN_PLUGIN_CONTAINER_TYPE miopen_plugin::MiopenContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE HipdnnMiopenHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE HipdnnMiopenContext
