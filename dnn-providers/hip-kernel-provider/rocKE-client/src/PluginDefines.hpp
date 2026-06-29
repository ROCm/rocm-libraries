// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "RocKEContainer.hpp"
#include "RocKEContext.hpp"
#include "RocKEHandle.hpp"
#include "version.h"

// Defines are formatted as required for hipdnn_plugin_sdk/EnginePluginImpl.inl.
#define HIPDNN_PLUGIN_NAME "rocKEclient"
#define HIPDNN_PLUGIN_VERSION ROCKE_CLIENT_VERSION_STRING
#define HIPDNN_PLUGIN_API_VERSION "1.0.0"
#define HIPDNN_PLUGIN_CONTAINER_TYPE rocke_client::RocKEContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE rocke_client::RocKEHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE rocke_client::RocKEContext
