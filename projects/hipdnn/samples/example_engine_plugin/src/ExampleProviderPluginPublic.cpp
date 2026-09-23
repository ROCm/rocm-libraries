// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ExampleProviderContainer.hpp"
#include "ExampleProviderHandle.hpp"
#include "version.h"

using namespace example_provider;

// TEMPLATE ADAPTATION: Update these 5 macros below for your plugin. HIPDNN_PLUGIN_NAME sets the
// display name. HIPDNN_PLUGIN_VERSION is the semantic version string generated from version.json
// and version.h.in via the CMake project's versioning infrastructure.
// The EnginePluginImpl.inl include generates all C API entry points from these macros. No other
// changes are needed in this file.

#define HIPDNN_PLUGIN_NAME "example_provider_plugin"
#define HIPDNN_PLUGIN_VERSION EXAMPLE_PROVIDER_VERSION_STRING // from version.h
#define HIPDNN_PLUGIN_CONTAINER_TYPE ExampleProviderContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE ExampleProviderHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE ExampleProviderContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
