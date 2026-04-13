// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ExampleProviderContainer.hpp"
#include "ExampleProviderHandle.hpp"

using namespace example_provider;

// TEMPLATE ADAPTATION: Update these 5 macros below for your plugin. HIPDNN_PLUGIN_NAME sets the
// display name. HIPDNN_PLUGIN_VERSION is the semantic version number of the plugin, maintained
// by the plugin developer.
// The EnginePluginImpl.inl include generates all C API entry points from these macros. No other
// changes are needed in this file.

#define HIPDNN_PLUGIN_NAME "example_provider"
#define HIPDNN_PLUGIN_VERSION "0.1.0"
#define HIPDNN_PLUGIN_CONTAINER_TYPE ExampleProviderContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE ExampleProviderHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE ExampleProviderContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
