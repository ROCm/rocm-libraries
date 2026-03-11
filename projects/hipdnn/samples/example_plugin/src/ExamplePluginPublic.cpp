// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ExamplePluginContainer.hpp"
#include "ExamplePluginHandle.hpp"

using namespace example_plugin;

#define HIPDNN_PLUGIN_NAME "example_plugin"
#define HIPDNN_PLUGIN_VERSION "0.1.0"
#define HIPDNN_PLUGIN_CONTAINER_TYPE ExamplePluginContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE ExamplePluginHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE ExamplePluginContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
