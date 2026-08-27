// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CodegenFixturePlugin.hpp"

using namespace codegen_fixture;

// These five macros plus the EnginePluginImpl.inl include generate every C API
// entry point this plugin exports, including hipdnnEnginePluginGetEngineName.
//
// HIPDNN_PLUGIN_API_VERSION is deliberately omitted so the host sees the 1.0.0
// baseline, as most plugins do.
#define HIPDNN_PLUGIN_NAME "codegen_fixture_plugin"
#define HIPDNN_PLUGIN_VERSION "1.0.0"
#define HIPDNN_PLUGIN_CONTAINER_TYPE CodegenFixtureContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE CodegenFixtureHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE CodegenFixtureContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
