// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CodegenFixturePlugin.hpp"

using namespace codegen_fixture;

// These five macros plus the EnginePluginImpl.inl include generate every C API
// entry point this plugin exports, including hipdnnEnginePluginGetEngineName.
//
// The optional sixth macro, HIPDNN_PLUGIN_API_VERSION, is deliberately omitted:
// the .inl then emits no hipdnnPluginGetApiVersion and the host falls back to
// the 1.0.0 baseline. That is the shape of most plugins in and out of tree, and
// the host must name their engines all the same, so the fixture keeps it.
#define HIPDNN_PLUGIN_NAME "codegen_fixture_plugin"
#define HIPDNN_PLUGIN_VERSION "1.0.0"
#define HIPDNN_PLUGIN_CONTAINER_TYPE CodegenFixtureContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE CodegenFixtureHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE CodegenFixtureContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
