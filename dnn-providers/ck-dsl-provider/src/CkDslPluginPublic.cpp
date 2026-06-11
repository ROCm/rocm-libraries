// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// hipDNN engine-plugin C API entry point. The EnginePluginImpl.inl facility
// generates the full C ABI (hipdnnEnginePlugin*) from the four plugin types.
#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"

using namespace ck_dsl_plugin;

#define HIPDNN_PLUGIN_NAME "ck_dsl_provider_plugin"
#define HIPDNN_PLUGIN_VERSION "1.0.0"
#define HIPDNN_PLUGIN_API_VERSION "0.0.1"
#define HIPDNN_PLUGIN_CONTAINER_TYPE ck_dsl_plugin::CkDslContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE CkDslHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE CkDslContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
