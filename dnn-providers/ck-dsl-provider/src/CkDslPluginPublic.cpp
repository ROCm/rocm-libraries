// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "version.h"

// Plugin C-ABI entry points are synthesised by the SDK's
// EnginePluginImpl.inl. Defining the five macros below and including
// the .inl wires every hipdnnPluginGet* / hipdnnEnginePlugin* symbol
// the backend dlopen()s us for. Do not hand-write those C exports.

#define HIPDNN_PLUGIN_NAME "ck_dsl_provider_plugin"
#define HIPDNN_PLUGIN_VERSION CK_DSL_PROVIDER_VERSION_STRING
#define HIPDNN_PLUGIN_API_VERSION "1.0.0"
#define HIPDNN_PLUGIN_CONTAINER_TYPE ck_dsl_provider::CkDslContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE CkDslHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE ck_dsl_provider::CkDslContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
