// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Container.hpp"
#include "Handle.hpp"
#include "version.h"

using namespace hip_kernel_provider::core;

#define HIPDNN_PLUGIN_NAME "hip_kernel_provider_plugin"
#define HIPDNN_PLUGIN_VERSION HIP_KERNEL_PROVIDER_VERSION_STRING
#define HIPDNN_PLUGIN_CONTAINER_TYPE Container
#define HIPDNN_PLUGIN_HANDLE_TYPE Handle
#define HIPDNN_PLUGIN_CONTEXT_TYPE Context

#define HIPDNN_PLUGIN_API_VERSION "1.2.0"
#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
