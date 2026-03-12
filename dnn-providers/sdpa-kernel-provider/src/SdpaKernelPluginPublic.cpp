// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "SdpaKernelContainer.hpp"
#include "SdpaKernelHandle.hpp"

using namespace sdpa_kernel_provider;

#define HIPDNN_PLUGIN_NAME "sdpa_kernel_provider_plugin"
#define HIPDNN_PLUGIN_VERSION "1.0.0"
// #define HIPDNN_PLUGIN_API_VERSION "0.0.1" // Uncomment once all API functions have been implemented
#define HIPDNN_PLUGIN_CONTAINER_TYPE SdpaKernelContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE SdpaKernelHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE SdpaKernelContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
