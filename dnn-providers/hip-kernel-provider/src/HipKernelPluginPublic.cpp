// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipKernelContainer.hpp"
#include "HipKernelHandle.hpp"
#include "version.h"

using namespace hip_kernel_provider;

#define HIPDNN_PLUGIN_NAME "hip_kernel_provider_plugin"
#define HIPDNN_PLUGIN_VERSION HIP_KERNEL_PROVIDER_VERSION_STRING
#define HIPDNN_PLUGIN_CONTAINER_TYPE HipKernelContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE HipKernelHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE HipKernelContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
