// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipdnn_plugin_sdk/PluginApi.h>

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetNameImpl(const char** name);

hipdnnPluginStatus_t hipdnnPluginGetVersionImpl(const char** version);

hipdnnPluginStatus_t hipdnnPluginGetTypeImpl(hipdnnPluginType_t* type);

void hipdnnPluginGetLastErrorStringImpl(const char** errorStr);

hipdnnPluginStatus_t hipdnnPluginSetLoggingCallbackImpl(hipdnnCallback_t callback);

} // extern "C"
