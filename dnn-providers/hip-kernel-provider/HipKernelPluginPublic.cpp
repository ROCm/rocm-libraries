// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>

#include "HipKernelPlugin.hpp"

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    return hipdnnPluginGetNameImpl(name);
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    return hipdnnPluginGetVersionImpl(version);
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    return hipdnnPluginGetTypeImpl(type);
}

void hipdnnPluginGetLastErrorString(const char** errorStr)
{
    hipdnnPluginGetLastErrorStringImpl(errorStr);
}

// Once plugins are loaded via plugin manager then logging will work for them
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    return hipdnnPluginSetLoggingCallbackImpl(callback);
}

} // extern "C"
