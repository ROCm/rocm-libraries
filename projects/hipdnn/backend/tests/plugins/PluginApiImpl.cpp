// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.
// It contains the API functions for the test plugin.

#include <hipdnn_sdk/plugin/PluginApi.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginHelpers.hpp>

#include "PluginApiImpl.hpp"

using namespace hipdnn_plugin;

// NOLINTNEXTLINE
thread_local char PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";
static hipdnnCallback_t loggingCallback = nullptr;

#ifdef THROW_IF_NULL
#error "THROW_IF_NULL is already defined"
#endif
#define THROW_IF_NULL(value) \
    PLUGIN_THROW_IF_NULL(value, HIPDNN_PLUGIN_STATUS_BAD_PARAM, #value " is null")

// Exported functions:

extern "C" hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    return hipdnn_plugin::tryCatch([&]() {
        THROW_IF_NULL(name);
        *name = PLUGIN_NAME;
    });
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    return hipdnn_plugin::tryCatch([&]() {
        THROW_IF_NULL(version);
        *version = PLUGIN_VERSION;
    });
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    return hipdnn_plugin::tryCatch([&]() {
        THROW_IF_NULL(type);
        *type = PLUGIN_TYPE;
    });
}

extern "C" hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback)
{
    if(callback == nullptr)
    {
        return PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                    "hipdnnPluginGetType: type is null");
    }
    loggingCallback = callback;
    loggingCallback(HIPDNN_SEV_INFO, "Logging callback successfully set for test plugin.");
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" void hipdnnPluginGetLastErrorString(const char** errorStr)
{
    if(errorStr == nullptr)
    {
        return;
    }
    *errorStr = PluginLastErrorManager::getLastError();
}
