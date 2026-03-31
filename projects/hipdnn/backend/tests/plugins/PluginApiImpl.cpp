// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.
// It contains the API functions for the test plugin.

#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>

#include "PluginApiImpl.hpp"

using namespace hipdnn_plugin_sdk;

// NOLINTNEXTLINE
thread_local char PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";
static hipdnnCallback_t loggingCallback = nullptr;
static hipdnnSeverity_t pluginLogLevel = HIPDNN_SEV_OFF;

#ifdef THROW_IF_NULL
#error "THROW_IF_NULL is already defined"
#endif
#define THROW_IF_NULL(value) \
    PLUGIN_THROW_IF_NULL(value, HIPDNN_PLUGIN_STATUS_BAD_PARAM, #value " is null")

// Exported functions:

extern "C" hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    return hipdnn_plugin_sdk::tryCatch([&]() {
        THROW_IF_NULL(name);
        *name = PLUGIN_NAME;
    });
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    return hipdnn_plugin_sdk::tryCatch([&]() {
        THROW_IF_NULL(version);
        *version = PLUGIN_VERSION;
    });
}

#ifndef HIPDNN_API_VERSION_UNDEFINED
extern "C" hipdnnPluginStatus_t hipdnnPluginGetApiVersion(const char** version)
{
    return hipdnn_plugin_sdk::tryCatch([&]() {
        THROW_IF_NULL(version);
        *version = PLUGIN_API_VERSION;
    });
}
#endif

extern "C" hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    return hipdnn_plugin_sdk::tryCatch([&]() {
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

extern "C" hipdnnPluginStatus_t hipdnnPluginSetLogLevel(hipdnnSeverity_t level)
{
    pluginLogLevel = level;
    if(loggingCallback != nullptr)
    {
        loggingCallback(HIPDNN_SEV_INFO, "pluginSetLogLevel called");
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnSeverity_t testPluginGetLogLevel()
{
    return pluginLogLevel;
}

extern "C" void hipdnnPluginGetLastErrorString(const char** errorStr)
{
    if(errorStr == nullptr)
    {
        return;
    }
    *errorStr = PluginLastErrorManager::getLastError();
}
