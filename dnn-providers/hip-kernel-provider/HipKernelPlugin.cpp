// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginDataTypeHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

#include "EngineManager.hpp"

static const char* pluginName = "hip_kernel_plugin";
static const char* pluginVersion = "1.0.0";

using namespace hipdnn_data_sdk::flatbuffer_utilities;
using namespace hipdnn_plugin_sdk;
using namespace hip_kernel_plugin;

// NOLINTNEXTLINE
thread_local char PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetNameImpl(const char** name)
{
    LOG_API_ENTRY("name_ptr={:p}", static_cast<void*>(name));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(name);

        *name = pluginName;

        LOG_API_SUCCESS(apiName, "pluginName={:p}", static_cast<void*>(name));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetVersionImpl(const char** version)
{
    LOG_API_ENTRY("versionPtr={:p}", static_cast<void*>(version));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(version);

        *version = pluginVersion;

        LOG_API_SUCCESS(apiName, "version={:p}", static_cast<void*>(version));
    });
}

hipdnnPluginStatus_t hipdnnPluginGetTypeImpl(hipdnnPluginType_t* type)
{
    LOG_API_ENTRY("typePtr={:p}", static_cast<void*>(type));

    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(type);

        *type = HIPDNN_PLUGIN_TYPE_ENGINE;

        LOG_API_SUCCESS(apiName, "type={}", *type);
    });
}

void hipdnnPluginGetLastErrorStringImpl(const char** errorStr)
{
    LOG_API_ENTRY("errorStrPtr={:p}", static_cast<void*>(errorStr));

    hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(errorStr);

        *errorStr = PluginLastErrorManager::getLastError();

        LOG_API_SUCCESS(apiName, "errorStr={:p}", static_cast<void*>(errorStr));
    });
}

// Once plugins are loaded via plugin manager then logging will work for them
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallbackImpl(hipdnnCallback_t callback)
{
    return hipdnn_plugin_sdk::tryCatch([&, apiName = __func__]() {
        throwIfNull(callback);
        hipdnn::logging::initializeCallbackLogging(pluginName, callback);
        LOG_API_SUCCESS(apiName, "", "");
    });
}

} // extern "C"
