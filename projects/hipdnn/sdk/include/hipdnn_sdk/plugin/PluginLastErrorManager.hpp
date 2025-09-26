// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginApi.h>
#include <hipdnn_sdk/plugin/PluginDataTypeHelpers.hpp>
#include <hipdnn_sdk/utilities/StringUtil.hpp>

// NOTE: The last_error variable must be defined in one of the plugin source files.
//       We also need to nolint it to avoid issues since thread_local static confused clang-tidy.
// NOLINTNEXTLINE
// thread_local char PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

namespace hipdnn_plugin
{

class PluginLastErrorManager
{
private:
    // We cannot use std::string in thread-local storage here because it requires a thread-local storage destructor.
    // This prevents the shared object (plugin) from being unloaded until the program terminates.
    // NOLINTNEXTLINE
    thread_local static char s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH];

public:
    static hipdnnPluginStatus_t setLastError(hipdnnPluginStatus_t status, const char* message)
    {
        if(status == HIPDNN_PLUGIN_STATUS_SUCCESS)
        {
            return status;
        }

        HIPDNN_LOG_ERROR("Error occured in status:{} message:{}", status, message);

        hipdnn_sdk::utilities::copyMaxSizeWithNullTerminator(
            s_lastError, message, sizeof(s_lastError));

        return status;
    }

    static hipdnnPluginStatus_t setLastError(hipdnnPluginStatus_t status,
                                             const std::string& message)
    {
        return setLastError(status, message.c_str());
    }

    static const char* getLastError()
    {
        return s_lastError;
    }
};

}
