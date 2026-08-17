// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/FunctionNameMacro.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

#include <iostream>

// Logging macros for plugin API entry/exit (stream-style)
// NOLINTBEGIN(bugprone-macro-parentheses) msg and func_name are stream expression args
#define LOG_API_ENTRY(msg) HIPDNN_PLUGIN_LOG_INFO("API called: [" << __func__ << "] " << msg)
#define LOG_API_SUCCESS(func_name, msg) \
    HIPDNN_PLUGIN_LOG_INFO("API success: [" << func_name << "] " << msg)
// NOLINTEND(bugprone-macro-parentheses)

namespace hipdnn_plugin_sdk
{

template <typename T>
void throwIfNull(T* value)
{
    if(value == nullptr)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    "Null pointer provided to "
                                        + std::string(HIPDNN_FUNCTION_NAME));
    }
}

template <class F>
hipdnnPluginStatus_t tryCatch(F f)
{
    try
    {
        f();
    }
    catch(const HipdnnPluginException& ex)
    {
        return PluginLastErrorManager::setLastError(ex.getStatus(), ex.what());
    }
    catch(const std::exception& ex)
    {
        return PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, ex.what());
    }
    catch(...)
    {
        return PluginLastErrorManager::setLastError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                    "Unknown exception occured");
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}
} // namespace hipdnn_plugin_sdk
