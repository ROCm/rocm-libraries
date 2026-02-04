// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLastErrorManager.hpp>

#include <iostream>

// Logging macros for plugin API entry/exit
// These are dual-mode: fmt-style with HIPDNN_PLUGIN_USE_SPDLOG, stream-style otherwise
#ifdef HIPDNN_PLUGIN_USE_SPDLOG
#define LOG_API_ENTRY(format, ...) \
    HIPDNN_LOG_INFO("API called: [{}] " format, __func__, __VA_ARGS__)
#define LOG_API_SUCCESS(func_name, format, ...) \
    HIPDNN_LOG_INFO("API success: [{}] " format, func_name, __VA_ARGS__)
#else
// Stream-style versions that don't support format strings
// Usage: LOG_API_ENTRY_STREAM("info " << value)
#define LOG_API_ENTRY(msg) HIPDNN_LOG_INFO("API called: [" << __func__ << "] " << msg)
#define LOG_API_SUCCESS(func_name, msg) \
    HIPDNN_LOG_INFO("API success: [" << func_name << "] " << msg)
#endif

namespace hipdnn_plugin_sdk
{

template <typename T>
void throwIfNull(T* value)
{
    if(value == nullptr)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                    std::string(typeid(T).name()) + " is nullptr");
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
