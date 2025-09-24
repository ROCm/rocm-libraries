// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/PluginException.hpp>
#include <hipdnn_sdk/plugin/PluginLastErrorManager.hpp>

#include <iostream>

#define LOG_API_ENTRY(format, ...) \
    HIPDNN_LOG_INFO("API called: [{}] " format, __func__, __VA_ARGS__)

#define LOG_API_SUCCESS(func_name, format, ...) \
    HIPDNN_LOG_INFO("API success: [{}] " format, func_name, __VA_ARGS__)

namespace hipdnn_plugin
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
} // namespace hipdnn_plugin
