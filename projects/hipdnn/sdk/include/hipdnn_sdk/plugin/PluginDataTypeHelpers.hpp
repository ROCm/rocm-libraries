// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ostream>

#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

inline const char* toString(hipdnnPluginStatus_t status)
{
    switch(status)
    {
    case HIPDNN_PLUGIN_STATUS_SUCCESS:
        return "HIPDNN_PLUGIN_STATUS_SUCCESS";
    case HIPDNN_PLUGIN_STATUS_BAD_PARAM:
        return "HIPDNN_PLUGIN_STATUS_BAD_PARAM";
    case HIPDNN_PLUGIN_STATUS_INVALID_VALUE:
        return "HIPDNN_PLUGIN_STATUS_INVALID_VALUE";
    case HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR:
        return "HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR";
    case HIPDNN_PLUGIN_STATUS_ALLOC_FAILED:
        return "HIPDNN_PLUGIN_STATUS_ALLOC_FAILED";
    default:
        return "HIPDNN_PLUGIN_STATUS_UNKNOWN";
    }
}

inline std::ostream& operator<<(std::ostream& os, hipdnnPluginStatus_t status)
{
    return os << toString(status);
}

template <>
struct fmt::formatter<hipdnnPluginStatus_t> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(hipdnnPluginStatus_t status, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(toString(status), ctx);
    }
};

inline const char* toString(hipdnnPluginType_t type)
{
    switch(type)
    {
    case HIPDNN_PLUGIN_TYPE_UNSPECIFIED:
        return "HIPDNN_PLUGIN_TYPE_UNSPECIFIED";
    case HIPDNN_PLUGIN_TYPE_ENGINE:
        return "HIPDNN_PLUGIN_TYPE_ENGINE";
    default:
        return "HIPDNN_PLUGIN_TYPE_UNKNOWN";
    }
}

inline std::ostream& operator<<(std::ostream& os, hipdnnPluginType_t type)
{
    return os << toString(type);
}

template <>
struct fmt::formatter<hipdnnPluginType_t> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(hipdnnPluginType_t type, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(toString(type), ctx);
    }
};
