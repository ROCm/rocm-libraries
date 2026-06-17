// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ostream>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

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
    case HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE:
        return "HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE";
    case HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED:
        return "HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED";
    default:
        return "HIPDNN_PLUGIN_STATUS_UNKNOWN";
    }
}

inline const char* toString(hipdnnPluginType_t type)
{
    switch(type)
    {
    case HIPDNN_PLUGIN_TYPE_UNSPECIFIED:
        return "HIPDNN_PLUGIN_TYPE_UNSPECIFIED";
    case HIPDNN_PLUGIN_TYPE_ENGINE:
        return "HIPDNN_PLUGIN_TYPE_ENGINE";
    case HIPDNN_PLUGIN_TYPE_HEURISTIC:
        return "HIPDNN_PLUGIN_TYPE_HEURISTIC";
    default:
        return "HIPDNN_PLUGIN_TYPE_UNKNOWN";
    }
}

inline std::ostream& operator<<(std::ostream& os, hipdnnPluginStatus_t status)
{
    return os << toString(status);
}

inline std::ostream& operator<<(std::ostream& os, hipdnnPluginType_t type)
{
    return os << toString(type);
}
