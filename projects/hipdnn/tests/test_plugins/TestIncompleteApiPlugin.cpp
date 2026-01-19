// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_plugin_sdk/PluginApiDataTypes.h"
#include <hipdnn_plugin_sdk/EnginePluginApi.h>

extern "C" {
hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    *name = "test_IncompleteApiPlugin";
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    *version = "1.0.0";
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    *type = HIPDNN_PLUGIN_TYPE_ENGINE;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}
}
