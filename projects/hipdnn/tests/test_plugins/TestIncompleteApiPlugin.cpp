#include "hipdnn_sdk/plugin/PluginApiDataTypes.h"
#include <hipdnn_sdk/plugin/EnginePluginApi.h>

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
