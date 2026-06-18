// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipblasltHandleFactory.hpp"
#include "HipdnnEnginePluginHandle.hpp"

using namespace hipdnn_plugin_sdk;

namespace hipblaslt_plugin
{

void HipblasltHandleFactory::createHipblasltHandle(hipdnnEnginePluginHandle_t* handle)
{
    if(handle == nullptr)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "handle is null");
    }

    *handle = new HipdnnEnginePluginHandle();

    hipblasStatus_t const status = hipblasLtCreate(&(*handle)->hipblasltHandle);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        delete *handle;
        *handle = nullptr;
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "Failed to create Hipblaslt handle");
    }
}

void HipblasltHandleFactory::destroyHipblasltHandle(hipdnnEnginePluginHandle_t handle)
{
    if(handle == nullptr)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM, "handle is null");
    }

    hipblasStatus_t const status = hipblasLtDestroy(handle->hipblasltHandle);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                    "Failed to destroy Hipblaslt handle");
    }
}

} // namespace hipblaslt_plugin
