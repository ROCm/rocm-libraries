// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipblaslt_plugin
{

class HipblasltHandleFactory
{
public:
    static void createHipblasltHandle(hipdnnEnginePluginHandle_t* handle);

    static void destroyHipblasltHandle(hipdnnEnginePluginHandle_t handle);
};

} // namespace hipblaslt
