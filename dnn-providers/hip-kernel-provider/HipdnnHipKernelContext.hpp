// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_plugin_sdk/ExecutionContextBase.hpp>
#include <hipdnn_plugin_sdk/PluginBaseTypes.hpp>

#include "HipdnnHipKernelSettings.hpp"

struct HipdnnHipKernelHandle;

struct HipdnnHipKernelContext
    : HipdnnEnginePluginExecutionContext,
      hipdnn_plugin_sdk::ExecutionContextBase<HipdnnHipKernelHandle, HipdnnHipKernelSettings>
{
};
