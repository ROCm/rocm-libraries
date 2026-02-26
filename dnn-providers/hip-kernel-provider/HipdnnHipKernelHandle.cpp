// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipdnnHipKernelHandle.hpp"
#include "HipKernelContainer.hpp"

hipdnn_plugin_sdk::
    EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>&
    HipdnnHipKernelHandle::getEngineManager()
{
    return container->getEngineManager();
}
