// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipKernelHandle.hpp"

#include "HipKernelContainer.hpp"

hipdnn_plugin_sdk::EngineManager<HipKernelHandle, HipKernelSettings, HipKernelContext>&
    HipKernelHandle::getEngineManager()
{
    return container->getEngineManager();
}
