// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelHandle.hpp"

#include "SdpaKernelContainer.hpp"

hipdnn_plugin_sdk::EngineManager<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>&
    SdpaKernelHandle::getEngineManager()
{
    return container->getEngineManager();
}
