// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ExampleProviderHandle.hpp"

#include "ExampleProviderContainer.hpp"

hipdnn_plugin_sdk::
    EngineManager<ExampleProviderHandle, ExampleProviderSettings, ExampleProviderContext>&
    ExampleProviderHandle::getEngineManager()
{
    return container->getEngineManager();
}
