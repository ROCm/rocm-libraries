// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ExamplePluginHandle.hpp"

#include "ExamplePluginContainer.hpp"

hipdnn_plugin_sdk::EngineManager<ExamplePluginHandle, ExamplePluginSettings, ExamplePluginContext>&
    ExamplePluginHandle::getEngineManager()
{
    return container->getEngineManager();
}
