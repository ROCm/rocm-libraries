// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Handle.hpp"

#include "Container.hpp"

hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>& Handle::getEngineManager()
{
    return container->getEngineManager();
}
