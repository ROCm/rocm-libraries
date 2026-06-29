// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RocKEHandle.hpp"

#include "RocKEContainer.hpp"

namespace rocke_client
{

hipdnn_plugin_sdk::EngineManager<RocKEHandle, RocKESettings, RocKEContext>&
    RocKEHandle::getEngineManager()
{
    return container->getEngineManager();
}

} // namespace rocke_client
