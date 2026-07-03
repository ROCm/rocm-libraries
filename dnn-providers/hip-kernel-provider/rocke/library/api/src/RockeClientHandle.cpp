// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RockeClientHandle.hpp"

#include "RockeClientContainer.hpp"

namespace rocke_client
{

hipdnn_plugin_sdk::EngineManager<RockeClientHandle, RockeClientSettings, RockeClientContext>&
    RockeClientHandle::getEngineManager()
{
    return container->getEngineManager();
}

} // namespace rocke_client
