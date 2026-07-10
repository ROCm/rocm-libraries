// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <hipdnn_plugin_sdk/EngineManager.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

#include "RockeClientHandle.hpp"

namespace rocke_client
{

class RockeClientContainer
{
public:
    RockeClientContainer();
    ~RockeClientContainer();

    // Copy engine IDs into a buffer.
    // If maxEngines == 0: Does not copy, only queries total count.
    // If maxEngines > 0: Copies up to maxEngines IDs into *engineIds, sets numEngines to number
    // copied. Returns: Total number of available engines (regardless of maxEngines value).
    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    hipdnn_plugin_sdk::EngineManager<RockeClientHandle, RockeClientSettings, RockeClientContext>&
        getEngineManager();

private:
    struct EngineDefinition
    {
        int64_t id;
        std::function<std::unique_ptr<hipdnn_plugin_sdk::IEngine<RockeClientHandle,
                                                                 RockeClientSettings,
                                                                 RockeClientContext>>()>
            createEngine;
    };

    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<hipdnn_plugin_sdk::
                        EngineManager<RockeClientHandle, RockeClientSettings, RockeClientContext>>
        _engineManager;
};

} // namespace rocke_client
