// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RockeClientContainer.hpp"

#include "engines/RockeClientEngine.hpp"

#include <algorithm>

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace rocke_client
{

const std::vector<RockeClientContainer::EngineDefinition>&
    RockeClientContainer::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        {hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID,
         []() -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<RockeClientHandle,
                                                            RockeClientSettings,
                                                            RockeClientContext>> {
             return std::make_unique<RockeClientEngine>();
         }},
    };

    return s_engineDefinitions;
}

uint32_t RockeClientContainer::copyEngineIds(int64_t* engineIds,
                                             uint32_t maxEngines,
                                             uint32_t& numEngines)
{
    const auto& engineDefinitions = getEngineDefinitions();
    const auto totalEngines = static_cast<uint32_t>(engineDefinitions.size());

    if(maxEngines == 0)
    {
        numEngines = totalEngines;
        return totalEngines;
    }

    const auto enginesToCopy = std::min(maxEngines, totalEngines);
    for(uint32_t i = 0; i < enginesToCopy; ++i)
    {
        engineIds[i] = engineDefinitions[i].id;
    }

    numEngines = enginesToCopy;
    return totalEngines;
}

RockeClientContainer::RockeClientContainer()
    : _engineManager(std::make_unique<hipdnn_plugin_sdk::EngineManager<RockeClientHandle,
                                                                       RockeClientSettings,
                                                                       RockeClientContext>>())
{
    HIPDNN_PLUGIN_LOG_INFO("Creating RockeClientContainer");

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(engineDefinition.createEngine());
    }
}

RockeClientContainer::~RockeClientContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying RockeClientContainer");
}

hipdnn_plugin_sdk::EngineManager<RockeClientHandle, RockeClientSettings, RockeClientContext>&
    RockeClientContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace rocke_client
