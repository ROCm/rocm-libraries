// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RocKEContainer.hpp"

#include "engines/RocKEEngine.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace rocke_client
{

uint32_t
    RocKEContainer::copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines)
{
    constexpr uint32_t TOTAL_ENGINES = 1;

    if(maxEngines == 0)
    {
        numEngines = TOTAL_ENGINES;
        return TOTAL_ENGINES;
    }

    engineIds[0] = hipdnn_data_sdk::utilities::ROCKE_ENGINE_ID;
    numEngines = 1;
    return TOTAL_ENGINES;
}

RocKEContainer::RocKEContainer()
    : _engineManager(std::make_unique<
                     hipdnn_plugin_sdk::EngineManager<RocKEHandle, RocKESettings, RocKEContext>>())
{
    HIPDNN_PLUGIN_LOG_INFO("Creating RocKEContainer");
    _engineManager->addEngine(std::make_unique<RocKEEngine>());
}

RocKEContainer::~RocKEContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying RocKEContainer");
}

hipdnn_plugin_sdk::EngineManager<RocKEHandle, RocKESettings, RocKEContext>&
    RocKEContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace rocke_client
