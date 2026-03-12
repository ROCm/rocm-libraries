// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaKernelContainer.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace sdpa_kernel_provider
{

// ============================================================================
// Engine Registration
// ============================================================================
// Use HIPDNN_REGISTER_ENGINE to register engine names here when adding engines.
// This will:
// 1. Create _NAME and _ID constants for the engine
// 2. Detect hash collisions with other registered engines
//
// Example:
// HIPDNN_REGISTER_ENGINE(SDPA_KERNEL_ENGINE, "SDPA_KERNEL_ENGINE")
// ============================================================================

const std::vector<SdpaKernelContainer::EngineDefinition>&
    SdpaKernelContainer::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        // ====================================================================
        // Engines will be added here as plan builders are implemented
        // ====================================================================
        // Example:
        // {SDPA_KERNEL_ENGINE_ID, []() -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<
        //     SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>> {
        //     auto engine = std::make_unique<SdpaKernelEngine>(SDPA_KERNEL_ENGINE_ID);
        //     engine->addPlanBuilder(std::make_unique<SomePlanBuilder>());
        //     return engine;
        // }}
        // ====================================================================
    };

    return s_engineDefinitions;
}

uint32_t SdpaKernelContainer::copyEngineIds(int64_t* engineIds,
                                            uint32_t maxEngines,
                                            uint32_t& numEngines)
{
    const auto& engineDefinitions = getEngineDefinitions();
    auto totalEngines = static_cast<uint32_t>(engineDefinitions.size());

    if(maxEngines == 0)
    {
        numEngines = totalEngines;
        return totalEngines;
    }

    auto enginesToCopy = std::min(maxEngines, totalEngines);
    for(uint32_t i = 0; i < enginesToCopy; ++i)
    {
        engineIds[i] = engineDefinitions[i].id;
    }

    numEngines = enginesToCopy;

    return totalEngines;
}

SdpaKernelContainer::SdpaKernelContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating SdpaKernelContainer");

    _engineManager = std::make_unique<hipdnn_plugin_sdk::EngineManager<SdpaKernelHandle,
                                                                       SdpaKernelSettings,
                                                                       SdpaKernelContext>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(engineDefinition.createEngine());
    }
}

SdpaKernelContainer::~SdpaKernelContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying SdpaKernelContainer");
}

hipdnn_plugin_sdk::EngineManager<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>&
    SdpaKernelContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace sdpa_kernel_provider
