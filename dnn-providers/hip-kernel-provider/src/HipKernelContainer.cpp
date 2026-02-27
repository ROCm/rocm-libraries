// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipKernelContainer.hpp"

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider
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
// HIPDNN_REGISTER_ENGINE(HIP_KERNEL_ENGINE, "HIP_KERNEL_ENGINE")
// ============================================================================

const std::vector<HipKernelContainer::EngineDefinition>& HipKernelContainer::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        // ====================================================================
        // Engines will be added here as plan builders are implemented
        // ====================================================================
        // Example:
        // {HIP_KERNEL_ENGINE_ID, []() -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<
        //     HipKernelHandle, HipKernelSettings, HipKernelContext>> {
        //     auto engine = std::make_unique<HipKernelEngine>(HIP_KERNEL_ENGINE_ID);
        //     engine->addPlanBuilder(std::make_unique<SomePlanBuilder>());
        //     return engine;
        // }}
        // ====================================================================
    };

    return s_engineDefinitions;
}

uint32_t
    HipKernelContainer::copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines)
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

HipKernelContainer::HipKernelContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating HipKernelContainer");

    _engineManager = std::make_unique<
        hipdnn_plugin_sdk::EngineManager<HipKernelHandle, HipKernelSettings, HipKernelContext>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(engineDefinition.createEngine());
    }
}

HipKernelContainer::~HipKernelContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying HipKernelContainer");
}

hipdnn_plugin_sdk::EngineManager<HipKernelHandle, HipKernelSettings, HipKernelContext>&
    HipKernelContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace hip_kernel_provider
