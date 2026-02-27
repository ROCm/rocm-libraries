// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipKernelContainer.hpp"
#include "engines/HipKernelEngine.hpp"
#include "engines/plans/BatchnormPlanBuilder.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

namespace hip_kernel_plugin
{

// ============================================================================
// Engine Registration
// ============================================================================
// For plugins that are not yet globally registered (by adding a call to
// HIPDNN_REGISTER_ENGINE() in "hipdnn_data_sdk/utilities/EngineNames.hpp"),
// use HIPDNN_REGISTER_ENGINE to register the engine names here. This will:
// 1. Create _NAME and _ID constants for the engine
// 2. Detect hash collisions with other formally-registered engines
//
// Example for new engines:
// HIPDNN_REGISTER_ENGINE(MY_CUSTOM_ENGINE, "MY_CUSTOM_ENGINE")
// HIPDNN_REGISTER_ENGINE(MY_OTHER_ENGINE, "MY_OTHER_ENGINE")
// ============================================================================
using namespace hipdnn_data_sdk::utilities;
HIPDNN_REGISTER_ENGINE(HIP_KERNEL_ENGINE, "HIP_KERNEL_ENGINE");

const std::vector<HipKernelContainer::EngineDefinition>& HipKernelContainer::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        // HIP_KERNEL_ENGINE
        {HIP_KERNEL_ENGINE_ID,
         []() -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<HipdnnHipKernelHandle,
                                                            HipdnnHipKernelSettings,
                                                            HipdnnHipKernelContext>> {
             auto engine = std::make_unique<HipKernelEngine>(HIP_KERNEL_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<BatchnormPlanBuilder>());
             // add more plan builders here as they are created, for example:
             // engine->addPlanBuilder(std::make_unique<HipKernelBatchnormPlanBuilder>());
             return engine;
         }}

        // ====================================================================
        // Additional engines would be added here
        // ====================================================================
        // Example:
        // ,{MY_CUSTOM_ENGINE_ID, []() -> std::unique_ptr<IEngine> {
        //     auto engine = std::make_unique<MyCustomEngine>(MY_CUSTOM_ENGINE_ID);
        //     engine->addPlanBuilder(std::make_unique<CustomPlanBuilder>());
        //     // ... configure plan builders for this engine
        //     return engine;
        // }}
        // ,{MY_OTHER_ENGINE_ID, []() -> std::unique_ptr<IEngine> {
        //     auto engine = std::make_unique<MyOtherEngine>(MY_OTHER_ENGINE_ID);
        //     engine->addPlanBuilder(std::make_unique<OtherPlanBuilder>());
        //     // ... configure plan builders for this engine
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
        // When maxEngines is 0, set numEngines to total count
        numEngines = totalEngines;
        return totalEngines;
    }

    // Copy up to maxEngines IDs using index-based loop
    auto enginesToCopy = std::min(maxEngines, totalEngines);
    for(uint32_t i = 0; i < enginesToCopy; ++i)
    {
        engineIds[i] = engineDefinitions[i].id;
    }

    // When maxEngines > 0, set numEngines to number copied
    numEngines = enginesToCopy;

    return totalEngines;
}

HipKernelContainer::HipKernelContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating HipKernelContainer");

    _engineManager = std::make_unique<hipdnn_plugin_sdk::EngineManager<HipdnnHipKernelHandle,
                                                                       HipdnnHipKernelSettings,
                                                                       HipdnnHipKernelContext>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(engineDefinition.createEngine());
    }
}

HipKernelContainer::~HipKernelContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying HipKernelContainer");
}

hipdnn_plugin_sdk::
    EngineManager<HipdnnHipKernelHandle, HipdnnHipKernelSettings, HipdnnHipKernelContext>&
    HipKernelContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace hip_kernel_plugin
