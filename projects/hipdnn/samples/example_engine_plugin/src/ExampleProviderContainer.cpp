// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ExampleProviderContainer.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "engines/ExampleProviderEngine.hpp"
#include "engines/plans/ConvFwdPlanBuilder.hpp"
#include "engines/plans/ReluPlanBuilder.hpp"
#include "hip/HipKernelCompiler.hpp"

namespace example_provider
{

// TEMPLATE ADAPTATION: Register and create engines. To adapt:
// (1) Update the engine names and IDs in the HIPDNN_REGISTER_ENGINE calls (one call for each
//     engine provided by this plugin).
// (2) Update s_engineDefinitions to create your PlanBuilders instead of
//     ReluPlanBuilder/ConvFwdPlanBuilder.
//
// The s_engineDefinitions vector is one approach to reduce coupling between creating
// engines (in the ExampleProviderContainer constructor) and returning the list of engine
// IDs (in (in ExampleProviderContainer::copyEngineIds()). Alternate approaches can be
// used if this approach is not suitable for your plugin.

// The HIPDNN_REGISTER_ENGINE() macro creates _NAME and _ID constants for each engine.
// E.g. HIPDNN_REGISTER_ENGINE(EXAMPLE_PROVIDER_RELU_ENGINE) will create
// EXAMPLE_PROVIDER_RELU_ENGINE_NAME with the value "EXAMPLE_PROVIDER_RELU_ENGINE" and
// EXAMPLE_PROVIDER_RELU_ENGINE_ID with the hash-derived integer ID for the engine.
HIPDNN_REGISTER_ENGINE(EXAMPLE_PROVIDER_RELU_ENGINE)
HIPDNN_REGISTER_ENGINE(EXAMPLE_PROVIDER_CONV_FWD_ENGINE)

const std::vector<ExampleProviderContainer::EngineDefinition>&
    ExampleProviderContainer::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        {EXAMPLE_PROVIDER_RELU_ENGINE_ID,
         [](const IKernelCompiler& compiler) -> ExampleProviderEnginePtr {
             auto engine = std::make_unique<ExampleProviderEngine>(EXAMPLE_PROVIDER_RELU_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<ReluPlanBuilder>(compiler));
             return engine;
         }},
        {EXAMPLE_PROVIDER_CONV_FWD_ENGINE_ID,
         [](const IKernelCompiler& compiler) -> ExampleProviderEnginePtr {
             auto engine
                 = std::make_unique<ExampleProviderEngine>(EXAMPLE_PROVIDER_CONV_FWD_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<ConvFwdPlanBuilder>(compiler));
             return engine;
         }},
    };

    return s_engineDefinitions;
}

uint32_t ExampleProviderContainer::copyEngineIds(int64_t* engineIds,
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

ExampleProviderContainer::ExampleProviderContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating ExampleProviderContainer");

    _kernelCompiler = std::make_unique<HipKernelCompiler>();

    _engineManager = std::make_unique<hipdnn_plugin_sdk::EngineManager<ExampleProviderHandle,
                                                                       ExampleProviderSettings,
                                                                       ExampleProviderContext>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(engineDefinition.createEngine(*_kernelCompiler));
    }
}

ExampleProviderContainer::~ExampleProviderContainer() noexcept
{
    try
    {
        HIPDNN_PLUGIN_LOG_INFO("Destroying ExampleProviderContainer");
    }
    catch(...) // NOLINT(bugprone-empty-catch)
    {
    }
}

hipdnn_plugin_sdk::
    EngineManager<ExampleProviderHandle, ExampleProviderSettings, ExampleProviderContext>&
    ExampleProviderContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace example_provider
