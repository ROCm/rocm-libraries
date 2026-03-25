// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ExamplePluginContainer.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "CurrentDevicePropertyProvider.hpp"
#include "engines/ExamplePluginEngine.hpp"
#include "engines/plans/ConvFwdPlanBuilder.hpp"
#include "engines/plans/ReluPlanBuilder.hpp"
#include "hip/HipKernelCompiler.hpp"

namespace example_plugin
{

// ============================================================================
// Engine Registration
// ============================================================================
// HIPDNN_REGISTER_ENGINE creates _NAME and _ID constants and an EngineRegistrar
// that detects hash collisions with other registered engines at startup.
// The using declarations bring the SDK types into scope for the macro.
// ============================================================================
using hipdnn_data_sdk::utilities::engineNameToId;
using hipdnn_data_sdk::utilities::EngineRegistrar;
HIPDNN_REGISTER_ENGINE(EXAMPLE_PLUGIN_RELU_ENGINE, "EXAMPLE_PLUGIN_RELU_ENGINE")
HIPDNN_REGISTER_ENGINE(EXAMPLE_PLUGIN_CONV_FWD_ENGINE, "EXAMPLE_PLUGIN_CONV_FWD_ENGINE")

const std::vector<ExamplePluginContainer::EngineDefinition>&
    ExamplePluginContainer::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        {EXAMPLE_PLUGIN_RELU_ENGINE_ID,
         [](const IKernelCompiler& compiler, const IDevicePropertyProvider& deviceProps)
             -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<ExamplePluginHandle,
                                                           ExamplePluginSettings,
                                                           ExamplePluginContext>> {
             auto engine = std::make_unique<ExamplePluginEngine>(EXAMPLE_PLUGIN_RELU_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<ReluPlanBuilder>(compiler, deviceProps));
             return engine;
         }},
        {EXAMPLE_PLUGIN_CONV_FWD_ENGINE_ID,
         [](const IKernelCompiler& compiler, const IDevicePropertyProvider& deviceProps)
             -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<ExamplePluginHandle,
                                                           ExamplePluginSettings,
                                                           ExamplePluginContext>> {
             auto engine = std::make_unique<ExamplePluginEngine>(EXAMPLE_PLUGIN_CONV_FWD_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<ConvFwdPlanBuilder>(compiler, deviceProps));
             return engine;
         }},
    };

    return s_engineDefinitions;
}

uint32_t ExamplePluginContainer::copyEngineIds(int64_t* engineIds,
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

ExamplePluginContainer::ExamplePluginContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Creating ExamplePluginContainer");

    _kernelCompiler = std::make_unique<HipKernelCompiler>();
    _devicePropertyProvider = std::make_unique<CurrentDevicePropertyProvider>();

    _engineManager = std::make_unique<hipdnn_plugin_sdk::EngineManager<ExamplePluginHandle,
                                                                       ExamplePluginSettings,
                                                                       ExamplePluginContext>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(
            engineDefinition.createEngine(*_kernelCompiler, *_devicePropertyProvider));
    }
}

ExamplePluginContainer::~ExamplePluginContainer()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying ExamplePluginContainer");
}

hipdnn_plugin_sdk::EngineManager<ExamplePluginHandle, ExamplePluginSettings, ExamplePluginContext>&
    ExamplePluginContainer::getEngineManager()
{
    return *_engineManager;
}

} // namespace example_plugin
