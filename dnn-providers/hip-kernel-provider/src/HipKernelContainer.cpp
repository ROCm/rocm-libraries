// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipKernelContainer.hpp"
#include "CurrentDevicePropertyProvider.hpp"
#include "engines/HipKernelEngine.hpp"
#include "engines/plans/BatchnormPlanBuilder.hpp"
#include "engines/plans/RMSnorm/RMSnormPlanBuilder.hpp"
#include "hip/HipKernelCompiler.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider
{

using namespace hipdnn_data_sdk::utilities;

const std::vector<HipKernelContainer::EngineDefinition>& HipKernelContainer::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = {
        // HIP_KERNEL_ENGINE
        {HIP_KERNEL_ENGINE_ID,
         [](const IKernelCompiler& kernelCompiler,
            const IDevicePropertyProvider& devicePropertyProvider)
             -> std::unique_ptr<
                 hipdnn_plugin_sdk::IEngine<HipKernelHandle, HipKernelSettings, HipKernelContext>> {
             auto engine = std::make_unique<HipKernelEngine>(HIP_KERNEL_ENGINE_ID);
             engine->addPlanBuilder(
                 std::make_unique<BatchnormPlanBuilder>(kernelCompiler, devicePropertyProvider));
             engine->addPlanBuilder(std::make_unique<rmsnorm::RMSnormPlanBuilder>(
                 kernelCompiler, devicePropertyProvider));
             return engine;
         }}};

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
    : _devicePropertyProvider(std::make_unique<CurrentDevicePropertyProvider>())
    , _kernelCompiler(std::make_unique<HipKernelCompiler>())
{
    HIPDNN_PLUGIN_LOG_INFO("Creating HipKernelContainer");

    _engineManager = std::make_unique<
        hipdnn_plugin_sdk::EngineManager<HipKernelHandle, HipKernelSettings, HipKernelContext>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(
            engineDefinition.createEngine(*_kernelCompiler, *_devicePropertyProvider));
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
