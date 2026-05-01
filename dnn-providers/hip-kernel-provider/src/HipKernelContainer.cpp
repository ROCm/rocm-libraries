// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipKernelContainer.hpp"
#include "CurrentDevicePropertyProvider.hpp"
#include "engines/HipKernelEngine.hpp"
#include "engines/plans/RMSnorm/RMSnormBwdPlanBuilder.hpp"
#include "engines/plans/RMSnorm/RMSnormPlanBuilder.hpp"
#include "engines/plans/batchnorm/BatchnormFwdTrainingPlanBuilder.hpp"
#include "engines/plans/batchnorm/BatchnormPlanBuilder.hpp"
#include "engines/plans/layernorm/LayernormPlanBuilder.hpp"
#include "hip/HipKernelCompiler.hpp"

#ifdef HIPDNN_ENGINE_ASM_SDPA
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#endif

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
             engine->addPlanBuilder(std::make_unique<batchnorm::BatchnormPlanBuilder>(
                 kernelCompiler, devicePropertyProvider));
             engine->addPlanBuilder(std::make_unique<batchnorm::BatchnormFwdTrainingPlanBuilder>(
                 kernelCompiler, devicePropertyProvider));
             engine->addPlanBuilder(std::make_unique<rmsnorm::RMSnormPlanBuilder>(
                 kernelCompiler, devicePropertyProvider));
             engine->addPlanBuilder(std::make_unique<rmsnorm::RMSnormBwdPlanBuilder>(
                 kernelCompiler, devicePropertyProvider));
             engine->addPlanBuilder(std::make_unique<layernorm::LayernormPlanBuilder>(
                 kernelCompiler, devicePropertyProvider));
             return engine;
         }},
#ifdef HIPDNN_ENGINE_ASM_SDPA
        // ASM_SDPA_ENGINE
        {ASM_SDPA_ENGINE_ID,
         [](const IKernelCompiler& /*kernelCompiler*/,
            const IDevicePropertyProvider& /*devicePropertyProvider*/)
             -> std::unique_ptr<
                 hipdnn_plugin_sdk::IEngine<HipKernelHandle, HipKernelSettings, HipKernelContext>> {
             auto engine = std::make_unique<asm_sdpa_engine::AsmSdpaEngine>();
             engine->addPlanBuilder(std::make_unique<asm_sdpa_engine::SdpaFwdPlanBuilder>());
             engine->addPlanBuilder(std::make_unique<asm_sdpa_engine::SdpaBwdPlanBuilder>());
             return engine;
         }},
#endif
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
