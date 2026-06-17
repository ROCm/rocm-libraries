// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Container.hpp"

#include "compilation/KernelCompiler.hpp"
#include "device/CurrentDevicePropertyProvider.hpp"

#ifdef HIPDNN_ENGINE_HIP_MLOPS
#include "engines/hip_mlops_engine/HipMlopsEngine.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormBwdPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdTrainingPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormPlanBuilder.hpp"
#endif

#ifdef HIPDNN_ENGINE_HIP_FLASH2
#include "engines/hip_flash2_engine/HipFlash2Engine.hpp"
#include "engines/hip_flash2_engine/HipFlash2FwdPlanBuilder_v2.hpp"
#endif

#ifdef HIPDNN_ENGINE_ASM_SDPA
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#endif

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::core
{

using namespace hipdnn_data_sdk::utilities;

const std::vector<Container::EngineDefinition>& Container::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = {
    // HIP_MLOPS_ENGINE
#ifdef HIPDNN_ENGINE_HIP_MLOPS
        {HIP_MLOPS_ENGINE_ID,
         [](const compilation::IKernelCompiler& kernelCompiler,
            const device::IDevicePropertyProvider& devicePropertyProvider)
             -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
             auto engine = std::make_unique<HipMlopsEngine>(HIP_MLOPS_ENGINE_ID);
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
#endif
#ifdef HIPDNN_ENGINE_ASM_SDPA
        // ASM_SDPA_ENGINE
        {ASM_SDPA_ENGINE_ID,
         [](const compilation::IKernelCompiler& /*kernelCompiler*/,
            const device::IDevicePropertyProvider& /*devicePropertyProvider*/)
             -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
             auto engine = std::make_unique<asm_sdpa_engine::AsmSdpaEngine>();
             engine->addPlanBuilder(std::make_unique<asm_sdpa_engine::SdpaFwdPlanBuilder>());
             engine->addPlanBuilder(std::make_unique<asm_sdpa_engine::SdpaBwdPlanBuilder>());
             return engine;
         }},
#endif // HIPDNN_ENGINE_ASM_SDPA
#ifdef HIPDNN_ENGINE_HIP_FLASH2
        // HIP_FLASH2_ENGINE: FP16 Flash-Attention 2 V7 (rocWMMA MFMA + causal tile skip)
        // Complements ASM_SDPA_ENGINE: handles FP16 on gfx942/gfx950.
        // Performance: 78.98 TFLOPS MI325X, 71.27 TFLOPS MI300X (seq=4096 causal D=128).
        {HIP_FLASH2_ENGINE_ID,
         [](const compilation::IKernelCompiler& kernelCompiler,
            const device::IDevicePropertyProvider& devicePropertyProvider)
             -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
             auto engine
                 = std::make_unique<hip_flash2_engine::HipFlash2Engine>(HIP_FLASH2_ENGINE_ID);
             engine->addPlanBuilder(std::make_unique<hip_flash2_engine::HipFlash2FwdPlanBuilder>());
             return engine;
         }},
#endif // HIPDNN_ENGINE_HIP_FLASH2
    };

    return s_engineDefinitions;
}

uint32_t Container::copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines)
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

Container::Container()
    : _devicePropertyProvider(std::make_unique<device::CurrentDevicePropertyProvider>())
    , _kernelCompiler(std::make_unique<compilation::KernelCompiler>())
{
    HIPDNN_PLUGIN_LOG_INFO("Creating Container");

    _engineManager
        = std::make_unique<hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(
            engineDefinition.createEngine(*_kernelCompiler, *_devicePropertyProvider));
    }
}

Container::~Container()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying Container");
}

hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>& Container::getEngineManager()
{
    return *_engineManager;
}

} // namespace hip_kernel_provider::core
