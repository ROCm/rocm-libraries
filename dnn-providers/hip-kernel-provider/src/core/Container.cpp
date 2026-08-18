// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Container.hpp"
#include "device/CurrentDevicePropertyProvider.hpp"

#ifdef HIPDNN_ENGINE_HIP_MLOPS
#include "engines/hip_mlops_engine/HipMlopsEngine.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormBwdPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdTrainingPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResamplePlanBuilder.hpp"
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

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#endif

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::core
{

using namespace hipdnn_data_sdk::utilities;

const std::vector<Container::EngineDefinition>& Container::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = [] {
        std::vector<EngineDefinition> definitions = {
        // HIP_MLOPS_ENGINE
#ifdef HIPDNN_ENGINE_HIP_MLOPS
            {HIP_MLOPS_ENGINE_ID,
             [](const device::IDevicePropertyProvider& devicePropertyProvider)
                 -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
                 auto engine = std::make_unique<HipMlopsEngine>(HIP_MLOPS_ENGINE_ID);
                 const compilation::IKernelCompiler& kernelCompiler = engine->getKernelCompiler();
                 engine->addPlanBuilder(std::make_unique<batchnorm::BatchnormPlanBuilder>(
                     kernelCompiler, devicePropertyProvider));
                 engine->addPlanBuilder(
                     std::make_unique<batchnorm::BatchnormFwdTrainingPlanBuilder>(
                         kernelCompiler, devicePropertyProvider));
                 engine->addPlanBuilder(std::make_unique<rmsnorm::RMSnormPlanBuilder>(
                     kernelCompiler, devicePropertyProvider));
                 engine->addPlanBuilder(std::make_unique<rmsnorm::RMSnormBwdPlanBuilder>(
                     kernelCompiler, devicePropertyProvider));
                 engine->addPlanBuilder(std::make_unique<layernorm::LayernormPlanBuilder>(
                     kernelCompiler, devicePropertyProvider));
                 engine->addPlanBuilder(std::make_unique<resample::ResamplePlanBuilder>(
                     kernelCompiler, devicePropertyProvider));
                 return engine;
             }},
#endif
#ifdef HIPDNN_ENGINE_HIP_FLASH2
            // HIP_FLASH2_ENGINE: FP16 Flash-Attention 2 V7 (rocWMMA MFMA + causal tile skip)
            // Complements ASM_SDPA_ENGINE: handles FP16 on gfx942/gfx950.
            // Performance: 78.98 TFLOPS MI325X, 71.27 TFLOPS MI300X (seq=4096 causal D=128).
            {HIP_FLASH2_ENGINE_ID,
             [](const device::IDevicePropertyProvider& /*devicePropertyProvider*/)
                 -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
                 auto engine = std::make_unique<hip_flash2_engine::HipFlash2Engine>();
                 engine->addPlanBuilder(
                     std::make_unique<hip_flash2_engine::HipFlash2FwdPlanBuilder>());
                 return engine;
             }},
#endif // HIPDNN_ENGINE_HIP_FLASH2
#ifdef HIPDNN_ENGINE_ASM_SDPA
            // ASM_SDPA_ENGINE
            {ASM_SDPA_ENGINE_ID,
             [](const device::IDevicePropertyProvider& /*devicePropertyProvider*/)
                 -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
                 auto engine = std::make_unique<asm_sdpa_engine::AsmSdpaEngine>();
                 engine->addPlanBuilder(std::make_unique<asm_sdpa_engine::SdpaFwdPlanBuilder>());
                 engine->addPlanBuilder(std::make_unique<asm_sdpa_engine::SdpaBwdPlanBuilder>());
                 return engine;
             }},
#endif
        };

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
        // One ingestor engine per discovered descriptor set. A loop rather than a row
        // per engine, so adding one costs no edit here.
        for(auto& set : kernel_ingestor_engine::discoverDescriptorSets())
        {
            try
            {
                // Registered at enumeration, where a collision is catchable and can name
                // the engine; the id is what hipDNN identifies it by.
                const auto engineId = kernel_ingestor_engine::registerEngineName(set.engine.name);
                definitions.push_back(
                    {engineId,
                     [set](const device::IDevicePropertyProvider& /*devicePropertyProvider*/)
                         -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
                         // Device facts are resolved per call from the handle, not from
                         // the construction-time provider.
                         return hipdnn_plugin_sdk::ingestor::makeEngine<Handle, Settings, Context>(
                             set, kernel_ingestor_engine::deviceResolver());
                     }});
            }
            catch(const std::exception& error)
            {
                // Per set, not around the loop. This list is a function-local static's
                // initializer: an escaping throw costs HIP_MLOPS and ASM_SDPA their rows
                // and leaves the static uninitialized, so the next call rebuilds the
                // whole vector and throws again. One bad set costs only itself.
                HIPDNN_PLUGIN_LOG_ERROR("ingestor: descriptor set '"
                                        << set.engine.name
                                        << "' failed to register its engine name and is "
                                           "excluded: "
                                        << error.what());
            }
        }
#endif

        return definitions;
    }();

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
{
    HIPDNN_PLUGIN_LOG_INFO("Creating Container");

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    // Every ingestor pack's native matchers, scorers, and dispatch handlers must be
    // registered before any descriptor-backed engine below can resolve the symbols its
    // UMDs, UHDs, and UDDs name. Safe to call on every Container construction: it
    // registers exactly once per process (see SharedContainerManager).
    kernel_ingestor_engine::registerNativeIngestorSymbols();
#endif

    _engineManager
        = std::make_unique<hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        _engineManager->addEngine(engineDefinition.createEngine(*_devicePropertyProvider));
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
