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

#ifdef HIPDNN_ENGINE_ASM_SDPA
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#endif

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <filesystem>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginApi.h>
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
        // One ingestor engine per discovered descriptor set, and a set is now a file on
        // disk: adding an engine is an install, not an edit here.
        for(const auto& set : kernel_ingestor_engine::discoverDescriptorSets())
        {
            // engineNameToId, not a provider-side registration: the loader already
            // interned this name and registered it, and a second registry behind one
            // process-wide string_view map is how a dangling view gets created.
            const auto engineId = engineNameToId(set.engine.name);
            definitions.push_back(
                {engineId,
                 // set is a reference into discoverDescriptorSets()'s memoized s_sets
                 // vector (static, process-lifetime) -- captured by reference, not
                 // value, is what that memoization exists to make safe. Do not change
                 // this back to [set]: it re-copies a DescriptorSet per engine.
                 [&set](const device::IDevicePropertyProvider& /*devicePropertyProvider*/)
                     -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
                     try
                     {
                         // Device facts are resolved per call from the handle, not from
                         // the construction-time provider.
                         return hipdnn_plugin_sdk::ingestor::makeEngine<Handle, Settings, Context>(
                             set, kernel_ingestor_engine::deviceResolver());
                     }
                     catch(const std::exception& error)
                     {
                         // The loader validates every set it returns, but its probe and
                         // this construction are two different objects, so that is a
                         // convention rather than a guarantee. Returning null costs this
                         // engine; throwing would cost HIP_MLOPS and ASM_SDPA too.
                         HIPDNN_PLUGIN_LOG_ERROR("ingestor: engine '"
                                                 << set.engine.name
                                                 << "' failed to construct and is excluded: "
                                                 << error.what());
                         return nullptr;
                     }
                 }});
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
        // Null only from a descriptor-backed engine that failed to construct, which has
        // already logged why. Its id stays advertised and simply never claims a graph --
        // indistinguishable, to a caller, from an engine that declines everything.
        if(auto engine = engineDefinition.createEngine(*_devicePropertyProvider))
        {
            _engineManager->addEngine(std::move(engine));
        }
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
