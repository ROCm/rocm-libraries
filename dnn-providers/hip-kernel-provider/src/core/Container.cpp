// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Container.hpp"
#include "device/CurrentDevicePropertyProvider.hpp"

#ifdef HIPDNN_ENGINE_HIP_FLASH2
#include "engines/hip_flash2_engine/HipFlash2Engine.hpp"
#include "engines/hip_flash2_engine/HipFlash2FwdPlanBuilder_v2.hpp"
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
#ifdef HIPDNN_ENGINE_HIP_FLASH2
            // HIP_FLASH2_ENGINE: FP16 Flash-Attention 2 V7 (rocWMMA MFMA + causal tile skip)
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
        };

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
        // One ingestor engine per discovered descriptor set, and a set is now a file on
        // disk: adding an engine is an install, not an edit here.
        for(const auto& set : kernel_ingestor_engine::discoverDescriptorSets())
        {
            // engineNameToId, not a provider-side registration: the loader already interned
            // and registered this name, and a second registry over the same process-wide
            // string_view map risks a dangling view.
            const auto engineId = engineNameToId(set.engine.name);
            definitions.push_back(
                {engineId,
                 // set aliases discoverDescriptorSets()'s memoized, process-lifetime vector.
                 // Capture by reference: [set] would re-copy a DescriptorSet per engine.
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
                         // The loader validates each set, but its probe and this construction
                         // are different objects, so that's convention, not a guarantee.
                         // Return null: throwing here would cost every other engine too.
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
    // Must run before any descriptor-backed engine below can resolve its UMD/UHD/UDD
    // symbols. Safe on every Container construction: registers exactly once per process
    // (see SharedContainerManager).
    kernel_ingestor_engine::registerNativeIngestorSymbols();
#endif

    _engineManager
        = std::make_unique<hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        // Null only when a descriptor-backed engine failed to construct (already logged).
        // Its id stays advertised but never claims a graph -- indistinguishable from an
        // engine that declines everything.
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
