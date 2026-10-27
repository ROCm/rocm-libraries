// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/BehaviorNote.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "engines/kernel_ingestor_engine/HandleDeviceResolver.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/serialization/SerializableSourceKind.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

namespace
{

/// descriptorSearchDirectory() given the already-read HIPDNN_DESCRIPTOR_DIR, so one
/// discovery reads (and warns about) the environment once.
std::filesystem::path providerDescriptorTree(std::filesystem::path replacement)
{
    // 1. HIPDNN_DESCRIPTOR_DIR. The SDK reader already drops a value that is not a real
    //    directory (stale build paths are common), so empty means fall through.
    if(!replacement.empty())
    {
        return replacement;
    }

    // 2. Where this plugin was actually loaded from.
    //    Measuring from the loaded module is correct wherever it lands. Keyed on a
    //    function's own address rather than a symbol name, since a name lookup can resolve
    //    to a different module when every provider exports the same plugin entry points.
    try
    {
        const auto candidate = hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForAddress(
                                   reinterpret_cast<const void*>(&descriptorSearchDirectory))
                               / HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR;
        std::error_code notFound;
        if(std::filesystem::is_directory(candidate, notFound))
        {
            return candidate;
        }
    }
    catch(const std::runtime_error& error)
    {
        // No module to measure from -- this TU linked straight into an executable, as in
        // the static test binaries. Not fatal, step 3 still answers; logged because on a
        // real install it would mean the relocatable path had silently stopped working.
        HIPDNN_PLUGIN_LOG_INFO("ingestor: no module-relative descriptor directory ("
                               << error.what() << "); using the configure-time path");
    }

    // 3. The configure-time prefix. Right for an install that never moved.
    return HIPKERNELPROVIDER_DESCRIPTOR_INSTALL_DIR;
}

} // namespace

std::filesystem::path descriptorSearchDirectory()
{
    return providerDescriptorTree(
        hipdnn_plugin_sdk::ingestor::environmentDescriptorRoots().replacement);
}

std::vector<std::filesystem::path> descriptorSearchDirectories()
{
    auto environment = hipdnn_plugin_sdk::ingestor::environmentDescriptorRoots();

    std::vector<std::filesystem::path> roots;
    roots.reserve(1 + environment.additional.size());
    roots.push_back(providerDescriptorTree(std::move(environment.replacement)));

    // Additive, where HIPDNN_DESCRIPTOR_DIR above replaces: this is where descriptors are
    // dropped in beside a shipped install rather than instead of it.
    for(auto& additional : environment.additional)
    {
        roots.push_back(std::move(additional));
    }

    return roots;
}

namespace
{

/// One SymbolScope per pack; a throwing pack rolls back and is skipped so its duplicate
/// symbol can't unregister the others. Runs under call_once, not static init, so the throw
/// is catchable instead of terminating the process during dlopen().
void registerNativeIngestorSymbolsOnce()
{
    for(const auto& pack : ingestorPacks())
    {
        hipdnn_plugin_sdk::ingestor::SymbolScope<Handle> scope;
        try
        {
            pack.registerSymbols(scope);
            scope.commit();
        }
        catch(const std::exception& error)
        {
            HIPDNN_PLUGIN_LOG_ERROR("ingestor: pack '"
                                    << pack.label
                                    << "' failed to register its native symbols and is excluded: "
                                    << error.what());
        }
    }
}

} // namespace

const HandleDeviceResolver& deviceResolver()
{
    static const HandleDeviceResolver s_deviceResolver;
    return s_deviceResolver;
}

void registerNativeIngestorSymbols()
{
    static std::once_flag s_registered;
    std::call_once(s_registered, registerNativeIngestorSymbolsOnce);
}

const hipdnn_plugin_sdk::ingestor::DescriptorCatalog& descriptorCatalog()
{
    // Memoized: descriptor engines and opaque engines' declared L1 models read the same
    // trees; one parse keeps them from disagreeing.
    static const hipdnn_plugin_sdk::ingestor::DescriptorCatalog s_catalog
        = hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog(descriptorSearchDirectories());

    return s_catalog;
}

namespace
{

using StateManager = hipdnn_plugin_sdk::ingestor::KernelIngestorStateManager<Handle>;

/// A validated state manager waiting for the first engine constructed from the set that has
/// @p engineId. Carries the id itself, so a take never relies on a position in another vector.
struct StateManagerSlot
{
    hipdnn_plugin_sdk::ingestor::DescriptorId engineId;
    std::unique_ptr<StateManager> stateManager;
};

/// What one validation of descriptorCatalog() produced.
struct Scan
{
    std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet> sets;
    std::vector<StateManagerSlot> slots;
};

/// Validates descriptorCatalog(), splitting each validated set from the state manager
/// validation built for it. Discovery and rediscoverStateManagersForTesting() both scan
/// through here.
Scan scanDescriptorSets()
{
    // Register before scanning: validation checks each descriptor's symbol against the
    // registry, so an unregistered pack drops its descriptors here instead of throwing at
    // first use.
    registerNativeIngestorSymbols();
    auto validated
        = hipdnn_plugin_sdk::ingestor::loadValidatedDescriptorSets<Handle>(descriptorCatalog());

    Scan scan;
    scan.sets.reserve(validated.size());
    scan.slots.reserve(validated.size());
    for(auto& entry : validated)
    {
        scan.slots.push_back({entry.set.engine.id, std::move(entry.stateManager)});
        scan.sets.push_back(std::move(entry.set));
    }
    return scan;
}

/// What discovery produced: the sets, and a slot holding the state manager validation built
/// for each, waiting for the first engine constructed from that set.
struct Discovery
{
    Discovery()
    {
        auto scan = scanDescriptorSets();
        sets = std::move(scan.sets);
        slots = std::move(scan.slots);
    }

    std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet> sets;
    std::vector<StateManagerSlot> slots;
    std::mutex slotsMutex;
};

Discovery& discovery()
{
    // Memoized: Container's static engine-id enumeration and its constructor both call
    // this and must agree on what shipped.
    static Discovery s_discovery;
    return s_discovery;
}

} // namespace

const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet>& discoverDescriptorSets()
{
    return discovery().sets;
}

std::unique_ptr<StateManager>
    takeDiscoveredStateManager(const hipdnn_plugin_sdk::ingestor::DescriptorId& engineId)
{
    auto& found = discovery();
    const std::lock_guard<std::mutex> guard(found.slotsMutex);
    for(auto& slot : found.slots)
    {
        if(slot.engineId == engineId)
        {
            return std::move(slot.stateManager);
        }
    }
    return nullptr;
}

void rediscoverStateManagersForTesting()
{
    auto& found = discovery();
    auto rescan = scanDescriptorSets();
    // A rescan that finds different engines than discovery retained would hand a state
    // manager built from other descriptors to an engine of a retained set.
    const auto sameEngine
        = [](const auto& lhs, const auto& rhs) { return lhs.engine.id == rhs.engine.id; };
    if(!std::equal(rescan.sets.begin(),
                   rescan.sets.end(),
                   found.sets.begin(),
                   found.sets.end(),
                   sameEngine))
    {
        throw std::logic_error("ingestor: rediscovery found different descriptor sets than "
                               "discovery retained; the state-manager slots are unchanged");
    }
    const std::lock_guard<std::mutex> guard(found.slotsMutex);
    found.slots = std::move(rescan.slots);
}

bool supportsExecutionPlanSerialization(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set)
{
    bool hasKernel = false;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            if(!serialization::isSerializableSourceKind(kernel.source.kind))
            {
                return false;
            }
            hasKernel = true;
        }
    }
    if(!hasKernel)
    {
        return false;
    }

    registerNativeIngestorSymbols();
    for(const auto& pack : set.packs)
    {
        const auto dispatch = std::find_if(
            set.dispatches.begin(), set.dispatches.end(), [&pack](const auto& candidate) {
                return candidate.id == pack.dispatchId;
            });
        if(dispatch == set.dispatches.end())
        {
            return false;
        }
        const auto* handler = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::tryResolve(
            dispatch->dispatchSymbol);
        if(handler == nullptr || !handler->supportsSaving())
        {
            return false;
        }
    }
    return true;
}

hipdnn_plugin_sdk::ingestor::EngineDescriptor
    engineWithComputedBehaviorNotes(const hipdnn_plugin_sdk::ingestor::DescriptorSet& set)
{
    constexpr auto SERIALIZATION_NOTE
        = static_cast<int32_t>(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION);

    auto engine = set.engine;
    auto& notes = engine.behaviorNotes;
    if(supportsExecutionPlanSerialization(set)
       && std::find(notes.begin(), notes.end(), SERIALIZATION_NOTE) == notes.end())
    {
        notes.push_back(SERIALIZATION_NOTE);
    }
    return engine;
}

hipdnn_plugin_sdk::ingestor::DescriptorSet
    withComputedBehaviorNotes(hipdnn_plugin_sdk::ingestor::DescriptorSet set)
{
    set.engine = engineWithComputedBehaviorNotes(set);
    return set;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
