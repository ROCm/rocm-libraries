// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <deque>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "engines/kernel_ingestor_engine/HandleDeviceResolver.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

namespace
{

/// Labels of packs whose registration failed; discoverDescriptorSets() excludes them.
std::unordered_set<std::string>& failedPackLabels()
{
    static std::unordered_set<std::string> s_failed;
    return s_failed;
}

/// Registers every pack's native symbols. A pack that throws is rolled back and
/// skipped. Runs under call_once: at static-init a throw would terminate the process.
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
            failedPackLabels().emplace(pack.label);
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

std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet> discoverDescriptorSets()
{
    // Must run first: the backend's first call arrives via the static engine-id path,
    // before any Container exists to trigger registration otherwise.
    registerNativeIngestorSymbols();

    // C++ stand-in for a descriptor-file scan. Memoized: read has a happens-before
    // edge on the sweep's write.
    static const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet> s_sets = [] {
        std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet> sets;
        for(const auto& pack : ingestorPacks())
        {
            if(failedPackLabels().count(std::string(pack.label)) != 0)
            {
                continue;
            }
            sets.push_back(pack.buildDescriptorSet());
        }
        return sets;
    }();

    return s_sets;
}

int64_t registerEngineName(const std::string& name)
{
    // Not namespace-scope: a throw during dlopen() there terminates the process.
    static std::mutex s_mutex;
    const std::lock_guard<std::mutex> lock(s_mutex);

    if(!hipdnn_data_sdk::utilities::isEngineNameRegistered(name))
    {
        // A run-time name must be interned somewhere outliving the registry: it keeps a
        // string_view, and every other caller registers a literal through the macro.
        // deque, not vector: reallocation would move a registered view's string.
        static std::deque<std::string> s_names;
        // The registrar is a constructor, not an object: it writes into the process-wide
        // registry and carries no state, so nothing needs to hold it.
        hipdnn_data_sdk::utilities::EngineRegistrar{s_names.emplace_back(name)};
    }
    return hipdnn_data_sdk::utilities::engineNameToId(name);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
