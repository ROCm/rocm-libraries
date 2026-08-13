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

/// Labels of packs whose symbol registration failed. Read by discoverDescriptorSets()
/// so a pack that could not register is excluded at startup rather than failing lazily
/// at its first query.
std::unordered_set<std::string>& failedPackLabels()
{
    static std::unordered_set<std::string> s_failed;
    return s_failed;
}

/// Registers every pack's native symbols, one SymbolScope per pack.
///
/// A pack that throws is rolled back and skipped; the sweep carries on, because one
/// pack's duplicate symbol must not unregister every other pack's. Runs under
/// call_once, where a throw is catchable; at static-init it would terminate the
/// process during dlopen().
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
    // Registration first, and this is load-bearing rather than defensive. The backend's
    // first call into a plugin is hipdnnEnginePluginGetAllEngineIds, from
    // EnginePluginManager::validateBeforeAdding at load time, which reaches here
    // through Container's *static* copyEngineIds -- before any Container exists and so
    // before the constructor's sweep would have run. Memoizing without this call means
    // s_sets is built with failedPackLabels() still empty on every real run, so a pack
    // that could not register its symbols is enumerated anyway, gets an engine id, and
    // then throws out of makeEngine (matcher and scorer symbols resolve eagerly) with
    // no catch above it, taking down the whole plugin. Idempotent: call_once.
    registerNativeIngestorSymbols();

    // The C++ stand-in for a descriptor-file scan. ALMIOPEN-2401 replaces this body
    // with a directory scan; nothing downstream changes.
    //
    // Memoized because two callers read it at different times: Container's static
    // engine-id enumeration and Container's constructor. "Read once at startup" has
    // to mean once, not once per caller, and post-2401 it stops two filesystem scans
    // from disagreeing. The call above also gives the read below a happens-before
    // edge on the write inside the sweep, which a bare unordered_set otherwise lacks.
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
    // Registered here rather than at namespace scope: EngineRegistrar throws on a
    // duplicate name or hash collision, and a throw from a namespace-scope constructor
    // during dlopen() terminates the process (measured, not assumed).
    static std::mutex s_mutex;
    const std::lock_guard<std::mutex> lock(s_mutex);

    if(!hipdnn_data_sdk::utilities::isEngineNameRegistered(name))
    {
        // EngineNames.hpp keeps std::string_view, not std::string: every other caller
        // registers a string literal through HIPDNN_REGISTER_ENGINE, so the storage is
        // static and nobody had to think about it. A descriptor-backed name is built at
        // run time, so it must be interned somewhere that outlives the registry or the
        // map is left holding a dangling view. This is the one thing a loader
        // registering names from parsed files must not get wrong.
        static std::deque<std::string> s_names;
        static std::vector<hipdnn_data_sdk::utilities::EngineRegistrar> s_registrars;
        // deque, not vector: reallocation would move the strings a registered view
        // points at, and a deque never relocates existing elements.
        s_registrars.emplace_back(s_names.emplace_back(name));
    }
    return hipdnn_data_sdk::utilities::engineNameToId(name);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
