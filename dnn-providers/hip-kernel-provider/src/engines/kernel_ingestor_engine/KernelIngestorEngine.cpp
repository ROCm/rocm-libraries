// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "engines/kernel_ingestor_engine/HandleDeviceResolver.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

std::filesystem::path descriptorSearchDirectory()
{
    // The install tree is the only path compiled in. A build-tree default would be an
    // absolute path from this machine baked into the shipped plugin, preferred over the
    // installed copy on any host where it happens to exist, and it would mean nothing
    // ever exercises the installed one. Tests and run-from-build-dir set the variable
    // instead, the same way the ASM engine takes HIPDNN_AITER_ASM_DIR.
    if(const auto override = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR");
       !override.empty())
    {
        return override;
    }

    return HIPDNN_DESCRIPTOR_INSTALL_DIR;
}

namespace
{

/// Registers every pack's native symbols, one SymbolScope per pack. A pack that throws is
/// rolled back and skipped, because one pack's duplicate symbol must not unregister every
/// other pack's. Runs under call_once, where a throw is catchable; at static-init it would
/// terminate the process during dlopen().
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

const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet>& discoverDescriptorSets()
{
    // Memoized because two callers read it at different times: Container's static
    // engine-id enumeration and Container's constructor. "Read once at startup" has to
    // mean once, not once per caller, so the two can never disagree about what shipped.
    static const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet> s_sets = [] {
        // Before the scan, not after: validation asks the registry whether each
        // descriptor's symbol exists, so an unregistered pack drops its descriptors here
        // rather than throwing later at first use.
        registerNativeIngestorSymbols();
        return hipdnn_plugin_sdk::ingestor::loadValidatedDescriptorSets<Handle>(
            descriptorSearchDirectory());
    }();

    return s_sets;
}


} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
