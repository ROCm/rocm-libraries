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
    // Three sources, in falling order of how much they know about this specific install.
    //
    // 1. The environment, which is the only one an operator or a test can set. Tests and
    //    run-from-build-dir use it, the same way the ASM engine takes HIPDNN_AITER_ASM_DIR.
    if(const auto override = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR");
       !override.empty())
    {
        return override;
    }

    // 2. Where this plugin was actually loaded from. HIPDNN_DESCRIPTOR_INSTALL_DIR below
    //    bakes in the prefix this build was *configured* with, which a DESTDIR-staged,
    //    relocated or repackaged install (ROCm packaging, conda, wheels) invalidates --
    //    and the cost of getting it wrong is the whole descriptor-backed engine inventory,
    //    reported only as "0 descriptor-backed engine(s) loaded from <a path that is not
    //    there>". Measuring from the loaded module instead is correct wherever it lands,
    //    because the descriptors install at a fixed offset from the plugin.
    //
    //    hipdnnPluginGetName is this plugin's own exported entry point, so an RTLD_DEFAULT
    //    lookup from inside the plugin finds this module even when several RTLD_LOCAL
    //    providers export that same name. No Windows counterpart takes a symbol yet, so
    //    there the chain is 1 then 3, exactly as before.
#if defined(__linux__)
    try
    {
        const auto candidate
            = hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForSymbol("hipdnnPluginGetName")
              / HIPDNN_DESCRIPTOR_SUBDIR;
        std::error_code notFound;
        if(std::filesystem::is_directory(candidate, notFound))
        {
            return candidate;
        }
    }
    catch(const std::runtime_error& error)
    {
        // No loaded module to measure from -- a statically linked test binary, say.
        // Not fatal: step 3 still answers. Logged because on a real install it would
        // mean the relocatable path silently stopped working.
        HIPDNN_PLUGIN_LOG_INFO("ingestor: no module-relative descriptor directory ("
                               << error.what() << "); using the configure-time path");
    }
#endif

    // 3. The configure-time prefix. Right for an install that never moved.
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
