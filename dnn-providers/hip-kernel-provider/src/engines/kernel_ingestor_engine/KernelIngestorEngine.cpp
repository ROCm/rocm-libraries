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
    //
    //    Taken only if it names a real directory. A stale value is not hypothetical: the
    //    install-tree CTestTestfile.cmake is generated from the same ENVIRONMENT list as
    //    the build-tree one, so it carries this build's absolute staging path, and on a
    //    test machine that path does not exist. Honouring it unconditionally would win
    //    over both branches below and load nothing at all -- the failure this whole
    //    function is ordered to avoid.
    if(const auto override = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR");
       !override.empty())
    {
        std::error_code notFound;
        if(std::filesystem::is_directory(override, notFound))
        {
            return override;
        }
        HIPDNN_PLUGIN_LOG_WARN("ingestor: HIPDNN_DESCRIPTOR_DIR is set to '"
                               << override
                               << "', which is not a directory; ignoring it and resolving "
                                  "the descriptor tree from the loaded module instead");
    }

    // 2. Where this plugin was actually loaded from. HIPDNN_DESCRIPTOR_INSTALL_DIR below
    //    bakes in the prefix this build was *configured* with, which a DESTDIR-staged,
    //    relocated or repackaged install (ROCm packaging, conda, wheels) invalidates --
    //    and the cost of getting it wrong is the whole descriptor-backed engine inventory,
    //    reported only as "0 descriptor-backed engine(s) loaded from <a path that is not
    //    there>". Measuring from the loaded module instead is correct wherever it lands,
    //    because the descriptors install at a fixed offset from the plugin.
    //
    //    Keyed on the address of this very function rather than on a symbol name. A name
    //    lookup goes through the dynamic linker's scope rules and can in principle answer
    //    for a different module -- every provider exports the same plugin entry points,
    //    and the backend opens each one RTLD_LOCAL. An address cannot be ambiguous: it
    //    already belongs to exactly one module, under any dlopen flag and on both
    //    platforms, and nothing has to be exported for it to work.
    try
    {
        const auto candidate = hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForAddress(
                                   reinterpret_cast<const void*>(&descriptorSearchDirectory))
                               / HIPDNN_DESCRIPTOR_SUBDIR;
        std::error_code notFound;
        if(std::filesystem::is_directory(candidate, notFound))
        {
            return candidate;
        }
    }
    catch(const std::runtime_error& error)
    {
        // No module to measure from -- this TU linked straight into an executable, as in
        // the static test binaries. Not fatal: step 3 still answers. Logged because on a
        // real install it would mean the relocatable path had silently stopped working.
        HIPDNN_PLUGIN_LOG_INFO("ingestor: no module-relative descriptor directory ("
                               << error.what() << "); using the configure-time path");
    }

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
