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
    // Three sources, in falling order of specificity. Env var first: it's the only one an
    // operator or test can set (tests and run-from-build-dir use it, like the ASM engine's
    // HIPDNN_AITER_ASM_DIR). The SDK's reader has already dropped it if it does not name a
    // real directory -- a stale value is common (install-tree CTestTestfile.cmake bakes in
    // the build's staging path, which won't exist on a test machine), and trusting it
    // blindly would load nothing at all -- so an empty replacement means "fall through".
    if(const auto replacement
       = hipdnn_plugin_sdk::ingestor::environmentDescriptorRoots().replacement;
       !replacement.empty())
    {
        return replacement;
    }

    // 2. Where this plugin was actually loaded from. HIPDNN_DESCRIPTOR_INSTALL_DIR bakes in
    //    the configure-time prefix, which a relocated or repackaged install invalidates;
    //    measuring from the loaded module is correct wherever it lands. Keyed on this
    //    function's own address rather than a symbol name, since a name lookup can resolve
    //    to a different module when every provider exports the same plugin entry points.
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
        // the static test binaries. Not fatal, step 3 still answers; logged because on a
        // real install it would mean the relocatable path had silently stopped working.
        HIPDNN_PLUGIN_LOG_INFO("ingestor: no module-relative descriptor directory ("
                               << error.what() << "); using the configure-time path");
    }

    // 3. The configure-time prefix. Right for an install that never moved.
    return HIPDNN_DESCRIPTOR_INSTALL_DIR;
}

std::vector<std::filesystem::path> descriptorSearchDirectories()
{
    std::vector<std::filesystem::path> roots{descriptorSearchDirectory()};

    // Additive, where HIPDNN_DESCRIPTOR_DIR above replaces: this is where descriptors are
    // dropped in beside a shipped install rather than instead of it.
    for(auto& additional : hipdnn_plugin_sdk::ingestor::environmentDescriptorRoots().additional)
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
    // Memoized: the descriptor-backed engines and every opaque engine's declared L1 model
    // (resolveDeclaredEnginePredictions) come out of the same trees, and walking them
    // once per consumer would both cost a second parse and let two consumers disagree.
    static const hipdnn_plugin_sdk::ingestor::DescriptorCatalog s_catalog
        = hipdnn_plugin_sdk::ingestor::loadDescriptorCatalog(descriptorSearchDirectories());

    return s_catalog;
}

const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet>& discoverDescriptorSets()
{
    // Memoized: Container's static engine-id enumeration and its constructor both call
    // this and must agree on what shipped.
    static const std::vector<hipdnn_plugin_sdk::ingestor::DescriptorSet> s_sets = [] {
        // Register before scanning: validation checks each descriptor's symbol against the
        // registry, so an unregistered pack drops its descriptors here instead of throwing
        // at first use.
        registerNativeIngestorSymbols();
        return hipdnn_plugin_sdk::ingestor::loadValidatedDescriptorSets<Handle>(
            descriptorCatalog());
    }();

    return s_sets;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
