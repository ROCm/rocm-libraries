// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "KpackKernelLoader.hpp"

#include "KpackProgram.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::compilation
{

std::unique_ptr<ICompiledProgram> KpackKernelLoader::load(const std::filesystem::path& archive,
                                                          const std::string& tocKey,
                                                          const std::string& deviceArch,
                                                          const std::string& symbol,
                                                          const std::string& descriptorLabel) const
{
    const std::string archivePath = archive.string();

    CachedKpackModule module;
    try
    {
        // The cache key is (archivePath, tocKey, deviceArch). `symbol` is deliberately
        // absent: kernels differing only by entry point name the same blob and must
        // share one hipModule_t. It is used below for the message only.
        module = _moduleCache.getOrLoad(archivePath, tocKey, deviceArch);
    }
    catch(const KpackModuleLoadFailure& failure)
    {
        // The cache knows the stage and what went wrong; only this layer knows who
        // asked. Prefixing here is what lets every message name the descriptor and the
        // symbol without either entering the key.
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "kpack kernel source for " + descriptorLabel
                                                           + ", symbol '" + symbol
                                                           + "': " + failure.what());
    }

    if(!module)
    {
        // Unreachable via KpackModuleCache::load, which throws rather than returning
        // null. Kept so a future loader that returns falsy cannot produce a null
        // dereference in KpackProgram instead of a message.
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "kpack kernel source for " + descriptorLabel + ", symbol '" + symbol
                + "': no module was produced for toc_key '" + tocKey + "' from archive '"
                + archivePath + "'");
    }

    return std::make_unique<KpackProgram>(std::move(module), descriptorLabel);
}

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
