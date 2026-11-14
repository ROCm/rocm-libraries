// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The conv engine group's runtime switch (conv/README.md). On this UHD training branch a
/// build that ships the group (HIPKERNELPROVIDER_ENABLE_CONV) serves its engines unless the
/// variable is 0/false/off/no, so the UHD pipeline's corpus, collection and validation
/// jobs see them without setting it; a false value serves exactly what a build without the
/// group serves.
inline constexpr const char* CONV_ENABLE_ENV = "HIP_KERNEL_PROVIDER_ENABLE_CONV";

/// Every engine of the conv group is named with this prefix (descriptors/conv/<family>).
inline constexpr std::string_view CONV_ENGINE_PREFIX = "hipkernel:CONV_";

/// Whether the group is served: unless CONV_ENABLE_ENV is set to a false value.
inline bool convEngineGroupEnabled()
{
    const auto raw = hipdnn_data_sdk::utilities::toLower(
        hipdnn_data_sdk::utilities::trim(hipdnn_data_sdk::utilities::getEnv(CONV_ENABLE_ENV)));
    return !(raw == "0" || raw == "false" || raw == "off" || raw == "no");
}

/// Takes the conv group's engines out of @p catalog, with the packs that bind them and the
/// standalone kernels only those packs name, so validation, discovery and every engine id
/// the provider advertises proceed as if the group were not installed.
inline void withdrawConvEngineGroup(hipdnn_plugin_sdk::ingestor::DescriptorCatalog& catalog)
{
    using hipdnn_plugin_sdk::ingestor::DescriptorId;
    using hipdnn_plugin_sdk::ingestor::DescriptorIdHash;

    std::unordered_set<DescriptorId, DescriptorIdHash> engines;
    for(auto it = catalog.engines.begin(); it != catalog.engines.end();)
    {
        if(std::string_view(it->second.descriptor.name).substr(0, CONV_ENGINE_PREFIX.size())
           == CONV_ENGINE_PREFIX)
        {
            engines.insert(it->first);
            it = catalog.engines.erase(it);
        }
        else
        {
            ++it;
        }
    }
    if(engines.empty())
    {
        return;
    }

    std::set<DescriptorId> withdrawnKernels;
    std::set<DescriptorId> keptKernels;
    for(auto it = catalog.packs.begin(); it != catalog.packs.end();)
    {
        const auto& pack = it->second.descriptor;
        const bool withdrawn = engines.count(pack.engineId) != 0;
        auto& kernels = withdrawn ? withdrawnKernels : keptKernels;
        kernels.insert(pack.kernelIds.begin(), pack.kernelIds.end());
        it = withdrawn ? catalog.packs.erase(it) : std::next(it);
    }
    for(auto it = catalog.kernels.begin(); it != catalog.kernels.end();)
    {
        const auto& id = it->first.first;
        const bool withdrawn = withdrawnKernels.count(id) != 0 && keptKernels.count(id) == 0;
        it = withdrawn ? catalog.kernels.erase(it) : std::next(it);
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
