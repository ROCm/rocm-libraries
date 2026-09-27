// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// Models are loaded only when a multi-candidate selection needs the architecture's
/// scorer. Explicitly unavailable architecture entries suppress default fallback.
/// @param knobs The UED's declared knobs and @p kmdFields the KMD's declared fields, the
///        two halves of RFC 0019 §6.3 check 2. Both are re-checked here rather than only in
///        the offline validator, because a descriptor set is drop-in: the set this provider
///        loads need not be the set the pipeline emitted.
inline std::shared_ptr<IKernelHeuristic>
    makeKernelHeuristic(const std::optional<HeuristicDescriptor>& descriptor,
                        const std::string& describedBy = {},
                        const std::vector<std::string>& knobs = {},
                        const std::unordered_set<std::string>& kmdFields = {},
                        const std::map<std::string, HeuristicDescriptor>& byArch = {},
                        const std::set<std::string>& unavailableArches = {})
{
    if(byArch.empty() && unavailableArches.empty() && descriptor)
    {
        if(descriptor->adapter == UhdAdapter::STATIC_ORDER)
        {
            return std::make_shared<UnrankedKernelHeuristic>();
        }
        if(descriptor->adapter == UhdAdapter::NATIVE && descriptor->featuresSignature.empty())
        {
            return std::make_shared<NativeKernelHeuristic>(
                descriptor->nativeSymbol,
                describeDescriptor("heuristic", descriptor->name, descriptor->id));
        }
    }
    auto entries = byArch;
    if(descriptor)
    {
        entries.emplace("default", *descriptor);
    }
    if(!entries.empty() || !unavailableArches.empty())
    {
        return UhdKernelHeuristic::makeArchResolver(
            entries, describedBy, knobs, kmdFields, unavailableArches);
    }
    HIPDNN_PLUGIN_LOG_WARN("ingestor: "
                           << (describedBy.empty() ? "engine" : describedBy)
                           << " ships no heuristic; kernels rank by priority, then descriptor id");
    return std::make_shared<UnrankedKernelHeuristic>();
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
