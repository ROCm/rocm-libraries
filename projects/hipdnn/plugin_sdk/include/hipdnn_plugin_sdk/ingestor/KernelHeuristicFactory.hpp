// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <map>
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
/// scorer. Explicitly unavailable (metric, architecture) entries suppress that metric's
/// `default` fallback.
/// @param descriptor A single ranker, bound as its metric's `default` entry unless
///        @p byMetric already names one there. A metric-less `static_order` or signature-less
///        `native` descriptor with nothing else bound is the whole ranking: it is the
///        default ranker for every metric (RFC 0019 §3.1), so it is built directly.
/// @param knobs The UED's declared knobs and @p kmdFields the KMD's declared fields, the
///        two halves of RFC 0019 §6.3 check 2. Both are re-checked here rather than only in
///        the offline validator, because a descriptor set is drop-in: the set this provider
///        loads need not be the set the pipeline emitted.
/// @param byMetric The engine's `sort_kernel_catalog` UHDs by metric (`""` for the
///        metric-less ranker), then architecture key -- DescriptorSet::heuristicsByMetric.
/// @param unavailable Metric to architecture keys whose named model was refused.
inline std::shared_ptr<IKernelHeuristic> makeKernelHeuristic(
    const std::optional<HeuristicDescriptor>& descriptor,
    const std::string& describedBy = {},
    const std::vector<std::string>& knobs = {},
    const std::unordered_set<std::string>& kmdFields = {},
    const std::map<std::string, std::map<std::string, HeuristicDescriptor>>& byMetric = {},
    const std::map<std::string, std::set<std::string>>& unavailable = {})
{
    if(byMetric.empty() && unavailable.empty() && descriptor && descriptor->score.metric.empty())
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
    auto entries = byMetric;
    if(descriptor)
    {
        entries[descriptor->score.metric].emplace("default", *descriptor);
    }
    if(!entries.empty() || !unavailable.empty())
    {
        return UhdKernelHeuristic::makeResolver(
            entries, describedBy, knobs, kmdFields, unavailable);
    }
    HIPDNN_PLUGIN_LOG_WARN("ingestor: "
                           << (describedBy.empty() ? "engine" : describedBy)
                           << " ships no heuristic; kernels rank by priority, then descriptor id");
    return std::make_shared<UnrankedKernelHeuristic>();
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
