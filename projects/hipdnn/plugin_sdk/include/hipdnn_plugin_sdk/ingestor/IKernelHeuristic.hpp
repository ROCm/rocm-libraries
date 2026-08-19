// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// Chooses which kernel within an engine to run. An implementation supplies only
/// `score()`, ranking one kernel at a time without seeing the catalog, so filtering
/// and ranking commute.
class IKernelHeuristic
{
public:
    virtual ~IKernelHeuristic() = default;

    virtual double score(const KernelDefinition& kernel, const MatchContext& context) const = 0;

    /// Orders @p catalog best-first, breaking ties on `priority`, then descriptor id
    /// bytes (stable across runs).
    ///
    /// A NaN score ranks last rather than poisoning the order. `score()` is supplied by
    /// the pack, so its value is outside this class's control, and NaN compares false
    /// against everything -- it would read as equivalent to every kernel while real
    /// scores stayed ordered among themselves, which is not a strict weak ordering and
    /// is undefined behaviour for stable_sort. Mapping it to -infinity keeps the order
    /// total, so a pack that returns NaN loses selection quality without costing
    /// determinism or reaching UB. Infinities are already well-ordered and pass through.
    std::vector<KernelDefinition> rank(const Catalog& catalog, const MatchContext& context) const
    {
        std::vector<std::pair<double, const KernelDefinition*>> scored;
        scored.reserve(catalog.entries.size());
        for(const auto& entry : catalog.entries)
        {
            const double raw = score(entry, context);
            scored.emplace_back(std::isnan(raw) ? -std::numeric_limits<double>::infinity() : raw,
                                &entry);
        }

        std::stable_sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
            if(lhs.first != rhs.first)
            {
                return lhs.first > rhs.first;
            }
            if(lhs.second->priority != rhs.second->priority)
            {
                return lhs.second->priority > rhs.second->priority;
            }
            return lhs.second->kernelId < rhs.second->kernelId;
        });

        std::vector<KernelDefinition> ranked;
        ranked.reserve(scored.size());
        for(const auto& [_, entry] : scored)
        {
            ranked.push_back(*entry);
        }
        return ranked;
    }
};

/// score() is a native function resolved by symbol, eagerly at construction: the
/// registry is fully populated and immutable by then, so a missing symbol is a build
/// fact, not a per-call race.
class NativeKernelHeuristic : public IKernelHeuristic
{
public:
    /// @throws std::runtime_error if @p scoreSymbol is not registered.
    explicit NativeKernelHeuristic(const std::string& scoreSymbol,
                                   const std::string& describedBy = {})
        : _scoreFn(ScoreRegistry::resolve(scoreSymbol, describedBy))
    {
    }

    double score(const KernelDefinition& kernel, const MatchContext& context) const override
    {
        return _scoreFn(kernel, context);
    }

private:
    ScoreFn _scoreFn;
};

/// Used when an engine ships no UHD: scores every kernel alike, so rank()'s tie-break
/// decides. Named for what it does -- it adds no ordering rule of its own and just
/// declines to rank. The tie-break it falls through to is `priority` descending then
/// descriptor id ascending, which is not authoring order: an id is a UUID and sorts by
/// its bytes. Ranking stays total and stable, so the absence of a model costs selection
/// quality, never determinism.
class UnrankedKernelHeuristic : public IKernelHeuristic
{
public:
    double score(const KernelDefinition& /*kernel*/, const MatchContext& /*context*/) const override
    {
        return 0.0;
    }
};

/// @param describedBy Engine named in the warning when @p descriptor is nullopt.
/// @throws std::invalid_argument if @p descriptor names a kind with no adapter yet.
inline std::shared_ptr<IKernelHeuristic>
    makeKernelHeuristic(const std::optional<HeuristicDescriptor>& descriptor,
                        const std::string& describedBy = {})
{
    if(!descriptor.has_value())
    {
        // Warn, not fail: an engine with no model still selects deterministically. The
        // warning is the point -- it separates an engine that declares its order from
        // one still waiting on a UHD, which otherwise look identical from the outside.
        HIPDNN_PLUGIN_LOG_WARN("ingestor: " << (describedBy.empty() ? "engine" : describedBy)
                                            << " ships no heuristic; kernels rank by priority, "
                                               "then descriptor id");
        return std::make_shared<UnrankedKernelHeuristic>();
    }

    switch(descriptor->kind)
    {
    case HeuristicKind::NATIVE:
        return std::make_shared<NativeKernelHeuristic>(
            descriptor->payload, describeDescriptor("heuristic", descriptor->name, descriptor->id));
    default:
        throw std::invalid_argument("heuristic '" + toString(descriptor->id)
                                    + "' names a kind with no adapter yet");
    }
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
