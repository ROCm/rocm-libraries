// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
    std::vector<KernelDefinition> rank(const Catalog& catalog, const MatchContext& context) const
    {
        std::vector<std::pair<double, const KernelDefinition*>> scored;
        scored.reserve(catalog.entries.size());
        for(const auto& entry : catalog.entries)
        {
            scored.emplace_back(score(entry, context), &entry);
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

/// @throws std::invalid_argument if @p descriptor names a kind with no adapter yet.
inline std::shared_ptr<IKernelHeuristic> makeKernelHeuristic(const HeuristicDescriptor& descriptor)
{
    switch(descriptor.kind)
    {
    case HeuristicKind::NATIVE:
        return std::make_shared<NativeKernelHeuristic>(
            descriptor.payload, describeDescriptor("heuristic", descriptor.name, descriptor.id));
    default:
        throw std::invalid_argument("heuristic '" + toString(descriptor.id)
                                    + "' names a kind with no adapter yet");
    }
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
