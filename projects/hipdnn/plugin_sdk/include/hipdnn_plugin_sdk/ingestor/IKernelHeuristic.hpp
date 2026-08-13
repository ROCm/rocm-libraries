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

/**
 * @brief Chooses which kernel within an engine to run, given the kernels that fit.
 *
 * One level below hipDNN's engine-selection heuristic, which picks the engine.
 *
 * An implementation supplies only `score()`: it ranks one kernel at a time and never
 * sees the catalog, so filtering and ranking commute (RFC 0017 §9.2). `rank()` is
 * non-virtual; a selector over the whole candidate set is the heuristic follow-up RFC.
 */
class IKernelHeuristic
{
public:
    virtual ~IKernelHeuristic() = default;

    /// @brief Scores one kernel for one problem. Higher wins.
    virtual double score(const KernelDefinition& kernel, const MatchContext& context) const = 0;

    /**
     * @brief Orders @p catalog best-first.
     *
     * Scores each entry once, then sorts descending, breaking ties on the kernel's
     * `priority`, then its descriptor id as bytes (stable across runs, load orders,
     * and machines).
     */
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

/**
 * @brief An IKernelHeuristic whose score() is a native function resolved by symbol.
 *
 * The UHD escape hatch: the descriptor names a symbol, this resolves it at
 * construction.
 *
 * Eager because the check is a fact about the build. The registry is fully populated
 * before any state manager exists and never mutates, so passing once means passing
 * forever, and failing means this binary does not ship what the descriptor names.
 * Resolving on first score() instead would run after isApplicable() already answered
 * true, past the point RFC 0017 §8.6 makes that a binding promise. It also drops a
 * mutexed lookup and a once_flag from every ranking call.
 *
 * That argument does not extend to loading a model artifact (HeuristicKind::MODEL).
 * An artifact is a fact about the environment: I/O, potentially large against §3/§8.1's
 * cheap-construction rule, and removable between load and first rank(). Such an adapter
 * should validate its descriptor eagerly and load its artifact lazily.
 *
 * Nor is a failure to rank a failure to serve. Ranking runs only from
 * sortedDefinitions(), over a catalog whose entries already passed every matcher, so
 * each is launchable. An adapter whose artifact will not load should degrade to
 * rank()'s priority-then-id order (§9.2's `static_order`) and log once rather than fail
 * a plan: losing the best kernel is a performance regression, losing the plan is an
 * outage.
 */
class NativeKernelHeuristic : public IKernelHeuristic
{
public:
    /// @param scoreSymbol Resolved immediately.
    /// @param describedBy The UHD naming @p scoreSymbol, for the failure message.
    ///
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

/**
 * @brief Builds the IKernelHeuristic a UHD names, keyed on its kind.
 *
 * @throws std::invalid_argument if @p descriptor names a kind with no adapter yet
 *         (HeuristicKind::MODEL).
 * @throws std::runtime_error if a NATIVE descriptor names a scorer this build does not
 *         ship.
 *
 * Both are descriptor errors, and both fail here rather than at the first rank(). An
 * adapter needing an artifact from disk should not load it here; see
 * NativeKernelHeuristic for why that boundary sits where it does.
 */
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
