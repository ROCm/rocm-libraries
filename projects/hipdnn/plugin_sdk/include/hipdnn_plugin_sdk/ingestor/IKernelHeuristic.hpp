// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <memory>
#include <mutex>
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
 * One level below hipDNN's engine-selection heuristic, which decides *which engine*
 * handles a graph and is untouched by this system.
 *
 * An implementation supplies `score()` and nothing else: it ranks one kernel at a
 * time and never sees the catalog, so a kernel's score cannot depend on which other
 * kernels are present. That is what makes filtering and ranking commute (RFC 0017
 * §9.2), so a knob-filtered subset ranks exactly as it did in the whole catalog.
 * `rank()` is therefore non-virtual; a selector that must reason over the candidate
 * set as a whole is the heuristic follow-up RFC's job.
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
     * Scores each entry independently and sorts descending, breaking ties first on
     * the kernel's explicit `priority`, then on its descriptor id compared as bytes
     * (stable across runs, load orders, and machines).
     *
     * Each kernel is scored exactly once, before sorting, rather than inside the
     * comparator: scoring from the comparator would run inference O(n log n) times
     * instead of n for an n-kernel catalog.
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
 * The UHD escape hatch: the descriptor names a symbol, this resolves it. The
 * data-driven form loads a model artifact instead (the UHD follow-up RFC), and slots
 * in here as another IKernelHeuristic.
 *
 * **Constructing a heuristic must stay cheap; whatever it selects with loads on first
 * use** (RFC 0017 §8.1, §3): an engine whose matchers reject a graph must never pay
 * for its selector. This one holds a symbol name and resolves it on first score(); a
 * LightGBM adapter would hold an artifact path and read the model there.
 */
class NativeKernelHeuristic : public IKernelHeuristic
{
public:
    /// @param scoreSymbol Resolved from the registry on first use, not here, so
    ///        constructing this costs nothing an engine that never ranks would pay.
    explicit NativeKernelHeuristic(std::string scoreSymbol)
        : _scoreSymbol(std::move(scoreSymbol))
    {
    }

    double score(const KernelDefinition& kernel, const MatchContext& context) const override
    {
        std::call_once(_resolved, [this]() { _scoreFn = ScoreRegistry::resolve(_scoreSymbol); });
        return _scoreFn(kernel, context);
    }

private:
    std::string _scoreSymbol;
    /// Resolved once, on the first call that needs an order. Mutable because score() is
    /// logically const: resolving is an implementation detail of answering, and a plan
    /// may rank from several threads at once.
    mutable std::once_flag _resolved;
    mutable ScoreFn _scoreFn = nullptr;
};

/**
 * @brief Builds the IKernelHeuristic a UHD names, keyed on its kind.
 *
 * @throws std::invalid_argument if @p descriptor names a kind with no adapter yet
 *         (HeuristicKind::MODEL). Fails at descriptor-assembly time, matching
 *         KernelIngestorStateManager's other cross-reference checks: a kind nothing
 *         can load is a load error, not a runtime surprise at the first rank().
 */
inline std::shared_ptr<IKernelHeuristic> makeKernelHeuristic(const HeuristicDescriptor& descriptor)
{
    switch(descriptor.kind)
    {
    case HeuristicKind::NATIVE:
        return std::make_shared<NativeKernelHeuristic>(descriptor.payload);
    default:
        throw std::invalid_argument("heuristic '" + toString(descriptor.id)
                                    + "' names a kind with no adapter yet");
    }
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
