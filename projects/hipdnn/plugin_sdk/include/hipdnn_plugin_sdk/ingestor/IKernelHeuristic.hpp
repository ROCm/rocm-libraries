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
 * The UHD escape hatch: the descriptor names a symbol, this resolves it on first
 * score(). Construction must stay cheap (RFC 0017 §8.1, §3). A data-driven form (UHD
 * follow-up RFC) would load a model artifact instead.
 */
class NativeKernelHeuristic : public IKernelHeuristic
{
public:
    /// @param scoreSymbol Resolved from the registry on first use, not here.
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
    /// Mutable: resolution is an implementation detail of a logically-const score(),
    /// which may be called from several threads at once.
    mutable std::once_flag _resolved;
    mutable ScoreFn _scoreFn = nullptr;
};

/**
 * @brief Builds the IKernelHeuristic a UHD names, keyed on its kind.
 *
 * @throws std::invalid_argument if @p descriptor names a kind with no adapter yet
 *         (HeuristicKind::MODEL). Fails at descriptor-assembly time, not at the
 *         first rank().
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
