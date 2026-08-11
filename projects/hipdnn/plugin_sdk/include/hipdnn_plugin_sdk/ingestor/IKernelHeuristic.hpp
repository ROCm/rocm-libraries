// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
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
 * An implementation supplies `score()` and nothing else. It ranks one kernel at a time
 * and never sees the catalog, so a kernel's score cannot depend on which other kernels
 * are present. That is what makes filtering and ranking commute: a knob-filtered subset
 * ranks exactly as it did in the whole catalog, so the kernel a knob setting selects is
 * the kernel the reported default named. The failure it prevents, a reported default
 * that a knob setting then contradicts, is silent and surfaces to a user rather than to
 * the author who caused it.
 *
 * RFC 0017 §9.2 makes that structural rather than advisory: the scorer interface "takes
 * one kernel at a time and is never handed the catalog, so a scorer that ranks relative
 * to its peers cannot be written against it". `rank()` is therefore non-virtual. A
 * selector that must reason over the candidate set as a whole is admitted by the
 * heuristic follow-up RFC, which owns deciding what a knob-filtered query means once
 * ranking is no longer per kernel; re-opening this for override then is additive.
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
     * Scores each entry independently and sorts descending, breaking ties
     * deterministically: first on the kernel's explicit `priority`, then on its
     * descriptor id compared as bytes. That id order carries no meaning — it is chosen
     * only for being stable across runs, load orders, and machines, so two processes
     * given the same catalog always choose the same kernel.
     *
     * Each kernel is scored exactly once, before sorting, rather than inside the
     * comparator. A scorer is a model evaluation in the data-driven form, so scoring
     * from the comparator would run inference O(n log n) times for an n-kernel catalog
     * instead of n, and would also let a scorer that is not perfectly deterministic
     * produce an inconsistent ordering.
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
 * The UHD escape hatch: the descriptor names a symbol, this resolves it. The data-driven
 * form loads a model artifact and assembles its feature vector from the bound token state
 * instead (the UHD follow-up RFC), and slots in here as another IKernelHeuristic.
 *
 * **Constructing a heuristic must stay cheap; whatever it selects with loads on first
 * use.** RFC 0017 §8.1 admits the heuristic at applicability only as a name: "The
 * heuristic is named but **not** loaded; nothing ranks yet", and §3 generalizes it,
 * "a heuristic model is not read until something needs the catalog ranked". An engine
 * whose matchers reject a graph must never pay for its selector.
 *
 * That contract lives here rather than one level up, because only the adapter knows what
 * "loading" costs it. This one holds a symbol name and resolves it on first score(); a
 * LightGBM adapter would hold an artifact path and read the model there. Either way the
 * object is constructible before any graph is seen, which is what lets the engine own it
 * outright instead of threading a factory through the state manager.
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

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
