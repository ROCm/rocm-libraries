// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
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
 * Two levels of interface, deliberately:
 *
 * - `score()` ranks one kernel at a time and never sees the catalog. Because a kernel's
 *   score cannot depend on which other kernels are present, filtering and ranking
 *   commute: a knob-filtered subset ranks exactly as it did in the whole catalog, so the
 *   kernel a knob setting selects is the kernel the reported default named. That failure
 *   — a reported default a knob setting then contradicts — is silent and surfaces to a
 *   user rather than to the author who caused it, which is why the property is
 *   structural here rather than advisory.
 *
 * - `rank()` is virtual and receives the whole catalog. A future selector that must
 *   reason over the candidate set as a whole (spreading a choice across a batch, or
 *   ranking by a criterion only meaningful relatively) overrides it.
 *
 * **Overriding `rank()` forfeits the consistency guarantee above.** An override must
 * state what a knob-filtered query then means for it; the default implementation is the
 * shape every day-one descriptor-backed heuristic should use.
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
     * The default implementation scores each entry independently and sorts descending,
     * breaking ties deterministically: first on the kernel's explicit `priority`, then on
     * its descriptor id compared as bytes. That id order carries no meaning — it is
     * chosen only for being stable across runs, load orders, and machines, so two
     * processes given the same catalog always choose the same kernel.
     */
    virtual std::vector<KernelDefinition> rank(const Catalog& catalog,
                                               const MatchContext& context) const
    {
        std::vector<KernelDefinition> ranked = catalog.entries;

        std::stable_sort(
            ranked.begin(),
            ranked.end(),
            [this, &context](const KernelDefinition& lhs, const KernelDefinition& rhs) {
                const double lhsScore = score(lhs, context);
                const double rhsScore = score(rhs, context);
                if(lhsScore != rhsScore)
                {
                    return lhsScore > rhsScore;
                }
                if(lhs.priority != rhs.priority)
                {
                    return lhs.priority > rhs.priority;
                }
                return lhs.kernelId < rhs.kernelId;
            });

        return ranked;
    }
};

/**
 * @brief An IKernelHeuristic whose score() is a native function resolved by symbol.
 *
 * The UHD escape hatch: the descriptor names a symbol, this resolves it. The
 * data-driven form loads a model artifact and assembles its feature vector from the
 * bound token state instead — the UHD follow-up RFC — and slots in here as another
 * IKernelHeuristic.
 */
class NativeKernelHeuristic : public IKernelHeuristic
{
public:
    explicit NativeKernelHeuristic(ScoreFn scoreFn)
        : _scoreFn(scoreFn)
    {
    }

    double score(const KernelDefinition& kernel, const MatchContext& context) const override
    {
        return _scoreFn(kernel, context);
    }

private:
    ScoreFn _scoreFn;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
