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

    /// Operands in the pipeline order every stage shares; see NativeRegistry.hpp.
    virtual double score(const MatchContext& context,
                         const BoundTokens& bound,
                         const KernelDefinition& kernel) const
        = 0;

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
    ///
    /// Virtual because an implementation may need the whole catalog at once where this
    /// default needs only one kernel at a time. A model-backed heuristic has two such
    /// needs: the problem and device parts of its feature row are the same for every
    /// candidate and should be computed once, and a model that fails partway through must
    /// abandon the whole ranking rather than leave a mix of real scores and sentinels,
    /// which would be neither the model's order nor the fallback's.
    /// What decided the order, for the §12 trace: a compiled scorer, or nothing at all.
    /// Overridden by the unranked fallback, which declines to rank.
    virtual std::string traceDecidedBy() const
    {
        return "native";
    }

    virtual std::vector<KernelDefinition> rank(const Catalog& catalog,
                                               const MatchContext& context) const
    {
        std::vector<std::pair<double, const KernelDefinition*>> scored;
        scored.reserve(catalog.entries.size());
        for(const auto& entry : catalog.entries)
        {
            const double raw = score(context, catalog.bound, entry);
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

        // RFC 0019 §12's selection trace, for every heuristic that ranks through this
        // default -- native scorers and the unranked fallback. UhdKernelHeuristic overrides
        // rank() and traces its own, with the model provenance §12 also asks for. Two of the
        // three shipped UHDs are `native` kind, so tracing only the model path would leave
        // most real selections invisible.
        if(!scored.empty() && ::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO))
        {
            std::ostringstream candidates;
            for(size_t i = 0; i < scored.size(); ++i)
            {
                candidates << (i == 0 ? "" : " ") << toString(scored[i].second->kernelId) << "="
                           << scored[i].first;
            }
            HIPDNN_PLUGIN_LOG_INFO("uhd trace: decided_by=" << traceDecidedBy()
                                   << " winner=" << toString(scored.front().second->kernelId)
                                   << " candidates=" << scored.size()
                                   << " arch=" << context.deviceProperties.gcnArchName
                                   << " ranked=[" << candidates.str() << "]");
        }

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

    double score(const MatchContext& context,
                 const BoundTokens& bound,
                 const KernelDefinition& kernel) const override
    {
        return _scoreFn(context, bound, kernel);
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
    /// §12 asks whether the model or a fallback decided. For this one it is always the
    /// fallback, and saying so is the point: an engine ranking on priority because it ships
    /// no UHD looks identical in the output to one whose model ranked that way.
    std::string traceDecidedBy() const override
    {
        return "declared_order";
    }

    double score(const MatchContext& /*context*/,
                 const BoundTokens& /*bound*/,
                 const KernelDefinition& /*kernel*/) const override
    {
        return 0.0;
    }
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
