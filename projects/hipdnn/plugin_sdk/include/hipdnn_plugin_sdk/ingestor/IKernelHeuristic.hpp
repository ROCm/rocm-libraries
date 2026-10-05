// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// One ranked candidate, as returned by selection (RFC 0019.13 §15.2).
struct ScoredKernel
{
    /// For a single-layer heuristic this is the complete answer.
    DescriptorId kernelId;
    double score;
    /// The group chosen by a two-layer heuristic (e.g. MIOpen's solver); `groupFeature()` names
    /// the field. Per candidate because a runner-up may sit in a different group than the
    /// winner. NaN means no group was decided; every real number is a legal value.
    double group = std::numeric_limits<double>::quiet_NaN();
};

namespace detail
{

/// `priority` descending, then id ascending (RFC 0019 §5 step 5). Never discovery order, which
/// varies by filesystem.
inline std::vector<KernelDefinition> declaredOrder(const std::vector<KernelDefinition>& entries)
{
    std::vector<KernelDefinition> ordered(entries);
    std::stable_sort(
        ordered.begin(), ordered.end(), [](const KernelDefinition& a, const KernelDefinition& b) {
            if(a.priority != b.priority)
            {
                return a.priority > b.priority;
            }
            return a.kernelId < b.kernelId;
        });
    return ordered;
}

/// Declared order as (id, score) pairs, scoring 0 for "no measurement" (RFC 0019 §5 step 7).
inline std::vector<ScoredKernel> asScored(const std::vector<KernelDefinition>& ordered)
{
    std::vector<ScoredKernel> scored;
    scored.reserve(ordered.size());
    for(const auto& entry : ordered)
    {
        scored.push_back({entry.kernelId, 0.0});
    }
    return scored;
}

} // namespace detail

/// Chooses which kernel within an engine to run. An implementation supplies only
/// `score()`, ranking one kernel at a time without seeing the catalog, so filtering
/// and ranking commute.
class IKernelHeuristic
{
public:
    virtual ~IKernelHeuristic() = default;

private:
    /// Mutable and atomic: ranking runs through a shared_ptr<const> from any thread.
    mutable std::atomic<bool> _reportedScorerFailure{false};

public:
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
    /// Results cross a plugin boundary as ids and scores (§15.2), never objects or references.
    using ScoredKernel = ingestor::ScoredKernel;

    /// Logs a scorer failure once per heuristic; the cause is the descriptor set, so it would
    /// otherwise repeat for every graph.
    void reportScorerFailureOnce(const char* what) const
    {
        if(_reportedScorerFailure.exchange(true))
        {
            return;
        }
        HIPDNN_PLUGIN_LOG_ERROR("uhd: scorer threw while ranking ("
                                << what << "); kernels rank by priority, then descriptor id. "
                                << "Further occurrences for this heuristic are not logged.");
    }

    /// What decided the order, for the §12 trace.
    virtual std::string traceDecidedBy() const
    {
        return "native";
    }

    /// Which field `ScoredKernel::group` holds, or nothing for a single-layer heuristic.
    virtual std::optional<std::string> groupFeature() const
    {
        return std::nullopt;
    }

    /// The ranking in §15.2's form. Override this, not rank(), which derives from it.
    virtual std::vector<ScoredKernel> rankScored(const Catalog& catalog,
                                                 const MatchContext& context) const
    {
        struct Ranked
        {
            double ordering; ///< NaN-free, so the comparator stays a strict weak ordering
            double reported; ///< exactly what score() returned, NaN included
            const KernelDefinition* entry;
        };

        std::vector<Ranked> scored;
        scored.reserve(catalog.entries.size());
        try
        {
            for(const auto& entry : catalog.entries)
            {
                // A non-finite score sorts last and is reported as 0 (§5 step 7). The keys stay
                // separate because NaN in the comparator is undefined behaviour.
                const double raw = score(context, catalog.bound, entry);
                const bool usable = std::isfinite(raw);
                scored.push_back({usable ? raw : -std::numeric_limits<double>::infinity(),
                                  usable ? raw : 0.0,
                                  &entry});
            }
        }
        catch(const std::exception& e)
        {
            // RFC 0019 §5 step 7: a throwing scorer degrades the whole ranking to declared
            // order; it must not fail the request, and a partial ranking is neither order.
            reportScorerFailureOnce(e.what());
            return detail::asScored(detail::declaredOrder(catalog.entries));
        }

        std::stable_sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
            if(lhs.ordering != rhs.ordering)
            {
                return lhs.ordering > rhs.ordering;
            }
            if(lhs.entry->priority != rhs.entry->priority)
            {
                return lhs.entry->priority > rhs.entry->priority;
            }
            return lhs.entry->kernelId < rhs.entry->kernelId;
        });

        // RFC 0019 §12 selection trace. UhdKernelHeuristic traces its own model path.
        if(!scored.empty() && ::hipdnn_data_sdk::logging::isLogLevelEnabled(HIPDNN_SEV_INFO))
        {
            std::ostringstream candidates;
            for(size_t i = 0; i < scored.size(); ++i)
            {
                candidates << (i == 0 ? "" : " ") << toString(scored[i].entry->kernelId) << "="
                           << scored[i].reported;
            }
            HIPDNN_PLUGIN_LOG_INFO("uhd trace: decided_by="
                                   << traceDecidedBy() << " metric=" << context.rankingMetric
                                   << " winner=" << toString(scored.front().entry->kernelId)
                                   << " candidates=" << scored.size()
                                   << " arch=" << context.deviceProperties.gcnArchName
                                   << " ranked=[" << candidates.str() << "]");
        }

        std::vector<ScoredKernel> ranked;
        ranked.reserve(scored.size());
        for(const auto& candidate : scored)
        {
            ranked.push_back({candidate.entry->kernelId, candidate.reported});
        }
        return ranked;
    }

    /// @brief Exact physical estimates in the context's ranking metric, best first; empty when
    ///        this ranker cannot calibrate for it.
    ///
    /// Only a model trained on exactly `context.rankingMetric` may answer (RFC 0019 §4.4, §11.3).
    /// Empty by default so a heuristic is never compared across engines by accident.
    virtual std::vector<ScoredKernel> calibratedRanking(const Catalog& /*catalog*/,
                                                        const MatchContext& /*context*/,
                                                        std::string& /*modelId*/) const
    {
        return {};
    }

    /// @brief Id of the model calibratedRanking() would use for @p metric on @p arch, from bound
    ///        state alone (no loading or ranking); empty when none could answer.
    virtual std::string calibratedModelId(const std::string& /*metric*/,
                                          const std::string& /*arch*/) const
    {
        return {};
    }

    /// The same order as rankScored(), as whole kernels. Non-virtual so the order is decided in
    /// one place.
    std::vector<KernelDefinition> rank(const Catalog& catalog, const MatchContext& context) const
    {
        const auto scored = rankScored(catalog, context);

        std::map<DescriptorId, const KernelDefinition*> byId;
        for(const auto& entry : catalog.entries)
        {
            byId.emplace(entry.kernelId, &entry);
        }

        std::vector<KernelDefinition> ordered;
        ordered.reserve(scored.size());
        // Named member, not a structured binding, so adding ScoredKernel fields needs no edit.
        for(const auto& candidate : scored)
        {
            if(const auto found = byId.find(candidate.kernelId); found != byId.end())
            {
                ordered.push_back(*found->second);
            }
        }
        return ordered;
    }
};

/// score() is a native function resolved by symbol, eagerly at construction: the
/// registry is fully populated and immutable by then, so a missing symbol is a build
/// fact, not a per-call race.
class NativeKernelHeuristic : public IKernelHeuristic
{
public:
    /// @param objective `min` means the scorer returns a cost, which score() negates.
    /// @param transform The UHD's `score.transform`.
    /// @param metric The UHD's `score.metric`; empty for a metric-less ranker.
    /// @throws std::runtime_error if @p scoreSymbol is not registered.
    explicit NativeKernelHeuristic(const std::string& scoreSymbol,
                                   const std::string& describedBy = {},
                                   const std::string& objective = "max",
                                   std::string transform = {},
                                   const std::string& metric = {})
        : _scoreFn(ScoreRegistry::resolve(scoreSymbol, describedBy))
        , _sign(objective == "min" ? -1.0 : 1.0)
        , _transform(std::move(transform))
        // Under `min` a cost must be positive: a native scorer's 0 means "no measurement", and
        // negated it would outrank every real cost. Otherwise §8.3's rule decides.
        , _positiveRequired(objective == "min"
                            || uhd::score_transform::isPhysicalScore(metric, _transform))
    {
    }

    /// The scorer's value inverse-transformed and oriented so higher wins. Recovered before
    /// any zero check, since a transformed 0 is a real value. When positivity is required, a
    /// non-positive or non-finite value becomes NaN (ranked last, reported as 0).
    double score(const MatchContext& context,
                 const BoundTokens& bound,
                 const KernelDefinition& kernel) const override
    {
        const double recovered
            = uhd::score_transform::applyInverse(_scoreFn(context, bound, kernel), _transform);
        if(_positiveRequired && !uhd::score_transform::isRankableScore(recovered, true))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return _sign * recovered;
    }

private:
    ScoreFn _scoreFn;
    double _sign;
    std::string _transform;
    bool _positiveRequired;
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
    /// Always the fallback; the trace must distinguish this from a model that ranked by priority.
    std::string traceDecidedBy() const override
    {
        return "declared_order";
    }

    /// Zero, which RFC 0019 §5 step 7 reports for "no measurement"; ordering is unaffected.
    double score(const MatchContext& /*context*/,
                 const BoundTokens& /*bound*/,
                 const KernelDefinition& /*kernel*/) const override
    {
        return 0.0;
    }
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
