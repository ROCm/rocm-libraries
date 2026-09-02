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
    /// RFC 0019.13 §15.2: what selection returns -- "an ordered sequence of `(UKD id, score)`,
    /// winner first".
    ///
    /// By id rather than object or reference, and that is the whole point: "The result crosses
    /// a plugin boundary. An object commits the ABI to a kernel-definition layout; a reference
    /// couples the caller's lifetime to the catalog."
    ///
    /// The score travels with the id because the callers §15.2 names need it -- a knob query
    /// reports the top-ranked value as its default, autotune walks the ranked list, and engine
    /// selection reads the top score as the engine's figure of merit (§11.1). Returning order
    /// alone makes the third impossible, which is what blocked §15 phase 7.
    struct ScoredKernel
    {
        DescriptorId kernelId;
        double score;
    };

    /// What decided the order, for the §12 trace: a compiled scorer, or nothing at all.
    /// Overridden by the unranked fallback, which declines to rank.
    virtual std::string traceDecidedBy() const
    {
        return "native";
    }

    /// The ranking, in §15.2's form. Overriding this rather than rank() keeps one
    /// implementation of the order: rank() is derived from it below.
    virtual std::vector<ScoredKernel> rankScored(const Catalog& catalog,
                                                 const MatchContext& context) const
    {
        struct Ranked
        {
            double ordering;   ///< NaN-free, so the comparator stays a strict weak ordering
            double reported;   ///< exactly what score() returned, NaN included
            const KernelDefinition* entry;
        };

        std::vector<Ranked> scored;
        scored.reserve(catalog.entries.size());
        for(const auto& entry : catalog.entries)
        {
            // NaN sorts last rather than poisoning the comparator, but it is reported to the
            // caller unchanged: §15.2's score is a figure of merit, and a heuristic that
            // computed none has to say so rather than hand back a number that reads as one.
            const double raw = score(context, catalog.bound, entry);
            scored.push_back({std::isnan(raw) ? -std::numeric_limits<double>::infinity() : raw,
                              raw,
                              &entry});
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
                candidates << (i == 0 ? "" : " ") << toString(scored[i].entry->kernelId) << "="
                           << scored[i].reported;
            }
            HIPDNN_PLUGIN_LOG_INFO("uhd trace: decided_by=" << traceDecidedBy()
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

    /// @brief Whether this heuristic's score is comparable against other engines' scores.
    ///
    /// RFC 0019 §11.3: a cross-engine score must be an absolute metric on a scale that means
    /// the same thing everywhere -- calibrated TFLOPS. Defaults to false, so a heuristic that
    /// has not said otherwise is never compared against another engine by accident.
    virtual bool scoreIsCalibrated() const { return false; }

    /// @brief This engine's predicted TFLOPS for @p catalog, or nullopt if it cannot say.
    ///
    /// RFC 0019 §11.1 defines `predict_engine_tflops` as the cheap proxy for engine ranking and
    /// then states it is not needed for v1: with a single descriptor engine there is nothing to
    /// rank against. It names the stopgap -- "an engine reports sort_kernel_catalog's best
    /// predicted score as its estimate, accepting the enumeration cost" -- which is what this
    /// is. A distinct estimate model, when one exists, replaces the body without moving the
    /// seam.
    ///
    /// Returns nullopt rather than a number whenever the number would not mean what the caller
    /// needs it to. An uncalibrated score ranks within one engine only (§15.1), and a NaN score
    /// is a ranking that computed no figure of merit at all (§15.2). Both are legal rankings and
    /// neither is an estimate, so the distinction is returned rather than flattened -- a caller
    /// comparing engines has to be able to tell "slow" from "declined to say".
    std::optional<double> estimateTflops(const Catalog& catalog, const MatchContext& context) const
    {
        if(!scoreIsCalibrated())
        {
            return std::nullopt;
        }

        const auto scored = rankScored(catalog, context);
        if(scored.empty() || std::isnan(scored.front().score))
        {
            return std::nullopt;
        }
        return scored.front().score;
    }

    /// The same order as rankScored(), as whole kernels.
    ///
    /// Kept because the catalog is what the state manager holds and re-sorts; §15.2's point is
    /// that the *result crossing a plugin boundary* is ids and scores, not that a caller
    /// already holding the catalog may not look at it. Non-virtual, so there is exactly one
    /// place the order is decided.
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
        for(const auto& [kernelId, _] : scored)
        {
            if(const auto found = byId.find(kernelId); found != byId.end())
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

    /// NaN, not zero: this heuristic ranks by declared order and computes no figure of merit.
    /// Zero is a value an engine could legitimately score, so reporting it here would let
    /// §15.2's engine-selection caller compare a fallback against a model on the same scale.
    double score(const MatchContext& /*context*/,
                 const BoundTokens& /*bound*/,
                 const KernelDefinition& /*kernel*/) const override
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
