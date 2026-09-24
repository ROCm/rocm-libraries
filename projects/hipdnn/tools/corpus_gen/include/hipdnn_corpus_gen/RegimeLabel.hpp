// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/ArgumentResolver.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <string>

/// @file RegimeLabel.hpp
/// @brief Which population a problem belongs to, as the operation itself declares them.
///
/// RFC 0019.13 §11.2 wants the per-regime table as the PRIMARY form of the regret report,
/// because an aggregate hides a model that is excellent on the dense middle and useless on
/// decode-shaped problems. `uhd_gen evaluate` reports it as UNAVAILABLE when no corpus column
/// carries a regime, so this label is what makes that report exist at all -- and it is also
/// what PoolAssembly.hpp's spread stratifies on, so a truncated pool keeps its mix.
///
/// The retired Python assembler this replaces computed the label in three hand-written
/// properties over SDPA field names. That is why that corpus was SDPA-only:
/// not the sampling, not the graph building, but the fact that "which population is this"
/// had one operation's answer compiled into it. Here the populations come from the
/// declaration's `regime_label` block, so an operation covers itself.
namespace hipdnn_corpus_gen
{

/// @brief Each facet's name and the label @p point takes under it, in declaration order.
///
/// The joined label is what a reader greps for; the facets are what a report groups by. The
/// Python this replaces emitted both (`phase`, `context`, `grouping` columns beside `regime`),
/// and it could only do so because the three facet names were compiled in. Recovering them by
/// splitting the joined label is not available here: a declared label may itself contain `_`.
///
/// First match wins rather than "exactly one must match", because the natural way to write
/// these is as a cascade -- decode, then cross, then prefill, else append -- where the later
/// clauses are only ever reached having already excluded the earlier ones. Requiring
/// mutual exclusivity would force every clause to restate its predecessors' negations, which
/// is how a stratification acquires a gap nobody notices.
///
/// A clause that throws or yields a non-boolean is treated as not matching, matching
/// `detail::satisfiesConstraints`' reading of a malformed relation: a label is a report about
/// a problem, and reporting one population as another is worse than falling through to the
/// axis' declared `otherwise`.
inline std::vector<std::pair<std::string, std::string>>
regimeFacets(const std::vector<RegimeAxis>& axes, const ProblemPoint& point)
{
    const auto context = detail::contextFor(point);

    std::vector<std::pair<std::string, std::string>> facets;
    facets.reserve(axes.size());
    for(const auto& axis : axes)
    {
        auto chosen = axis.otherwise;
        auto work = axis.clauses.workspace();
        for(size_t i = 0; i < axis.clauses.size(); ++i)
        {
            try
            {
                const auto& value = axis.clauses.evaluate(i, context, work);
                const auto* held = std::get_if<bool>(&value.raw);
                if(held != nullptr && *held)
                {
                    chosen = i < axis.labels.size() ? axis.labels[i] : axis.otherwise;
                    break;
                }
            }
            catch(const std::exception&)
            {
                continue;
            }
        }

        facets.emplace_back(axis.name, chosen);
    }
    return facets;
}

/// @brief The label @p point carries under @p axes: each facet's first matching clause, joined
/// with `_`.
inline std::string regimeLabel(const std::vector<RegimeAxis>& axes, const ProblemPoint& point)
{
    std::string label;
    for(const auto& facet : regimeFacets(axes, point))
    {
        if(!label.empty())
        {
            label += "_";
        }
        label += facet.second;
    }
    return label;
}

/// @brief The label @p point carries under @p metadata's declared facets.
inline std::string regimeLabel(const OperationMetadata& metadata, const ProblemPoint& point)
{
    return regimeLabel(metadata.regimeLabel, point);
}

} // namespace hipdnn_corpus_gen
