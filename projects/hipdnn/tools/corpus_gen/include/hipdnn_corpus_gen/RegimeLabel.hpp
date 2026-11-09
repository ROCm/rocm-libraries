// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/ArgumentResolver.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <string>

/// @file RegimeLabel.hpp
/// @brief Which population a problem belongs to, from the operation's `regime_label` block.
///
/// Feeds the per-regime regret report (RFC 0019.13 §11.2) and PoolAssembly's stratification.
namespace hipdnn_corpus_gen
{

/// @brief Each facet's name and the label @p point takes under it, in declaration order.
///
/// Facets are returned separately because a declared label may itself contain `_`, so the
/// joined label cannot be split back. First matching clause wins, so clauses can be written
/// as a cascade. A clause that fails to resolve or is non-boolean counts as not matching.
inline std::vector<std::pair<std::string, std::string>>
    regimeFacets(const std::vector<RegimeAxis>& axes, const ProblemPoint& point)
{
    const auto context = detail::contextFor(point);

    std::vector<std::pair<std::string, std::string>> facets;
    facets.reserve(axes.size());
    for(const auto& axis : axes)
    {
        auto chosen = axis.otherwise;
        for(size_t i = 0; i < axis.clauses.size(); ++i)
        {
            const auto value = axis.clauses.evaluate(i, context);
            if(value.isBool() && value.asBool())
            {
                chosen = i < axis.labels.size() ? axis.labels[i] : axis.otherwise;
                break;
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
