// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/CorpusOutput.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <algorithm>
#include <string>
#include <vector>

/// @file PointFilter.hpp
/// @brief Restricting a corpus to the facets an engine actually serves.
///
/// An engine that cannot serve a facet contributes declines, not data: AITER's gfx950 kernels
/// are unmasked while a general SDPA corpus is ~90% causal, so a 900-graph draw yielded 106
/// usable measurements and a model with a large bias. Filtering the corpus to the engine's own
/// facets is not a convenience, it is the difference between a trained model and a broken one.
///
/// Stated as `q.<parameter>=<value>` -- the corpus's own column spelling, so what is typed here
/// matches what appears in `manifest.csv` -- and op-general because nothing here names a
/// parameter. A filter naming a parameter the operation does not declare is refused rather than
/// ignored: silently keeping everything is how a run meant to be narrow comes back broad.
namespace hipdnn_corpus_gen
{

/// One `q.<parameter>=<value>` clause. Values compare as text, against @ref asText, so a filter
/// is written the same way for an enum, a bool and an integer.
struct KeepClause
{
    std::string parameter;
    std::string value;
};

/// @brief Parses `--keep` clauses, refusing anything the operations do not declare.
///
/// @p known is every parameter name any loaded declaration declares. A clause outside it is a
/// typo -- `q.headdim=128` for `head_dim` -- and a typo that filtered nothing would report a
/// full corpus as a narrow one.
inline bool parseKeepClauses(const std::vector<std::string>& clauses,
                             const std::vector<std::string>& known,
                             std::vector<KeepClause>& parsed,
                             std::string& error)
{
    parsed.clear();
    for(const auto& clause : clauses)
    {
        const auto equals = clause.find('=');
        if(equals == std::string::npos || equals == 0 || equals + 1 == clause.size())
        {
            error = "--keep expects q.<parameter>=<value>, got '" + clause + "'";
            return false;
        }

        auto name = clause.substr(0, equals);
        if(name.rfind("q.", 0) == 0)
        {
            name = name.substr(2);
        }

        if(std::find(known.begin(), known.end(), name) == known.end())
        {
            error = "--keep names '" + name + "', which no loaded declaration declares";
            return false;
        }
        parsed.push_back(KeepClause{name, clause.substr(equals + 1)});
    }
    return true;
}

/// @brief Whether @p point satisfies every clause.
///
/// Clauses naming *different* parameters are AND-ed; clauses repeating the *same* parameter are
/// OR-ed. `--keep q.head_dim=64 --keep q.head_dim=128` therefore means "either head dim", which
/// is the only reading that is ever wanted: AND-ing them names an empty corpus, and a filter
/// that can only express one value per facet cannot describe an engine whose kernel table
/// covers two.
///
/// A point missing a filtered parameter fails: it belongs to an operation that does not have
/// that facet, and admitting it would put problems the filter was written to exclude into a
/// corpus that reports itself as filtered.
inline bool keeps(const std::vector<KeepClause>& clauses, const ProblemPoint& point)
{
    for(const auto& clause : clauses)
    {
        const auto found = point.find(clause.parameter);
        if(found == point.end())
        {
            return false;
        }

        const auto held      = asText(found->second);
        const auto satisfied = std::any_of(clauses.begin(), clauses.end(),
                                           [&](const KeepClause& alternative) {
                                               return alternative.parameter == clause.parameter
                                                      && alternative.value == held;
                                           });
        if(!satisfied)
        {
            return false;
        }
    }
    return true;
}

} // namespace hipdnn_corpus_gen
