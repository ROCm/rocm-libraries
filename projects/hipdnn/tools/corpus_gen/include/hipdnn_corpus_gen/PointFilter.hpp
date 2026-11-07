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
/// Clauses use the corpus column spelling `q.<parameter>=<value>`, as in `manifest.csv`.
namespace hipdnn_corpus_gen
{

/// One `q.<parameter>=<value>` clause. Values compare as text via @ref asText, so enums, bools
/// and integers are written the same way.
struct KeepClause
{
    std::string parameter;
    std::string value;
};

/// @brief Parses `--keep` clauses into @p parsed; false with @p error on a malformed clause or
/// a parameter not in @p known, since a typo that filters nothing would look like a full corpus.
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
/// Clauses on different parameters are AND-ed; repeats of the same parameter are OR-ed, so
/// `--keep q.head_dim=64 --keep q.head_dim=128` means either. A point lacking a filtered
/// parameter fails.
inline bool keeps(const std::vector<KeepClause>& clauses, const ProblemPoint& point)
{
    for(const auto& clause : clauses)
    {
        const auto found = point.find(clause.parameter);
        if(found == point.end())
        {
            return false;
        }

        const auto held = asText(found->second);
        const auto satisfied
            = std::any_of(clauses.begin(), clauses.end(), [&](const KeepClause& alternative) {
                  return alternative.parameter == clause.parameter && alternative.value == held;
              });
        if(!satisfied)
        {
            return false;
        }
    }
    return true;
}

} // namespace hipdnn_corpus_gen
