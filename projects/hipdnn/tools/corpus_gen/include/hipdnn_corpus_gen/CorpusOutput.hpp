// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <string>
#include <type_traits>
#include <variant>
#include <vector>

/// @file CorpusOutput.hpp
/// @brief Rendering a problem point into the corpus's two output forms.
///
/// Lifted out of the tool's main file so it can be tested. It was not testable there, and the
/// consequence is specific: a corpus is only useful if the `q.*` columns of a CSV row and the
/// `--query` argument of the command that produced it describe the same problem. Nothing
/// checked that, and the two are rendered by separate code paths that must agree forever.
///
/// The column names are the feature signature `uhd_gen` hashes (tools/uhd_gen/features.py),
/// so they are emitted in the form the trainer expects rather than renamed downstream. A
/// renaming step is how the two sides drift while the hash still matches.
namespace hipdnn_corpus_gen
{

/// One parameter value, rendered the way both output forms render it.
inline std::string asText(const ParameterValue& value)
{
    std::string text;
    std::visit(
        [&text](const auto& held) {
            using Held = std::decay_t<decltype(held)>;
            if constexpr(std::is_same_v<Held, std::string>)
            {
                text += held;
            }
            else if constexpr(std::is_same_v<Held, bool>)
            {
                // Spelled, not printed as 0/1: the column is read back as a categorical value
                // and "false" is what the declaration calls it.
                text += held ? "true" : "false";
            }
            else
            {
                text += std::to_string(held);
            }
        },
        value);
    return text;
}

/// The CSV half: `q.N,q.C,...` when @p namesOnly, otherwise the row of values.
///
/// Both are generated from the same ordered traversal of the point, so a header and its rows
/// cannot disagree about column order -- which would silently transpose two `q.*` features and
/// train a model on the wrong ones.
inline std::string asQueryColumns(const ProblemPoint& point, bool namesOnly)
{
    std::string text;
    for(const auto& entry : point)
    {
        if(!text.empty())
        {
            text += ",";
        }
        text += namesOnly ? "q." + entry.first : asText(entry.second);
    }
    return text;
}

/// The command half: `name=value,...` as `hipdnn_bench --query` takes it.
inline std::string asQueryArgument(const ProblemPoint& point)
{
    std::string text;
    for(const auto& entry : point)
    {
        if(!text.empty())
        {
            text += ",";
        }
        text += entry.first + "=" + asText(entry.second);
    }
    return text;
}

/// @brief Reads a `--query` argument back into name/value pairs.
///
/// The inverse of asQueryArgument, and the reason it exists: without a parser nothing can check
/// that what the generator emits is what the benchmark can read. Values stay strings because
/// this is a transport check, not a re-typing of the problem -- the declaration owns the types.
///
/// Returns an empty vector for malformed input rather than a partial parse, so a caller cannot
/// mistake half a problem for a whole one.
inline std::vector<std::pair<std::string, std::string>> parseQueryArgument(const std::string& text)
{
    std::vector<std::pair<std::string, std::string>> parsed;
    size_t start = 0;
    while(start <= text.size())
    {
        const auto comma = text.find(',', start);
        const auto field
            = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if(field.empty())
        {
            return {};
        }

        const auto equals = field.find('=');
        if(equals == std::string::npos || equals == 0 || equals + 1 == field.size())
        {
            return {};
        }
        parsed.emplace_back(field.substr(0, equals), field.substr(equals + 1));

        if(comma == std::string::npos)
        {
            break;
        }
        start = comma + 1;
    }
    return parsed;
}

} // namespace hipdnn_corpus_gen
