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
/// The CSV `q.*` columns and the `--query` argument must describe the same problem. Column
/// names are the feature signature `uhd_gen` hashes (tools/uhd_gen/features.py); do not rename.
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
                // Spelled, not 0/1: read back as a categorical and the declaration says "false".
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
/// Header and rows come from the same ordered traversal so column order cannot disagree.
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
/// The inverse of asQueryArgument. Values stay strings; the declaration owns the types.
/// Returns an empty vector for malformed input rather than a partial parse.
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

/// @brief @p text as one POSIX shell word: single-quoted, an embedded quote closed, escaped
/// and reopened, so a space or a `$` in a path or value reaches the program unchanged.
inline std::string shellWord(const std::string& text)
{
    std::string word = "'";
    for(const char character : text)
    {
        word += character == '\'' ? std::string("'\\''") : std::string(1, character);
    }
    return word + "'";
}

/// @brief One line of `commands.txt`: hipdnn_bench sweeping @p graph, its CSV (with its own
/// header) written to @p csv. One file per graph: graphs whose catalogs expose different
/// knobs have different `kernel.*` columns, so their rows cannot share one header.
inline std::string benchCommandLine(const std::string& bench,
                                    const std::vector<std::string>& pluginDirs,
                                    const std::string& graph,
                                    const std::string& engineName,
                                    const std::string& problemId,
                                    const ProblemPoint& point,
                                    const std::string& csv)
{
    std::string line = shellWord(bench);
    for(const auto& dir : pluginDirs)
    {
        line += " --plugin-dir " + shellWord(dir);
    }
    line += " --graph " + shellWord(graph);
    if(!engineName.empty())
    {
        line += " --engine-name " + shellWord(engineName);
    }
    line += " --sweep --problem-id " + shellWord(problemId);
    line += " --query " + shellWord(asQueryArgument(point));
    return line + " > " + shellWord(csv);
}

} // namespace hipdnn_corpus_gen
