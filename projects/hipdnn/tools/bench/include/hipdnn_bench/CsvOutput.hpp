// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

/// @file CsvOutput.hpp
/// @brief CSV field quoting for benchmark rows.
namespace hipdnn_bench
{

/// A CSV field, quoted only when it has to be (RFC 4180). An unquoted comma would silently
/// shift every later column rather than fail to parse, and an unquoted line break would end
/// the row.
inline std::string csvField(const std::string& text)
{
    if(text.find_first_of(",\"\r\n") == std::string::npos)
    {
        return text;
    }

    std::string quoted = "\"";
    for(const char character : text)
    {
        if(character == '"')
        {
            quoted += '"'; // RFC 4180 escapes an embedded quote by doubling it
        }
        quoted += character;
    }
    return quoted + "\"";
}

/// The problem columns of a catalog sweep: every feature the runtime extracted for the graph,
/// under the name the catalog page publishes it, so the dataset can tell one problem from
/// another without a `--query`. Booleans are written as `True`/`False`, as `is_valid` is.
inline std::vector<std::pair<std::string, std::string>>
    problemColumns(const nlohmann::json& problemFeatures)
{
    std::vector<std::pair<std::string, std::string>> columns;
    columns.reserve(problemFeatures.size());
    for(const auto& [name, value] : problemFeatures.items())
    {
        std::string text;
        if(value.is_boolean())
        {
            text = value.get<bool>() ? "True" : "False";
        }
        else if(value.is_string())
        {
            text = value.get<std::string>();
        }
        else
        {
            text = value.dump();
        }
        columns.emplace_back(name, std::move(text));
    }
    return columns;
}

/// The CSV header: the problem and its graph's content key, the device and problem columns
/// (named in full), the `--query` parameters (`q.*`), the knobs (`kernel.*`), then the row's
/// outcome. Timing
/// names follow RFC 0019.13 §8.3 (camelCase); `numerically_valid` is three-valued text,
/// kept apart from `is_valid` (RFC 0019 §13.2).
inline std::string csvHeader(const std::vector<std::string>& featureColumns,
                             const std::vector<std::string>& queryNames,
                             const std::vector<std::string>& knobNames)
{
    std::string header = "problem,benchmark";
    for(const auto& column : featureColumns)
    {
        header += "," + csvField(column);
    }
    for(const auto& name : queryNames)
    {
        header += "," + csvField("q." + name);
    }
    for(const auto& knob : knobNames)
    {
        header += "," + csvField("kernel." + knob);
    }
    return header
           + ",engine,rank,succeeded,is_valid,numerically_valid,validation,skip_reason,"
             "minTimeMs,avgTimeMs,robustMeanMs,stddevMs,iters,converged,workspace_bytes";
}

/// A timing cell: the value when the candidate was timed, else empty (a missing value). A
/// zero would read as a measurement, and a row that is both a measurement and a failure is
/// refused by the dataset (RFC 0019.13 §8.3).
inline std::string csvTiming(bool timed, float milliseconds)
{
    if(!timed)
    {
        return {};
    }
    std::ostringstream text;
    text << milliseconds;
    return text.str();
}

} // namespace hipdnn_bench
