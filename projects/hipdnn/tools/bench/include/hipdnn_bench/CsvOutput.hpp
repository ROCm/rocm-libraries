// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
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

/// Why an untimed candidate has no measurement, or empty when it was timed. RFC 0019.13
/// §7.4/§8: untimed pairs are kept with is_valid=False and this reason for coverage auditing.
inline std::string skipReason(bool succeeded, int64_t iterationsRun)
{
    if(!succeeded)
    {
        return "config_not_applicable: engine declined or failed to run this configuration";
    }
    if(iterationsRun == 0)
    {
        // Emitting this as valid would put a zero time in the training set.
        return "not_timed: autotune reported success without running an iteration";
    }
    return {};
}

/// One candidate's outcome, as the CSV columns after its knobs and the JSON `results[]`
/// entry report it.
struct CandidateOutcome
{
    std::string engine;
    int64_t rank = 0;
    bool succeeded = false;
    int64_t iterationsRun = 0;
    /// `numerically_valid` as CSV text (verdictText) and as JSON (true, false or null), and
    /// its reason.
    std::string verdict;
    nlohmann::json numericallyValid = nullptr;
    std::string validation;
    float minTimeMs = 0.0F;
    float avgTimeMs = 0.0F;
    float robustTimeMs = 0.0F;
    float stddevMs = 0.0F;
    bool converged = false;
    int64_t workspaceBytes = 0;
};

/// One CSV row, in @ref csvHeader's column order: the problem and its graph's content key,
/// the feature, query and knob values given in their header's order, then the outcome.
///
/// Every text field goes through csvField: the problem id defaults to the graph's path, and
/// knob values and engine names are free text. Training filters on is_valid.
inline std::string csvRow(const std::string& problem,
                          const std::string& benchmark,
                          const std::vector<std::string>& features,
                          const std::vector<std::string>& queries,
                          const std::vector<std::string>& knobs,
                          const CandidateOutcome& outcome)
{
    std::string row = csvField(problem) + "," + csvField(benchmark);
    for(const auto* values : {&features, &queries, &knobs})
    {
        for(const auto& value : *values)
        {
            row += "," + csvField(value);
        }
    }

    const bool timed = outcome.succeeded && outcome.iterationsRun > 0;

    std::ostringstream tail;
    tail << "," << csvField(outcome.engine) << "," << outcome.rank << ","
         << (outcome.succeeded ? 1 : 0) << "," << (timed ? "True" : "False") << ","
         << outcome.verdict << "," << csvField(outcome.validation) << ","
         << csvField(skipReason(outcome.succeeded, outcome.iterationsRun)) << ","
         << csvTiming(timed, outcome.minTimeMs) << "," << csvTiming(timed, outcome.avgTimeMs) << ","
         << csvTiming(timed, outcome.robustTimeMs) << "," << csvTiming(timed, outcome.stddevMs)
         << "," << outcome.iterationsRun << "," << (outcome.converged ? 1 : 0) << ","
         << outcome.workspaceBytes;
    return row + tail.str();
}

} // namespace hipdnn_bench
