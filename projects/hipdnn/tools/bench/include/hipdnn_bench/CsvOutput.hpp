// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <string>
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

/// The CSV header: the problem and its graph's content key, the device columns, the
/// `--query` parameters (`q.*`), the knobs (`kernel.*`), then the row's outcome. Timing
/// names follow RFC 0019.13 §8.3 (camelCase); `numerically_valid` is three-valued text,
/// kept apart from `is_valid` (RFC 0019 §13.2).
inline std::string csvHeader(const std::vector<std::string>& deviceColumns,
                             const std::vector<std::string>& queryNames,
                             const std::vector<std::string>& knobNames)
{
    std::string header = "problem,benchmark";
    for(const auto& column : deviceColumns)
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
