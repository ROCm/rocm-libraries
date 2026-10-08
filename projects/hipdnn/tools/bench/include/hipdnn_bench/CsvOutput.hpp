// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

/// @file CsvOutput.hpp
/// @brief CSV field quoting for benchmark rows.
namespace hipdnn_bench
{

/// A CSV field, quoted only when it has to be (RFC 4180). An unquoted comma would silently
/// shift every later column rather than fail to parse.
inline std::string csvField(const std::string& text)
{
    if(text.find_first_of(",\"\n") == std::string::npos)
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

} // namespace hipdnn_bench
