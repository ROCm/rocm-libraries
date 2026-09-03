// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

/// @file CsvOutput.hpp
/// @brief Writing benchmark rows that survive being read back.
///
/// In a header rather than the tool's main file so it can be tested. The failure it guards is
/// not a crash: a harvest CSV that mis-quotes one field still parses, and the damage shows up
/// as a model trained on columns that are one place to the left.
namespace hipdnn_bench
{

/// A CSV field, quoted only when it has to be (RFC 4180).
///
/// `skip_reason` (RFC 0019.13 §7.4) is the one free-text column a harvest emits, and the only
/// one that can carry a comma or a quote. An unquoted comma does not fail to parse -- it shifts
/// every column to its right, so the timings land under the wrong headers and the trainer reads
/// transposed features without noticing.
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
