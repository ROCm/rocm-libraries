// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

/// @file GraphFormat.hpp
/// @brief Telling a JSON graph from a FlatBuffer by its content.
namespace hipdnn_bench
{

/// @brief Whether @p bytes are a JSON graph: its first byte other than JSON whitespace is `{`.
///
/// A FlatBuffer opens with a little-endian root offset whose bytes after the first are zero
/// for any graph under 256 bytes, so skipping leading whitespace and then meeting `{`
/// identifies a JSON document in practice. Only content decides, never the file extension.
inline bool isJsonGraph(const std::vector<uint8_t>& bytes)
{
    for(const uint8_t byte : bytes)
    {
        if(byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r')
        {
            continue;
        }
        return byte == '{';
    }
    return false;
}

} // namespace hipdnn_bench
