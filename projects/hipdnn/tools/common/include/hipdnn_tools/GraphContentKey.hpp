// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

/// @file GraphContentKey.hpp
/// @brief The key a graph is measured, stored, deduplicated and held out under.
namespace hipdnn_tools
{

/// @brief The content key of a serialized graph (or of the graph inside a graph-and-plan
/// container): `GraphContentKey::hash()` in decimal.
///
/// The key ignores the graph's name and id and numbers tensors by position, so two graphs
/// that compute the same thing share it. hipdnn_bench reports it as `graph_key`, and
/// hipdnn_corpus_gen writes it as a manifest's `benchmark`.
inline std::string graphContentKey(const uint8_t* data, size_t size)
{
    const auto graph
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(data,
                                                                                         size);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey key{graph};
    if(!key.isUsable())
    {
        throw std::runtime_error("graph has no verifiable serialized content to key");
    }
    return std::to_string(key.hash());
}

} // namespace hipdnn_tools
