// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>

#include <hipdnn_corpus_gen/CorpusOutput.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <hipdnn_tools/GraphContentKey.hpp>

#include <cstdint>
#include <string>
#include <vector>

/// @file GraphIdentity.hpp
/// @brief A written graph's id, and the content key that identifies it.
///
/// A graph is identified by its **content key**, the kernel ingestor's `GraphContentKey`: it
/// ignores the graph's name and id and numbers tensors by position, so two graphs that
/// compute the same thing share it however they were made. It is the manifest's `benchmark`,
/// the key a measurement is stored, deduplicated and held out under. Every written graph is
/// also stamped with a stable id, since an id-less one gets a random id at load.
namespace hipdnn_corpus_gen
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

/// @brief A content-derived graph id as raw UUID bytes: SHA-256 of the graph, stamped as a
/// version 8 (custom) UUID with the RFC 4122 variant. `graph.fbs` permits any 128-bit id.
inline hipdnn_flatbuffers_sdk::utilities::UuidBytes graphIdentityBytes(const uint8_t* data,
                                                                       size_t size)
{
    const auto digest = hipdnn_plugin_sdk::uhd::sha256(data, size);

    const auto nibble = [&digest](size_t index) {
        const char character = digest[index];
        return static_cast<uint8_t>(character <= '9' ? character - '0' : (character - 'a') + 10);
    };

    hipdnn_flatbuffers_sdk::utilities::UuidBytes bytes{};
    for(size_t i = 0; i < bytes.size(); ++i)
    {
        bytes[i] = static_cast<uint8_t>((nibble(i * 2) << 4U) | nibble(i * 2 + 1));
    }

    bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0fU) | 0x80U); // version 8: custom
    bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3fU) | 0x80U); // RFC 4122 variant
    return bytes;
}

/// @brief @ref graphIdentityBytes formatted as the graph JSON carries it.
///
/// Uses the SDK's `formatUuid` so the string matches what `to_json` renders exactly.
inline std::string graphIdentity(const uint8_t* data, size_t size)
{
    return hipdnn_flatbuffers_sdk::utilities::formatUuid(graphIdentityBytes(data, size));
}

using hipdnn_tools::graphContentKey;

/// The bytes of one graph and the identities it now carries.
struct IdentifiedGraph
{
    std::vector<uint8_t> bytes;

    /// The formatted id written into the document.
    std::string id;

    /// Its content key (@ref graphContentKey): the manifest's `benchmark`.
    std::string key;

    /// The name it was given.
    std::string name;
};

/// @brief Names @p bytes and gives it an id derived from its own content.
///
/// The digest is taken after renaming and with the id cleared, so restamping is idempotent
/// and the id depends on the name.
inline IdentifiedGraph stampGraphIdentity(const std::vector<uint8_t>& bytes,
                                          const std::string& name)
{
    auto object = fb::UnPackGraph(bytes.data());
    object->name = name;
    object->id.reset();

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(fb::CreateGraph(builder, object.get()));

    const auto identity = graphIdentityBytes(builder.GetBufferPointer(), builder.GetSize());
    object->id
        = std::make_unique<fb::Uuid>(hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid(identity));

    builder.Clear();
    builder.Finish(fb::CreateGraph(builder, object.get()));

    const auto* stamped = builder.GetBufferPointer();
    return {{stamped, stamped + builder.GetSize()},
            hipdnn_flatbuffers_sdk::utilities::formatUuid(identity),
            graphContentKey(stamped, builder.GetSize()),
            name};
}

/// The longest graph name. The name is a file stem (`<name>.fb`, `results/<name>.csv`), so
/// it stays well under the 255-byte file-name limit of common filesystems.
inline constexpr size_t MAX_GRAPH_NAME = 200;

/// @brief The name the graph document carries: the operation, its regime, and its parameters.
///
/// Also the file stem, `--problem-id` and `<op>.problems.csv` key. Scoring and L2 collection
/// join on this name alone, so it must be unique: every parameter is spelled out. A name
/// longer than @ref MAX_GRAPH_NAME keeps its start and ends in a hash of the whole, so it
/// stays unique and the same on every run.
inline std::string
    graphNameFor(const std::string& operation, const std::string& regime, const ProblemPoint& point)
{
    std::string name = operation;
    if(!regime.empty())
    {
        name += "_" + regime;
    }
    for(const auto& parameter : point)
    {
        name += "_" + parameter.first + asText(parameter.second);
    }
    if(name.size() > MAX_GRAPH_NAME)
    {
        constexpr size_t HASH_LENGTH = 16;
        const auto digest = hipdnn_plugin_sdk::uhd::sha256(
            reinterpret_cast<const uint8_t*>(name.data()), name.size());
        name = name.substr(0, MAX_GRAPH_NAME - HASH_LENGTH - 1) + "_"
               + digest.substr(0, HASH_LENGTH);
    }
    return name;
}

} // namespace hipdnn_corpus_gen
