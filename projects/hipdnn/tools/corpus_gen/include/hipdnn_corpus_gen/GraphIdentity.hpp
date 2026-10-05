// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>

#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>

#include <cstdint>
#include <string>
#include <vector>

/// @file GraphIdentity.hpp
/// @brief Content-derived names and ids for written graphs.
///
/// The id must equal the manifest's `benchmark` column so bench results join to the corpus.
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

/// The bytes of one graph and the identity it now carries.
struct IdentifiedGraph
{
    std::vector<uint8_t> bytes;

    /// The formatted id, as the bench will report it.
    std::string id;

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
            name};
}

} // namespace hipdnn_corpus_gen
