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
/// @brief The name and the id a written graph carries, and why they are not incidental.
///
/// Both are how a measurement finds its way back to the problem that produced it.
/// `hipdnn_bench` reports `graph_id` from the document's own `id` field, and
/// `uhd_gen/corpus_io.py:31-34` treats `graph_id` and `benchmark` as one identity under two
/// spellings -- so the manifest's `benchmark` column and this id must be the same string or
/// the two files do not join and RFC 0019.13 §11.2's per-regime table reports UNAVAILABLE.
///
/// A graph written without an id is not rejected: `GraphDescriptor::finalize` mints a UUID v4
/// for it at load. That is the failure worth naming, because it is silent -- every run and
/// every machine measures the same corpus under different identities, and nothing reports an
/// error. Deriving the id from the graph's own bytes is what makes `tools/CLAUDE.md` Step 1's
/// promise ("`benchmark` ids are content-derived, so rows join across runs and machines") true
/// for the graphs this tool writes, as it already is for the ones `uhd_gen` mints ids for.
namespace hipdnn_corpus_gen
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

/// @brief A content-derived identity for a graph, as raw UUID bytes.
///
/// Version 8 (RFC 9562, custom) with the RFC 4122 variant. It is neither v4 random nor the v5
/// SHA-1 name hash `uhd_gen generate` mints for id-less JSON graphs, and labelling it as
/// either would misdescribe it. `graph.fbs:121` permits it outright -- "readers preserve any
/// 128-bit value without enforcing a UUID version or variant" -- and the schema's one
/// substantive requirement, that one id never name two different graph contents, is what a
/// digest gives by construction. Nothing cross-checks these against `generate.py`'s; that path
/// declines to mint for binary graphs (`generate.py:385-390`).
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

/// @brief @ref graphIdentityBytes in the spelling the graph document's JSON carries.
///
/// Formatted through the SDK's own `formatUuid` rather than by assembling the hex here,
/// because this string has to be character for character what `to_json` renders from the same
/// bytes (`utilities/json/Graph.hpp:173`). That equality is the entire join.
inline std::string graphIdentity(const uint8_t* data, size_t size)
{
    return hipdnn_flatbuffers_sdk::utilities::formatUuid(graphIdentityBytes(data, size));
}

/// The bytes of one graph and the identity it now carries.
struct IdentifiedGraph
{
    std::vector<uint8_t> bytes;

    /// The formatted id, as the graph document now carries it and as the bench will report it.
    std::string id;

    /// The name it was given, returned so a caller recording both does not have to keep its
    /// own copy in step with this one.
    std::string name;
};

/// @brief Names @p bytes and gives it an id derived from its own content.
///
/// Done by unpacking and repacking rather than by threading a name and an id through every
/// builder, because the identity has to be a function of the finished graph: a builder cannot
/// digest bytes it has not produced yet. The round trip is the same `UnPack`/`CreateGraph`
/// pair the backend performs on every graph it loads, so it is faithful by the same argument.
///
/// The digest is taken over the graph *after* renaming and with the id cleared, which makes
/// the operation idempotent -- restamping an already-stamped graph reproduces the same id
/// rather than digesting the previous one. It also makes the id sensitive to the name, which
/// is wanted: two problems that differ only in a parameter the builder ignores would otherwise
/// collide, and the name is where that parameter still shows.
inline IdentifiedGraph stampGraphIdentity(const std::vector<uint8_t>& bytes,
                                          const std::string& name)
{
    auto object = fb::UnPackGraph(bytes.data());
    object->name = name;
    object->id.reset();

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(fb::CreateGraph(builder, object.get()));

    const auto identity = graphIdentityBytes(builder.GetBufferPointer(), builder.GetSize());
    object->id = std::make_unique<fb::Uuid>(
        hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid(identity));

    builder.Clear();
    builder.Finish(fb::CreateGraph(builder, object.get()));

    const auto* stamped = builder.GetBufferPointer();
    return {{stamped, stamped + builder.GetSize()},
            hipdnn_flatbuffers_sdk::utilities::formatUuid(identity),
            name};
}

} // namespace hipdnn_corpus_gen
