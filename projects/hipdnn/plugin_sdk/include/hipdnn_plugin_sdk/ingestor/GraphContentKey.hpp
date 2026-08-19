// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The graph half of a winner-cache key: content, never identity.
///
/// # What "equal" means here
///
/// This type defines one specific equivalence relation: **two graphs are equal when a
/// kernel measurement taken on one is a valid measurement for the other.** It is
/// deliberately coarser than "these are the same request" and deliberately finer than
/// "these compute the same thing". Everything below follows from that single question,
/// and any future change to the excluded set must answer it: *would a benchmark result
/// transfer?*
///
/// # Two jobs, split
///
/// Conflating them is what makes a content key expensive:
///
///  - `hash` narrows the lookup to a bucket. It may be lossy. It is folded from a few
///    cheap facts read in place -- no allocation, no serialization.
///  - `content` decides the match, via the schema-generated
///    `operator==(const GraphT&, const GraphT&)`. That comparison walks every tensor,
///    every node and the whole attribute union, and codegen keeps it current, so a new
///    `graph.fbs` field cannot silently stop being compared.
///
/// A hash match with differing content is therefore a miss, not a wrong kernel.
///
/// # The excluded fields, and why each transfers
///
/// Excluded by clearing them on the unpacked copy, so exclusion is code that runs rather
/// than a list someone remembered to skip:
///
///  - `id` -- minted per finalize by `generateUuidV4()`, so it differs on every run of
///    the same program. Including it would make every lookup a miss and defeat the cache
///    outright.
///  - `preferred_engine_id` -- names which engine the caller would like to handle the
///    graph. It selects *who* runs the computation, never *what* is computed, and by the
///    time this engine is building a plan the preference has already been honoured or
///    not. Two callers wanting the same computation from different engines still measure
///    the same kernels.
///  - `is_override_shape_enabled` -- permits shapes to be overridden; it does not change
///    them. The tensor dims and strides in this same graph are the shapes that will
///    actually run, and they are compared in full. So the flag alters what the caller is
///    *allowed* to do later, never the geometry a kernel was timed against -- a
///    measurement transfers across it unchanged.
///
/// Everything else is compared, including tensor dims and strides, every node's
/// attributes, and all three graph-level data types -- each of those genuinely changes
/// what a kernel does, so a measurement does not transfer across them.
///
/// # Cost
///
/// One `UnPack` per key: a deep copy, since `NodeT` owns a `std::string` and a
/// `NodeAttributesUnion` that heap-allocates its payload. Paid once per `buildPlan`, on a
/// path otherwise about to rank the catalog or run a device benchmark sweep, and it buys
/// the ability to skip the former outright.
class GraphContentKey
{
public:
    /// The unpacked, policy-stripped graph this key matched on. Shared because a record
    /// outlives the plan that produced it, and copies of the key must not deep-copy the
    /// graph.
    using Content = std::shared_ptr<const hipdnn_flatbuffers_sdk::data_objects::GraphT>;

    GraphContentKey() = default;

    explicit GraphContentKey(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
        : _content(unpackWithoutIdentity(graph))
        , _hash(fold(graph))
    {
    }

    uint64_t hash() const
    {
        return _hash;
    }

    const Content& content() const
    {
        return _content;
    }

    /// Hash first because it rejects almost every non-match without touching the graphs;
    /// the structural comparison then decides the survivors.
    bool operator==(const GraphContentKey& other) const
    {
        if(_hash != other._hash)
        {
            return false;
        }
        if(_content == nullptr || other._content == nullptr)
        {
            return _content == other._content;
        }
        return *_content == *other._content;
    }

    bool operator!=(const GraphContentKey& other) const
    {
        return !(*this == other);
    }

private:
    /// Unpacks into an owned object so the excluded fields can simply be cleared. The
    /// generated `operator==` compares `id` (graph_generated.h:1313) and both policy
    /// fields (:1310-1311), so clearing them here is what makes equality mean "same
    /// computation" rather than "same request".
    static Content
        unpackWithoutIdentity(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
    {
        auto content = std::shared_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT>(
            graph.getGraph().UnPack());
        if(content != nullptr)
        {
            // Each is cleared to the value a default-constructed GraphT carries, so two
            // graphs differing only in these compare equal. The types differ -- `id` is a
            // unique_ptr, `preferred_engine_id` a flatbuffers::Optional -- so this cannot
            // be one uniform reset.
            content->id.reset();
            content->preferred_engine_id = ::flatbuffers::nullopt;
            content->is_override_shape_enabled = false;
        }
        return content;
    }

    /// Cheap, allocation-free facts only: enough to scatter distinct graphs across
    /// buckets, never enough to decide a match. Everything read here is a view into the
    /// caller's buffer.
    ///
    /// The version tag and node count are emitted before any content, so a degenerate or
    /// empty graph still folds a non-empty stream -- `fnv1aHash` collapses null and empty
    /// input to sentinel `0`, and a key of `0` would alias every unkeyable graph onto one
    /// bucket.
    static uint64_t fold(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
    {
        constexpr uint32_t KEY_FORMAT_VERSION = 1;

        std::vector<uint8_t> stream;
        appendTrivial(stream, KEY_FORMAT_VERSION);

        const auto nodeCount = graph.nodeCount();
        appendTrivial(stream, nodeCount);

        const auto& raw = graph.getGraph();
        appendTrivial(stream, raw.compute_data_type());
        appendTrivial(stream, raw.io_data_type());
        appendTrivial(stream, raw.intermediate_data_type());

        for(uint32_t index = 0; index < nodeCount; ++index)
        {
            const auto& node = graph.getNode(index);
            appendTrivial(stream, node.attributes_type());
            appendTrivial(stream, node.compute_data_type());
        }

        // Tensors come out of an unordered_map, so fold an order-independent summary
        // rather than iterating: a per-tensor fold combined with XOR, which commutes.
        // Sorting the uids first would also work and costs an allocation this does not.
        uint64_t tensorFold = 0;
        for(const auto& [uid, attributes] : graph.getTensorMap())
        {
            if(attributes == nullptr)
            {
                continue;
            }
            std::vector<uint8_t> tensorStream;
            appendTrivial(tensorStream, uid);
            appendTrivial(tensorStream, attributes->data_type());
            appendVector(tensorStream, attributes->dims());
            appendVector(tensorStream, attributes->strides());
            tensorFold
                ^= hipdnn_data_sdk::utilities::fnv1aHash(tensorStream.data(), tensorStream.size());
        }
        appendTrivial(stream, tensorFold);

        return hipdnn_data_sdk::utilities::fnv1aHash(stream.data(), stream.size());
    }

    template <typename T>
    static void appendTrivial(std::vector<uint8_t>& stream, const T& value)
    {
        const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
        stream.insert(stream.end(), bytes, bytes + sizeof(T));
    }

    /// Length first, so {1,2} and {1,2,0} cannot fold to the same bytes.
    template <typename TVector>
    static void appendVector(std::vector<uint8_t>& stream, const TVector* values)
    {
        const size_t size = values == nullptr ? 0 : values->size();
        appendTrivial(stream, size);
        for(size_t index = 0; index < size; ++index)
        {
            appendTrivial(stream, values->Get(static_cast<uint32_t>(index)));
        }
    }

    Content _content;
    uint64_t _hash = 0;
};

} // namespace hipdnn_plugin_sdk::ingestor

namespace std
{

template <>
struct hash<hipdnn_plugin_sdk::ingestor::GraphContentKey>
{
    size_t operator()(const hipdnn_plugin_sdk::ingestor::GraphContentKey& key) const noexcept
    {
        return static_cast<size_t>(key.hash());
    }
};

} // namespace std

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
