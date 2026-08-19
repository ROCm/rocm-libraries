// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/cachekey_generated.h>
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
///  - `hash` narrows the lookup to a bucket. It may be lossy.
///  - `logicallyEqual` decides the match.
///
/// Both come from `cachekey_generated.h`, generated from `graph.fbs`, so a new schema
/// field is hashed and compared the day it is added -- the policy is opt-out. Both read
/// the buffer in place: no `UnPack`, no allocation, no reflection.
///
/// A hash match with differing content is therefore a miss, not a wrong kernel.
///
/// # The excluded fields, and why each transfers
///
/// Exclusion lives on the field in `graph.fbs`, as `(cache_ignore)`, next to the comment
/// explaining what the field means. Nothing here re-states the list, and nothing has to
/// remember to skip anything:
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
///  - `min_required_engine_api_version` -- derived, never independent content: stamped
///    from `computeMinimumEnginePluginApiVersion(...)`. Comparing it would silently
///    undo the `is_override_shape_enabled` exclusion above, since that flag alone moves
///    the stamped version. Its content-bearing inputs -- pass-by-value, ragged offsets,
///    alignment -- are each compared directly on the `TensorAttributes` that carry them.
///  - the `name` on the graph, each node and each tensor -- labels for humans. Two
///    identically-shaped graphs run the same kernels whatever they are called.
///
/// Everything else is compared, including tensor dims and strides, every node's
/// attributes, and all three graph-level data types -- each of those genuinely changes
/// what a kernel does, so a measurement does not transfer across them.
///
/// # Cost
///
/// One contiguous copy of the serialized graph per key, because a cached record outlives
/// the caller's buffer. That is a single `memcpy` of a few KB and the retained bytes are
/// the wire format itself -- no per-node allocation, and the comparison reads them
/// directly.
class GraphContentKey
{
public:
    /// The serialized graph this key matched on, kept as the wire bytes. Shared because
    /// a record outlives the plan that produced it, and copies of the key must not
    /// re-copy the buffer.
    using Content = std::shared_ptr<const std::vector<uint8_t>>;

    GraphContentKey() = default;

    explicit GraphContentKey(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
        : _content(retain(graph))
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

    /// Hash first because it rejects almost every non-match without reading either
    /// graph; the structural comparison then decides the survivors.
    ///
    /// An unkeyable graph matches **nothing, including another unkeyable graph**. Two
    /// graphs whose bytes could not be retained are not known to be the same
    /// computation -- they are two graphs we know nothing about, and treating them as
    /// equal would serve one's measured ranking for the other. That is the wrong-kernel
    /// outcome the whole key exists to prevent, so absence of content is a permanent
    /// miss rather than a wildcard.
    bool operator==(const GraphContentKey& other) const
    {
        if(_hash != other._hash)
        {
            return false;
        }
        const auto* left = root();
        const auto* right = other.root();
        if(left == nullptr || right == nullptr)
        {
            return false;
        }
        return hipdnn_flatbuffers_sdk::data_objects::cachekey::logicallyEqual(left, right);
    }

    /// Whether this key can identify a graph at all. False when the graph was invalid or
    /// its implementation does not supply `bytes()`; callers must not cache under it.
    bool isUsable() const
    {
        return root() != nullptr;
    }

    bool operator!=(const GraphContentKey& other) const
    {
        return !(*this == other);
    }

protected:
    /// Test seam: overrides the narrowing hash so a collision can be forced. Nothing in
    /// production calls this. It exists because the header's central claim -- the hash
    /// only narrows, the content decides -- is unobservable otherwise: `operator==`
    /// short-circuits on a hash mismatch, so the structural comparison it promises can
    /// never be reached in a test unless the hashes are made to agree deliberately.
    void forceHash(uint64_t hash)
    {
        _hash = hash;
    }

private:
    /// Copies the verified buffer. `IGraph` is a view over storage this key does not
    /// own and will outlive, and the bytes are already the comparison's input format,
    /// so the copy is one memcpy rather than a graph reconstruction.
    static Content retain(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
    {
        const auto bytes = graph.bytes();
        if(bytes.data == nullptr || bytes.size == 0)
        {
            return nullptr;
        }
        return std::make_shared<const std::vector<uint8_t>>(bytes.data, bytes.data + bytes.size);
    }

    const hipdnn_flatbuffers_sdk::data_objects::Graph* root() const
    {
        if(_content == nullptr)
        {
            return nullptr;
        }
        // Verified by GraphWrapper before bytes() would hand them over.
        return ::flatbuffers::GetRoot<hipdnn_flatbuffers_sdk::data_objects::Graph>(
            _content->data());
    }

    /// The same generated walk the comparison uses, so the two can never disagree about
    /// which fields matter: a hash collision is possible, a hash *disagreement* on
    /// equal content is not.
    static uint64_t fold(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
    {
        if(!graph.isValid())
        {
            return 0;
        }
        hipdnn_flatbuffers_sdk::data_objects::cachekey::Hasher hasher;
        hipdnn_flatbuffers_sdk::data_objects::cachekey::hashAppend(hasher, &graph.getGraph());
        return hasher.value();
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
