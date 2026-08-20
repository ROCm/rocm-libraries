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
/// Two graphs are equal when a kernel measurement taken on one is valid for the other;
/// any future change to the excluded field set must answer that question.
///
/// `hash` narrows the lookup to a bucket and may be lossy; `logicallyEqual` decides the
/// match. Both come from `cachekey_generated.h`, generated from `graph.fbs`, so a new
/// schema field is hashed and compared automatically. Both read the buffer in place: no
/// `UnPack`, no allocation, no reflection.
///
/// Exclusion lives on the field in `graph.fbs` as `(cache_ignore)`; nothing here
/// re-states the list.
///
/// Tensors and nodes are compared in vector order, which `IGraph`'s topological-order
/// precondition already fixes: the frontend's sort is deterministic, so the same graph
/// built the same way keys the same. Two different construction orders of one logical
/// DAG miss rather than mismatch.
///
/// One contiguous copy of the serialized graph per key, since a cached record outlives
/// the caller's buffer.
class GraphContentKey
{
public:
    /// The serialized graph this key matched on, kept as wire bytes. Shared because a
    /// record outlives the plan that produced it; copies of the key must not re-copy it.
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

    /// Hash first rejects most non-matches cheaply; the structural comparison decides
    /// the survivors.
    ///
    /// An unkeyable graph matches nothing, including another unkeyable graph: absence
    /// of content is a permanent miss, not a wildcard.
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
    /// Test seam: overrides the narrowing hash to force a collision. `operator==`
    /// short-circuits on a hash mismatch, so the structural comparison is otherwise
    /// unreachable in a test.
    void forceHash(uint64_t hash)
    {
        _hash = hash;
    }

private:
    /// Copies the verified buffer: `IGraph` is a view over storage this key does not
    /// own, and the bytes are already the comparison's input format.
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

    /// The same generated walk the comparison uses, so hash and comparison can never
    /// disagree about which fields matter.
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
