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

/// The graph half of a winner-cache key: content, never identity. Two graphs are equal
/// when a kernel measurement taken on one is valid for the other.
///
/// `hash` narrows the lookup and may be lossy; `logicallyEqual` decides the match. Both
/// are generated from `graph.fbs` into `cachekey_generated.h`, so a new schema field
/// participates automatically, reading the buffer in place with no `UnPack` or
/// allocation.
///
/// Field policy lives in `graph.fbs`: `(cache_ignore)` drops a field, `(cache_uid)`
/// folds a tensor reference as its ordinal in `Graph.tensors`, not its caller-assigned
/// uid -- renumbering keys the same, rewiring an operand does not.
///
/// Tensors and nodes compare in vector order, fixed by `IGraph`'s topological-order
/// precondition: two construction orders of one logical DAG miss rather than mismatch.
class GraphContentKey
{
public:
    /// Shared because a record outlives the plan that produced it.
    using Content = std::shared_ptr<const std::vector<uint8_t>>;

    GraphContentKey() = default;

    /// Retained before `fold()` reads it, so hash and comparison walk the same bytes.
    explicit GraphContentKey(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
        : _content(retain(graph))
        , _hash(fold())
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

    /// False when the graph was invalid or supplied no `bytes()`; callers must not
    /// cache under it.
    bool isUsable() const
    {
        return root() != nullptr;
    }

    bool operator!=(const GraphContentKey& other) const
    {
        return !(*this == other);
    }

protected:
    /// Test seam: forces a hash collision so a test can reach the structural
    /// comparison, which `operator==` otherwise short-circuits before.
    void forceHash(uint64_t hash)
    {
        _hash = hash;
    }

private:
    /// Copies the verified buffer: `IGraph` is a view this key does not own.
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

    /// Hash 0 means unkeyable, agreeing with `isUsable()` and `operator==`.
    uint64_t fold() const
    {
        const auto* graph = root();
        if(graph == nullptr)
        {
            return 0;
        }
        hipdnn_flatbuffers_sdk::data_objects::cachekey::Hasher hasher;
        hipdnn_flatbuffers_sdk::data_objects::cachekey::hashAppend(hasher, graph);
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
