// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

/// @file InputDomain.hpp
/// @brief What the bench must write into a graph's inputs: the tensors the validation fill must
///        keep positive, and the contents of those that decide how much work runs.
namespace hipdnn_bench
{

/// @brief Whether a pointwise @p mode is undefined on negative values (and on zero, which the
///        fill never writes): LOG, SQRT and RSQRT.
inline bool undefinedOnNegatives(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    return mode == PointwiseMode::LOG || mode == PointwiseMode::SQRT
           || mode == PointwiseMode::RSQRT;
}

/// @brief The uids of the tensors an operation undefined on negative values reads, in the
///        serialized graph (or graph-and-plan container) @p data.
///
/// The fill keeps those it writes positive. One it does not write -- a tensor the graph
/// computes -- can be negative, so a NaN there may be the correct answer.
inline std::set<int64_t> positiveDomainReads(const uint8_t* data, size_t size)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    const auto wrapper
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(data,
                                                                                         size);
    std::set<int64_t> reads;
    const auto* nodes = wrapper.getGraph().nodes();
    if(nodes == nullptr)
    {
        return reads;
    }
    for(const auto* node : *nodes)
    {
        const auto* pointwise = node->attributes_as_PointwiseAttributes();
        if(pointwise != nullptr && undefinedOnNegatives(pointwise->operation()))
        {
            reads.insert(pointwise->in_0_tensor_uid());
        }
    }
    return reads;
}

/// @brief Input contents by uid, in element index order.
using DeclaredContents = std::map<int64_t, std::vector<int64_t>>;

namespace detail
{
namespace fb = hipdnn_flatbuffers_sdk::data_objects;

inline const fb::TensorAttributes* findTensor(const fb::Graph& graph, int64_t uid)
{
    if(graph.tensors() == nullptr)
    {
        return nullptr;
    }
    for(const auto* tensor : *graph.tensors())
    {
        if(tensor->uid() == uid)
        {
            return tensor;
        }
    }
    return nullptr;
}

inline const fb::TensorAttributes* findTensor(const fb::Graph& graph,
                                              flatbuffers::Optional<int64_t> uid)
{
    return uid.has_value() ? findTensor(graph, *uid) : nullptr;
}

/// Extent @p axis of @p tensor, when it has one and it is positive.
inline std::optional<int64_t> extent(const fb::TensorAttributes* tensor, size_t axis)
{
    if(tensor == nullptr || tensor->dims() == nullptr || tensor->dims()->size() <= axis
       || tensor->dims()->Get(static_cast<flatbuffers::uoffset_t>(axis)) <= 0)
    {
        return std::nullopt;
    }
    return tensor->dims()->Get(static_cast<flatbuffers::uoffset_t>(axis));
}

/// Elements of @p tensor, when every extent is known and non-negative.
inline std::optional<int64_t> elementCount(const fb::TensorAttributes* tensor)
{
    if(tensor == nullptr || tensor->dims() == nullptr)
    {
        return std::nullopt;
    }
    int64_t count = 1;
    for(const auto dim : *tensor->dims())
    {
        if(dim < 0)
        {
            return std::nullopt;
        }
        count *= dim;
    }
    return count;
}

/// Collects contents, leaving out a uid two rules give different contents.
class ContentsBuilder
{
public:
    /// Element i of @p tensor is @p first + i x @p step.
    void ramp(const fb::TensorAttributes* tensor, int64_t step, int64_t first = 0)
    {
        const auto count = elementCount(tensor);
        if(!count)
        {
            return;
        }
        std::vector<int64_t> values(static_cast<size_t>(*count));
        for(size_t index = 0; index < values.size(); ++index)
        {
            values[index] = first + (static_cast<int64_t>(index) * step);
        }
        add(tensor->uid(), std::move(values));
    }

    /// Every element of @p tensor is @p value.
    void fill(const fb::TensorAttributes* tensor, int64_t value)
    {
        ramp(tensor, 0, value);
    }

    void add(int64_t uid, std::vector<int64_t> values)
    {
        if(_conflicted.count(uid) != 0)
        {
            return;
        }
        const auto [found, inserted] = _contents.emplace(uid, values);
        if(!inserted && found->second != values)
        {
            _contents.erase(found);
            _conflicted.insert(uid);
        }
    }

    DeclaredContents take()
    {
        return std::move(_contents);
    }

private:
    DeclaredContents _contents;
    std::set<int64_t> _conflicted;
};

/// Valid lengths: every element Q's declared sequence extent (axis 2 of [B, H, S, D]), and
/// @p keys for K.
template <typename TAttention>
void declareLengths(const fb::Graph& graph,
                    const TAttention& op,
                    std::optional<int64_t> keys,
                    ContentsBuilder& contents)
{
    if(const auto queries = extent(findTensor(graph, op.q_tensor_uid()), 2))
    {
        contents.fill(findTensor(graph, op.seq_len_q_tensor_uid()), *queries);
    }
    if(keys)
    {
        contents.fill(findTensor(graph, op.seq_len_kv_tensor_uid()), *keys);
    }
}

inline void declareAttention(const fb::Graph& graph,
                             const fb::SdpaAttributes& op,
                             ContentsBuilder& contents)
{
    const auto* k = findTensor(graph, op.k_tensor_uid());
    const auto* v = findTensor(graph, op.v_tensor_uid());
    std::optional<int64_t> keys = extent(k, 2);
    if(op.page_table_k_tensor_uid().has_value())
    {
        // A paged cache holds [pages, heads, page size, D]: the keys per sequence are the stated
        // maximum, else the table's pages per sequence times the page size.
        const auto pages = extent(findTensor(graph, op.page_table_k_tensor_uid()), 2);
        if(op.max_seq_len_kv().has_value() && *op.max_seq_len_kv() > 0)
        {
            keys = *op.max_seq_len_kv();
        }
        else
        {
            keys = pages && keys ? std::optional<int64_t>{*pages * *keys} : std::nullopt;
        }
    }
    declareLengths(graph, op, keys, contents);
    if(const auto queries = extent(findTensor(graph, op.q_tensor_uid()), 2))
    {
        contents.ramp(findTensor(graph, op.cu_seq_len_q_tensor_uid()), *queries);
    }
    if(keys)
    {
        contents.ramp(findTensor(graph, op.cu_seq_len_kv_tensor_uid()), *keys);
    }
    // Every sequence its own pages: entry i of a table is block i, when the cache holds them.
    for(const auto& [uid, cache] :
        {std::pair{op.page_table_k_tensor_uid(), k}, std::pair{op.page_table_v_tensor_uid(), v}})
    {
        const auto* table = findTensor(graph, uid);
        const auto blocks = extent(cache, 0);
        const auto count = elementCount(table);
        if(blocks && count && *count <= *blocks)
        {
            contents.ramp(table, 1);
        }
    }
    contents.fill(findTensor(graph, op.block_mask_tensor_uid()), 1);
}

/// Group g starts at g x rows / groups: every declared row routed, groups as even as they go.
inline void declareRouting(const fb::TensorAttributes* offsets,
                           std::optional<int64_t> rows,
                           ContentsBuilder& contents)
{
    const auto groups = elementCount(offsets);
    if(!rows || !groups || *groups == 0)
    {
        return;
    }
    std::vector<int64_t> values(static_cast<size_t>(*groups));
    for(size_t group = 0; group < values.size(); ++group)
    {
        values[group] = static_cast<int64_t>(group) * *rows / *groups;
    }
    contents.add(offsets->uid(), std::move(values));
}

/// A ragged tensor's batches where the dense layout puts them: offset i is i batch strides,
/// in units of its multiplier.
inline void declareRaggedOffsets(const fb::Graph& graph, ContentsBuilder& contents)
{
    if(graph.tensors() == nullptr)
    {
        return;
    }
    for(const auto* tensor : *graph.tensors())
    {
        const auto* offsets = findTensor(graph, tensor->ragged_offset_tensor_uid());
        const auto multiplier = tensor->ragged_offset_multiplier();
        if(offsets == nullptr || tensor->strides() == nullptr || tensor->strides()->size() == 0
           || multiplier < 1 || tensor->strides()->Get(0) % multiplier != 0)
        {
            continue;
        }
        contents.ramp(offsets, tensor->strides()->Get(0) / multiplier);
    }
}
} // namespace detail

/// @brief The inputs whose contents decide how much of a node's declared work runs, each with
///        the contents under which the run is the problem the graph declares: the one
///        `graph.flops` counts (RFC 0019 section 6.8). @p data is a serialized graph (or a
///        graph-and-plan container).
///
/// Without them a timing buffer's zeros give valid lengths and offsets of 0 and a block mask
/// keeping nothing, and the validation fill's 1s and 2s lengths of a token or two, so a label
/// would time a near-empty problem against the declared problem's work. Contents:
/// - a ragged offset tensor: element i = i x the ragged tensor's batch stride / its multiplier;
/// - SDPA forward and backward `seq_len_q` / `seq_len_kv`: every element Q's or K's sequence
///   extent, a paged cache's declared keys for K;
/// - SDPA `cu_seq_len_q` / `cu_seq_len_kv`: element i = i x that length;
/// - SDPA `page_table_k` / `page_table_v`: element i = block i, when the cache holds them all;
/// - SDPA `block_mask`: every block kept;
/// - MoE grouped matmul `first_token_offset` (both directions): group g starts at
///   g x rows / groups, rows being the declared routed rows.
/// An operand without contents here (a missing extent, a table larger than its cache, two
/// rules disagreeing) keeps the generic fills.
inline DeclaredContents declaredContents(const uint8_t* data, size_t size)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    const auto wrapper
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(data,
                                                                                         size);
    const auto& graph = wrapper.getGraph();
    detail::ContentsBuilder contents;
    detail::declareRaggedOffsets(graph, contents);
    if(graph.nodes() == nullptr)
    {
        return contents.take();
    }
    for(const auto* node : *graph.nodes())
    {
        if(const auto* sdpa = node->attributes_as_SdpaAttributes())
        {
            detail::declareAttention(graph, *sdpa, contents);
        }
        else if(const auto* backward = node->attributes_as_SdpaBackwardAttributes())
        {
            const auto keys
                = detail::extent(detail::findTensor(graph, backward->k_tensor_uid()), 2);
            detail::declareLengths(graph, *backward, keys, contents);
        }
        else if(const auto* moe = node->attributes_as_MoeGroupedMatmulAttributes())
        {
            // Gathering routes the token index's rows, the other modes the token's.
            const auto* rows = moe->mode() == fb::MoeGroupedMatmulMode::GATHER
                                   ? detail::findTensor(graph, moe->token_index_tensor_uid())
                                   : detail::findTensor(graph, moe->token_tensor_uid());
            detail::declareRouting(detail::findTensor(graph, moe->first_token_offset_tensor_uid()),
                                   detail::extent(rows, 1),
                                   contents);
        }
        else if(const auto* moeBackward = node->attributes_as_MoeGroupedMatmulBwdAttributes())
        {
            detail::declareRouting(
                detail::findTensor(graph, moeBackward->first_token_offset_tensor_uid()),
                detail::extent(detail::findTensor(graph, moeBackward->token_tensor_uid()), 1),
                contents);
        }
    }
    return contents.take();
}

} // namespace hipdnn_bench
