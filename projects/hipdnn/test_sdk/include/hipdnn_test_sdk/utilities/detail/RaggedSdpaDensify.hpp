// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedTokenBoundaries.hpp>

namespace hipdnn_test_sdk::detail
{

// A ragged (RFC-0014) SDPA graph run as its dense padded equivalent. The references implement
// every SDPA feature for dense tensors with per-batch lengths (seq_len_q/kv + padding_mask), and a
// ragged tensor is that same data packed token by token, so a ragged graph the native ragged plans
// decline (backward, bias, dropout, sinks, paged K/V with ragged Q, dtypes they do not register)
// is rewritten to the dense graph, its packed buffers are unpacked into dense ones, the dense plans
// run, and the outputs are packed back. Per-batch lengths come from the ragged offsets; when the
// graph also carries seq_len tensors (cuDNN's ragged + padding form) those give the live lengths
// and must not exceed the packed ones.

// A packed primary of the node: the tensor, its offset table and its two layouts.
struct RaggedSdpaPrimary
{
    int64_t uid = 0;
    int64_t offsetUid = 0;
    size_t offsetElementSize = 4;
    int64_t multiplier = 1;
    std::vector<int64_t> dims; // logical [B, H, S, D] (LSE: [B, H, S, 1])
    std::vector<int64_t> packedStrides; // token-major strides of the packed tensor
    size_t elementSize = 0;
    bool queryPacking = true; // packed like Q (Q, O, dO, dQ, LSE) or like K (K, V, dK, dV)
    bool output = false;
};

struct DenseSdpaFromRagged
{
    std::vector<uint8_t> graph; // the dense graph, serialized
    std::vector<RaggedSdpaPrimary> primaries;
    int64_t batch = 0;
    int64_t maxSeqQ = 0;
    int64_t maxSeqKv = 0;
    // seq_len tensors the rewrite added (INT32 [B, 1, 1, 1]); the caller fills them with the
    // packed lengths. Absent when the graph already had that tensor.
    std::optional<int64_t> addedSeqLenQUid;
    std::optional<int64_t> addedSeqLenKvUid;
    // The dense graph's seq_len tensors (added or the graph's own).
    int64_t seqLenQUid = 0;
    int64_t seqLenKvUid = 0;
};

namespace ragged_densify_detail
{

inline size_t elementSizeOf(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    return std::visit([](auto value) { return sizeof(value); },
                      hipdnn_test_sdk::utilities::datatypeToNativeVariant(type));
}

inline std::vector<int64_t> contiguousStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

struct Role
{
    int64_t uid;
    bool queryPacking;
    bool output;
    // Row statistics (LSE, Max, Sum_exp) may be dense beside a ragged Q; the other primaries of a
    // packing group are all ragged or all dense.
    bool stats = false;
};

template <typename AttrsT>
std::vector<Role> sdpaRoles(const AttrsT& attrs);

template <>
inline std::vector<Role>
    sdpaRoles(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT& attrs)
{
    std::vector<Role> roles = {{attrs.q_tensor_uid, true, false, false},
                               {attrs.k_tensor_uid, false, false, false},
                               {attrs.v_tensor_uid, false, false, false},
                               {attrs.o_tensor_uid, true, true, false}};
    for(const auto& stats :
        {attrs.stats_tensor_uid, attrs.max_tensor_uid, attrs.sum_exp_tensor_uid})
    {
        if(stats.has_value())
        {
            roles.push_back({stats.value(), true, true, true});
        }
    }
    return roles;
}

template <>
inline std::vector<Role>
    sdpaRoles(const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributesT& attrs)
{
    return {{attrs.q_tensor_uid, true, false, false},
            {attrs.k_tensor_uid, false, false, false},
            {attrs.v_tensor_uid, false, false, false},
            {attrs.o_tensor_uid, true, false, false},
            {attrs.do_tensor_uid, true, false, false},
            {attrs.stats_tensor_uid, true, false, true},
            {attrs.dq_tensor_uid, true, true, false},
            {attrs.dk_tensor_uid, false, true, false},
            {attrs.dv_tensor_uid, false, true, false}};
}

template <typename AttrsT>
std::optional<DenseSdpaFromRagged> densify(hipdnn_flatbuffers_sdk::data_objects::GraphT& graph,
                                           AttrsT& attrs)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    using hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT;

    std::unordered_map<int64_t, TensorAttributesT*> tensors;
    int64_t maxUid = 0;
    for(auto& tensor : graph.tensors)
    {
        tensors[tensor->uid] = tensor.get();
        maxUid = std::max(maxUid, tensor->uid);
    }
    const auto roles = sdpaRoles(attrs);
    std::unordered_map<int64_t, Role> roleOf;
    for(const auto& role : roles)
    {
        if(tensors.count(role.uid) == 0)
        {
            return std::nullopt;
        }
        roleOf.emplace(role.uid, role);
    }
    // Only the node's primaries (and a packed LSE) may be ragged.
    for(const auto& [uid, tensor] : tensors)
    {
        if(tensor->ragged_offset_tensor_uid.has_value() && roleOf.count(uid) == 0)
        {
            return std::nullopt;
        }
    }
    // A partly ragged packing group (e.g. ragged Q/K/V with a dense O) is not a layout either
    // reference defines, so it is declined rather than guessed at (RFC-0014).
    for(const bool queryGroup : {true, false})
    {
        int ragged = 0;
        int dense = 0;
        for(const auto& role : roles)
        {
            if(role.queryPacking == queryGroup && !role.stats)
            {
                (tensors.at(role.uid)->ragged_offset_tensor_uid.has_value() ? ragged : dense) += 1;
            }
        }
        if(ragged > 0 && dense > 0)
        {
            return std::nullopt;
        }
    }

    DenseSdpaFromRagged result;
    std::vector<int64_t> offsetUids;
    for(const auto& role : roles)
    {
        auto* tensor = tensors.at(role.uid);
        if(!tensor->ragged_offset_tensor_uid.has_value())
        {
            continue;
        }
        const auto offsetUid = tensor->ragged_offset_tensor_uid.value();
        const auto offsetIt = tensors.find(offsetUid);
        if(tensor->dims.size() != 4 || offsetIt == tensors.end() || tensor->virtual_
           || (offsetIt->second->data_type != DataType::INT32
               && offsetIt->second->data_type != DataType::INT64)
           || tensor->ragged_offset_multiplier < 1
           || !isTokenMajorRaggedLayout(tensor->dims, tensor->strides))
        {
            return std::nullopt;
        }
        RaggedSdpaPrimary primary;
        primary.uid = role.uid;
        primary.offsetUid = offsetUid;
        primary.offsetElementSize = elementSizeOf(offsetIt->second->data_type);
        primary.multiplier = tensor->ragged_offset_multiplier;
        primary.dims = tensor->dims;
        primary.packedStrides = tensor->strides;
        primary.elementSize = elementSizeOf(tensor->data_type);
        primary.queryPacking = role.queryPacking;
        primary.output = role.output;
        result.primaries.push_back(primary);
        offsetUids.push_back(offsetUid);

        tensor->strides = contiguousStrides(tensor->dims);
        tensor->ragged_offset_tensor_uid.reset();
        tensor->ragged_offset_multiplier = 1;
    }
    if(result.primaries.empty())
    {
        return std::nullopt;
    }

    // The offset tables are inputs of the ragged graph only; drop them unless something else
    // still refers to them.
    graph.tensors.erase(
        std::remove_if(graph.tensors.begin(),
                       graph.tensors.end(),
                       [&](const auto& tensor) {
                           return std::find(offsetUids.begin(), offsetUids.end(), tensor->uid)
                                  != offsetUids.end();
                       }),
        graph.tensors.end());

    const auto& q = *tensors.at(attrs.q_tensor_uid);
    const auto& k = *tensors.at(attrs.k_tensor_uid);
    result.batch = q.dims.at(0);
    result.maxSeqQ = q.dims.at(2);
    result.maxSeqKv = k.dims.at(2);

    // Lengths: per-batch seq_len tensors, added where the graph has none.
    const auto addSeqLen = [&](const char* name) {
        auto tensor = std::make_unique<TensorAttributesT>();
        tensor->uid = ++maxUid;
        tensor->name = name;
        tensor->data_type = DataType::INT32;
        tensor->dims = {result.batch, 1, 1, 1};
        tensor->strides = {1, 1, 1, 1};
        graph.tensors.push_back(std::move(tensor));
        return maxUid;
    };
    const bool qRagged = std::any_of(result.primaries.begin(),
                                     result.primaries.end(),
                                     [](const auto& p) { return p.queryPacking; });
    const bool kvRagged = std::any_of(result.primaries.begin(),
                                      result.primaries.end(),
                                      [](const auto& p) { return !p.queryPacking; });
    if(!attrs.seq_len_q_tensor_uid.has_value())
    {
        if(!qRagged)
        {
            return std::nullopt; // no lengths for a dense Q beside a ragged K
        }
        result.addedSeqLenQUid = addSeqLen("seq_len_q (from ragged_offset)");
        attrs.seq_len_q_tensor_uid = result.addedSeqLenQUid;
    }
    if(!attrs.seq_len_kv_tensor_uid.has_value())
    {
        if(!kvRagged)
        {
            return std::nullopt;
        }
        result.addedSeqLenKvUid = addSeqLen("seq_len_kv (from ragged_offset)");
        attrs.seq_len_kv_tensor_uid = result.addedSeqLenKvUid;
    }
    attrs.padding_mask = true;
    result.seqLenQUid = attrs.seq_len_q_tensor_uid.value();
    result.seqLenKvUid = attrs.seq_len_kv_tensor_uid.value();

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::Graph::Pack(builder, &graph));
    result.graph.assign(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
    return result;
}

} // namespace ragged_densify_detail

/// The dense padded graph equivalent to a single-node ragged SDPA forward or backward graph, or
/// nullopt when the graph has no ragged primary or cannot be expressed densely (a ragged tensor
/// that is not one of the node's primaries, a non-token-major layout, a dense Q beside a ragged K
/// without seq_len_q, or the reverse).
inline std::optional<DenseSdpaFromRagged> densifyRaggedSdpaGraph(const void* buffer, size_t size)
{
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buffer), size);
    if(!verifier.VerifyBuffer<hipdnn_flatbuffers_sdk::data_objects::Graph>())
    {
        return std::nullopt;
    }
    auto graph = hipdnn_flatbuffers_sdk::data_objects::UnPackGraph(buffer);
    if(graph == nullptr || graph->nodes.size() != 1)
    {
        return std::nullopt;
    }
    auto& node = *graph->nodes.front();
    if(auto* fwd = node.attributes.AsSdpaAttributes(); fwd != nullptr)
    {
        return ragged_densify_detail::densify(*graph, *fwd);
    }
    if(auto* bwd = node.attributes.AsSdpaBackwardAttributes(); bwd != nullptr)
    {
        return ragged_densify_detail::densify(*graph, *bwd);
    }
    return std::nullopt;
}

/// Element offset of each batch's first token: ragged_offset[b] * multiplier, read from a host copy
/// of the offset table (B + 1 INT32 or INT64 entries).
inline std::vector<int64_t>
    raggedElementOffsets(const RaggedSdpaPrimary& primary, const void* offsetsHost, int64_t batch)
{
    std::vector<int64_t> offsets(static_cast<size_t>(batch + 1));
    for(size_t b = 0; b < offsets.size(); ++b)
    {
        int64_t value = 0;
        if(primary.offsetElementSize == 8)
        {
            std::memcpy(&value, static_cast<const uint8_t*>(offsetsHost) + b * 8, 8);
        }
        else
        {
            int32_t narrow = 0;
            std::memcpy(&narrow, static_cast<const uint8_t*>(offsetsHost) + b * 4, 4);
            value = narrow;
        }
        offsets[b] = value * primary.multiplier;
    }
    return offsets;
}

/// Per-batch token counts of a packed primary (validated as RaggedTokenBoundaries does).
inline std::vector<int64_t> raggedLengths(const RaggedSdpaPrimary& primary,
                                          const std::vector<int64_t>& elementOffsets)
{
    const auto tokens = raggedTokenBoundaries(elementOffsets,
                                              primary.packedStrides.at(2),
                                              primary.dims.at(2),
                                              "ragged SDPA densify",
                                              "primary");
    std::vector<int64_t> lengths;
    for(size_t b = 1; b < tokens.size(); ++b)
    {
        lengths.push_back(tokens[b] - tokens[b - 1]);
    }
    return lengths;
}

/// Bytes a packed primary spans from its base pointer (what a device copy must cover).
inline size_t raggedPackedBytes(const RaggedSdpaPrimary& primary,
                                const std::vector<int64_t>& elementOffsets,
                                const std::vector<int64_t>& lengths)
{
    int64_t end = 0;
    for(size_t b = 0; b < lengths.size(); ++b)
    {
        if(lengths[b] == 0)
        {
            continue;
        }
        int64_t last = elementOffsets[b] + (lengths[b] - 1) * primary.packedStrides[2];
        for(const size_t axis : {size_t{1}, size_t{3}})
        {
            last += (primary.dims[axis] - 1) * primary.packedStrides[axis];
        }
        end = std::max(end, last + 1);
    }
    return static_cast<size_t>(end) * primary.elementSize;
}

/// Bytes of the dense (contiguous [B, H, S, D]) form of a primary.
inline size_t raggedDenseBytes(const RaggedSdpaPrimary& primary)
{
    int64_t count = 1;
    for(const auto d : primary.dims)
    {
        count *= d;
    }
    return static_cast<size_t>(count) * primary.elementSize;
}

/// Copies every live token between the packed and the dense form. toDense: packed -> dense (the
/// dense buffer's padding is left as the caller initialised it); otherwise dense -> packed (packed
/// bytes outside live tokens are not touched).
inline void copyRaggedTokens(const RaggedSdpaPrimary& primary,
                             const std::vector<int64_t>& elementOffsets,
                             const std::vector<int64_t>& lengths,
                             uint8_t* packed,
                             uint8_t* dense,
                             bool toDense)
{
    const auto denseStrides = ragged_densify_detail::contiguousStrides(primary.dims);
    const auto heads = primary.dims[1];
    const auto width = primary.dims[3];
    const auto size = primary.elementSize;
    for(size_t b = 0; b < lengths.size(); ++b)
    {
        for(int64_t h = 0; h < heads; ++h)
        {
            for(int64_t s = 0; s < lengths[b]; ++s)
            {
                for(int64_t d = 0; d < width; ++d)
                {
                    const int64_t p = elementOffsets[b] + h * primary.packedStrides[1]
                                      + s * primary.packedStrides[2] + d * primary.packedStrides[3];
                    const int64_t q = static_cast<int64_t>(b) * denseStrides[0]
                                      + h * denseStrides[1] + s * denseStrides[2]
                                      + d * denseStrides[3];
                    uint8_t* from = toDense ? packed + p * static_cast<int64_t>(size)
                                            : dense + q * static_cast<int64_t>(size);
                    uint8_t* to = toDense ? dense + q * static_cast<int64_t>(size)
                                          : packed + p * static_cast<int64_t>(size);
                    std::memcpy(to, from, size);
                }
            }
        }
    }
}

/// The packed lengths of each packing group, checked to agree within the group. Returns
/// {query lengths, key/value lengths}; a group with no ragged primary is empty.
inline std::pair<std::vector<int64_t>, std::vector<int64_t>>
    raggedGroupLengths(const DenseSdpaFromRagged& dense,
                       const std::vector<std::vector<int64_t>>& lengthsPerPrimary)
{
    std::vector<int64_t> q;
    std::vector<int64_t> kv;
    for(size_t i = 0; i < dense.primaries.size(); ++i)
    {
        auto& group = dense.primaries[i].queryPacking ? q : kv;
        if(group.empty())
        {
            group = lengthsPerPrimary[i];
        }
        else if(group != lengthsPerPrimary[i])
        {
            throw std::invalid_argument("ragged SDPA densify: tensors packed like "
                                        + std::string(dense.primaries[i].queryPacking ? "Q" : "K")
                                        + " have different per-batch lengths");
        }
    }
    return {q, kv};
}

/// Supplied seq_len values (cuDNN's ragged + padding form) must not exceed the packed lengths.
inline void requireSeqLensWithinPacked(const int32_t* seqLens,
                                       const std::vector<int64_t>& packed,
                                       const char* name)
{
    for(size_t b = 0; b < packed.size(); ++b)
    {
        if(seqLens[b] < 0 || seqLens[b] > packed[b])
        {
            throw std::invalid_argument(std::string("ragged SDPA densify: ") + name + "["
                                        + std::to_string(b) + "] = " + std::to_string(seqLens[b])
                                        + " exceeds the packed length "
                                        + std::to_string(packed[b]));
        }
    }
}

} // namespace hipdnn_test_sdk::detail
