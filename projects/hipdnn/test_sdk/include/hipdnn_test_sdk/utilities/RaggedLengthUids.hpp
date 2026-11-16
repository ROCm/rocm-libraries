// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaDensify.hpp>

namespace hipdnn_test_sdk::utilities
{

/// Maps each ragged tensor uid of an SDPA graph to the uid of its node's per-batch length tensor.
/// Lengths belong to the node (seq_len_q / seq_len_kv), while ragged_offset_tensor_uid stays the
/// tensor's per-batch origin.
///  - seq_len_q:  Q, O, Stats, Max, Sum_exp (forward); Q, O, dO, dQ, Stats (backward).
///  - seq_len_kv: K, V (forward); K, V, dK, dV (backward).
/// A tensor has no entry when it is not ragged or when its node has no seq_len for its role.
/// Throws std::invalid_argument when one ragged tensor is bound to two different seq_len tensors
/// (the same tensor in a seq_len_q and a seq_len_kv role, or shared by two SDPA nodes).
inline std::unordered_map<int64_t, int64_t>
    raggedLengthUids(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
{
    std::unordered_map<int64_t, int64_t> lengthUids;
    if(graph.tensors() == nullptr || graph.nodes() == nullptr)
    {
        return lengthUids;
    }

    std::unordered_set<int64_t> raggedUids;
    for(const auto* tensor : *graph.tensors())
    {
        if(tensor->ragged_offset_tensor_uid().has_value())
        {
            raggedUids.insert(tensor->uid());
        }
    }

    // The q-packed roles take seq_len_q, the k-packed roles seq_len_kv (sdpaRoles' queryPacking).
    const auto bind = [&](const std::vector<detail::ragged_densify_detail::Role>& roles,
                          const std::optional<int64_t>& seqLenQ,
                          const std::optional<int64_t>& seqLenKv) {
        for(const bool queryPacking : {true, false})
        {
            const auto& lengthUid = queryPacking ? seqLenQ : seqLenKv;
            if(!lengthUid.has_value())
            {
                continue;
            }
            for(const auto& role : roles)
            {
                if(role.queryPacking != queryPacking || raggedUids.count(role.uid) == 0)
                {
                    continue;
                }
                const auto [it, inserted] = lengthUids.emplace(role.uid, lengthUid.value());
                if(!inserted && it->second != lengthUid.value())
                {
                    throw std::invalid_argument(
                        "raggedLengthUids: ragged tensor " + std::to_string(role.uid)
                        + " is bound to two seq_len tensors, " + std::to_string(it->second)
                        + " and " + std::to_string(lengthUid.value()));
                }
            }
        }
    };

    for(const auto* node : *graph.nodes())
    {
        if(const auto* fwd = node->attributes_as_SdpaAttributes(); fwd != nullptr)
        {
            const std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT> attrs(
                fwd->UnPack());
            bind(detail::ragged_densify_detail::sdpaRoles(*attrs),
                 attrs->seq_len_q_tensor_uid,
                 attrs->seq_len_kv_tensor_uid);
        }
        else if(const auto* bwd = node->attributes_as_SdpaBackwardAttributes(); bwd != nullptr)
        {
            const std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributesT>
                attrs(bwd->UnPack());
            bind(detail::ragged_densify_detail::sdpaRoles(*attrs),
                 attrs->seq_len_q_tensor_uid,
                 attrs->seq_len_kv_tensor_uid);
        }
    }

    return lengthUids;
}

} // namespace hipdnn_test_sdk::utilities
