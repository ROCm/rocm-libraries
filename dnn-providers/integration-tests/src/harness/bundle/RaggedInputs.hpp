// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaDensify.hpp>

#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/LiveRows.hpp"

namespace hipdnn_integration_tests::bundle::detail
{

using TensorAttributeMap
    = std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>;

// The offset tensor of every ragged tensor in `attributes`, and the seq_len_q and
// seq_len_kv tensors of every SDPA node of `graph` that has a ragged tensor, in ascending
// order.
inline std::vector<int64_t>
    raggedIndexUids(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                    const TensorAttributeMap& attributes)
{
    std::set<int64_t> uids;
    std::set<int64_t> raggedUids;
    for(const auto& [uid, attrs] : attributes)
    {
        if(attrs->ragged_offset_tensor_uid().has_value())
        {
            raggedUids.insert(uid);
            uids.insert(attrs->ragged_offset_tensor_uid().value());
        }
    }
    if(graph.nodes() == nullptr)
    {
        return {uids.begin(), uids.end()};
    }
    const auto addSeqLens = [&](const auto& sdpa) {
        bool ragged = false;
        for(const auto& role : hipdnn_test_sdk::detail::ragged_densify_detail::sdpaRoles(sdpa))
        {
            ragged = ragged || raggedUids.count(role.uid) != 0;
        }
        if(!ragged)
        {
            return;
        }
        for(const auto& seqLen : {sdpa.seq_len_q_tensor_uid, sdpa.seq_len_kv_tensor_uid})
        {
            if(seqLen.has_value())
            {
                uids.insert(seqLen.value());
            }
        }
    };
    for(const auto* node : *graph.nodes())
    {
        if(const auto* fwd = node->attributes_as_SdpaAttributes(); fwd != nullptr)
        {
            const std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT> sdpa(
                fwd->UnPack());
            addSeqLens(*sdpa);
        }
        else if(const auto* bwd = node->attributes_as_SdpaBackwardAttributes(); bwd != nullptr)
        {
            const std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributesT>
                sdpa(bwd->UnPack());
            addSeqLens(*sdpa);
        }
    }
    return {uids.begin(), uids.end()};
}

// dims and strides of `attrs`, as std::vector.
inline std::pair<std::vector<int64_t>, std::vector<int64_t>>
    raggedDimsAndStrides(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs)
{
    return {hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrs.dims()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrs.strides())};
}

// The ragged offset tensor `attrs` names, found in `tensors`. Throws std::invalid_argument
// when it has no data.
inline const std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>&
    findRaggedOffsetTensor(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                           const TensorMap& tensors)
{
    const int64_t offsetUid = attrs.ragged_offset_tensor_uid().value();
    const auto offsetIt = tensors.find(offsetUid);
    if(offsetIt == tensors.end())
    {
        throw std::invalid_argument("offset tensor " + std::to_string(offsetUid) + " has no data");
    }
    return offsetIt->second;
}

// The ragged layout of the tensor `attrs`, read from its offset tensor and, when its node
// has one, its seq_len tensor in `tensors`. Throws std::invalid_argument when a tensor is
// missing or the values do not form a valid ragged layout.
inline hipdnn_data_sdk::utilities::detail::RaggedFootprint
    raggedLayoutFootprint(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                          const TensorMap& tensors,
                          const std::unordered_map<int64_t, int64_t>& lengthUids)
{
    namespace data = hipdnn_data_sdk::utilities;
    const auto [dims, strides] = raggedDimsAndStrides(attrs);
    if(dims.size() != 4 || strides.size() != 4)
    {
        throw std::invalid_argument("a ragged tensor must be rank 4");
    }
    const int64_t multiplier = attrs.ragged_offset_multiplier();
    if(multiplier < 1)
    {
        throw std::invalid_argument("ragged_offset_multiplier " + std::to_string(multiplier)
                                    + " is less than 1");
    }

    const int64_t offsetUid = attrs.ragged_offset_tensor_uid().value();
    const auto& offsetTensor = findRaggedOffsetTensor(attrs, tensors);
    auto offsets = readLengths(*offsetTensor);
    if(static_cast<int64_t>(offsets.size()) != dims[0] + 1)
    {
        throw std::invalid_argument("offset tensor " + std::to_string(offsetUid) + " holds "
                                    + std::to_string(offsets.size()) + " values for "
                                    + std::to_string(dims[0]) + " batches");
    }
    for(auto& offset : offsets)
    {
        offset *= multiplier;
    }

    std::optional<std::vector<int64_t>> lengths;
    if(const auto lengthsTensor
       = hipdnn_test_sdk::utilities::findRaggedLengths(lengthUids, tensors, attrs.uid());
       lengthsTensor != nullptr)
    {
        lengths = readLengths(*lengthsTensor);
    }

    const auto batches = data::detail::raggedBatches(
        offsets, lengths.has_value() ? &*lengths : nullptr, strides[2], dims[2]);
    return data::detail::raggedFootprint(dims, strides, data::SDPA_SEQ_AXIS, batches);
}

// A ragged view of the dense host buffer `data` of the tensor `attrs`, with the origins
// and lengths of its offset and seq_len tensors in `tensors`. The caller keeps `data`
// and `tensors` alive while the view lives. Throws std::invalid_argument when the
// offsets and lengths do not form a valid ragged layout.
inline std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>
    makeRaggedView(void* data,
                   const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attrs,
                   const TensorMap& tensors,
                   const std::unordered_map<int64_t, int64_t>& lengthUids)
{
    // Not structured-bound: a structured binding captured by the lambda below is a
    // C++20 extension that this C++17 build rejects under -Wpedantic -Werror.
    const auto dimsAndStrides = raggedDimsAndStrides(attrs);
    const auto& dims = dimsAndStrides.first;
    const auto& strides = dimsAndStrides.second;
    const auto& offsetTensor = findRaggedOffsetTensor(attrs, tensors);
    const auto lengths
        = hipdnn_test_sdk::utilities::findRaggedLengths(lengthUids, tensors, attrs.uid());
    return hipdnn_test_sdk::detail::visitNativeType(
        attrs.data_type(), [&](auto tag) -> std::unique_ptr<hipdnn_data_sdk::utilities::ITensor> {
            using T = typename decltype(tag)::type;
            return std::make_unique<hipdnn_data_sdk::utilities::ShallowRaggedTensor<T>>(
                data,
                dims,
                strides,
                hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS,
                offsetTensor,
                std::nullopt,
                attrs.ragged_offset_multiplier(),
                lengths);
        });
}

} // namespace hipdnn_integration_tests::bundle::detail
