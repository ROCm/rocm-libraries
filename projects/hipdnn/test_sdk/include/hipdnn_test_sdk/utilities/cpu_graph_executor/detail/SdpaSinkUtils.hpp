// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

// Whether an SDPA forward reference can take the node's sink_token: absent, or a non-ragged
// [1, H_q, 1, 1] tensor (H_q from Q's dims) of FLOAT, or of Q's data type when that is HALF or
// BFLOAT16 (the frontend contract). Any other sink makes the plan not applicable.
inline bool isSdpaSinkSupported(
    const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& nodeAttributes,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto sinkUid = nodeAttributes.sink_token_tensor_uid();
    if(!sinkUid.has_value())
    {
        return true;
    }
    const auto sinkIt = tensorMap.find(sinkUid.value());
    const auto qIt = tensorMap.find(nodeAttributes.q_tensor_uid());
    if(sinkIt == tensorMap.end() || qIt == tensorMap.end())
    {
        return false;
    }
    const auto* sink = sinkIt->second;
    const auto* q = qIt->second;
    if(sink->ragged_offset_tensor_uid().has_value() || sink->dims() == nullptr
       || q->dims() == nullptr || q->dims()->size() < 2)
    {
        return false;
    }
    const auto* dims = sink->dims();
    if(dims->size() != 4 || dims->Get(0) != 1 || dims->Get(1) != q->dims()->Get(1)
       || dims->Get(2) != 1 || dims->Get(3) != 1)
    {
        return false;
    }
    const auto sinkType = sink->data_type();
    const auto qType = q->data_type();
    return sinkType == DataType::FLOAT
           || (sinkType == qType && (qType == DataType::HALF || qType == DataType::BFLOAT16));
}

// Reads the sink_token operand from the variant pack as one float per query head.
inline std::vector<float>
    readSdpaSinkPerHead(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& sink,
                        const std::unordered_map<int64_t, void*>& variantPack,
                        int64_t numHeads)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    void* ptr = variantPack.at(sink.uid);
    switch(sink.data_type)
    {
    case DataType::FLOAT:
        return utilities::CpuFpReferenceSdpa::readSinkPerHead(
            *createShallowTensor<float>(sink, ptr), numHeads);
    case DataType::HALF:
        return utilities::CpuFpReferenceSdpa::readSinkPerHead(
            *createShallowTensor<utilities::half>(sink, ptr), numHeads);
    case DataType::BFLOAT16:
        return utilities::CpuFpReferenceSdpa::readSinkPerHead(
            *createShallowTensor<utilities::bfloat16>(sink, ptr), numHeads);
    default:
        throw std::invalid_argument(
            "SDPA sink_token must be FLOAT, HALF or BFLOAT16 in the reference");
    }
}

} // namespace hipdnn_test_sdk::detail
