// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>

/// Result of extracting diagonal band mask parameters from SDPA node attributes.
struct DiagonalBandParams
{
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
};

/// Extracts diagonal band mask parameters from SDPA attributes (forward or backward).
/// A deprecated causal_mask / causal_mask_bottom_right flag is merged the way cuDNN's setters
/// compose: it sets right_bound=0 (and BOTTOM_RIGHT alignment for causal_mask_bottom_right), then
/// any left_bound, right_bound other than -1, and (for causal_mask) diagonal_alignment apply on
/// top. Setting both flags is rejected.
/// Works with both SdpaAttributes and SdpaBackwardAttributes FlatBuffers types
/// (they expose identical accessor signatures for mask fields).
template <typename SdpaAttributesType>
DiagonalBandParams extractDiagonalBandParams(const SdpaAttributesType& nodeAttributes,
                                             const char* planName)
{
    const int64_t leftBound
        = nodeAttributes.left_bound().has_value() ? nodeAttributes.left_bound().value() : -1;
    const int64_t rightBound
        = nodeAttributes.right_bound().has_value() ? nodeAttributes.right_bound().value() : -1;

    if(leftBound < -1 || rightBound < -1)
    {
        throw std::invalid_argument(
            std::string(planName) + ": left_bound and right_bound must be >= -1 (got left_bound="
            + std::to_string(leftBound) + ", right_bound=" + std::to_string(rightBound) + ")");
    }

    const bool isTopLeft = nodeAttributes.diagonal_alignment()
                           == hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT;
    const bool causalDeprecated = nodeAttributes.causal_mask();
    const bool bottomRightDeprecated = nodeAttributes.causal_mask_bottom_right();

    if(causalDeprecated && bottomRightDeprecated)
    {
        throw std::invalid_argument(std::string(planName)
                                    + ": cannot set both causal_mask and causal_mask_bottom_right. "
                                      "Use diagonal_alignment={TOP_LEFT|BOTTOM_RIGHT} with "
                                      "left_bound=-1, right_bound=0 instead.");
    }

    if(causalDeprecated || bottomRightDeprecated)
    {
        return {leftBound, rightBound >= 0 ? rightBound : 0, causalDeprecated && isTopLeft};
    }

    return {leftBound, rightBound, isTopLeft};
}

/// Whether the SDPA node's padding attributes are ones the references implement. Padding is
/// given by both seq_len_q and seq_len_kv, each INT32 with one value per batch ([B, 1, 1, 1]).
/// padding_mask without both tensors, or only one tensor, is declined (cuDNN requires both). The
/// tensors without padding_mask are accepted as padding, because hipDNN graphs omit the flag
/// (SdpaSemanticsRegister.md SEM-008). Works for forward and backward attributes.
template <typename SdpaAttributesType, typename TensorMap>
bool isSupportedSdpaPadding(const SdpaAttributesType& nodeAttributes, const TensorMap& tensorMap)
{
    const auto seqLenQ = nodeAttributes.seq_len_q_tensor_uid();
    const auto seqLenKv = nodeAttributes.seq_len_kv_tensor_uid();
    if(!seqLenQ.has_value() && !seqLenKv.has_value())
    {
        return !nodeAttributes.padding_mask();
    }
    if(!seqLenQ.has_value() || !seqLenKv.has_value())
    {
        return false;
    }
    const auto qIt = tensorMap.find(nodeAttributes.q_tensor_uid());
    if(qIt == tensorMap.end() || qIt->second->dims() == nullptr || qIt->second->dims()->size() == 0)
    {
        return false;
    }
    const auto batch = qIt->second->dims()->Get(0);
    for(const auto uid : {seqLenQ.value(), seqLenKv.value()})
    {
        const auto it = tensorMap.find(uid);
        if(it == tensorMap.end()
           || it->second->data_type() != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32
           || it->second->dims() == nullptr || it->second->dims()->size() == 0
           || it->second->dims()->Get(0) != batch)
        {
            return false;
        }
        int64_t count = 1;
        for(const auto dim : *it->second->dims())
        {
            count *= dim;
        }
        if(count != batch)
        {
            return false;
        }
    }
    return true;
}

/// Logical key extent of a paged SDPA forward node: max_seq_len_kv when set, otherwise
/// tableSize * blockSize of the K side (or V when only V is paged). Returns -1 when the node is
/// not a paging layout the references implement:
/// - paging needs padding (both seq_len tensors), as cuDNN requires;
/// - each page table is INT32 [B, 1, tableSize, 1] over a rank-4 [blocks, H, blockSize, D]
///   container;
/// - max_seq_len_kv must fit every page table (tableSize * blockSize);
/// - a non-paged K or V must already be dense [B, H, maxSeqKv, D];
/// - a bias's last dim must be maxSeqKv or 1.
/// Returns 0 for a non-paged node, which must not set max_seq_len_kv (cuDNN rejects it).
template <typename TensorMap>
int64_t sdpaPagedMaxSeqKv(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                          const TensorMap& tensorMap)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto pageK = attrs.page_table_k_tensor_uid();
    const auto pageV = attrs.page_table_v_tensor_uid();
    if(!pageK.has_value() && !pageV.has_value())
    {
        return attrs.max_seq_len_kv().has_value() ? -1 : 0;
    }
    if(!attrs.seq_len_q_tensor_uid().has_value() || !attrs.seq_len_kv_tensor_uid().has_value())
    {
        return -1;
    }
    const auto dimsOf = [&tensorMap](int64_t uid) -> std::vector<int64_t> {
        const auto it = tensorMap.find(uid);
        if(it == tensorMap.end() || it->second->dims() == nullptr)
        {
            return {};
        }
        return {it->second->dims()->begin(), it->second->dims()->end()};
    };
    const auto qDims = dimsOf(attrs.q_tensor_uid());
    if(qDims.size() != 4)
    {
        return -1;
    }
    const auto batch = qDims[0];

    // Capacity (tableSize * blockSize) of one paged side, or -1 if its layout is unsupported.
    const auto pagedCapacity = [&](int64_t tableUid, int64_t containerUid) -> int64_t {
        const auto it = tensorMap.find(tableUid);
        const auto tableDims = dimsOf(tableUid);
        const auto containerDims = dimsOf(containerUid);
        if(it == tensorMap.end() || it->second->data_type() != DataType::INT32
           || tableDims.size() != 4 || tableDims[0] != batch || tableDims[1] != 1
           || tableDims[3] != 1 || containerDims.size() != 4)
        {
            return -1;
        }
        return tableDims[2] * containerDims[2];
    };
    int64_t capacityK = -1;
    int64_t capacityV = -1;
    if(pageK.has_value())
    {
        capacityK = pagedCapacity(pageK.value(), attrs.k_tensor_uid());
        if(capacityK < 0)
        {
            return -1;
        }
    }
    if(pageV.has_value())
    {
        capacityV = pagedCapacity(pageV.value(), attrs.v_tensor_uid());
        if(capacityV < 0)
        {
            return -1;
        }
    }

    int64_t maxSeqKv = pageK.has_value() ? capacityK : capacityV;
    if(attrs.max_seq_len_kv().has_value())
    {
        maxSeqKv = static_cast<int64_t>(attrs.max_seq_len_kv().value());
    }
    if(maxSeqKv <= 0 || (pageK.has_value() && maxSeqKv > capacityK)
       || (pageV.has_value() && maxSeqKv > capacityV))
    {
        return -1;
    }
    for(const auto& [paged, uid] : {std::pair{pageK.has_value(), attrs.k_tensor_uid()},
                                    std::pair{pageV.has_value(), attrs.v_tensor_uid()}})
    {
        const auto dims = dimsOf(uid);
        if(!paged && (dims.size() != 4 || dims[0] != batch || dims[2] != maxSeqKv))
        {
            return -1;
        }
    }
    if(attrs.attn_mask_tensor_uid().has_value())
    {
        const auto maskDims = dimsOf(attrs.attn_mask_tensor_uid().value());
        if(maskDims.empty() || (maskDims.back() != maxSeqKv && maskDims.back() != 1))
        {
            return -1;
        }
    }
    return maxSeqKv;
}

/// Whether an SDPA forward node's attention sink is one the references implement: absent, or a
/// FLOAT tensor [1, H, 1, 1] with H the query head count (cuDNN's shape).
template <typename TensorMap>
bool isSupportedSdpaSink(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& attrs,
                         const TensorMap& tensorMap)
{
    if(!attrs.sink_token_tensor_uid().has_value())
    {
        return true;
    }
    const auto sinkIt = tensorMap.find(attrs.sink_token_tensor_uid().value());
    const auto qIt = tensorMap.find(attrs.q_tensor_uid());
    if(sinkIt == tensorMap.end() || qIt == tensorMap.end() || sinkIt->second->dims() == nullptr
       || qIt->second->dims() == nullptr || qIt->second->dims()->size() != 4
       || sinkIt->second->data_type() != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
    {
        return false;
    }
    const std::vector<int64_t> dims(sinkIt->second->dims()->begin(), sinkIt->second->dims()->end());
    return dims == std::vector<int64_t>{1, qIt->second->dims()->Get(1), 1, 1};
}

#define CHECK_TENSOR_EXISTS(tensor_map, tensor_uid) \
    do                                              \
    {                                               \
        auto it = (tensor_map).find((tensor_uid));  \
        if(it == (tensor_map).end())                \
        {                                           \
            return false;                           \
        }                                           \
    } while(0)

#define CHECK_TENSOR_TYPE(tensor_map, tensor_uid, datatype_enum) \
    do                                                           \
    {                                                            \
        auto tensor = (tensor_map).at((tensor_uid));             \
        if(tensor->data_type() != (datatype_enum))               \
        {                                                        \
            return false;                                        \
        }                                                        \
    } while(0)

#define CHECK_OPTIONAL_TENSOR_EXISTS(tensor_map, optional_tensor_uid) \
    do                                                                \
    {                                                                 \
        if(!(optional_tensor_uid).has_value())                        \
        {                                                             \
            return false;                                             \
        }                                                             \
        CHECK_TENSOR_EXISTS(tensor_map, *(optional_tensor_uid));      \
    } while(0)

#define CHECK_OPTIONAL_TENSOR_TYPE(tensor_map, optional_tensor_uid, datatype_enum) \
    CHECK_TENSOR_TYPE(tensor_map, *(optional_tensor_uid), datatype_enum)

// The CPU references do not support ragged tensors. Reject any graph that
// contains one so the plan builder is reported as not applicable.
#define CHECK_NO_RAGGED_TENSORS(tensor_map)                                        \
    do                                                                             \
    {                                                                              \
        if(!hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds((tensor_map))) \
        {                                                                          \
            return false;                                                          \
        }                                                                          \
    } while(0)
