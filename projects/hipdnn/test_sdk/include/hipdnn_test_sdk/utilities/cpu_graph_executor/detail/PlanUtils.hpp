// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

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
