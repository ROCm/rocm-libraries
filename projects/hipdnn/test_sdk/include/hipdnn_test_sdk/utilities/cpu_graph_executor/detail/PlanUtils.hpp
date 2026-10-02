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
/// A deprecated causal_mask / causal_mask_bottom_right flag maps to a causal band on its own and
/// is rejected alongside left_bound / right_bound, the other flag, or a contradicting alignment.
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

    if((causalDeprecated || bottomRightDeprecated)
       && (nodeAttributes.left_bound().has_value() || nodeAttributes.right_bound().has_value()))
    {
        throw std::invalid_argument(std::string(planName)
                                    + ": deprecated causal_mask / causal_mask_bottom_right cannot "
                                      "be combined with left_bound or right_bound");
    }

    if(causalDeprecated && !isTopLeft)
    {
        throw std::invalid_argument(std::string(planName)
                                    + ": deprecated causal_mask cannot be combined with "
                                      "diagonal_alignment=BOTTOM_RIGHT");
    }

    if(causalDeprecated)
    {
        return {-1, 0, true};
    }
    if(bottomRightDeprecated)
    {
        return {-1, 0, false};
    }

    return {leftBound, rightBound, isTopLeft};
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
