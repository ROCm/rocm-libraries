// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>

/// Result of extracting diagonal band mask parameters from SDPA node attributes.
struct DiagonalBandParams
{
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
};

/// Extracts diagonal band mask parameters from SDPA attributes (forward or backward).
/// Handles deprecated causal_mask / causal_mask_bottom_right flags.
/// Works with both SdpaAttributes and SdpaBackwardAttributes FlatBuffers types
/// (they expose identical accessor signatures for mask fields).
template <typename SdpaAttributesType>
DiagonalBandParams extractDiagonalBandParams(const SdpaAttributesType& nodeAttributes,
                                             const char* planName)
{
    int64_t leftBound
        = nodeAttributes.left_bound().has_value() ? nodeAttributes.left_bound().value() : -1;
    int64_t rightBound
        = nodeAttributes.right_bound().has_value() ? nodeAttributes.right_bound().value() : -1;

    if(leftBound < -1 || rightBound < -1)
    {
        throw std::invalid_argument(
            std::string(planName) + ": left_bound and right_bound must be >= -1 (got left_bound="
            + std::to_string(leftBound) + ", right_bound=" + std::to_string(rightBound) + ")");
    }

    bool isTopLeft = nodeAttributes.diagonal_alignment()
                     == hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT;

    // Validate mutually exclusive deprecated attributes
    if(nodeAttributes.causal_mask() && nodeAttributes.causal_mask_bottom_right())
    {
        throw std::invalid_argument("Cannot set both causal_mask and causal_mask_bottom_right. "
                                    "Use diagonal_alignment={TOP_LEFT|BOTTOM_RIGHT} with "
                                    "left_bound=-1, right_bound=0 instead.");
    }

    // Check deprecated attributes
    if(nodeAttributes.causal_mask())
    {
        leftBound = -1;
        rightBound = 0;
        isTopLeft = true;
    }
    if(nodeAttributes.causal_mask_bottom_right())
    {
        leftBound = -1;
        rightBound = 0;
        isTopLeft = false;
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
