// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_attributes_generated.h>

namespace hipdnn_flatbuffers_sdk::utilities
{

/// Mode-dependent routing rules for a forward MoE grouped matmul, expressed over
/// plain values so the backend operation descriptor and the CPU reference plan
/// builder evaluate the same contract from their own tensor representations.
struct MoeGroupedMatmulRouting
{
    data_objects::MoeGroupedMatmulMode mode = data_objects::MoeGroupedMatmulMode::NONE;
    bool hasTokenIndex = false;
    bool hasTokenKs = false;
    data_objects::DataType firstTokenOffsetDataType = data_objects::DataType::UNSET;
    data_objects::DataType tokenIndexDataType = data_objects::DataType::UNSET;
    data_objects::DataType tokenKsDataType = data_objects::DataType::UNSET;
    int32_t topK = 0;
    int64_t expertCount = 0; ///< weight dims[0]; must be positive to bound top_k
};

/// Returns nullptr when the routing contract holds, otherwise a static reason
/// string the caller wraps in its own error type.
inline const char* checkMoeGroupedMatmulRouting(const MoeGroupedMatmulRouting& routing)
{
    using Mode = data_objects::MoeGroupedMatmulMode;
    using DataType = data_objects::DataType;

    if(routing.firstTokenOffsetDataType != DataType::INT32)
    {
        return "FIRST_TOKEN_OFFSET tensor must have INT32 data type";
    }

    switch(routing.mode)
    {
    case Mode::NONE:
        if(routing.hasTokenIndex)
        {
            return "NONE mode forbids the TOKEN_INDEX tensor";
        }
        if(routing.hasTokenKs)
        {
            return "NONE mode forbids the TOKEN_KS tensor";
        }
        if(routing.topK != 0)
        {
            return "NONE mode requires top_k to equal 0";
        }
        break;
    case Mode::GATHER:
        if(!routing.hasTokenIndex)
        {
            return "GATHER mode requires the TOKEN_INDEX tensor";
        }
        if(routing.hasTokenKs)
        {
            return "GATHER mode forbids the TOKEN_KS tensor";
        }
        if(routing.topK != 0)
        {
            return "GATHER mode requires top_k to equal 0";
        }
        break;
    case Mode::SCATTER:
        if(!routing.hasTokenIndex)
        {
            return "SCATTER mode requires the TOKEN_INDEX tensor";
        }
        if(!routing.hasTokenKs)
        {
            return "SCATTER mode requires the TOKEN_KS tensor";
        }
        if(routing.topK < 1)
        {
            return "SCATTER mode requires top_k to be at least 1";
        }
        if(routing.expertCount <= 0)
        {
            return "expert count must be positive to bound top_k";
        }
        if(routing.topK > routing.expertCount)
        {
            return "top_k must not exceed the number of experts";
        }
        break;
    default:
        return "unknown routing mode";
    }

    if(routing.hasTokenIndex && routing.tokenIndexDataType != DataType::INT32)
    {
        return "TOKEN_INDEX tensor must have INT32 data type";
    }
    if(routing.hasTokenKs && routing.tokenKsDataType != DataType::INT32)
    {
        return "TOKEN_KS tensor must have INT32 data type";
    }

    return nullptr;
}

} // namespace hipdnn_flatbuffers_sdk::utilities
