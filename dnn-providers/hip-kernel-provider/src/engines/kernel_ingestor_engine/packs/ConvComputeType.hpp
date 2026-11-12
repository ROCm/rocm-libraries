// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>

/**
 * @file ConvComputeType.hpp
 * @brief The convolution node compute types the conv engines serve, per element type.
 *
 * cuDNN's set for a convolution (cudnn-frontend docs/developer/graph-api.mdx, "Compute Data
 * Type"): FLOAT for FLOAT, HALF and BFLOAT16 elements, and HALF as well for HALF elements.
 * UNSET is read as FLOAT. Every conv engine accumulates in fp32 with one final rounding, at
 * least the precision any of these asks for; a HALF compute type permits less but does not
 * require it. Every other pairing (BFLOAT16 compute, HALF compute on FLOAT or BFLOAT16
 * elements, DOUBLE, integers) is declined.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

inline bool convServesComputeType(hipdnn_flatbuffers_sdk::data_objects::DataType computeType,
                                  hipdnn_flatbuffers_sdk::data_objects::DataType elementType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(computeType)
    {
    case DataType::UNSET:
    case DataType::FLOAT:
        return true;
    case DataType::HALF:
        return elementType == DataType::HALF;
    default:
        return false;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
