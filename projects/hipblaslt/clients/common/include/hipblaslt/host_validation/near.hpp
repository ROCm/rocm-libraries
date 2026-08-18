// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt tolerance policy.

#include "hipblaslt_ostream.hpp"

#include <hipblaslt/host_validation/Types.hpp>
#include <limits>

// Returns the per-term accumulation error bound for the compute type. Former
// GFX11 input/output-specific allowances were removed after native GFX1151
// tests passed with compute-type epsilon.
inline double sum_error_tolerance_for_compute_type(hipDataType computeType)
{
    switch(computeType)
    {
    case HIP_R_32F:
    case HIP_C_32F:
        return std::numeric_limits<float>::epsilon();
    case HIP_R_64F:
    case HIP_C_64F:
        return std::numeric_limits<double>::epsilon();
    case HIP_R_16F:
        return std::numeric_limits<hipblasLtHalf>::epsilon();
    case HIP_R_32I:
        return std::numeric_limits<int32_t>::epsilon();
    default:
        hipblaslt_cerr << "Error type in sum_error_tolerance_for_compute_type" << std::endl;
        return 0.0;
    }
}
