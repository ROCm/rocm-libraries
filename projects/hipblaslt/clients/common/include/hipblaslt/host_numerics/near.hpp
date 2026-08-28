// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt tolerance policy.

#include "hipblaslt_ostream.hpp"

#include <hipblaslt/host_numerics/Types.hpp>
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

inline double gfx11_low_precision_accumulation_tolerance_coefficient(hipDataType computeType,
                                                                     size_t      reductionLength)
{
    // GFX11 matrix instructions may combine partial sums in a different order from the
    // sequential host reference. The symmetric comparison below accounts for cancellation and
    // result scaling. Native gfx1151 FP16 testing required a factor of 7; 8 is the next power of
    // two above that observed bound.
    //
    // The caller uses this coefficient with the host-numerics symmetric comparison:
    //   |gpu - reference| < tolerance * (|gpu| + |reference| + 1).
    constexpr double accumulationSafetyFactor = 8.0;
    return accumulationSafetyFactor * static_cast<double>(reductionLength)
           * sum_error_tolerance_for_compute_type(computeType);
}

inline constexpr double bfloat16_output_rounding_tolerance_coefficient()
{
    // Native gfx1151 BF16 checks across K=128, 256, and 512 differed from the sequential host
    // reference by one BF16 output step, with a maximum symmetric difference of 0.00377003.
    // Half a BF16 epsilon is 0.00390625 and bounds that difference.
    return 0x1p-8;
}
