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

inline double gfx11_low_precision_accumulation_tolerance_coefficient(hipDataType computeType,
                                                                     size_t      reductionLength)
{
    // GFX11 matrix instructions may combine partial sums in a different order from the
    // sequential host reference. With FP16/BF16 inputs, cancellation can therefore leave a few
    // compute-precision ULPs on the GPU even when the reference result is exactly zero. A pure
    // K * epsilon absolute bound failed the native gfx1151 smoke matrix by as much as 7x, and it
    // does not scale when alpha or an epilogue scale increases the result.
    //
    // The caller uses this coefficient with the host-validation symmetric comparison:
    //   |gpu - reference| < tolerance * (|gpu| + |reference| + 1).
    // The +1 supplies the required absolute floor around cancellation, while the magnitude terms
    // scale the allowance for nonzero results. Eight is the next power of two above the observed
    // 7 * K * epsilon worst case and remains over 10,000x tighter for FP16, and over 100,000x
    // tighter for BF16, than the former K * 0.01 / K * 0.1 allowances.
    constexpr double accumulationSafetyFactor = 8.0;
    return accumulationSafetyFactor * static_cast<double>(reductionLength)
           * sum_error_tolerance_for_compute_type(computeType);
}
