// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference validator kernels - compiled at runtime via HipRTC.
// DATA_TYPE must be defined at compile time via -DDATA_TYPE=<type>.
// COMPUTE_TYPE must be defined for accumulation precision.
//
// Both reference and implementation tensors share the same element type.

#include "GpuRefTypes.h"
#include "GpuRefValidatorArgs.h"

// Floating-point allClose validation kernel.
// For each element i: passes if |impl[i] - ref[i]| <= atol + rtol * |ref[i]|
// Fails on NaN or Inf in either tensor.
// resultFlags[i]: 1 = pass, 0 = fail
extern "C" __global__ void validateAllClose(ValidatorArgs args)
{
    auto idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= args.totalElements)
    {
        return;
    }

    const auto* ref = static_cast<const DATA_TYPE*>(args.reference);
    const auto* impl = static_cast<const DATA_TYPE*>(args.implementation);

    auto refVal = toAccum(ref[idx]);
    auto implVal = toAccum(impl[idx]);

    // Check for NaN or Inf
    if(refVal != refVal || implVal != implVal) // NaN check
    {
        args.resultFlags[idx] = 0;
        return;
    }

    // Inf check: if either is very large (beyond representable range for double)
    auto absRef = refVal < static_cast<COMPUTE_TYPE>(0) ? -refVal : refVal;
    auto absImpl = implVal < static_cast<COMPUTE_TYPE>(0) ? -implVal : implVal;

    // Use a large threshold to detect infinity-like values
    auto infThreshold = static_cast<COMPUTE_TYPE>(1e300);
    if(absRef > infThreshold || absImpl > infThreshold)
    {
        args.resultFlags[idx] = 0;
        return;
    }

    auto diff = implVal - refVal;
    auto absDiff = diff < static_cast<COMPUTE_TYPE>(0) ? -diff : diff;
    auto threshold = static_cast<COMPUTE_TYPE>(args.absoluteTolerance)
                     + static_cast<COMPUTE_TYPE>(args.relativeTolerance) * absRef;

    args.resultFlags[idx] = (absDiff <= threshold) ? 1 : 0;
}

// Integer exact-equality validation kernel.
// resultFlags[i]: 1 = pass, 0 = fail (values must be exactly equal)
extern "C" __global__ void validateExact(ValidatorArgs args)
{
    auto idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= args.totalElements)
    {
        return;
    }

    const auto* ref = static_cast<const DATA_TYPE*>(args.reference);
    const auto* impl = static_cast<const DATA_TYPE*>(args.implementation);

    args.resultFlags[idx] = (ref[idx] == impl[idx]) ? 1 : 0;
}
