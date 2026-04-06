// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference validator kernels - compiled at runtime via HipRTC.
// DATA_TYPE must be defined at compile time via -DDATA_TYPE=<type>.
// COMPUTE_TYPE must be defined for accumulation precision.
//
// Both reference and implementation tensors share the same element type.

#include "GpuRefTypes.h"
#include "GpuRefValidatorArgs.h"

using namespace gpu_ref;

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

    if(isnan(refVal) || isnan(implVal))
    {
        args.resultFlags[idx] = 0;
        return;
    }

    if(isinf(refVal) || isinf(implVal))
    {
        args.resultFlags[idx] = 0;
        return;
    }

    auto absRef = fabs(refVal);

    auto diff = implVal - refVal;
    auto absDiff = fabs(diff);
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
