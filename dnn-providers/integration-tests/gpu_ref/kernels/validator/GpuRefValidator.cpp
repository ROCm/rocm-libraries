// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference validator kernels - compiled at runtime via HipRTC.
// DATA_TYPE must be defined at compile time via -DDATA_TYPE=<type>.
// COMPUTE_TYPE must be defined for accumulation precision.
//
// Both reference and implementation tensors share the same element type.
// Kernels set a single atomic failureFlag (0 = all passed, 1 = any failed).

#include "GpuRefTypes.h"
#include "GpuRefValidatorArgs.h"

using namespace gpu_ref;

// Floating-point allClose validation kernel.
// For each element i: passes if |impl[i] - ref[i]| <= atol + rtol * |ref[i]|
// Fails on NaN or Inf in either tensor.
// Sets failureFlag to 1 atomically if any element fails.
extern "C" __global__ void validateAllClose(ValidatorArgs args)
{
    auto idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= args.totalElements)
    {
        return;
    }

    if(*args.failureFlag != 0)
    {
        return;
    }

    const auto* ref = static_cast<const DATA_TYPE*>(args.reference);
    const auto* impl = static_cast<const DATA_TYPE*>(args.implementation);

    auto refVal = toAccum(ref[idx]);
    auto implVal = toAccum(impl[idx]);

    if(isnan(refVal) || isnan(implVal) || isinf(refVal) || isinf(implVal))
    {
        atomicMax(args.failureFlag, 1);
        return;
    }

    auto absDiff = fabs(implVal - refVal);
    auto threshold = static_cast<COMPUTE_TYPE>(args.absoluteTolerance)
                     + static_cast<COMPUTE_TYPE>(args.relativeTolerance) * fabs(refVal);

    if(absDiff > threshold)
    {
        atomicMax(args.failureFlag, 1);
    }
}

// Integer exact-equality validation kernel.
// Sets failureFlag to 1 atomically if any element differs.
extern "C" __global__ void validateExact(ValidatorArgs args)
{
    auto idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= args.totalElements)
    {
        return;
    }

    if(*args.failureFlag != 0)
    {
        return;
    }

    const auto* ref = static_cast<const DATA_TYPE*>(args.reference);
    const auto* impl = static_cast<const DATA_TYPE*>(args.implementation);

    if(ref[idx] != impl[idx])
    {
        atomicMax(args.failureFlag, 1);
    }
}
