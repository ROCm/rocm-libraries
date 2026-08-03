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

// Decompose a linear element index into strided offsets for two tensors.
// Computes multi-dimensional coordinates from linearIdx using dims (innermost-last),
// then dot-products with each tensor's strides to get physical offsets.
__device__ inline void decomposeAndComputeOffsets(long long linearIdx,
                                                  const long long* dims,
                                                  const long long* refStrides,
                                                  const long long* implStrides,
                                                  int ndim,
                                                  long long& refOffset,
                                                  long long& implOffset)
{
    refOffset = 0;
    implOffset = 0;
    for(int d = ndim - 1; d >= 0; --d)
    {
        long long coord = linearIdx % dims[d];
        linearIdx /= dims[d];
        refOffset += coord * refStrides[d];
        implOffset += coord * implStrides[d];
    }
}

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

    long long refIdx = idx;
    long long implIdx = idx;
    if(args.ndim > 0)
    {
        decomposeAndComputeOffsets(
            idx, args.dims, args.refStrides, args.implStrides, args.ndim, refIdx, implIdx);
    }

    auto refVal = toAccum(ref[refIdx]);
    auto implVal = toAccum(impl[implIdx]);

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

    long long refIdx = idx;
    long long implIdx = idx;
    if(args.ndim > 0)
    {
        decomposeAndComputeOffsets(
            idx, args.dims, args.refStrides, args.implStrides, args.ndim, refIdx, implIdx);
    }

    if(ref[refIdx] != impl[implIdx])
    {
        atomicMax(args.failureFlag, 1);
    }
}
