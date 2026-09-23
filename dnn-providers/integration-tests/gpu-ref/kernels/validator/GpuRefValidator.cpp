// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference validator kernels - compiled at runtime via HipRTC.
// DATA_TYPE must be defined at compile time via -DDATA_TYPE=<type>.
// COMPUTE_TYPE must be defined for accumulation precision.
// LOCAL_SIZE must be defined as the launch block size (validateRms reduces per block).
//
// Both reference and implementation tensors share the same element type.
// validateAllClose/validateExact set a single atomic failureFlag (0 = all passed,
// 1 = any failed); validateRms accumulates the totals the host needs for the ratio.

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

// Relative-RMS accumulation kernel: MIOpen's aggregate check, the device half of
// CpuFpReferenceMiopenRmsValidation. The host forms
//   sqrt(sum((ref - impl)^2)) / (sqrt(n) * max(max|ref|, max|impl|))
// from the totals this leaves in args.accumulators. Each block reduces its slice in
// shared memory and folds it in with one atomic per total, so a large tensor costs one
// atomic per block rather than per element. Any NaN/Inf fails the whole check, as on
// the host: an unwritten output element is still sentinel-filled.
extern "C" __global__ void validateRms(RmsValidatorArgs args)
{
    __shared__ double squareDifference[LOCAL_SIZE];
    __shared__ double maxRefMagnitude[LOCAL_SIZE];
    __shared__ double maxImplMagnitude[LOCAL_SIZE];

    const unsigned int lid = threadIdx.x;
    const auto idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    double diffSquared = 0.0;
    double absRef = 0.0;
    double absImpl = 0.0;

    // Out-of-range threads still take part in the reduction below, contributing zeros.
    if(idx < args.totalElements)
    {
        const auto* ref = static_cast<const DATA_TYPE*>(args.reference);
        const auto* impl = static_cast<const DATA_TYPE*>(args.implementation);

        long long refIdx = idx;
        long long implIdx = idx;
        if(args.ndim > 0)
        {
            decomposeAndComputeOffsets(
                idx, args.dims, args.refStrides, args.implStrides, args.ndim, refIdx, implIdx);
        }

        const auto refVal = static_cast<double>(toAccum(ref[refIdx]));
        const auto implVal = static_cast<double>(toAccum(impl[implIdx]));

        if(isnan(refVal) || isnan(implVal) || isinf(refVal) || isinf(implVal))
        {
            atomicMax(&args.accumulators->nanOrInf, 1);
        }
        else
        {
            const double diff = refVal - implVal;
            diffSquared = diff * diff;
            absRef = fabs(refVal);
            absImpl = fabs(implVal);
        }
    }

    squareDifference[lid] = diffSquared;
    maxRefMagnitude[lid] = absRef;
    maxImplMagnitude[lid] = absImpl;
    __syncthreads();

    for(unsigned int stride = LOCAL_SIZE >> 1; stride > 0; stride >>= 1)
    {
        if(lid < stride)
        {
            squareDifference[lid] += squareDifference[lid + stride];
            maxRefMagnitude[lid] = fmax(maxRefMagnitude[lid], maxRefMagnitude[lid + stride]);
            maxImplMagnitude[lid] = fmax(maxImplMagnitude[lid], maxImplMagnitude[lid + stride]);
        }
        __syncthreads();
    }

    if(lid == 0)
    {
        atomicAdd(&args.accumulators->squareDifference, squareDifference[0]);
        atomicMax(&args.accumulators->maxRefMagnitudeBits,
                  static_cast<unsigned long long>(__double_as_longlong(maxRefMagnitude[0])));
        atomicMax(&args.accumulators->maxImplMagnitudeBits,
                  static_cast<unsigned long long>(__double_as_longlong(maxImplMagnitude[0])));
    }
}
