// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference validator kernels - compiled at runtime via HipRTC.
// DATA_TYPE must be defined at compile time via -DDATA_TYPE=<type>.
// COMPUTE_TYPE must be defined for accumulation precision.
// LOCAL_SIZE must be defined as the launch block size (validateRms reduces per block).
//
// Both reference and implementation tensors share the same element type.
// validateAllClose/validateExact set a single atomic failureFlag (0 = all passed,
// 1 = any failed) and accept ragged (RFC 0014) layouts; validateRms accumulates the
// totals the host needs for the ratio and is dense-only.

#include "GpuRefTypes.h"
#include "GpuRefValidatorArgs.h"

using namespace gpu_ref;

// True when x is the value Tensor::fillWithSentinelValue() leaves in an unwritten integer
// output: the type's maximum. CpuIntReferenceValidation rejects it on either side, and so
// must validateExact or the two sites grade the same data differently. Floating-point
// types return false: their sentinel is NaN, which validateAllClose already rejects.
// Every kernel here is compiled for every DATA_TYPE, so this must compile for all of them.
template <typename T>
__device__ inline bool isIntegerSentinel(T x)
{
    if constexpr(__is_same(T, signed char))
    {
        return x == __INT8_MAX__;
    }
    else if constexpr(__is_same(T, unsigned char))
    {
        return x == __UINT8_MAX__;
    }
    else if constexpr(__is_same(T, int))
    {
        return x == __INT32_MAX__;
    }
    else
    {
        return false;
    }
}

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

// Ragged counterpart of decomposeAndComputeOffsets (RFC 0014). The batch coordinate picks
// a base from args.rowOffsets in place of a batch stride, as RaggedTensorBase::getIndexImpl
// does. Returns false for a position past its batch's stored sequence rows: it has no
// storage of its own (it aliases another batch or runs past the buffer) and must not be read.
__device__ inline bool decomposeRaggedAndComputeOffsets(long long linearIdx,
                                                        const ValidatorArgs& args,
                                                        long long& refOffset,
                                                        long long& implOffset)
{
    refOffset = 0;
    implOffset = 0;
    long long seqCoord = 0;
    for(int d = args.ndim - 1; d >= 1; --d)
    {
        long long coord = linearIdx % args.dims[d];
        linearIdx /= args.dims[d];
        if(d == args.seqAxis)
        {
            seqCoord = coord;
        }
        refOffset += coord * args.refStrides[d];
        implOffset += coord * args.implStrides[d];
    }

    // What remains of linearIdx is the batch coordinate.
    const long long base = args.rowOffsets[linearIdx];
    const long long seqExtent = (args.rowOffsets[linearIdx + 1] - base) / args.seqStride;
    if(seqCoord >= seqExtent)
    {
        return false;
    }
    refOffset += base;
    implOffset += base;
    return true;
}

// Maps a logical element index to each tensor's element offset for whichever layout args
// describe. Returns false when the index is a ragged out-of-block position to skip.
__device__ inline bool
    resolveOffsets(long long idx, const ValidatorArgs& args, long long& refIdx, long long& implIdx)
{
    refIdx = idx;
    implIdx = idx;
    if(args.rowOffsets != nullptr)
    {
        return decomposeRaggedAndComputeOffsets(idx, args, refIdx, implIdx);
    }
    if(args.ndim > 0)
    {
        decomposeAndComputeOffsets(
            idx, args.dims, args.refStrides, args.implStrides, args.ndim, refIdx, implIdx);
    }
    return true;
}

// Floating-point allClose validation kernel.
// For each element i: passes if |impl[i] - ref[i]| <= atol + rtol * |ref[i]|
// Fails on NaN or Inf in either tensor.
// Sets failureFlag to 1 atomically if any element fails.
// For a ragged layout, out-of-block positions and positions where the reference is NaN are
// skipped (a ragged reference marks the positions it leaves undefined with NaN, RFC 0014),
// and comparedFlag records that at least one element was graded, as CpuFpReferenceValidation
// does.
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

    long long refIdx;
    long long implIdx;
    if(!resolveOffsets(idx, args, refIdx, implIdx))
    {
        return;
    }

    auto refVal = toAccum(ref[refIdx]);
    auto implVal = toAccum(impl[implIdx]);

    if(args.rowOffsets != nullptr)
    {
        if(isnan(refVal))
        {
            return;
        }
        // Checked first so a large tensor does not pay one atomic per element.
        if(*args.comparedFlag == 0)
        {
            atomicMax(args.comparedFlag, 1);
        }
    }

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
// Sets failureFlag to 1 atomically if any element differs, or if either side holds the
// unwritten-output sentinel (see isIntegerSentinel). For a ragged layout, out-of-block
// positions are skipped, as CpuIntReferenceValidation does.
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

    long long refIdx;
    long long implIdx;
    if(!resolveOffsets(idx, args, refIdx, implIdx))
    {
        return;
    }

    const auto refVal = ref[refIdx];
    const auto implVal = impl[implIdx];
    if(isIntegerSentinel(refVal) || isIntegerSentinel(implVal) || refVal != implVal)
    {
        atomicMax(args.failureFlag, 1);
    }
}

// validateRms's tree reduction halves its stride from LOCAL_SIZE / 2, so it needs
// LOCAL_SIZE to be a power of two and to equal blockDim.x (the host launches with the
// same constant it passes as LOCAL_SIZE).
static_assert((LOCAL_SIZE & (LOCAL_SIZE - 1)) == 0, "LOCAL_SIZE must be a power of two");

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
