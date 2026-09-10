// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference reduction kernel.
// Compiled via HipRTC with -DINPUT_TYPE=<type> -DOUTPUT_TYPE=<type> -DCOMPUTE_TYPE=<type>
// -DMODE=<mode> -DNUM_DIMS=<value> -DLOCAL_SIZE=<value>.
// Each thread block computes one output element, reducing over the relevant
// reduction dimensions of the input tensor in parallel across the threads.

#include "GpuRefTypes.h"

using namespace gpu_ref;

constexpr ReductionMode mode = static_cast<ReductionMode>(MODE);

template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<double>
{
    static constexpr double maxVal = 1.7976931348623157e+308;
    static constexpr double minVal = -1.7976931348623157e+308;
};

template <>
struct NumericLimits<float>
{
    static constexpr float maxVal = 3.402823466e+38f;
    static constexpr float minVal = -3.402823466e+38f;
};

template <>
struct NumericLimits<_Float16>
{
    static constexpr _Float16 maxVal = static_cast<_Float16>(65504.0);
    static constexpr _Float16 minVal = static_cast<_Float16>(-65504.0);
};

template <>
struct NumericLimits<__bf16>
{
    static constexpr __bf16 maxVal = static_cast<__bf16>(0x1.fep+127);
    static constexpr __bf16 minVal = static_cast<__bf16>(-0x1.fep+127);
};

__device__ inline COMPUTE_TYPE initAccumulator()
{
    if constexpr(mode == ReductionMode::MUL || mode == ReductionMode::MUL_NO_ZEROS)
    {
        return static_cast<COMPUTE_TYPE>(1);
    }
    else if constexpr(mode == ReductionMode::MIN_OP)
    {
        return NumericLimits<COMPUTE_TYPE>::maxVal;
    }
    else if constexpr(mode == ReductionMode::MAX_OP)
    {
        return NumericLimits<COMPUTE_TYPE>::minVal;
    }
    else
    {
        return static_cast<COMPUTE_TYPE>(0);
    }
}

__device__ inline void accumulate(COMPUTE_TYPE* acc, COMPUTE_TYPE val)
{
    if constexpr(mode == ReductionMode::ADD || mode == ReductionMode::AVG)
    {
        *acc = *acc + val;
    }
    else if constexpr(mode == ReductionMode::MUL)
    {
        *acc = *acc * val;
    }
    else if constexpr(mode == ReductionMode::MIN_OP)
    {
        *acc = (*acc < val) ? *acc : val;
    }
    else if constexpr(mode == ReductionMode::MAX_OP)
    {
        *acc = (*acc > val) ? *acc : val;
    }
    else if constexpr(mode == ReductionMode::AMAX)
    {
        COMPUTE_TYPE absVal = (val < static_cast<COMPUTE_TYPE>(0)) ? -val : val;
        *acc = (*acc > absVal) ? *acc : absVal;
    }
    else if constexpr(mode == ReductionMode::NORM1)
    {
        COMPUTE_TYPE absVal = (val < static_cast<COMPUTE_TYPE>(0)) ? -val : val;
        *acc = *acc + absVal;
    }
    else if constexpr(mode == ReductionMode::NORM2)
    {
        *acc = *acc + (val * val);
    }
    else if constexpr(mode == ReductionMode::MUL_NO_ZEROS)
    {
        if(val != static_cast<COMPUTE_TYPE>(0))
        {
            *acc = *acc * val;
        }
    }
}

__device__ inline void combine(COMPUTE_TYPE* acc, COMPUTE_TYPE val)
{
    if constexpr(mode == ReductionMode::ADD || mode == ReductionMode::AVG
                 || mode == ReductionMode::NORM1 || mode == ReductionMode::NORM2)
    {
        *acc = *acc + val;
    }
    else if constexpr(mode == ReductionMode::MUL || mode == ReductionMode::MUL_NO_ZEROS)
    {
        *acc = *acc * val;
    }
    else if constexpr(mode == ReductionMode::MIN_OP)
    {
        *acc = (*acc < val) ? *acc : val;
    }
    else if constexpr(mode == ReductionMode::MAX_OP || mode == ReductionMode::AMAX)
    {
        *acc = (*acc > val) ? *acc : val;
    }
}

__device__ inline COMPUTE_TYPE finalize(COMPUTE_TYPE acc, long long count)
{
    if constexpr(mode == ReductionMode::AVG)
    {
        return acc / static_cast<COMPUTE_TYPE>(count);
    }
    else if constexpr(mode == ReductionMode::NORM2)
    {
        return static_cast<COMPUTE_TYPE>(sqrt(static_cast<double>(acc)));
    }
    else
    {
        return acc;
    }
}

extern "C" __global__ void ReductionRef(ReductionArgs args)
{
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* output = static_cast<OUTPUT_TYPE*>(args.output);
    constexpr long long localSize = static_cast<long long>(LOCAL_SIZE);

    // Each block handles one output element
    // Threads in the block collaboratively reduce over the relevant
    // reduction dimensions of the input tensor
    const long long gid = static_cast<long long>(blockIdx.x);
    const long long lid = static_cast<long long>(threadIdx.x);

    constexpr long long numDims = static_cast<long long>(NUM_DIMS);
    const long long reductionRank = args.reductionRank;
    const long long reductionDomainSize = args.reductionDomainSize;

    // Decompose the linear block index into multi-dimensional indices
    // to determine the corresponding output indices for this block
    long long inputIndices[numDims];
    long long outputIndices[numDims];
    {
        long long remaining = gid;
        for(long long d = 0; d < numDims; ++d)
        {
            long long idx = remaining / args.outputLogicalStrides[d];
            remaining %= args.outputLogicalStrides[d];
            outputIndices[d] = idx;
            inputIndices[d] = idx; // Set default input index to output index
        }
    }

    __shared__ COMPUTE_TYPE ltmp[localSize];
    COMPUTE_TYPE pacc = initAccumulator();

    for(long long flatIdx = lid; flatIdx < reductionDomainSize; flatIdx += localSize)
    {
        // Decompose the flat index into multi-dimensional indices to set
        // the corresponding input indices for the reduction domain
        long long remaining = flatIdx;
        for(long long r = 0; r < reductionRank; ++r)
        {
            long long stride = args.reductionDomainStride[r];
            inputIndices[args.reductionDomainAxes[r]] = remaining / stride;
            remaining %= stride;
        }

        // Compute the linear offset into the input tensor based on the
        // input indices and strides
        long long inputOffset = 0;
        for(long long d = 0; d < numDims; ++d)
        {
            inputOffset += inputIndices[d] * args.inputStrides[d];
        }

        COMPUTE_TYPE val = toAccum(input[inputOffset]);
        accumulate(&pacc, val);
    }

    // Reduce the partial results from all threads in the block to compute
    // the final result for this output element
    ltmp[lid] = pacc;
    __syncthreads();
    for(long long i = localSize >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            combine(&ltmp[lid], ltmp[lid + i]);
        }
        __syncthreads();
    }

    // Thread 0 writes the final result to the output tensor
    if(lid == 0)
    {
        COMPUTE_TYPE result = finalize(ltmp[0], reductionDomainSize);

        long long outputOffset = 0;
        for(long long d = 0; d < numDims; ++d)
        {
            outputOffset += outputIndices[d] * args.outputStrides[d];
        }

        OUTPUT_TYPE* tag = nullptr;
        output[outputOffset] = fromAccum(result, tag);
    }
}
