// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference reduction kernel.
// Compiled via HipRTC with -DINPUT_TYPE=<type> -DOUTPUT_TYPE=<type> -DCOMPUTE_TYPE=<type>
// -DREDUCTION_MODE=<mode> -DTOTAL_RANK=<value> -DLOCAL_SIZE=<value>.
// Each thread block computes one output element, reducing over the relevant
// reduction dimensions of the input tensor in parallel across the threads.

#include "GpuRefTypes.h"

using namespace gpu_ref;

enum class ReductionMode : int
{
    ADD = 1,
    MUL = 2,
    MIN_OP = 3,
    MAX_OP = 4,
    AMAX = 5,
    AVG = 6,
    NORM1 = 7,
    NORM2 = 8,
    MUL_NO_ZEROS = 9
};

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

__device__ inline COMPUTE_TYPE initAccumulator(ReductionMode mode)
{
    switch(mode)
    {
    case ReductionMode::MUL:
    case ReductionMode::MUL_NO_ZEROS:
        return static_cast<COMPUTE_TYPE>(1);
    case ReductionMode::MIN_OP:
        return NumericLimits<COMPUTE_TYPE>::maxVal;
    case ReductionMode::MAX_OP:
        return NumericLimits<COMPUTE_TYPE>::minVal;
    case ReductionMode::ADD:
    case ReductionMode::AVG:
    case ReductionMode::AMAX:
    case ReductionMode::NORM1:
    case ReductionMode::NORM2:
    default:
        return static_cast<COMPUTE_TYPE>(0);
    }
}

__device__ inline void accumulate(COMPUTE_TYPE* acc, COMPUTE_TYPE val, ReductionMode mode)
{
    switch(mode)
    {
    case ReductionMode::ADD:
    case ReductionMode::AVG:
        *acc = *acc + val;
        break;
    case ReductionMode::MUL:
        *acc = *acc * val;
        break;
    case ReductionMode::MIN_OP:
        *acc = (*acc < val) ? *acc : val;
        break;
    case ReductionMode::MAX_OP:
        *acc = (*acc > val) ? *acc : val;
        break;
    case ReductionMode::AMAX:
    {
        COMPUTE_TYPE absVal = (val < static_cast<COMPUTE_TYPE>(0)) ? -val : val;
        *acc = (*acc > absVal) ? *acc : absVal;
        break;
    }
    case ReductionMode::NORM1:
    {
        COMPUTE_TYPE absVal = (val < static_cast<COMPUTE_TYPE>(0)) ? -val : val;
        *acc = *acc + absVal;
        break;
    }
    case ReductionMode::NORM2:
        *acc = *acc + (val * val);
        break;
    case ReductionMode::MUL_NO_ZEROS:
        if(val != static_cast<COMPUTE_TYPE>(0))
        {
            *acc = *acc * val;
        }
        break;
    default:
        break;
    }
}

__device__ inline void combine(COMPUTE_TYPE* acc, COMPUTE_TYPE val, ReductionMode mode)
{
    switch(mode)
    {
    case ReductionMode::ADD:
    case ReductionMode::AVG:
    case ReductionMode::NORM1:
    case ReductionMode::NORM2:
        *acc = *acc + val;
        break;
    case ReductionMode::MUL:
    case ReductionMode::MUL_NO_ZEROS:
        *acc = *acc * val;
        break;
    case ReductionMode::MIN_OP:
        *acc = (*acc < val) ? *acc : val;
        break;
    case ReductionMode::MAX_OP:
    case ReductionMode::AMAX:
        *acc = (*acc > val) ? *acc : val;
        break;
    default:
        break;
    }
}

__device__ inline COMPUTE_TYPE finalize(COMPUTE_TYPE acc, long long count, ReductionMode mode)
{
    switch(mode)
    {
    case ReductionMode::AVG:
        return acc / static_cast<COMPUTE_TYPE>(count);
    case ReductionMode::NORM2:
        return static_cast<COMPUTE_TYPE>(sqrt(static_cast<double>(acc)));
    default:
        return acc;
    }
}

extern "C" __global__ void ReductionRef(ReductionArgs args)
{
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* output = static_cast<OUTPUT_TYPE*>(args.output);
    const ReductionMode reductionMode = static_cast<ReductionMode>(REDUCTION_MODE);
    constexpr long long localSize = static_cast<long long>(LOCAL_SIZE);

    // Each block handles one output element
    // Threads in the block collaboratively reduce over the relevant
    // reduction dimensions of the input tensor
    const long long gid = static_cast<long long>(blockIdx.x);
    const long long lid = static_cast<long long>(threadIdx.x);

    constexpr long long totalRank = static_cast<long long>(TOTAL_RANK);
    const long long reductionRank = args.reductionRank;
    const long long reductionDomainSize = args.reductionDomainSize;

    // Decompose the linear block index into multi-dimensional indices
    // to determine the corresponding output indices for this block
    long long inputIndices[totalRank];
    long long outputIndices[totalRank];
    {
        long long remaining = gid;
        for(long long d = 0; d < totalRank; ++d)
        {
            long long idx = remaining / args.outputShapeStrides[d];
            remaining %= args.outputShapeStrides[d];
            outputIndices[d] = idx;
            inputIndices[d] = idx; // Set default input index to output index
        }
    }

    __shared__ COMPUTE_TYPE ltmp[localSize];
    COMPUTE_TYPE pacc = initAccumulator(reductionMode);

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
        for(long long d = 0; d < totalRank; ++d)
        {
            inputOffset += inputIndices[d] * args.inputStrides[d];
        }

        COMPUTE_TYPE val = toAccum(input[inputOffset]);
        accumulate(&pacc, val, reductionMode);
    }

    // Reduce the partial results from all threads in the block to compute
    // the final result for this output element
    ltmp[lid] = pacc;
    __syncthreads();
    for(long long i = localSize >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            combine(&ltmp[lid], ltmp[lid + i], reductionMode);
        }
        __syncthreads();
    }

    // Thread 0 writes the final result to the output tensor
    if(lid == 0)
    {
        COMPUTE_TYPE result = finalize(ltmp[0], reductionDomainSize, reductionMode);

        long long outputOffset = 0;
        for(long long d = 0; d < totalRank; ++d)
        {
            outputOffset += outputIndices[d] * args.outputStrides[d];
        }

        OUTPUT_TYPE* tag = nullptr;
        output[outputOffset] = fromAccum(result, tag);
    }
}
