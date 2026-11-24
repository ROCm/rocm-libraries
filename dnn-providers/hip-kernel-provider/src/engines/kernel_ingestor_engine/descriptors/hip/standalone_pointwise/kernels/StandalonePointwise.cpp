// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "StandalonePointwiseContract.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using namespace hkp_standalone_pointwise;

__device__ float
    readOperand(const void* pointer, const Metadata& meta, std::size_t operand, int64_t index)
{
    const auto position = offset(meta, operand, index);
    switch(meta.types[operand])
    {
    case FLOAT:
        return static_cast<const float*>(pointer)[position];
    case HALF:
        return __half2float(static_cast<const __half*>(pointer)[position]);
    case BFLOAT16:
        return static_cast<float>(static_cast<const hip_bfloat16*>(pointer)[position]);
    default:
        return NAN;
    }
}

extern "C" __global__ void
    StandalonePointwise(const void* a, const void* b, void* output, Metadata meta)
{
    const auto step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for(auto index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < meta.elements;
        index += step)
    {
        const float x = readOperand(a, meta, 0, index);
        const float y = b != nullptr ? readOperand(b, meta, 1, index) : 0.0f;
        const float value = evaluate(meta, x, y);
        const auto position = offset(meta, 2, index);
        switch(meta.types[2])
        {
        case FLOAT:
            static_cast<float*>(output)[position] = value;
            break;
        case HALF:
            static_cast<__half*>(output)[position] = __float2half_rn(value);
            break;
        case BFLOAT16:
            static_cast<hip_bfloat16*>(output)[position] = hip_bfloat16(value);
            break;
        case BOOLEAN:
            static_cast<bool*>(output)[position] = value != 0.0f;
            break;
        }
    }
}
