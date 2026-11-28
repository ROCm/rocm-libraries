// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "StandalonePointwiseContractV2.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using namespace hkp_standalone_pointwise_v2;

__device__ float
    readOperand(const void* pointer, const Metadata& meta, std::size_t operand, int64_t index)
{
    if((meta.scalarMask & (1 << operand)) != 0)
    {
        return meta.scalars[operand];
    }
    const auto position = offset(meta, operand, index);
    switch(meta.types[operand])
    {
    case FLOAT:
        return static_cast<const float*>(pointer)[position];
    case HALF:
        return __half2float(static_cast<const __half*>(pointer)[position]);
    case BFLOAT16:
        return static_cast<float>(static_cast<const hip_bfloat16*>(pointer)[position]);
    case BOOLEAN:
        return static_cast<const bool*>(pointer)[position] ? 1.0f : 0.0f;
    default:
        return NAN;
    }
}

extern "C" __global__ void StandalonePointwiseV2(
    const void* a, const void* b, const void* mask, void* output, Metadata meta)
{
    const auto step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for(auto index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < meta.elements;
        index += step)
    {
        const float x = readOperand(a, meta, 0, index);
        const float y
            = b != nullptr || (meta.scalarMask & 2) != 0 ? readOperand(b, meta, 1, index) : 0.0f;
        const bool condition = mask != nullptr || (meta.scalarMask & 4) != 0
                                   ? readOperand(mask, meta, 2, index) != 0.0f
                                   : false;
        auto coordinate = index;
        for(auto axis = meta.dims.size() - 1; axis > static_cast<std::size_t>(meta.axis); --axis)
        {
            coordinate /= meta.dims[axis];
        }
        coordinate %= meta.dims[static_cast<std::size_t>(meta.axis)];
        const float value = meta.operation == 22 ? static_cast<float>(coordinate)
                                                 : evaluate(meta, x, y, condition);
        const auto position = offset(meta, 3, index);
        switch(meta.types[3])
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
        case INT32:
            static_cast<int32_t*>(output)[position] = static_cast<int32_t>(coordinate);
            break;
        }
    }
}
