// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <math.h>
#include <stdint.h>

#ifdef __HIPCC__
#define HKP_POINTWISE_DEVICE __host__ __device__
#else
#define HKP_POINTWISE_DEVICE
#endif

namespace hkp_standalone_pointwise_v2
{

enum Storage : int32_t
{
    Float = 0,
    Half = 1,
    Bfloat16 = 2,
    Boolean = 3,
    Int32 = 4
};

// Fixed-size launch metadata, not compile-time specialization: every rank, shape,
// stride, mode, storage combination and scalar shares one code object per arch.
struct Metadata
{
    int64_t elements = 0;
    int64_t dims[5] = {};
    int64_t strides[4][5] = {};
    int32_t types[4] = {};
    int32_t scalarMask = 0;
    int32_t operation = 0;
    float lower = 0;
    float upper = 0;
    float slope = 0;
    float beta = 1;
    float scalars[3] = {};
    int32_t reluParameterized = 0;
    float alpha = 1;
    int32_t axis = 0;
};

static_assert(sizeof(Metadata) == 272, "standalone pointwise launch ABI");

HKP_POINTWISE_DEVICE inline int64_t offset(const Metadata& meta, int operand, int64_t linear)
{
    int64_t result = 0;
    for(int axis = 4; axis >= 0; --axis)
    {
        result += (linear % meta.dims[axis]) * meta.strides[operand][axis];
        linear /= meta.dims[axis];
    }
    return result;
}

// Mode numbers are the stable FlatBuffers PointwiseMode values. The native matcher
// restricts the catalog to the CPU-reference-defined subset below.
HKP_POINTWISE_DEVICE inline float evaluate(const Metadata& meta, float a, float b, bool mask)
{
    switch(meta.operation)
    {
    case 1:
        return fabsf(a);
    case 2:
        return a + b;
    case 4:
        return mask ? a : b;
    case 8:
        return a > b ? 1.0f : 0.0f;
    case 13:
        return a * (b > 0.0f ? 1.0f : meta.alpha * expf(b));
    case 14:
        return a > 0.0f ? a : meta.alpha * expm1f(a);
    case 19:
    {
        const float inner = sqrtf(2.0f / 3.14159265f) * (a + 0.044715f * a * a * a);
        return 0.5f * a * (1.0f + tanhf(inner));
    }
    case 21:
        return 0.5f * a * (1.0f + erff(a / sqrtf(2.0f)));
    case 23:
        return a;
    case 28:
        return a < b ? b : a;
    case 29:
        return b < a ? b : a;
    case 30:
        return a * b;
    case 31:
        return -a;
    case 33:
        return a
               * (meta.reluParameterized == 0
                      ? (b > 0.0f ? 1.0f : 0.0f)
                      : (b <= meta.lower ? meta.slope : (b > meta.upper ? 0.0f : 1.0f)));
    case 34:
        return a <= meta.lower ? meta.slope * (a - meta.lower) + meta.lower
                               : (a >= meta.upper ? meta.upper : a);
    case 36:
    {
        const float sigmoid = 1.0f / (1.0f + expf(-b));
        return a * (sigmoid * (1.0f - sigmoid));
    }
    case 37:
        return 1.0f / (1.0f + expf(-a));
    case 39:
    {
        const auto value = meta.beta * b;
        const auto sigmoid
            = value >= 0.0f ? 1.0f / (1.0f + expf(-value)) : expf(value) / (1.0f + expf(value));
        return a * sigmoid;
    }
    case 40:
    {
        const auto value = meta.beta * a;
        return (value > 0.0f ? value + log1pf(expf(-value)) : log1pf(expf(value))) / meta.beta;
    }
    case 42:
        return a - b;
    case 44:
        return a / (1.0f + expf(-meta.beta * a));
    case 43:
    {
        const auto sigmoid = 1.0f / (1.0f + expf(-meta.beta * b));
        return a * (sigmoid + meta.beta * b * sigmoid * (1.0f - sigmoid));
    }
    case 46:
    {
        const float value = tanhf(b);
        return a * (1.0f - value * value);
    }
    case 47:
        return tanhf(a);
    default:
        return NAN;
    }
}

} // namespace hkp_standalone_pointwise_v2

#undef HKP_POINTWISE_DEVICE
