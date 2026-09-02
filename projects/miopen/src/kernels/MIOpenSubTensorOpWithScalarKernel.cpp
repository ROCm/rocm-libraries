// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "miopen_cstdint.hpp"

#ifndef MIOPEN_USE_64BIT_INDEX
#define MIOPEN_USE_64BIT_INDEX 0
#endif

#if MIOPEN_USE_64BIT_INDEX
using index_t = size_t;
#else
using index_t = unsigned int;
#endif

#if MIOPEN_USE_INT8 == 1
#define FLOAT char
#endif

#if MIOPEN_USE_INT32 == 1
#define FLOAT int
#endif

#ifndef WORK_LENGTH_0
#define WORK_LENGTH_0 1
#endif

#ifndef WORK_LENGTH_1
#define WORK_LENGTH_1 1
#endif

#ifndef WORK_LENGTH_2
#define WORK_LENGTH_2 1
#endif

#ifndef WORK_LENGTH_3
#define WORK_LENGTH_3 1
#endif

#ifndef WORK_LENGTH_4
#define WORK_LENGTH_4 1
#endif

constexpr index_t work_stride_4 = 1;
constexpr index_t work_stride_3 = WORK_LENGTH_4 * work_stride_4;
constexpr index_t work_stride_2 = WORK_LENGTH_3 * work_stride_3;
constexpr index_t work_stride_1 = WORK_LENGTH_2 * work_stride_2;
constexpr index_t work_stride_0 = WORK_LENGTH_1 * work_stride_1;

// SUBTENSOR_OP_WITH_SCALAR set to 0 for set operation, and 1 for multiply operation
constexpr bool is_set_op = (SUBTENSOR_OP_WITH_SCALAR == 0);

template <bool set_op>
struct subtensor;

template <>
struct subtensor<true>
{
    static __forceinline__ __device__ void scalar_op(FLOAT& t, const FLOAT& a) { t = a; }
};

template <>
struct subtensor<false>
{
    static __forceinline__ __device__ void scalar_op(FLOAT& t, const FLOAT& a) { t *= a; }
};

extern "C" __global__ __launch_bounds__(LOCAL_SIZE) void SubTensorOpWithScalar1d(
    FLOAT* dst, const FLOAT alpha, const index_t offset, const index_t stride0, const index_t len0)
{
    index_t itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const index_t did0_begin = itmp / work_stride_0;

    for(index_t did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        const index_t i = stride0 * did0;
        subtensor<is_set_op>::scalar_op(dst[i + offset], alpha);
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithScalar2d(FLOAT* dst,
                                                           const FLOAT alpha,
                                                           const index_t offset,
                                                           const index_t stride0,
                                                           const index_t stride1,
                                                           const index_t len0,
                                                           const index_t len1)
{
    index_t itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const index_t did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const index_t did1_begin = itmp / work_stride_1;

    for(index_t did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(index_t did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            const index_t i = stride0 * did0 + stride1 * did1;
            subtensor<is_set_op>::scalar_op(dst[i + offset], alpha);
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithScalar3d(FLOAT* dst,
                                                           const FLOAT alpha,
                                                           const index_t offset,
                                                           const index_t stride0,
                                                           const index_t stride1,
                                                           const index_t stride2,
                                                           const index_t len0,
                                                           const index_t len1,
                                                           const index_t len2)
{
    index_t itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const index_t did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const index_t did1_begin = itmp / work_stride_1;

    itmp -= did1_begin * work_stride_1;

    const index_t did2_begin = itmp / work_stride_2;

    for(index_t did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(index_t did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            for(index_t did2 = did2_begin; did2 < len2; did2 += WORK_LENGTH_2)
            {
                const index_t i = stride0 * did0 + stride1 * did1 + stride2 * did2;
                subtensor<is_set_op>::scalar_op(dst[i + offset], alpha);
            }
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithScalar4d(FLOAT* dst,
                                                           const FLOAT alpha,
                                                           const index_t offset,
                                                           const index_t stride0,
                                                           const index_t stride1,
                                                           const index_t stride2,
                                                           const index_t stride3,
                                                           const index_t len0,
                                                           const index_t len1,
                                                           const index_t len2,
                                                           const index_t len3)
{
    index_t itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const index_t did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const index_t did1_begin = itmp / work_stride_1;

    itmp -= did1_begin * work_stride_1;

    const index_t did2_begin = itmp / work_stride_2;

    itmp -= did2_begin * work_stride_2;

    const index_t did3_begin = itmp / work_stride_3;

    for(index_t did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(index_t did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            for(index_t did2 = did2_begin; did2 < len2; did2 += WORK_LENGTH_2)
            {
                for(index_t did3 = did3_begin; did3 < len3; did3 += WORK_LENGTH_3)
                {
                    const index_t i =
                        stride0 * did0 + stride1 * did1 + stride2 * did2 + stride3 * did3;
                    subtensor<is_set_op>::scalar_op(dst[i + offset], alpha);
                }
            }
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithScalar5d(FLOAT* dst,
                                                           const FLOAT alpha,
                                                           const index_t offset,
                                                           const index_t stride0,
                                                           const index_t stride1,
                                                           const index_t stride2,
                                                           const index_t stride3,
                                                           const index_t stride4,
                                                           const index_t len0,
                                                           const index_t len1,
                                                           const index_t len2,
                                                           const index_t len3,
                                                           const index_t len4)
{
    index_t itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const index_t did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const index_t did1_begin = itmp / work_stride_1;

    itmp -= did1_begin * work_stride_1;

    const index_t did2_begin = itmp / work_stride_2;

    itmp -= did2_begin * work_stride_2;

    const index_t did3_begin = itmp / work_stride_3;

    itmp -= did3_begin * work_stride_3;

    const index_t did4_begin = itmp / work_stride_4;

    for(index_t did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(index_t did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            for(index_t did2 = did2_begin; did2 < len2; did2 += WORK_LENGTH_2)
            {
                for(index_t did3 = did3_begin; did3 < len3; did3 += WORK_LENGTH_3)
                {
                    for(index_t did4 = did4_begin; did4 < len4; did4 += WORK_LENGTH_4)
                    {
                        const index_t i = stride0 * did0 + stride1 * did1 + stride2 * did2 +
                                          stride3 * did3 + stride4 * did4;
                        subtensor<is_set_op>::scalar_op(dst[i + offset], alpha);
                    }
                }
            }
        }
    }
}
