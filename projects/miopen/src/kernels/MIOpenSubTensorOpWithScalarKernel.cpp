// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

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

constexpr unsigned int work_stride_4 = 1;
constexpr unsigned int work_stride_3 = WORK_LENGTH_4 * work_stride_4;
constexpr unsigned int work_stride_2 = WORK_LENGTH_3 * work_stride_3;
constexpr unsigned int work_stride_1 = WORK_LENGTH_2 * work_stride_2;
constexpr unsigned int work_stride_0 = WORK_LENGTH_1 * work_stride_1;

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
    FLOAT* dst, const FLOAT alpha, const int offset, const int stride0, const int len0)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    for(unsigned int did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        const unsigned int i = stride0 * did0;
        subtensor<is_set_op>::scalar_op(dst[i + offset], alpha);
    }
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE) void SubTensorOpWithScalar2d(FLOAT* dst,
                                                                                 const FLOAT alpha,
                                                                                 const int offset,
                                                                                 const int stride0,
                                                                                 const int stride1,
                                                                                 const int len0,
                                                                                 const int len1)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const unsigned int did1_begin = itmp / work_stride_1;

    for(unsigned int did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(unsigned int did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            const unsigned int i = stride0 * did0 + stride1 * did1;
            subtensor<is_set_op>::scalar_op(dst[i + offset], alpha);
        }
    }
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE) void SubTensorOpWithScalar3d(FLOAT* dst,
                                                                                 const FLOAT alpha,
                                                                                 const int offset,
                                                                                 const int stride0,
                                                                                 const int stride1,
                                                                                 const int stride2,
                                                                                 const int len0,
                                                                                 const int len1,
                                                                                 const int len2)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const unsigned int did1_begin = itmp / work_stride_1;

    itmp -= did1_begin * work_stride_1;

    const unsigned int did2_begin = itmp / work_stride_2;

    for(unsigned int did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(unsigned int did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            for(unsigned int did2 = did2_begin; did2 < len2; did2 += WORK_LENGTH_2)
            {
                const unsigned int i = stride0 * did0 + stride1 * did1 + stride2 * did2;
                subtensor<is_set_op>::scalar_op(dst[i + offset], alpha);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE) void SubTensorOpWithScalar4d(FLOAT* dst,
                                                                                 const FLOAT alpha,
                                                                                 const int offset,
                                                                                 const int stride0,
                                                                                 const int stride1,
                                                                                 const int stride2,
                                                                                 const int stride3,
                                                                                 const int len0,
                                                                                 const int len1,
                                                                                 const int len2,
                                                                                 const int len3)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const unsigned int did1_begin = itmp / work_stride_1;

    itmp -= did1_begin * work_stride_1;

    const unsigned int did2_begin = itmp / work_stride_2;

    itmp -= did2_begin * work_stride_2;

    const unsigned int did3_begin = itmp / work_stride_3;

    for(unsigned int did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(unsigned int did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            for(unsigned int did2 = did2_begin; did2 < len2; did2 += WORK_LENGTH_2)
            {
                for(unsigned int did3 = did3_begin; did3 < len3; did3 += WORK_LENGTH_3)
                {
                    const unsigned int i =
                        stride0 * did0 + stride1 * did1 + stride2 * did2 + stride3 * did3;
                    subtensor<is_set_op>::scalar_op(dst[i + offset], alpha);
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE) void SubTensorOpWithScalar5d(FLOAT* dst,
                                                                                 const FLOAT alpha,
                                                                                 const int offset,
                                                                                 const int stride0,
                                                                                 const int stride1,
                                                                                 const int stride2,
                                                                                 const int stride3,
                                                                                 const int stride4,
                                                                                 const int len0,
                                                                                 const int len1,
                                                                                 const int len2,
                                                                                 const int len3,
                                                                                 const int len4)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const unsigned int did1_begin = itmp / work_stride_1;

    itmp -= did1_begin * work_stride_1;

    const unsigned int did2_begin = itmp / work_stride_2;

    itmp -= did2_begin * work_stride_2;

    const unsigned int did3_begin = itmp / work_stride_3;

    itmp -= did3_begin * work_stride_3;

    const unsigned int did4_begin = itmp / work_stride_4;

    for(unsigned int did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(unsigned int did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            for(unsigned int did2 = did2_begin; did2 < len2; did2 += WORK_LENGTH_2)
            {
                for(unsigned int did3 = did3_begin; did3 < len3; did3 += WORK_LENGTH_3)
                {
                    for(unsigned int did4 = did4_begin; did4 < len4; did4 += WORK_LENGTH_4)
                    {
                        const unsigned int i = stride0 * did0 + stride1 * did1 + stride2 * did2 +
                                               stride3 * did3 + stride4 * did4;
                        subtensor<is_set_op>::scalar_op(dst[i + offset], alpha);
                    }
                }
            }
        }
    }
}
