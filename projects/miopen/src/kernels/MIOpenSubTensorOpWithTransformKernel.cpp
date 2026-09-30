// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

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

constexpr bool beta_is_zero = (MIOPEN_BETA_IS_ZERO == 1);
constexpr bool alpha_is_one = (MIOPEN_ALPHA_IS_ONE == 1);

template <bool alpha_one, bool beta_zero>
struct subtensor;

template <>
struct subtensor<true, true>
{
    static __forceinline__ __device__ void
    transform(FLOAT& dst, const FLOAT& src, const FLOAT&, const FLOAT)
    {
        dst = src;
    }
};

template <>
struct subtensor<false, true>
{
    static __forceinline__ __device__ void
    transform(FLOAT& dst, const FLOAT& src, const FLOAT& alpha, const FLOAT&)
    {
        dst = src * alpha;
    }
};

template <>
struct subtensor<true, false>
{
    static __forceinline__ __device__ void
    transform(FLOAT& dst, const FLOAT& src, const FLOAT&, const FLOAT& beta)
    {
        dst = fma(dst, beta, src);
    }
};

template <>
struct subtensor<false, false>
{
    static __forceinline__ __device__ void
    transform(FLOAT& dst, const FLOAT& src, const FLOAT& alpha, const FLOAT& beta)
    {
        dst = fma(src, alpha, dst * beta);
    }
};

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithTransform1d(FLOAT* __restrict__ src,
                                                              const FLOAT alpha,
                                                              FLOAT* __restrict__ dst,
                                                              const FLOAT beta,
                                                              const index_t src_offset,
                                                              const index_t dst_offset,
                                                              const index_t src_stride0,
                                                              const index_t dst_stride0,
                                                              const index_t len0)
{
    index_t itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const index_t did0_begin = itmp / work_stride_0;

    for(index_t did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        index_t si = src_stride0 * did0 + src_offset;
        index_t di = dst_stride0 * did0 + dst_offset;
        subtensor<alpha_is_one, beta_is_zero>::transform(dst[di], src[si], alpha, beta);
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithTransform2d(FLOAT* __restrict__ src,
                                                              const FLOAT alpha,
                                                              FLOAT* __restrict__ dst,
                                                              const FLOAT beta,
                                                              const index_t src_offset,
                                                              const index_t dst_offset,
                                                              const index_t src_stride0,
                                                              const index_t src_stride1,
                                                              const index_t dst_stride0,
                                                              const index_t dst_stride1,
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
            index_t si = src_stride0 * did0 + src_stride1 * did1 + src_offset;
            index_t di = dst_stride0 * did0 + dst_stride1 * did1 + dst_offset;
            subtensor<alpha_is_one, beta_is_zero>::transform(dst[di], src[si], alpha, beta);
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithTransform3d(FLOAT* __restrict__ src,
                                                              const FLOAT alpha,
                                                              FLOAT* __restrict__ dst,
                                                              const FLOAT beta,
                                                              const index_t src_offset,
                                                              const index_t dst_offset,
                                                              const index_t src_stride0,
                                                              const index_t src_stride1,
                                                              const index_t src_stride2,
                                                              const index_t dst_stride0,
                                                              const index_t dst_stride1,
                                                              const index_t dst_stride2,
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
                index_t si =
                    src_stride0 * did0 + src_stride1 * did1 + src_stride2 * did2 + src_offset;
                index_t di =
                    dst_stride0 * did0 + dst_stride1 * did1 + dst_stride2 * did2 + dst_offset;
                subtensor<alpha_is_one, beta_is_zero>::transform(dst[di], src[si], alpha, beta);
            }
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithTransform4d(FLOAT* __restrict__ src,
                                                              const FLOAT alpha,
                                                              FLOAT* __restrict__ dst,
                                                              const FLOAT beta,
                                                              const index_t src_offset,
                                                              const index_t dst_offset,
                                                              const index_t src_stride0,
                                                              const index_t src_stride1,
                                                              const index_t src_stride2,
                                                              const index_t src_stride3,
                                                              const index_t dst_stride0,
                                                              const index_t dst_stride1,
                                                              const index_t dst_stride2,
                                                              const index_t dst_stride3,
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
                    index_t si = src_stride0 * did0 + src_stride1 * did1 + src_stride2 * did2 +
                                 src_stride3 * did3 + src_offset;
                    index_t di = dst_stride0 * did0 + dst_stride1 * did1 + dst_stride2 * did2 +
                                 dst_stride3 * did3 + dst_offset;
                    subtensor<alpha_is_one, beta_is_zero>::transform(dst[di], src[si], alpha, beta);
                }
            }
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithTransform5d(FLOAT* __restrict__ src,
                                                              const FLOAT alpha,
                                                              FLOAT* __restrict__ dst,
                                                              const FLOAT beta,
                                                              const index_t src_offset,
                                                              const index_t dst_offset,
                                                              const index_t src_stride0,
                                                              const index_t src_stride1,
                                                              const index_t src_stride2,
                                                              const index_t src_stride3,
                                                              const index_t src_stride4,
                                                              const index_t dst_stride0,
                                                              const index_t dst_stride1,
                                                              const index_t dst_stride2,
                                                              const index_t dst_stride3,
                                                              const index_t dst_stride4,
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
                        index_t si = src_stride0 * did0 + src_stride1 * did1 + src_stride2 * did2 +
                                     src_stride3 * did3 + src_stride4 * did4 + src_offset;
                        index_t di = dst_stride0 * did0 + dst_stride1 * did1 + dst_stride2 * did2 +
                                     dst_stride3 * did3 + dst_stride4 * did4 + dst_offset;
                        subtensor<alpha_is_one, beta_is_zero>::transform(
                            dst[di], src[si], alpha, beta);
                    }
                }
            }
        }
    }
}
