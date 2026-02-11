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
                                                              const unsigned int src_offset,
                                                              const unsigned int dst_offset,
                                                              const unsigned int src_stride0,
                                                              const unsigned int dst_stride0,
                                                              const unsigned int len0)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    for(unsigned int did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        unsigned int si = src_stride0 * did0 + src_offset;
        unsigned int di = dst_stride0 * did0 + dst_offset;
        subtensor<alpha_is_one, beta_is_zero>::transform(dst[di], src[si], alpha, beta);
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithTransform2d(FLOAT* __restrict__ src,
                                                              const FLOAT alpha,
                                                              FLOAT* __restrict__ dst,
                                                              const FLOAT beta,
                                                              const unsigned int src_offset,
                                                              const unsigned int dst_offset,
                                                              const unsigned int src_stride0,
                                                              const unsigned int src_stride1,
                                                              const unsigned int dst_stride0,
                                                              const unsigned int dst_stride1,
                                                              const unsigned int len0,
                                                              const unsigned int len1)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const unsigned int did1_begin = itmp / work_stride_1;

    for(unsigned int did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(unsigned int did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            unsigned int si = src_stride0 * did0 + src_stride1 * did1 + src_offset;
            unsigned int di = dst_stride0 * did0 + dst_stride1 * did1 + dst_offset;
            subtensor<alpha_is_one, beta_is_zero>::transform(dst[di], src[si], alpha, beta);
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithTransform3d(FLOAT* __restrict__ src,
                                                              const FLOAT alpha,
                                                              FLOAT* __restrict__ dst,
                                                              const FLOAT beta,
                                                              const unsigned int src_offset,
                                                              const unsigned int dst_offset,
                                                              const unsigned int src_stride0,
                                                              const unsigned int src_stride1,
                                                              const unsigned int src_stride2,
                                                              const unsigned int dst_stride0,
                                                              const unsigned int dst_stride1,
                                                              const unsigned int dst_stride2,
                                                              const unsigned int len0,
                                                              const unsigned int len1,
                                                              const unsigned int len2)
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
                unsigned int si =
                    src_stride0 * did0 + src_stride1 * did1 + src_stride2 * did2 + src_offset;
                unsigned int di =
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
                                                              const unsigned int src_offset,
                                                              const unsigned int dst_offset,
                                                              const unsigned int src_stride0,
                                                              const unsigned int src_stride1,
                                                              const unsigned int src_stride2,
                                                              const unsigned int src_stride3,
                                                              const unsigned int dst_stride0,
                                                              const unsigned int dst_stride1,
                                                              const unsigned int dst_stride2,
                                                              const unsigned int dst_stride3,
                                                              const unsigned int len0,
                                                              const unsigned int len1,
                                                              const unsigned int len2,
                                                              const unsigned int len3)
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
                    unsigned int si = src_stride0 * did0 + src_stride1 * did1 + src_stride2 * did2 +
                                      src_stride3 * did3 + src_offset;
                    unsigned int di = dst_stride0 * did0 + dst_stride1 * did1 + dst_stride2 * did2 +
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
                                                              const unsigned int src_offset,
                                                              const unsigned int dst_offset,
                                                              const unsigned int src_stride0,
                                                              const unsigned int src_stride1,
                                                              const unsigned int src_stride2,
                                                              const unsigned int src_stride3,
                                                              const unsigned int src_stride4,
                                                              const unsigned int dst_stride0,
                                                              const unsigned int dst_stride1,
                                                              const unsigned int dst_stride2,
                                                              const unsigned int dst_stride3,
                                                              const unsigned int dst_stride4,
                                                              const unsigned int len0,
                                                              const unsigned int len1,
                                                              const unsigned int len2,
                                                              const unsigned int len3,
                                                              const unsigned int len4)
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
                        unsigned int si = src_stride0 * did0 + src_stride1 * did1 +
                                          src_stride2 * did2 + src_stride3 * did3 +
                                          src_stride4 * did4 + src_offset;
                        unsigned int di = dst_stride0 * did0 + dst_stride1 * did1 +
                                          dst_stride2 * did2 + dst_stride3 * did3 +
                                          dst_stride4 * did4 + dst_offset;
                        subtensor<alpha_is_one, beta_is_zero>::transform(
                            dst[di], src[si], alpha, beta);
                    }
                }
            }
        }
    }
}
