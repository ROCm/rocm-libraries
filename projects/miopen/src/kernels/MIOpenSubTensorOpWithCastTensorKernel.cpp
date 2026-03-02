// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

#if MIOPEN_SRC_TYPE == 0
#define _FLOAT_SRC char
#elif MIOPEN_SRC_TYPE == 1
#define _FLOAT_SRC int
#elif MIOPEN_SRC_TYPE == 2
#define _FLOAT_SRC half
#ifdef __HIP_PLATFORM_AMD__
#define _FLOAT_SRC _Float16
#else
#define _FLOAT_SRC half
#endif
#elif MIOPEN_SRC_TYPE == 3
#define _FLOAT_SRC float
#else /* BFloat16 */
#define _FLOAT_SRC ushort
#endif

#if MIOPEN_DST_TYPE == 0
#define _FLOAT_DST char
#ifndef INT8_MAX
#define MAX_VAL 127 /* max value */
#else
#define MAX_VAL INT8_MAX
#endif
#elif MIOPEN_DST_TYPE == 1
#define _FLOAT_DST int
#ifndef INT32_MAX
#define MAX_VAL 2147483647 /* max value */
#else
#define MAX_VAL INT32_MAX
#endif
#elif MIOPEN_DST_TYPE == 2
#ifdef __HIP_PLATFORM_AMD__
#define _FLOAT_DST _Float16
#else
#define _FLOAT_DST half
#endif
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#elif MIOPEN_DST_TYPE == 3
#define _FLOAT_DST float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#else /* BFloat16 */
#define _FLOAT_DST ushort
#define MAX_VAL 0x7F7F /* max value */
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

__forceinline__ __device__ void cast_impl(const _FLOAT_SRC* __restrict__ src,
                                          const int& srcOffset,
                                          const unsigned int& sindex,
                                          _FLOAT_DST* __restrict__ dst,
                                          const int& dstOffset,
                                          const unsigned int& dindex,
                                          const float& alpha,
                                          [[maybe_unused]] const int& clamping)
{
    _FLOAT_SRC temp_src = *(src + sindex + srcOffset);
    if constexpr(MIOPEN_SRC_TYPE == 3 && MIOPEN_DST_TYPE == 4)
    {
        temp_src *= alpha;
        *(dst + dindex + dstOffset) = float_to_bfloat16(temp_src);
    }
    else
    {
        bool over_flow = (clamping != 0) && (alpha * ((float)temp_src)) >= ((float)MAX_VAL);
        *(dst + dindex + dstOffset) = (_FLOAT_DST)(over_flow ? MAX_VAL : alpha * ((float)temp_src));
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithCastTensor1d(const _FLOAT_SRC* __restrict__ src,
                                                               const float alpha,
                                                               const int clamping,
                                                               const int srcOffset,
                                                               const int srcStride0,
                                                               const int srcLen0,
                                                               _FLOAT_DST* __restrict__ dst,
                                                               const int dstOffset,
                                                               const int dstStride0)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    for(unsigned int did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        const unsigned int sindex = srcStride0 * did0;
        const unsigned int dindex = dstStride0 * did0;
        cast_impl(src, srcOffset, sindex, dst, dstOffset, dindex, alpha, clamping);
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithCastTensor2d(const _FLOAT_SRC* __restrict__ src,
                                                               const float alpha,
                                                               const int clamping,
                                                               const int srcOffset,
                                                               const int srcStride0,
                                                               const int srcStride1,
                                                               const int srcLen0,
                                                               const int srcLen1,
                                                               _FLOAT_DST* __restrict__ dst,
                                                               const int dstOffset,
                                                               const int dstStride0,
                                                               const int dstStride1)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const unsigned int did1_begin = itmp / work_stride_1;

    for(unsigned int did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(unsigned int did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            const unsigned int sindex = srcStride0 * did0 + srcStride1 * did1;
            const unsigned int dindex = dstStride0 * did0 + dstStride1 * did1;
            cast_impl(src, srcOffset, sindex, dst, dstOffset, dindex, alpha, clamping);
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithCastTensor3d(const _FLOAT_SRC* __restrict__ src,
                                                               const float alpha,
                                                               const int clamping,
                                                               const int srcOffset,
                                                               const int srcStride0,
                                                               const int srcStride1,
                                                               const int srcStride2,
                                                               const int srcLen0,
                                                               const int srcLen1,
                                                               const int srcLen2,
                                                               _FLOAT_DST* __restrict__ dst,
                                                               const int dstOffset,
                                                               const int dstStride0,
                                                               const int dstStride1,
                                                               const int dstStride2)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const unsigned int did1_begin = itmp / work_stride_1;

    itmp -= did1_begin * work_stride_1;

    const unsigned int did2_begin = itmp / work_stride_2;

    for(unsigned int did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(unsigned int did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            for(unsigned int did2 = did2_begin; did2 < srcLen2; did2 += WORK_LENGTH_2)
            {
                const unsigned int sindex =
                    srcStride0 * did0 + srcStride1 * did1 + srcStride2 * did2;
                const unsigned int dindex =
                    dstStride0 * did0 + dstStride1 * did1 + dstStride2 * did2;

                cast_impl(src, srcOffset, sindex, dst, dstOffset, dindex, alpha, clamping);
            }
        }
    }
}
extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithCastTensor4d(const _FLOAT_SRC* __restrict__ src,
                                                               const float alpha,
                                                               const int clamping,
                                                               const int srcOffset,
                                                               const int srcStride0,
                                                               const int srcStride1,
                                                               const int srcStride2,
                                                               const int srcStride3,
                                                               const int srcLen0,
                                                               const int srcLen1,
                                                               const int srcLen2,
                                                               const int srcLen3,
                                                               _FLOAT_DST* __restrict__ dst,
                                                               const int dstOffset,
                                                               const int dstStride0,
                                                               const int dstStride1,
                                                               const int dstStride2,
                                                               const int dstStride3)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const unsigned int did1_begin = itmp / work_stride_1;

    itmp -= did1_begin * work_stride_1;

    const unsigned int did2_begin = itmp / work_stride_2;

    itmp -= did2_begin * work_stride_2;

    const unsigned int did3_begin = itmp / work_stride_3;

    for(unsigned int did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(unsigned int did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            for(unsigned int did2 = did2_begin; did2 < srcLen2; did2 += WORK_LENGTH_2)
            {
                for(unsigned int did3 = did3_begin; did3 < srcLen3; did3 += WORK_LENGTH_3)
                {
                    const unsigned int sindex = srcStride0 * did0 + srcStride1 * did1 +
                                                srcStride2 * did2 + srcStride3 * did3;
                    const unsigned int dindex = dstStride0 * did0 + dstStride1 * did1 +
                                                dstStride2 * did2 + dstStride3 * did3;
                    cast_impl(src, srcOffset, sindex, dst, dstOffset, dindex, alpha, clamping);
                }
            }
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithCastTensor5d(const _FLOAT_SRC* __restrict__ src,
                                                               const float alpha,
                                                               const int clamping,
                                                               const int srcOffset,
                                                               const int srcStride0,
                                                               const int srcStride1,
                                                               const int srcStride2,
                                                               const int srcStride3,
                                                               const int srcStride4,
                                                               const int srcLen0,
                                                               const int srcLen1,
                                                               const int srcLen2,
                                                               const int srcLen3,
                                                               const int srcLen4,
                                                               _FLOAT_DST* __restrict__ dst,
                                                               const int dstOffset,
                                                               const int dstStride0,
                                                               const int dstStride1,
                                                               const int dstStride2,
                                                               const int dstStride3,
                                                               const int dstStride4)
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

    for(unsigned int did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(unsigned int did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            for(unsigned int did2 = did2_begin; did2 < srcLen2; did2 += WORK_LENGTH_2)
            {
                for(unsigned int did3 = did3_begin; did3 < srcLen3; did3 += WORK_LENGTH_3)
                {
                    for(unsigned int did4 = did4_begin; did4 < srcLen4; did4 += WORK_LENGTH_4)
                    {
                        const unsigned int sindex = srcStride0 * did0 + srcStride1 * did1 +
                                                    srcStride2 * did2 + srcStride3 * did3 +
                                                    srcStride4 * did4;
                        const unsigned int dindex = dstStride0 * did0 + dstStride1 * did1 +
                                                    dstStride2 * did2 + dstStride3 * did3 +
                                                    dstStride4 * did4;
                        cast_impl(src, srcOffset, sindex, dst, dstOffset, dindex, alpha, clamping);
                    }
                }
            }
        }
    }
}
