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

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithSubTensor1d(const FLOAT* __restrict__ src,
                                                              const index_t srcOffset,
                                                              const index_t srcStride0,
                                                              const index_t srcLen0,
                                                              FLOAT* __restrict__ dst,
                                                              const index_t dstOffset,
                                                              const index_t dstStride0)
{
    index_t itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const index_t did0_begin = itmp / work_stride_0;

    for(index_t did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        const index_t sindex    = srcStride0 * did0;
        const index_t dindex    = dstStride0 * did0;
        dst[dindex + dstOffset] = src[sindex + srcOffset];
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithSubTensor2d(const FLOAT* __restrict__ src,
                                                              const index_t srcOffset,
                                                              const index_t srcStride0,
                                                              const index_t srcStride1,
                                                              const index_t srcLen0,
                                                              const index_t srcLen1,
                                                              FLOAT* __restrict__ dst,
                                                              const index_t dstOffset,
                                                              const index_t dstStride0,
                                                              const index_t dstStride1)
{
    index_t itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const index_t did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const index_t did1_begin = itmp / work_stride_1;

    for(index_t did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(index_t did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            const index_t sindex    = srcStride0 * did0 + srcStride1 * did1;
            const index_t dindex    = dstStride0 * did0 + dstStride1 * did1;
            dst[dindex + dstOffset] = src[sindex + srcOffset];
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithSubTensor3d(const FLOAT* __restrict__ src,
                                                              const index_t srcOffset,
                                                              const index_t srcStride0,
                                                              const index_t srcStride1,
                                                              const index_t srcStride2,
                                                              const index_t srcLen0,
                                                              const index_t srcLen1,
                                                              const index_t srcLen2,
                                                              FLOAT* __restrict__ dst,
                                                              const index_t dstOffset,
                                                              const index_t dstStride0,
                                                              const index_t dstStride1,
                                                              const index_t dstStride2)
{
    index_t itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const index_t did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const index_t did1_begin = itmp / work_stride_1;

    itmp -= did1_begin * work_stride_1;

    const index_t did2_begin = itmp / work_stride_2;

    for(index_t did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(index_t did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            for(index_t did2 = did2_begin; did2 < srcLen2; did2 += WORK_LENGTH_2)
            {
                const index_t sindex    = srcStride0 * did0 + srcStride1 * did1 + srcStride2 * did2;
                const index_t dindex    = dstStride0 * did0 + dstStride1 * did1 + dstStride2 * did2;
                dst[dindex + dstOffset] = src[sindex + srcOffset];
            }
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithSubTensor4d(const FLOAT* __restrict__ src,
                                                              const index_t srcOffset,
                                                              const index_t srcStride0,
                                                              const index_t srcStride1,
                                                              const index_t srcStride2,
                                                              const index_t srcStride3,
                                                              const index_t srcLen0,
                                                              const index_t srcLen1,
                                                              const index_t srcLen2,
                                                              const index_t srcLen3,
                                                              FLOAT* __restrict__ dst,
                                                              const index_t dstOffset,
                                                              const index_t dstStride0,
                                                              const index_t dstStride1,
                                                              const index_t dstStride2,
                                                              const index_t dstStride3)
{
    index_t itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const index_t did0_begin = itmp / work_stride_0;

    itmp -= did0_begin * work_stride_0;

    const index_t did1_begin = itmp / work_stride_1;

    itmp -= did1_begin * work_stride_1;

    const index_t did2_begin = itmp / work_stride_2;

    itmp -= did2_begin * work_stride_2;

    const index_t did3_begin = itmp / work_stride_3;

    for(index_t did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(index_t did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            for(index_t did2 = did2_begin; did2 < srcLen2; did2 += WORK_LENGTH_2)
            {
                for(index_t did3 = did3_begin; did3 < srcLen3; did3 += WORK_LENGTH_3)
                {
                    const index_t sindex = srcStride0 * did0 + srcStride1 * did1 +
                                           srcStride2 * did2 + srcStride3 * did3;
                    const index_t dindex = dstStride0 * did0 + dstStride1 * did1 +
                                           dstStride2 * did2 + dstStride3 * did3;

                    dst[dindex + dstOffset] = src[sindex + srcOffset];
                }
            }
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithSubTensor5d(const FLOAT* __restrict__ src,
                                                              const index_t srcOffset,
                                                              const index_t srcStride0,
                                                              const index_t srcStride1,
                                                              const index_t srcStride2,
                                                              const index_t srcStride3,
                                                              const index_t srcStride4,
                                                              const index_t srcLen0,
                                                              const index_t srcLen1,
                                                              const index_t srcLen2,
                                                              const index_t srcLen3,
                                                              const index_t srcLen4,
                                                              FLOAT* __restrict__ dst,
                                                              const index_t dstOffset,
                                                              const index_t dstStride0,
                                                              const index_t dstStride1,
                                                              const index_t dstStride2,
                                                              const index_t dstStride3,
                                                              const index_t dstStride4)
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

    for(index_t did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(index_t did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            for(index_t did2 = did2_begin; did2 < srcLen2; did2 += WORK_LENGTH_2)
            {
                for(index_t did3 = did3_begin; did3 < srcLen3; did3 += WORK_LENGTH_3)
                {
                    for(index_t did4 = did4_begin; did4 < srcLen4; did4 += WORK_LENGTH_4)
                    {
                        const index_t sindex = srcStride0 * did0 + srcStride1 * did1 +
                                               srcStride2 * did2 + srcStride3 * did3 +
                                               srcStride4 * did4;
                        const index_t dindex = dstStride0 * did0 + dstStride1 * did1 +
                                               dstStride2 * did2 + dstStride3 * did3 +
                                               dstStride4 * did4;

                        dst[dindex + dstOffset] = src[sindex + srcOffset];
                    }
                }
            }
        }
    }
}
