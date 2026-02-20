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

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithSubTensor1d(const FLOAT* __restrict__ src,
                                                              const int srcOffset,
                                                              const int srcStride0,
                                                              const int srcLen0,
                                                              FLOAT* __restrict__ dst,
                                                              const int dstOffset,
                                                              const int dstStride0)
{
    unsigned int itmp = blockIdx.x * LOCAL_SIZE + threadIdx.x;

    const unsigned int did0_begin = itmp / work_stride_0;

    for(unsigned int did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        const unsigned int sindex = srcStride0 * did0;
        const unsigned int dindex = dstStride0 * did0;
        dst[dindex + dstOffset]   = src[sindex + srcOffset];
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithSubTensor2d(const FLOAT* __restrict__ src,
                                                              const int srcOffset,
                                                              const int srcStride0,
                                                              const int srcStride1,
                                                              const int srcLen0,
                                                              const int srcLen1,
                                                              FLOAT* __restrict__ dst,
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
            dst[dindex + dstOffset]   = src[sindex + srcOffset];
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithSubTensor3d(const FLOAT* __restrict__ src,
                                                              const int srcOffset,
                                                              const int srcStride0,
                                                              const int srcStride1,
                                                              const int srcStride2,
                                                              const int srcLen0,
                                                              const int srcLen1,
                                                              const int srcLen2,
                                                              FLOAT* __restrict__ dst,
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
                dst[dindex + dstOffset] = src[sindex + srcOffset];
            }
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithSubTensor4d(const FLOAT* __restrict__ src,
                                                              const int srcOffset,
                                                              const int srcStride0,
                                                              const int srcStride1,
                                                              const int srcStride2,
                                                              const int srcStride3,
                                                              const int srcLen0,
                                                              const int srcLen1,
                                                              const int srcLen2,
                                                              const int srcLen3,
                                                              FLOAT* __restrict__ dst,
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

                    dst[dindex + dstOffset] = src[sindex + srcOffset];
                }
            }
        }
    }
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SubTensorOpWithSubTensor5d(const FLOAT* __restrict__ src,
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
                                                              FLOAT* __restrict__ dst,
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

                        dst[dindex + dstOffset] = src[sindex + srcOffset];
                    }
                }
            }
        }
    }
}
