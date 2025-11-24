// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

// Given the value of x & y, computes the value r such that r = x - k*y,
// where is the quotient `k` is returned as an output parameter.
// The `k*y` multiplication is done in 24 bits. It is the responsibility of the solver
// to ensure `x` & `y` don't exceed 2^24, otherwise this code should be updated to
// avoid UB.
inline __device__ unsigned int iRemquo(unsigned int x, unsigned int y, unsigned int& k)
{
    k = x / y;
    return x - __mul24(k, y);
}

static_assert((OUT_CHANNEL_STRIDE % WRITE_UNIT) == 0);
constexpr unsigned int out_channel_stride_aligned = OUT_CHANNEL_STRIDE / WRITE_UNIT;

static_assert((OUT_STRIDE % WRITE_UNIT) == 0);
constexpr unsigned int out_stride_aligned = OUT_STRIDE / WRITE_UNIT;

extern "C" __global__
__launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) void SubSample(const FLOAT* __restrict in,
                                                             FLOAT* __restrict out)
{
    const unsigned int stack_pos = blockIdx.x * LOCAL_SIZE_X + threadIdx.x;
    const unsigned int batch_id  = blockIdx.y * LOCAL_SIZE_Y + threadIdx.y;

    unsigned int map_id;
    unsigned int pix_pos = iRemquo(stack_pos, out_channel_stride_aligned, map_id);

    // Offsets are stored in 32-bit variable, it is the responsibiliy of the solver to ensure
    // this won't overflow, otherwise this should be updated to 64-bit variable.
    unsigned int out_y;
    unsigned int out_x   = iRemquo(pix_pos, out_stride_aligned, out_y) * WRITE_UNIT;
    unsigned int out_off = batch_id * IN_BATCH_STRIDE + stack_pos * WRITE_UNIT;

    unsigned int in_y = out_y * FILTER0_STRIDE1;
    unsigned int in_x = out_x * FILTER0_STRIDE0;
    unsigned int in_off =
        batch_id * IN0_BATCH_STRIDE + map_id * IN0_CHANNEL_STRIDE + in_y * IN0_STRIDE + in_x;

    const FLOAT* in_ptr = &in[in_off];
    FLOAT* out_ptr      = &out[out_off];

    for(unsigned int i = 0; i < WRITE_UNIT; ++i, in_ptr += FILTER0_STRIDE0, out_ptr++)
    {
        *out_ptr = *in_ptr;
    }
}

static_assert((IN0_CHANNEL_STRIDE % WRITE_UNIT) == 0);
constexpr unsigned int in_channel_stride_aligned = IN0_CHANNEL_STRIDE / WRITE_UNIT;

static_assert((IN0_STRIDE % WRITE_UNIT) == 0);
constexpr unsigned int in_stride_aligned = IN0_STRIDE / WRITE_UNIT;

/*
 *  Upsample kernel requires output to already be zero initalized to avoid undefined values in gaps
 *  between strides. See zero initalization in miopen/src/conv/invokers/gcn_asm_1x1u_us.cpp with
 *  TODO comment for finding a way to avoid this pre-initalization to improve performance.
 */
extern "C" __global__
__launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) void UpSample(const FLOAT* __restrict in,
                                                            FLOAT* __restrict out)
{
    const unsigned int stack_pos = blockIdx.x * LOCAL_SIZE_X + threadIdx.x;
    const unsigned int batch_id  = blockIdx.y * LOCAL_SIZE_Y + threadIdx.y;

    unsigned int map_id;
    unsigned int pix_pos = iRemquo(stack_pos, in_channel_stride_aligned, map_id);

    // Offsets are stored in 32-bit variables, it is the responsibiliy of the solver to ensure
    // this won't overflow, otherwise this should be updated to 64-bit variable.
    unsigned int in_y   = pix_pos / in_stride_aligned;
    unsigned int in_x   = iRemquo(pix_pos, in_stride_aligned, in_y) * WRITE_UNIT;
    unsigned int in_off = batch_id * IN_BATCH_STRIDE + stack_pos * WRITE_UNIT;

    unsigned int out_y = in_y * FILTER0_STRIDE1;
    unsigned int out_x = in_x * FILTER0_STRIDE0;
    unsigned int out_off =
        batch_id * IN0_BATCH_STRIDE + map_id * OUT_CHANNEL_STRIDE + out_y * OUT_STRIDE + out_x;

    const FLOAT* in_ptr = &in[in_off];
    FLOAT* out_ptr      = &out[out_off];

    for(unsigned int i = 0; i < WRITE_UNIT; ++i, in_ptr++, out_ptr += FILTER0_STRIDE0)
    {
        *out_ptr = *in_ptr;
    }
}
