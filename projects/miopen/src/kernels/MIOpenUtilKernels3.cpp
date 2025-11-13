// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

inline __device__ unsigned int iMod(unsigned int v, unsigned int u, unsigned int d)
{
    return v - __mul24(u, d);
}

constexpr unsigned int out_channel_stride_aligned = OUT_CHANNEL_STRIDE / WRITE_UNIT;
constexpr unsigned int out_stride_aligned         = OUT_STRIDE / WRITE_UNIT;

extern "C" __global__
__launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) void SubSample(const FLOAT* __restrict in,
                                                             FLOAT* __restrict out)
{
    const unsigned int stack_pos = blockIdx.x * LOCAL_SIZE_X + threadIdx.x;
    const unsigned int batch_id  = blockIdx.y * LOCAL_SIZE_Y + threadIdx.y;

    unsigned int map_id  = stack_pos / out_channel_stride_aligned;
    unsigned int pix_pos = iMod(stack_pos, map_id, out_channel_stride_aligned);
    unsigned int out_y   = pix_pos / out_stride_aligned;
    unsigned int out_x   = iMod(pix_pos, out_y, out_stride_aligned) * WRITE_UNIT;

    unsigned int out_off = batch_id * IN_BATCH_STRIDE + stack_pos * WRITE_UNIT;
    unsigned int in_y    = out_y * FILTER0_STRIDE1;
    unsigned int in_x    = out_x * FILTER0_STRIDE0;
    unsigned int in_off =
        batch_id * IN0_BATCH_STRIDE + map_id * IN0_CHANNEL_STRIDE + in_y * IN0_STRIDE + in_x;

    const FLOAT* in_ptr = &in[in_off];
    FLOAT* out_ptr      = &out[out_off];

    for(unsigned int i = 0; i < WRITE_UNIT; ++i, in_ptr += FILTER0_STRIDE0, out_ptr++)
    {
        *out_ptr = *in_ptr;
    }
}

constexpr unsigned int in_channel_stride_aligned = IN0_CHANNEL_STRIDE / WRITE_UNIT;
constexpr unsigned int in_stride_aligned         = IN0_STRIDE / WRITE_UNIT;

extern "C" __global__
__launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) void UpSample(const FLOAT* __restrict in,
                                                            FLOAT* __restrict out)
{
    const unsigned int stack_pos = blockIdx.x * LOCAL_SIZE_X + threadIdx.x;
    const unsigned int batch_id  = blockIdx.y * LOCAL_SIZE_Y + threadIdx.y;

    unsigned int map_id  = stack_pos / in_channel_stride_aligned;
    unsigned int pix_pos = iMod(stack_pos, map_id, in_channel_stride_aligned);
    unsigned int in_y    = pix_pos / in_stride_aligned;
    unsigned int in_x    = iMod(pix_pos, in_y, in_stride_aligned) * WRITE_UNIT;

    unsigned int in_off = batch_id * IN_BATCH_STRIDE + stack_pos * WRITE_UNIT;
    unsigned int out_y  = in_y * FILTER0_STRIDE1;
    unsigned int out_x  = in_x * FILTER0_STRIDE0;
    unsigned int out_off =
        batch_id * IN0_BATCH_STRIDE + map_id * OUT_CHANNEL_STRIDE + out_y * OUT_STRIDE + out_x;

    const FLOAT* in_ptr = &in[in_off];
    FLOAT* out_ptr      = &out[out_off];

    for(unsigned int i = 0; i < WRITE_UNIT; ++i, in_ptr++, out_ptr += FILTER0_STRIDE0)
    {
        *out_ptr = *in_ptr;
    }
}
