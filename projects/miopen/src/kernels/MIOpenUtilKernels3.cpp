/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
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
