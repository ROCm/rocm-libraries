// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "hip_math_ops.hpp"

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
