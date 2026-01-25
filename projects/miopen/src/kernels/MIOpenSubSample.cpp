// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "hip_math_ops.hpp"

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
