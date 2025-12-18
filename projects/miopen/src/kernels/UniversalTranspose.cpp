// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#endif

#include "float_types.h"
#include "miopen_cstdint.hpp"

extern "C" __global__ __launch_bounds__(256) void UniversalTranspose(const FLOAT* __restrict__ in,
                                                                     FLOAT* __restrict__ out,
                                                                     uint64_t lens_n,
                                                                     uint64_t lens_c,
                                                                     uint64_t lens_d,
                                                                     uint64_t lens_h,
                                                                     uint64_t lens_w,
                                                                     uint64_t in_strides_n,
                                                                     uint64_t in_strides_c,
                                                                     uint64_t in_strides_d,
                                                                     uint64_t in_strides_h,
                                                                     uint64_t in_strides_w,
                                                                     uint64_t out_strides_n,
                                                                     uint64_t out_strides_c,
                                                                     uint64_t out_strides_d,
                                                                     uint64_t out_strides_h,
                                                                     uint64_t out_strides_w)
{
    const uint64_t global_size = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const uint64_t global_id   = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const uint64_t lens_wh    = lens_w * lens_h;
    const uint64_t lens_whd   = lens_wh * lens_d;
    const uint64_t lens_whdc  = lens_whd * lens_c;
    const uint64_t lens_whdcn = lens_whdc * lens_n;

    for(uint64_t id = global_id; id < lens_whdcn; id += global_size)
    {
        const uint64_t n     = id / lens_whdc;
        const uint64_t rem_n = id - n * lens_whdc;
        const uint64_t c     = rem_n / lens_whd;
        const uint64_t rem_c = rem_n - c * lens_whd;
        const uint64_t d     = rem_c / lens_wh;
        const uint64_t rem_d = rem_c - d * lens_wh;
        const uint64_t h     = rem_d / lens_w;
        const uint64_t w     = rem_d - h * lens_w;

        const uint64_t in_id = n * in_strides_n + c * in_strides_c + d * in_strides_d +
                               h * in_strides_h + w * in_strides_w;

        const uint64_t out_id = n * out_strides_n + c * out_strides_c + d * out_strides_d +
                                h * out_strides_h + w * out_strides_w;

        out[out_id] = in[in_id];
    }
}
