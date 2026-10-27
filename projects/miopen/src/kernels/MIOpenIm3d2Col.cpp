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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "float_types.h"
#include "miopen_cstdint.hpp"

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#ifndef MIOPEN_USE_BFP16
#define MIOPEN_USE_BFP16 0
#endif

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#ifndef MIOPEN_USE_INT32
#define MIOPEN_USE_INT32 0
#endif

#if MIOPEN_USE_INT8
typedef char data_t;
#elif MIOPEN_USE_INT32
typedef int data_t;
#elif(MIOPEN_USE_FP16 || MIOPEN_USE_BFP16)
// As the half type degrades the performance, use short instead of half in the
// im2col, which has no match op. May change back to half when compile can
// deliver equal performance as short
typedef short data_t;
#elif MIOPEN_USE_FP32
typedef float data_t;
#endif

#if(LAYOUT_NHWC == 1)
extern "C" __global__ void Im3d2Col(data_t* const __restrict im,
                                    const uint64_t im_offset,
                                    const uint64_t im_c_size,
                                    const uint64_t im_d_size,
                                    const uint64_t im_h_size,
                                    const uint64_t im_w_size,
                                    const uint64_t wei_d_size,
                                    const uint64_t wei_h_size,
                                    const uint64_t wei_w_size,
                                    const uint64_t out_d_size,
                                    const uint64_t out_h_size,
                                    const uint64_t out_w_size,
                                    const uint64_t pad_d_size,
                                    const uint64_t pad_h_size,
                                    const uint64_t pad_w_size,
                                    const uint64_t stride_d_size,
                                    const uint64_t stride_h_size,
                                    const uint64_t stride_w_size,
                                    const uint64_t dilation_d_size,
                                    const uint64_t dilation_h_size,
                                    const uint64_t dilation_w_size,
                                    data_t* __restrict col)
{
    const uint64_t num_groups         = GROUPS;
    const uint64_t channels_per_group = im_c_size / num_groups;
    const uint64_t inner_size = (uint64_t)wei_d_size * wei_h_size * wei_w_size * channels_per_group;
    const uint64_t col_group_size = (uint64_t)out_d_size * out_h_size * out_w_size * inner_size;
    const uint64_t col_size       = col_group_size * num_groups;

    const uint64_t gtid        = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t global_size = (uint64_t)blockDim.x * gridDim.x;

    for(uint64_t tid = gtid; tid < col_size; tid += global_size)
    {
        const uint64_t group_id     = tid / col_group_size;
        const uint64_t tid_in_group = tid - group_id * col_group_size;

        // "col" matrix row and colome id
        const uint64_t col_i = tid_in_group / inner_size;
        const uint64_t col_j = tid_in_group - col_i * inner_size;

        // output tensor out_d, out_h, out_w id
        const uint64_t out_hw = (uint64_t)out_h_size * out_w_size;
        const uint64_t out_d  = col_i / out_hw;
        uint64_t tmp          = col_i - out_d * out_hw;
        const uint64_t out_h  = tmp / out_w_size;
        const uint64_t out_w  = tmp - out_h * out_w_size;

        // weight tensor wei_d, wei_h, wei_w, wei_c
        const uint64_t wei_hwc = (uint64_t)wei_h_size * wei_w_size * channels_per_group;
        const uint64_t wei_d   = col_j / wei_hwc;
        tmp                    = col_j - wei_d * wei_hwc;
        const uint64_t wei_wc  = (uint64_t)wei_w_size * channels_per_group;
        const uint64_t wei_h   = tmp / wei_wc;
        tmp -= wei_h * (wei_w_size * channels_per_group);
        const uint64_t wei_w          = tmp / channels_per_group;
        const uint64_t wei_c_in_group = tmp - wei_w * channels_per_group;

        const uint64_t wei_c = wei_c_in_group + group_id * channels_per_group;

        // input tensor im_d, im_h, im_w id
        const int64_t im_d = (int64_t)stride_d_size * (int64_t)out_d +
                             (int64_t)dilation_d_size * (int64_t)wei_d - (int64_t)pad_d_size;
        const int64_t im_h = (int64_t)stride_h_size * (int64_t)out_h +
                             (int64_t)dilation_h_size * (int64_t)wei_h - (int64_t)pad_h_size;
        const int64_t im_w = (int64_t)stride_w_size * (int64_t)out_w +
                             (int64_t)dilation_w_size * (int64_t)wei_w - (int64_t)pad_w_size;

        const uint64_t im_idx = im_offset + (uint64_t)im_d * im_h_size * im_w_size * im_c_size +
                                (uint64_t)im_h * im_w_size * im_c_size +
                                (uint64_t)im_w * im_c_size + wei_c;

        // NdHWC Memory Access
        data_t value = (im_d >= 0 && im_d < im_d_size && im_h >= 0 && im_h < im_h_size &&
                        im_w >= 0 && im_w < im_w_size && wei_c < im_c_size)
                           ? im[im_idx]
                           : 0;

        col[tid] = value;
    }
}

#else
extern "C" __global__ void Im3d2Col(data_t* const __restrict im,
                                    const uint64_t im_offset,
                                    const uint64_t im_c_size,
                                    const uint64_t im_d_size,
                                    const uint64_t im_h_size,
                                    const uint64_t im_w_size,
                                    const uint64_t wei_d_size,
                                    const uint64_t wei_h_size,
                                    const uint64_t wei_w_size,
                                    const uint64_t out_d_size,
                                    const uint64_t out_h_size,
                                    const uint64_t out_w_size,
                                    const uint64_t pad_d_size,
                                    const uint64_t pad_h_size,
                                    const uint64_t pad_w_size,
                                    const uint64_t stride_d_size,
                                    const uint64_t stride_h_size,
                                    const uint64_t stride_w_size,
                                    const uint64_t dilation_d_size,
                                    const uint64_t dilation_h_size,
                                    const uint64_t dilation_w_size,
                                    data_t* __restrict col)
{
    // Use size_t to prevent overflow for large tensors (>4GB elements)
    size_t col_size = (size_t)out_d_size * (size_t)out_h_size * (size_t)out_w_size *
                      (size_t)wei_d_size * (size_t)wei_h_size * (size_t)wei_w_size *
                      (size_t)im_c_size;

    size_t gtid        = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t global_size = (size_t)blockDim.x * (size_t)gridDim.x;
    for(size_t tid = gtid; tid < col_size; tid += global_size)
    {
        // "col" matrix row and colume id
        size_t out_spatial_size = (size_t)out_d_size * (size_t)out_h_size * (size_t)out_w_size;
        size_t col_i            = tid / out_spatial_size;
        size_t col_j            = tid - col_i * out_spatial_size;

        // output tensor out_d, out_h, out_w id
        unsigned out_d = col_j / (out_h_size * out_w_size);
        unsigned tmp   = col_j - out_d * (out_h_size * out_w_size);
        unsigned out_h = tmp / out_w_size;
        unsigned out_w = tmp - out_h * out_w_size;

        // weight tensor wei_c, wei_d, wei_h, wei_d id
        unsigned wei_c = col_i / (wei_d_size * wei_h_size * wei_w_size);
        tmp            = col_i - wei_c * (wei_d_size * wei_h_size * wei_w_size);
        unsigned wei_d = tmp / (wei_h_size * wei_w_size);
        tmp -= wei_d * (wei_h_size * wei_w_size);
        unsigned wei_h = tmp / wei_w_size;
        unsigned wei_w = tmp - wei_h * wei_w_size;

        // input tensor im_d, im_h, im_w id
        int im_d = (int)(stride_d_size * out_d + dilation_d_size * wei_d) - (int)(pad_d_size);
        int im_h = (int)(stride_h_size * out_h + dilation_h_size * wei_h) - (int)(pad_h_size);
        int im_w = (int)(stride_w_size * out_w + dilation_w_size * wei_w) - (int)(pad_w_size);

        data_t value =
            (im_d >= 0 && im_d < im_d_size && im_h >= 0 && im_h < im_h_size && im_w >= 0 &&
             im_w < im_w_size)
                ? im[(size_t)im_offset +
                     (size_t)wei_c * (size_t)im_d_size * (size_t)im_h_size * (size_t)im_w_size +
                     (size_t)im_d * (size_t)im_h_size * (size_t)im_w_size +
                     (size_t)im_h * (size_t)im_w_size + (size_t)im_w]
                : 0;

        col[tid] = value;
    }
}

#endif
