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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#endif

#include "miopen_limits.hpp"
#include "pooling_functions.h"

#ifndef USE_GLOBAL_INDEX
#define USE_GLOBAL_INDEX 1
#endif

#if defined(MLO_POOLING_SAVE_INDEX) && (MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX)
#define USE_MASK 1
#else
#define USE_MASK 0
#endif

#if (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE) || (MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE)
#define AVERAGE_OPS 1
#else
#define AVERAGE_OPS 0
#endif

// Let's use extended-precision accumulator only in FP16 pooling and only for averaging.
// For all other ops and datatypes, redefine macros used for accum-float conversion
// and accum types, so they do nothing, i.e. treate FLOAT_ACCUM as FLOAT.
#if !(AVERAGE_OPS && MIOPEN_USE_FP16)
#define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 1
#endif

#include "float_types.h"

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
#define MLO_POOLING_OP(A, B) (fmax((A), (B)))
#elif AVERAGE_OPS
#define MLO_POOLING_OP(A, B) ((A) + (B))
#endif

#define BOT_TILE_W ((TOP_W_PER_WORK - 1) * STRIDE_W + KERNEL_SZ_W)
#define BOT_TILE_H ((TOP_H_PER_WORK - 1) * STRIDE_H + KERNEL_SZ_H)
#define BOT_TILE_D ((TOP_D_PER_WORK - 1) * STRIDE_D + KERNEL_SZ_D)

extern "C" __global__ __launch_bounds__(MLO_POOLING_GROUP_SZ0) //
    void mloPoolingNDFwd(const FLOAT* bot,
                         FLOAT* top,
                         [[maybe_unused]] index_t* mask,
                         const unsigned int pad_d,
                         const unsigned int pad_h,
                         const unsigned int pad_w,
                         const unsigned int batch,
                         const unsigned int chal,
                         const unsigned int bot_d,
                         const unsigned int bot_h,
                         const unsigned int bot_w,
                         const unsigned int top_d,
                         const unsigned int top_h,
                         const unsigned int top_w,
                         const unsigned int bot_str_b,
                         const unsigned int bot_str_c,
                         const unsigned int bot_str_d,
                         const unsigned int bot_str_h,
                         const unsigned int top_str_b,
                         const unsigned int top_str_c,
                         const unsigned int top_str_d,
                         const unsigned int top_str_h,
                         const unsigned int total_work)
{

    int top_blk_w = (top_w + TOP_W_PER_WORK - 1) / TOP_W_PER_WORK;
    int top_blk_h = (top_h + TOP_H_PER_WORK - 1) / TOP_H_PER_WORK;
    int top_blk_d = (top_d + TOP_D_PER_WORK - 1) / TOP_D_PER_WORK;

    top_blk_w = max(top_blk_w, 1);
    top_blk_h = max(top_blk_h, 1);
    top_blk_d = max(top_blk_d, 1);

    for(unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total_work;
        gid += MAX_ACTIV_WORKITEM)
    {
        int b_id = gid / chal / top_blk_w / top_blk_h / top_blk_d;
        int c_id = (gid / top_blk_w / top_blk_h / top_blk_d) % chal;

        int top_d_id = ((gid / top_blk_w / top_blk_h) % top_blk_d) * TOP_D_PER_WORK;
        int top_h_id = ((gid / top_blk_w) % top_blk_h) * TOP_H_PER_WORK;
        int top_w_id = (gid % top_blk_w) * TOP_W_PER_WORK;

        FLOAT bot_data[BOT_TILE_D][BOT_TILE_H][BOT_TILE_W];

        for(unsigned int h = 0; h < BOT_TILE_D; ++h)
        {
            int run_z = top_d_id * STRIDE_D + h - pad_d;
            for(unsigned int j = 0; j < BOT_TILE_H; ++j)
            {
                int run_y = top_h_id * STRIDE_H + j - pad_h;
                for(unsigned int i = 0; i < BOT_TILE_W; ++i)
                {
                    int run_x       = top_w_id * STRIDE_W + i - pad_w;
                    int bot_gbl_off = b_id * bot_str_b + c_id * bot_str_c + run_z * bot_str_d +
                                      run_y * bot_str_h + run_x;
                    bool vis = ((run_z >= 0 && run_z < bot_d) && (run_y >= 0 && run_y < bot_h) &&
                                (run_x >= 0 && run_x < bot_w)) &&
                               b_id < batch;

                    bot_data[h][j][i] = (vis) ? bot[bot_gbl_off] :
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                                              (FLOAT)(-MAX_VAL);
#elif AVERAGE_OPS
                                              (FLOAT)(0);
#endif
                }
            }
        }

#pragma unroll
        for(unsigned int m = 0; m < TOP_D_PER_WORK; m++)
        {
#if AVERAGE_OPS
            int dstart = (top_d_id + m) * STRIDE_D - pad_d;
            int dend   = min((dstart + KERNEL_SZ_D), (int)bot_d);
            dstart     = max(dstart, 0);
#endif
            for(unsigned int k = 0; k < TOP_H_PER_WORK; k++)
            {
#if AVERAGE_OPS
                int hstart = (top_h_id + k) * STRIDE_H - pad_h;
                int hend   = min((hstart + KERNEL_SZ_H), (int)bot_h);
                hstart     = max(hstart, 0);
#endif
                for(unsigned int l = 0; l < TOP_W_PER_WORK; l++)
                {

#if AVERAGE_OPS
                    int wstart = (top_w_id + l) * STRIDE_W - pad_w;
                    int wend   = min((wstart + KERNEL_SZ_W), (int)bot_w);
                    wstart     = max(wstart, 0);
                    unsigned int pool_size =
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE_INCLUSIVE
                        KERNEL_SZ_W * KERNEL_SZ_H * KERNEL_SZ_D;
                    (void)wend;
                    (void)hend;
                    (void)dend;
#else
                        (dend - dstart) * (hend - hstart) * (wend - wstart);
#endif
                    pool_size = (pool_size == 0) ? 1 : pool_size;
#endif

                    FLOAT_ACCUM top_val =
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
                        (FLOAT_ACCUM)(-MAX_VAL_ACCUM);
#elif AVERAGE_OPS
                        (FLOAT_ACCUM)(0);
#endif

#if USE_MASK
                    index_t mask_idx = 0;
#endif

                    for(unsigned int h = 0; h < KERNEL_SZ_D; h++)
                    {
                        for(unsigned int j = 0; j < KERNEL_SZ_H; j++)
                        {
                            for(unsigned int i = 0; i < KERNEL_SZ_W; i++)
                            {

                                FLOAT_ACCUM bot_val = CVT_FLOAT2ACCUM(
                                    bot_data[h + m * STRIDE_D][j + k * STRIDE_H][i + l * STRIDE_W]);

#if USE_MASK
                                if(bot_val > top_val)
                                {
                                    top_val = bot_val;

#if USE_GLOBAL_INDEX
                                    mask_idx =
                                        ((top_w_id + l) * STRIDE_W + i - pad_w) +
                                        bot_w * ((top_h_id + k) * STRIDE_H + j - pad_h) +
                                        bot_w * bot_h * ((top_d_id + m) * STRIDE_D + h - pad_d);
#else
                                    mask_idx = i + KERNEL_SZ_W * (j + KERNEL_SZ_H * h);
#endif
                                }
#else
                                top_val = MLO_POOLING_OP(top_val, bot_val);
#endif
                            }
                        }
                    }

#if AVERAGE_OPS
                    top_val *= CVT_FP32_2ACCUM(1.f) / (FLOAT_ACCUM)pool_size;
#endif

                    if(top_d_id + m < top_d && top_h_id + k < top_h && top_w_id + l < top_w &&
                       b_id < batch)
                    {
                        unsigned int top_idx = b_id * top_str_b + c_id * top_str_c +
                                               (top_d_id + m) * top_str_d +
                                               (top_h_id + k) * top_str_h + top_w_id + l;

                        top[top_idx] = top_val;
#if USE_MASK
                        mask[top_idx] = mask_idx;
#endif
                    }
                }
            }
        }
    }
}
