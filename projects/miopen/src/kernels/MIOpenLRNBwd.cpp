// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "miopen_math.hpp"

constexpr unsigned group_size_z      = 1u;
constexpr unsigned local_data_width  = GROUP_SIZE_X * HORIZ_OUT_PIX + KERNEL_SIZE - 1;
constexpr unsigned local_data_height = GROUP_SIZE_Y * VERT_OUT_PIX + KERNEL_SIZE - 1;
constexpr bool low_channel_count     = N_INPUTS < KERNEL_SIZE;

/*
This is a naive implementation.
The "sliding window" -based implementation is in MIOpenLRNFwd.cpp file
*/
__launch_bounds__(GROUP_SIZE_X* GROUP_SIZE_Y* group_size_z) extern "C" __global__
    void MIOpenLRNWithinChannelBwd(const FLOAT* top,
                                   const FLOAT* bot,
                                   const FLOAT* top_df,
                                   const FLOAT* scale,
                                   FLOAT* bot_df,
                                   FLOAT /*ratio*/,
                                   FLOAT alpha,
                                   FLOAT beta)
{
    __shared__ FLOAT top_df_data[local_data_width * local_data_height];
    __shared__ FLOAT ratio_data[local_data_width * local_data_height];
    const int x          = blockIdx.x * GROUP_SIZE_X * HORIZ_OUT_PIX;
    const int y          = blockIdx.y * GROUP_SIZE_Y * VERT_OUT_PIX;
    const int lcl_id0    = threadIdx.x;
    const int lcl_id1    = threadIdx.y;
    const int ob         = blockIdx.z * group_size_z + threadIdx.z; // output * batch_sz
    const int o          = ob / BATCH_SIZE;
    const int b          = ob - o * BATCH_SIZE;
    const int top_x      = x;
    const int top_y      = y;
    const int top_df_off = b * TOPDF_BATCH_STRIDE + o * TOPDF_CHANNEL_STRIDE;
    const int scale_off  = b * SCALE_BATCH_STRIDE + o * SCALE_CHANNEL_STRIDE;
    const int bot_x      = x + lcl_id0 * HORIZ_OUT_PIX;
    const int bot_y      = y + lcl_id1 * VERT_OUT_PIX;

// load top_diff and scale tiles
#pragma unroll VERT_OUT_PIX
    for(int b_j = lcl_id1; b_j < local_data_height; b_j += GROUP_SIZE_Y)
    {
        int top_y_act         = top_y + b_j - PAD;
        const bool invisibleY = (top_y_act < 0) || (top_y_act >= TOP_HEIGHT);
        top_y_act             = (invisibleY) ? 0 : top_y_act;

        const int top_df_y_off = top_y_act * TOPDF_STRIDE;
        const int scale_y_off  = top_y_act * SCALE_STRIDE;
        const int lcl_off_v    = b_j * local_data_width;

/*
  Note: Code duplication from manual loop peeling

  The HIP compiler does not unroll this loop if the loop statement is written as
  `for(int b_i = lcl_id0; b_i < local_data_width; b_i += GROUP_SIZE_X)`, leading
  to a loss in performance.

  To enable unrolling the last loop iteration is manually peeled. As this is the
  iteration that may not be uniformly executed by all work-items in a work-group.

  Unfortunately there is code duplication in the body is the loop/if statements, as
  refactoring it into a __alwaysinline__ free function seems inhibits compiler
  optimizations.
*/
#pragma unroll
        for(int i = 0; i < HORIZ_OUT_PIX; i++)
        {
            const int b_i         = i * GROUP_SIZE_X + lcl_id0;
            int top_x_act         = top_x + b_i - PAD;
            const bool invisibleX = (top_x_act < 0) || (top_x_act >= TOP_WIDTH);
            top_x_act             = (invisibleX) ? 0 : top_x_act;

            FLOAT top_df_val = top_df[top_df_off + top_df_y_off + top_x_act];
            FLOAT scale_val  = scale[scale_off + scale_y_off + top_x_act];

            top_df_val = (invisibleX || invisibleY) ? 0 : top_df_val;
            scale_val  = (invisibleX || invisibleY) ? FLOAT(1.f) : scale_val;

            top_df_data[lcl_off_v + b_i] = top_df_val;
            ratio_data[lcl_off_v + b_i]  = scale_val;
        }
        if(lcl_id0 < (KERNEL_SIZE - 1))
        {
            const int b_i         = GROUP_SIZE_X * HORIZ_OUT_PIX + lcl_id0;
            int top_x_act         = top_x + b_i - PAD;
            const bool invisibleX = (top_x_act < 0) || (top_x_act >= TOP_WIDTH);
            top_x_act             = (invisibleX) ? 0 : top_x_act;

            FLOAT top_df_val = top_df[top_df_off + top_df_y_off + top_x_act];
            FLOAT scale_val  = scale[scale_off + scale_y_off + top_x_act];

            top_df_val = (invisibleX || invisibleY) ? 0 : top_df_val;
            scale_val  = (invisibleX || invisibleY) ? FLOAT(1.f) : scale_val;

            top_df_data[lcl_off_v + b_i] = top_df_val;
            ratio_data[lcl_off_v + b_i]  = scale_val;
        }
    }

    __syncthreads();

    // actual top_diffs and scales
    FLOAT prv_exp_scale[VERT_OUT_PIX][HORIZ_OUT_PIX];
    for(int j = 0; j < VERT_OUT_PIX; ++j)
    {
        const int lcl_off_v = (lcl_id1 * VERT_OUT_PIX + PAD + j) * local_data_width;
        for(int i = 0; i < HORIZ_OUT_PIX; i++)
        {
            const FLOAT scale_ratio     = ratio_data[lcl_off_v + lcl_id0 * HORIZ_OUT_PIX + PAD + i];
            const FLOAT scale_ratio_log = miopen::detail::log(scale_ratio);
            prv_exp_scale[j][i]         = miopen::detail::exp(-beta * scale_ratio_log);
        }
    }

    __syncthreads();

    // read top and load ratio tile
    const int top_off = b * TOP_BATCH_STRIDE + o * TOP_CHANNEL_STRIDE;
#pragma unroll VERT_OUT_PIX
    for(int b_j = lcl_id1; b_j < local_data_height; b_j += GROUP_SIZE_Y)
    {
        int top_y_act         = top_y + b_j - PAD;
        const bool invisibleY = (top_y_act < 0) || (top_y_act >= TOP_HEIGHT);
        top_y_act             = (invisibleY) ? 0 : top_y_act;

        const int top_y_off = top_y_act * TOP_STRIDE;
        const int lcl_off_v = b_j * local_data_width;
/*
  Note: Code duplication from manual loop peeling

  The HIP compiler does not unroll this loop if the loop statement is written as
  `for(int b_i = lcl_id0; b_i < local_data_width; b_i += GROUP_SIZE_X)`, leading
  to a loss in performance.

  To enable unrolling the last loop iteration is manually peeled. As this is the
  iteration that may not be uniformly executed by all work-items in a work-group.

  Unfortunately there is code duplication in the body is the loop/if statements, as
  refactoring it into a __alwaysinline__ free function seems inhibits compiler
  optimizations.
*/
#pragma unroll
        for(int i = 0; i < HORIZ_OUT_PIX; i++)
        {
            int b_i         = i * GROUP_SIZE_X + lcl_id0;
            int top_x_act   = top_x + b_i - PAD;
            bool invisibleX = (top_x_act < 0) || (top_x_act >= TOP_WIDTH);
            top_x_act       = (invisibleX) ? 0 : top_x_act;

            FLOAT top_val = top[top_off + top_y_off + top_x_act];
            top_val       = (invisibleX || invisibleY) ? 0 : top_val;

            const FLOAT top_df_val = top_df_data[lcl_off_v + b_i];
            const FLOAT scale_val  = ratio_data[lcl_off_v + b_i];

            // scale val is not 0
            FLOAT ratio_dta = (top_df_val * top_val) / scale_val;
            // replacing scale with ratio
            ratio_data[lcl_off_v + b_i] = ratio_dta;
        }
        if(lcl_id0 < (KERNEL_SIZE - 1))
        {
            int b_i         = GROUP_SIZE_X * HORIZ_OUT_PIX + lcl_id0;
            int top_x_act   = top_x + b_i - PAD;
            bool invisibleX = (top_x_act < 0) || (top_x_act >= TOP_WIDTH);
            top_x_act       = (invisibleX) ? 0 : top_x_act;

            FLOAT top_val = top[top_off + top_y_off + top_x_act];
            top_val       = (invisibleX || invisibleY) ? 0 : top_val;

            const FLOAT top_df_val = top_df_data[lcl_off_v + b_i];
            const FLOAT scale_val  = ratio_data[lcl_off_v + b_i];

            // scale val is not 0
            FLOAT ratio_dta = (top_df_val * top_val) / scale_val;
            // replacing scale with ratio
            ratio_data[lcl_off_v + b_i] = ratio_dta;
        }
    }

    __syncthreads();

    // caculate bot diff
    FLOAT prv_bot_diff[VERT_OUT_PIX][HORIZ_OUT_PIX];
    for(int j = 0; j < VERT_OUT_PIX; ++j)
    {
        const int v_off_v = (lcl_id1 * VERT_OUT_PIX + j);
        const int hstart  = y + v_off_v - PAD;
        const int hend    = min(hstart + KERNEL_SIZE, TOP_HEIGHT + PRE_PAD);

        // value offset, vertical
        int lcl_v_off_v = (v_off_v + PAD) * local_data_width;
        for(int i = 0; i < HORIZ_OUT_PIX; i++)
        {
            FLOAT prv_ratio_accum(0);
            const int v_off_h = lcl_id0 * HORIZ_OUT_PIX + i;

            const int wstart        = x + v_off_h - PAD;
            const int wend          = min(wstart + KERNEL_SIZE, TOP_WIDTH + PRE_PAD);
            const int adj_area_size = (hend - hstart) * (wend - wstart);

            // accum offset, horiz
            const int lcl_a_off_h = v_off_h;
            //	value offset, horiz
            const int lcl_v_off_h = lcl_a_off_h + PAD;
            for(int k = 0; k < KERNEL_SIZE; k++)
            {
                for(int l = 0; l < KERNEL_SIZE; l++)
                {
                    prv_ratio_accum +=
                        ratio_data[(v_off_v + k) * local_data_width + lcl_a_off_h + l];
                }
            }

            const FLOAT top_df_val  = top_df_data[lcl_v_off_v + lcl_v_off_h];
            const unsigned bot_off0 = BOT_BATCH_STRIDE * b + BOT_CHANNEL_STRIDE * o +
                                      BOT_STRIDE * (y + v_off_v) + x + v_off_h;

            const unsigned bot_off = (y + v_off_v < BOT_HEIGHT && x + v_off_h < BOT_WIDTH &&
                                      b < BATCH_SIZE && o < OUT_CHANNELS)
                                         ? bot_off0
                                         : BATCH_SIZE * BOT_BATCH_STRIDE - 1;

            const FLOAT bot_dta         = (y + v_off_v < BOT_HEIGHT && x + v_off_h < BOT_WIDTH &&
                                   b < BATCH_SIZE && o < OUT_CHANNELS)
                                              ? bot[bot_off]
                                              : 0;
            const FLOAT adj_ratio       = FLOAT(2.f) * alpha * beta / adj_area_size;
            const FLOAT prv_accum_ratio = adj_ratio * bot_dta * prv_ratio_accum;
            prv_bot_diff[j][i]          = prv_exp_scale[j][i] * top_df_val - prv_accum_ratio;
        }
    }

    for(int j = 0; j < VERT_OUT_PIX; j++)
    {
        for(int i = 0; i < HORIZ_OUT_PIX; i++)
        {
            if(bot_y + j < BOT_HEIGHT && bot_x + i < BOT_WIDTH && b < BATCH_SIZE &&
               o < OUT_CHANNELS)
            {
                bot_df[BOTDF_BATCH_STRIDE * b + BOTDF_CHANNEL_STRIDE * o +
                       BOTDF_STRIDE * (bot_y + j) + bot_x + i] = prv_bot_diff[j][i];
            }
        }
    }
}

__launch_bounds__(GROUP_SIZE_X* GROUP_SIZE_Y* group_size_z) extern "C" __global__
    void MIOpenLRNAcrossChannelsBwd1(const FLOAT* top,
                                     const FLOAT* bot,
                                     const FLOAT* top_df,
                                     const FLOAT* scale,
                                     FLOAT* bot_df,
                                     FLOAT ratio,
                                     FLOAT /* alpha */,
                                     FLOAT beta)
{
    const int x       = blockIdx.x * GROUP_SIZE_X + threadIdx.x; // channel x
    const int y       = blockIdx.y * GROUP_SIZE_Y + threadIdx.y; // channel y
    const int b       = blockIdx.z * group_size_z + threadIdx.z; // batch
    FLOAT accum_ratio = 0;
    FLOAT top_df_in[KERNEL_SIZE];
    FLOAT scale_in[KERNEL_SIZE];
    FLOAT ratio_dta[KERNEL_SIZE];
    int c_i = 0, c_o = 0;

    for(; c_i < PRE_PAD; c_i++)
    {
        top_df_in[c_i] =
            top_df[TOPDF_BATCH_STRIDE * b + TOPDF_CHANNEL_STRIDE * c_i + TOPDF_STRIDE * y + x];
        scale_in[c_i] =
            scale[SCALE_BATCH_STRIDE * b + SCALE_CHANNEL_STRIDE * c_i + SCALE_STRIDE * y + x];
        const FLOAT top_dta =
            top[TOP_BATCH_STRIDE * b + TOP_CHANNEL_STRIDE * c_i + TOP_STRIDE * y + x];

        ratio_dta[c_i] = (top_df_in[c_i] * top_dta) / scale_in[c_i];

        if constexpr(low_channel_count)
        {
            ratio_dta[c_i] = (c_i < OUT_CHANNELS) ? ratio_dta[c_i] : 0;
        }

        accum_ratio = accum_ratio + ratio_dta[c_i];
    }

    for(; c_i < KERNEL_SIZE; c_i++, c_o++)
    {
        top_df_in[c_i] =
            top_df[TOPDF_BATCH_STRIDE * b + TOPDF_CHANNEL_STRIDE * c_i + TOPDF_STRIDE * y + x];
        scale_in[c_i] =
            scale[SCALE_BATCH_STRIDE * b + SCALE_CHANNEL_STRIDE * c_i + SCALE_STRIDE * y + x];
        const FLOAT top_dta =
            top[TOP_BATCH_STRIDE * b + TOP_CHANNEL_STRIDE * c_i + TOP_STRIDE * y + x];
        ratio_dta[c_i] = (top_df_in[c_i] * top_dta) / scale_in[c_i];
        if constexpr(low_channel_count)
        {
            ratio_dta[c_i] = (c_i < OUT_CHANNELS) ? ratio_dta[c_i] : 0;
        }

        accum_ratio = accum_ratio + ratio_dta[c_i];
        if(!low_channel_count || c_o < N_INPUTS)
        {
            const FLOAT bot_dta =
                bot[BOT_BATCH_STRIDE * b + BOT_CHANNEL_STRIDE * c_o + BOT_STRIDE * y + x];

            const FLOAT prv_scale       = scale_in[c_o];
            const FLOAT prv_scale_log   = miopen::detail::log(prv_scale);
            const FLOAT exp_scale       = miopen::detail::exp(-beta * prv_scale_log);
            const FLOAT prv_accum_ratio = ratio * bot_dta * accum_ratio;

            const FLOAT out_val = top_df_in[c_o] * exp_scale - prv_accum_ratio;

            const int bot_df_off =
                BOTDF_BATCH_STRIDE * b + BOTDF_CHANNEL_STRIDE * c_o + BOTDF_STRIDE * y + x;

            bot_df[bot_df_off] = out_val;
        }
    }

    for(; c_i < N_INPUTS; c_i++, c_o++)
    {
        const FLOAT prv_top_df_in =
            top_df[TOPDF_BATCH_STRIDE * b + TOPDF_CHANNEL_STRIDE * c_i + TOPDF_STRIDE * y + x];
        const FLOAT prv_scale_in =
            scale[SCALE_BATCH_STRIDE * b + SCALE_CHANNEL_STRIDE * c_i + SCALE_STRIDE * y + x];
        const FLOAT top_dta =
            top[TOP_BATCH_STRIDE * b + TOP_CHANNEL_STRIDE * c_i + TOP_STRIDE * y + x];
        FLOAT prv_ratio_dta = prv_top_df_in * top_dta / prv_scale_in;
        if constexpr(low_channel_count)
        {
            prv_ratio_dta = (c_i < OUT_CHANNELS) ? prv_ratio_dta : 0;
        }

        accum_ratio = accum_ratio + prv_ratio_dta;
        accum_ratio = accum_ratio - ratio_dta[0];

        for(int i = 0; i < KERNEL_SIZE - 1; i++)
        {
            top_df_in[i] = top_df_in[i + 1];
            scale_in[i]  = scale_in[i + 1];
            ratio_dta[i] = ratio_dta[i + 1];
        }

        top_df_in[KERNEL_SIZE - 1] = prv_top_df_in;
        scale_in[KERNEL_SIZE - 1]  = prv_scale_in;
        ratio_dta[KERNEL_SIZE - 1] = prv_ratio_dta;

        if(!low_channel_count || c_o < N_INPUTS)
        {
            const FLOAT bot_dta =
                bot[BOT_BATCH_STRIDE * b + BOT_CHANNEL_STRIDE * c_o + BOT_STRIDE * y + x];

            const FLOAT prv_scale = scale_in[PAD];
            const FLOAT exp_scale = miopen::detail::exp(-beta * miopen::detail::log(prv_scale));
            const FLOAT prv_accum_ratio = ratio * bot_dta * accum_ratio;
            const FLOAT out_val         = top_df_in[PAD] * exp_scale - prv_accum_ratio;

            const int bot_df_off =
                BOTDF_BATCH_STRIDE * b + BOTDF_CHANNEL_STRIDE * c_o + BOTDF_STRIDE * y + x;

            bot_df[bot_df_off] = out_val;
        }
    }

    for(; c_i < N_INPUTS + PRE_PAD; c_i++, c_o++)
    {
        accum_ratio = accum_ratio - ratio_dta[0];

        for(int i = 0; i < KERNEL_SIZE - 1; i++)
        {
            top_df_in[i] = top_df_in[i + 1];
            scale_in[i]  = scale_in[i + 1];
            ratio_dta[i] = ratio_dta[i + 1];
        }

        if(!low_channel_count || c_o < N_INPUTS)
        {
            const FLOAT bot_dta =
                bot[BOT_BATCH_STRIDE * b + BOT_CHANNEL_STRIDE * c_o + BOT_STRIDE * y + x];

            const FLOAT prv_scale = scale_in[PAD];
            const FLOAT exp_scale = miopen::detail::exp(-beta * miopen::detail::log(prv_scale));
            const FLOAT prv_accum_ratio = ratio * bot_dta * accum_ratio;
            const FLOAT out_val         = top_df_in[PAD] * exp_scale - prv_accum_ratio;

            const int bot_df_off =
                BOTDF_BATCH_STRIDE * b + BOTDF_CHANNEL_STRIDE * c_o + BOTDF_STRIDE * y + x;

            bot_df[bot_df_off] = out_val;
        }
    }
}
