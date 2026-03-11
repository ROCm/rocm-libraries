// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "miopen_math.hpp"
#include "hip_math_ops.hpp"

using ReadType = typename miopen::mapped_vector_type<FLOAT, READ_UNIT>::type;

constexpr unsigned group_size_z = 1u;
constexpr unsigned left_pad0    = ((PRE_PAD0 + READ_UNIT - 1) / READ_UNIT) * READ_UNIT;
constexpr unsigned right_side =
    ((GROUP_SIZE_X * HORIZ_OUT_PIX + PAD0 + READ_UNIT - 1) / READ_UNIT) * READ_UNIT;
constexpr unsigned data_width        = left_pad0 + right_side;
constexpr unsigned local_read4       = data_width / READ_UNIT;
constexpr unsigned local_data_height = GROUP_SIZE_Y * VERT_OUT_PIX + KERNEL_SIZE - 1;
constexpr bool low_channel_count     = N_INPUTS < KERNEL_SIZE;

__launch_bounds__(GROUP_SIZE_X* GROUP_SIZE_Y* group_size_z) extern "C" __global__
    void MIOpenLRNWithinChannel_PS(const FLOAT* bot,
                                   FLOAT* top,
#if DO_SCALE
                                   FLOAT* scale,
#endif
                                   FLOAT alphaoverarea,
                                   FLOAT beta,
                                   FLOAT K)
{
    // Taken from POOLING AVE with stride = 1
    __shared__ FLOAT bot_data[data_width * local_data_height];
    const int x       = blockIdx.x * GROUP_SIZE_X * HORIZ_OUT_PIX;
    const int y       = blockIdx.y * GROUP_SIZE_Y * VERT_OUT_PIX;
    const int lcl_id0 = threadIdx.x;
    const int lcl_id1 = threadIdx.y;
    const int ob      = blockIdx.z * group_size_z + threadIdx.z; // output * batch_sz
    const int o       = iDiv(ob, BATCH_SIZE);
    const int b       = iMod(ob, o, BATCH_SIZE);
    const int bot_x   = x;
    const int bot_y   = y;
    const int bot_off = b * BOT_BATCH_STRIDE + o * BOT_CHANNEL_STRIDE;

    // load tile
    for(int b_j = lcl_id1; b_j < local_data_height; b_j += GROUP_SIZE_Y)
    {
        const int bot_y_act   = bot_y + b_j - PRE_PAD1;
        const bool invisibleY = (bot_y_act < 0) || (bot_y_act >= BOT_HEIGHT);
        const int bot_y_off   = bot_y_act * BOT_STRIDE;
        const int lcl_off_v   = __mul24(b_j, (int)data_width);

        for(int b_i = lcl_id0; b_i < local_read4; b_i += GROUP_SIZE_X)
        {
            const int bot_x_act = bot_x + (b_i * READ_UNIT) - left_pad0;
            for(int i = 0; i < READ_UNIT; ++i)
            {
                int bot_off_x         = bot_off + bot_y_off + bot_x_act + i;
                const bool invisibleX = (bot_x_act + i < 0) || (bot_x_act + i >= BOT_WIDTH);
                bot_off_x             = (invisibleX || invisibleY) ? 0 : bot_off_x;

                FLOAT bot_val = bot[bot_off_x];
                bot_val       = (invisibleX || invisibleY) ? 0 : bot_val;

                bot_data[lcl_off_v + (b_i * READ_UNIT) + i] = bot_val;
            }
        }
    }

    __syncthreads();
#if HORIZ_OUT_PIX > 1
    FLOAT partial_sum_x[HORIZ_OUT_PIX - 1]; // horizontal partial sum
#endif
#if VERT_OUT_PIX > 1
    FLOAT partial_sum_xy[VERT_OUT_PIX - 1][HORIZ_OUT_PIX]; // horizontal-vertical partial sums.
#endif
    FLOAT accum[VERT_OUT_PIX][HORIZ_OUT_PIX]; // accumulator

    const int top_y = __mul24(lcl_id1, VERT_OUT_PIX) + y;
    const int top_x = __mul24(lcl_id0, HORIZ_OUT_PIX) + x;

    const int lcl_y   = __mul24(lcl_id1, VERT_OUT_PIX);
    const int lcl_x   = __mul24(lcl_id0, HORIZ_OUT_PIX) + (left_pad0 - PRE_PAD0);
    const int lcl_off = __mul24(lcl_y, data_width) + lcl_x;

    for(int j = 0; j < VERT_OUT_PIX; ++j)
    {
        for(int i = 0; i < HORIZ_OUT_PIX; ++i)
        {
            accum[j][i] = 0;
        }
    }

#if VERT_OUT_PIX > 1
    for(int j = 0; j < VERT_OUT_PIX - 1; ++j)
    {
        for(int i = 0; i < HORIZ_OUT_PIX; ++i)
        {
            partial_sum_xy[j][i] = 0;
        }
    }
#endif

    // running window summation
    int jj = 0;
    int ii = 0;

    // first to get vertical partial sums
#if VERT_OUT_PIX > 1
    for(; jj < (VERT_OUT_PIX - 1); ++jj)
    {
        for(ii = 0; ii < (HORIZ_OUT_PIX - 1); ++ii)
        {
            const FLOAT bot_val   = bot_data[lcl_off + jj * data_width + ii];
            const FLOAT accum_tmp = bot_val * bot_val;

#if HORIZ_OUT_PIX > 1
            // save horizontal partial sums
            partial_sum_x[ii] = accum_tmp;
#endif
            // accumulate in vert-horizontal(0)
            partial_sum_xy[jj][0] += accum_tmp;
        }

        for(; ii < KERNEL_SIZE0; ++ii)
        {
            const FLOAT bot_val   = bot_data[lcl_off + jj * data_width + ii];
            const FLOAT accum_tmp = bot_val * bot_val;
            // accumulate in vert horizontal(0)
            partial_sum_xy[jj][0] += accum_tmp;
        }

        // running horizontal window
        for(; ii < (KERNEL_SIZE0 + HORIZ_OUT_PIX - 1); ++ii)
        {
            const FLOAT bot_val   = bot_data[lcl_off + jj * data_width + ii];
            const FLOAT accum_tmp = bot_val * bot_val;
            // calculate all vertical-horizontal partial sums
            partial_sum_xy[jj][ii - KERNEL_SIZE0 + 1] =
                partial_sum_xy[jj][ii - KERNEL_SIZE0] + (accum_tmp
#if HORIZ_OUT_PIX > 1
                                                         - partial_sum_x[ii - KERNEL_SIZE0]
#endif
                                                        );
        }

        // put into accumulator[0][i]
        // whatever has been accumulated so far
        for(int i = 0; i < HORIZ_OUT_PIX; ++i)
        {
            accum[0][i] += partial_sum_xy[jj][i];
        }
    }
#endif

    FLOAT mov_accum;
    // calculate row 0 accumulators
    for(; jj < KERNEL_SIZE1; ++jj)
    {
        mov_accum = 0;

        for(ii = 0; ii < (HORIZ_OUT_PIX - 1); ++ii)
        {
            const FLOAT bot_val   = bot_data[lcl_off + jj * data_width + ii];
            const FLOAT accum_tmp = bot_val * bot_val;
#if HORIZ_OUT_PIX > 1
            partial_sum_x[ii] = accum_tmp;
#endif
            mov_accum += accum_tmp;
        }

        for(; ii < KERNEL_SIZE0; ++ii)
        {
            const FLOAT bot_val   = bot_data[lcl_off + jj * data_width + ii];
            const FLOAT accum_tmp = bot_val * bot_val;
            mov_accum += accum_tmp;
        }

        accum[0][0] += mov_accum;
        // running horizontal window
        for(; ii < (KERNEL_SIZE0 + HORIZ_OUT_PIX - 1); ++ii)
        {
            const FLOAT bot_val   = bot_data[lcl_off + jj * data_width + ii];
            const FLOAT accum_tmp = bot_val * bot_val;
            // running horizontal window
            mov_accum += (accum_tmp
#if HORIZ_OUT_PIX > 1
                          - partial_sum_x[ii - KERNEL_SIZE0]
#endif
            );
            accum[0][ii - KERNEL_SIZE0 + 1] += mov_accum;
        }
    }

    // accumulate all other rows besides 0
    for(; jj < (KERNEL_SIZE1 + VERT_OUT_PIX - 1); ++jj)
    {
        // first running horizontal winodw as before
        mov_accum = 0;
        for(ii = 0; ii < (HORIZ_OUT_PIX - 1); ++ii)
        {
            const FLOAT bot_val   = bot_data[lcl_off + jj * data_width + ii];
            const FLOAT accum_tmp = bot_val * bot_val;
#if HORIZ_OUT_PIX > 1
            partial_sum_x[ii] = accum_tmp;
#endif
            accum[jj - KERNEL_SIZE1 + 1][0] += accum_tmp;
        }
        for(; ii < KERNEL_SIZE0; ++ii)
        {
            const FLOAT bot_val   = bot_data[lcl_off + jj * data_width + ii];
            const FLOAT accum_tmp = bot_val * bot_val;
            accum[jj - KERNEL_SIZE1 + 1][0] += accum_tmp;
        }
        // running horizontal window
        int ii1 = ii;
        for(; ii < (KERNEL_SIZE0 + HORIZ_OUT_PIX - 1); ++ii)
        {
            const FLOAT bot_val   = bot_data[lcl_off + jj * data_width + ii];
            const FLOAT accum_tmp = bot_val * bot_val;

            accum[jj - KERNEL_SIZE1 + 1][ii - KERNEL_SIZE0 + 1] =
                accum[jj - KERNEL_SIZE1 + 1][ii - KERNEL_SIZE0] + accum_tmp;
#if HORIZ_OUT_PIX > 1
            accum[jj - KERNEL_SIZE1 + 1][ii - KERNEL_SIZE0 + 1] -= partial_sum_x[ii - KERNEL_SIZE0];
#endif
        }

        // finally running vertical window
        for(ii = ii1; ii < (KERNEL_SIZE0 + HORIZ_OUT_PIX - 1); ++ii)
        {
            // finish horizontal summation
            // add/substarct vertical patial sum
            accum[jj - KERNEL_SIZE1 + 1][ii - KERNEL_SIZE0 + 1] +=
                accum[jj - KERNEL_SIZE1][ii - KERNEL_SIZE0 + 1];
#if VERT_OUT_PIX > 1
            accum[jj - KERNEL_SIZE1 + 1][ii - KERNEL_SIZE0 + 1] -=
                partial_sum_xy[jj - KERNEL_SIZE1][ii - KERNEL_SIZE0 + 1];
#endif
        }
#if VERT_OUT_PIX > 1
        accum[jj - KERNEL_SIZE1 + 1][0] -= partial_sum_xy[jj - KERNEL_SIZE1][0];
#endif
        accum[jj - KERNEL_SIZE1 + 1][0] += accum[jj - KERNEL_SIZE1][0];
    }

    // normalization
    FLOAT prv_scale[VERT_OUT_PIX][HORIZ_OUT_PIX];
    const FLOAT adj_alphaoverarea = alphaoverarea;
    for(int k = 0; k < VERT_OUT_PIX; k++)
    {
        for(int l = 0; l < HORIZ_OUT_PIX; l++)
        {
            prv_scale[k][l] = K + accum[k][l] * adj_alphaoverarea;
        }
    }

    const int top_off = b * TOP_BATCH_STRIDE + o * TOP_CHANNEL_STRIDE + top_y * TOP_STRIDE + top_x;
#if DO_SCALE
    int scale_off =
        b * SCALE_BATCH_STRIDE + o * SCALE_CHANNEL_STRIDE + top_y * SCALE_STRIDE + top_x;
#endif

    /*
       The HIP compiler doesn't automatically unroll this nested loop so we need to
       use pragma unroll to encourage that for better performance. Additionally, when access is not
       aligned in the horizontal or vertical access we lift the if condition out of the loop
       termination if the height/width is a multiple of the vert/horiz pixels, enabling SIMD
       vectorizatiom of the loop body.
       These optimizations were being done automatically by the OpenCL compiler prior to porting to
       HIP.
    */
    constexpr bool is_vert_aligned        = (VERT_ALIGNED == 1);
    constexpr bool is_horiz_aligned       = (HORIZ_ALIGNED == 1);
    constexpr bool sink_vert_align_check  = !is_vert_aligned && ((TOP_HEIGHT % VERT_OUT_PIX) != 0);
    constexpr bool sink_horiz_align_check = !is_horiz_aligned && ((TOP_WIDTH % HORIZ_OUT_PIX) != 0);
    constexpr bool lift_vert_align_check  = !is_vert_aligned && ((TOP_HEIGHT % VERT_OUT_PIX) == 0);
    constexpr bool lift_horiz_align_check = !is_horiz_aligned && ((TOP_WIDTH % HORIZ_OUT_PIX) == 0);

    // final output
    if(!lift_vert_align_check || (lift_vert_align_check && (top_y < TOP_HEIGHT)))
    {
#pragma unroll VERT_OUT_PIX
        for(int k = 0; k < VERT_OUT_PIX; k++)
        {
            if(sink_vert_align_check && !(top_y + k < TOP_HEIGHT))
            {
                break;
            }

            if(!lift_horiz_align_check || (lift_horiz_align_check && (top_x < TOP_WIDTH)))
            {
#pragma unroll HORIZ_OUT_PIX
                for(int l = 0; l < HORIZ_OUT_PIX; l++)
                {
                    if(sink_horiz_align_check && !(top_x + l < TOP_WIDTH))
                    {
                        break;
                    }
                    const FLOAT s =
                        miopen::detail::exp(FLOAT(-beta) * miopen::detail::log(prv_scale[k][l]));
                    int offset    = __mul24((k + PRE_PAD1), data_width) + (l + PRE_PAD0);
                    FLOAT bot_val = bot_data[lcl_off + offset];
#if DO_SCALE
                    scale[scale_off + k * SCALE_STRIDE + l] = prv_scale[k][l];
#endif
                    top[top_off + k * TOP_STRIDE + l] = bot_val * s;
                }
            }
        }
    }
}

__launch_bounds__(GROUP_SIZE_X* GROUP_SIZE_Y* group_size_z) extern "C" __global__
    void MIOpenLRNAcrossChannels4(const FLOAT* bottom,
                                  FLOAT* top,
#if DO_SCALE
                                  FLOAT* scale,
#endif
                                  FLOAT alphaoverarea,
                                  FLOAT beta,
                                  FLOAT K)
{
    const int pix_id = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    const int batch  = blockIdx.z * group_size_z + threadIdx.z;
    ReadType accum(0);
    ReadType bot_in2[KERNEL_SIZE];
    ReadType bot_in[KERNEL_SIZE];
    for(int i = 0; i < KERNEL_SIZE; ++i)
    {
        bot_in2[i] = ReadType(0);
        bot_in[i]  = ReadType(0);
    }

    int top_off = 0;
#if DO_SCALE
    int scale_off;
#endif

    int c_i = 0, c_o = 0; // accumulated throughout the kernel
    for(; c_i < PAD; c_i++)
    {
        ReadType prv_in(0);
        if(!low_channel_count || c_i < N_INPUTS)
        {
            const auto offset =
                BOT_BATCH_STRIDE * batch + BOT_CHANNEL_STRIDE * c_i + (pix_id * READ_UNIT);
            // if the last one
            if(C1x1_PIXLEFT > 0 && (pix_id == MAP_SZ_4 - 1))
            {
                for(int j = 0; j < C1x1_PIXLEFT; ++j)
                {
                    FLOAT* prv_in_as_scalar = reinterpret_cast<FLOAT*>(&prv_in);
                    prv_in_as_scalar[j]     = bottom[offset + j];
                }
            }
            else
            {
                const FLOAT* bottom_offset = bottom + offset;
                prv_in                     = *reinterpret_cast<const ReadType*>(bottom_offset);
            }
        }

        bot_in2[c_i] = prv_in * prv_in;
        bot_in[c_i]  = prv_in;
        accum        = accum + bot_in2[c_i];
    }

    for(; c_i < KERNEL_SIZE; c_i++, c_o++)
    {
        ReadType prv_in(0);
        if(!low_channel_count || c_i < N_INPUTS)
        {
            const auto offset =
                BOT_BATCH_STRIDE * batch + BOT_CHANNEL_STRIDE * c_i + (pix_id * READ_UNIT);
            if(C1x1_PIXLEFT > 0 && (pix_id == MAP_SZ_4 - 1))
            {
                for(int j = 0; j < C1x1_PIXLEFT; ++j)
                {
                    FLOAT* prv_in_as_scalar = reinterpret_cast<FLOAT*>(&prv_in);
                    prv_in_as_scalar[j]     = bottom[offset + j];
                }
            }
            else
            {
                const FLOAT* bottom_offset = bottom + offset;
                prv_in                     = *reinterpret_cast<const ReadType*>(bottom_offset);
            }
        }

        bot_in2[c_i] = prv_in * prv_in;
        bot_in[c_i]  = prv_in;
        accum        = accum + bot_in2[c_i];

        top_off = batch * TOP_BATCH_STRIDE + c_o * TOP_CHANNEL_STRIDE + (pix_id * READ_UNIT);
#if DO_SCALE
        scale_off = batch * SCALE_BATCH_STRIDE + c_o * SCALE_CHANNEL_STRIDE + (pix_id * READ_UNIT);
#endif
        const ReadType prv_scale     = (ReadType(K) + accum * ReadType(alphaoverarea));
        const ReadType prv_scale_log = miopen::log<ReadType>(prv_scale);
        const ReadType exp_scale     = miopen::exp<ReadType>(ReadType(-beta) * prv_scale_log);
        const ReadType prv_out       = bot_in[c_o];
        const ReadType out_val       = prv_out * exp_scale;
        if(!low_channel_count || c_o < N_INPUTS)
        {
            // if the last one
            if(C1x1_PIXLEFT > 0 && (pix_id == MAP_SZ_4 - 1))
            {
                for(int j = 0; j < C1x1_PIXLEFT; ++j)
                {
                    const FLOAT* out_val_as_scalar = reinterpret_cast<const FLOAT*>(&out_val);
                    top[top_off + j]               = out_val_as_scalar[j];
#if DO_SCALE
                    const FLOAT* prv_scale_as_scalar = reinterpret_cast<const FLOAT*>(&prv_scale);
                    scale[scale_off + j]             = prv_scale_as_scalar[j];
#endif
                }
            }
            else
            {
                FLOAT* top_offset                        = top + top_off;
                *reinterpret_cast<ReadType*>(top_offset) = out_val;

#if DO_SCALE
                FLOAT* scale_offset                        = scale + top_off;
                *reinterpret_cast<ReadType*>(scale_offset) = prv_scale;
#endif
            }
        }
    }

    for(; c_i < N_INPUTS; c_i++, c_o++)
    {
        ReadType prv_in(0);
        auto offset = BOT_BATCH_STRIDE * batch + BOT_CHANNEL_STRIDE * c_i + (pix_id * READ_UNIT);
        // if the last one
        if(C1x1_PIXLEFT > 0 && (pix_id == MAP_SZ_4 - 1))
        {
            for(int j = 0; j < C1x1_PIXLEFT; ++j)
            {
                FLOAT* prv_in_as_scalar = reinterpret_cast<FLOAT*>(&prv_in);
                prv_in_as_scalar[j]     = bottom[offset + j];
            }
        }
        else
        {
            const FLOAT* bottom_offset = bottom + offset;
            prv_in                     = *reinterpret_cast<const ReadType*>(bottom_offset);
        }

        const ReadType prv_bot_in2 = prv_in * prv_in;
        accum                      = accum + prv_bot_in2;
        accum                      = accum - bot_in2[0];
        for(int i = 0; i < KERNEL_SIZE - 1; i++)
        {
            bot_in2[i] = bot_in2[i + 1];
            bot_in[i]  = bot_in[i + 1];
        }

        bot_in2[KERNEL_SIZE - 1] = prv_bot_in2;
        bot_in[KERNEL_SIZE - 1]  = prv_in;

        top_off = batch * TOP_BATCH_STRIDE + c_o * TOP_CHANNEL_STRIDE + (pix_id * READ_UNIT);
#if DO_SCALE
        scale_off = batch * SCALE_BATCH_STRIDE + c_o * SCALE_CHANNEL_STRIDE + (pix_id * READ_UNIT);
#endif
        const ReadType prv_scale     = (ReadType(K) + accum * ReadType(alphaoverarea));
        const ReadType prv_scale_log = miopen::log<ReadType>(prv_scale);
        const ReadType exp_scale     = miopen::exp<ReadType>(ReadType(-beta) * prv_scale_log);
        const ReadType prv_out       = bot_in[PRE_PAD];
        const ReadType out_val       = prv_out * exp_scale;
        if(!low_channel_count || c_o < N_INPUTS)
        {
            // if the last one
            if(C1x1_PIXLEFT > 0 && (pix_id == MAP_SZ_4 - 1))
            {
                for(int j = 0; j < C1x1_PIXLEFT; ++j)
                {
                    const FLOAT* out_val_as_scalar = reinterpret_cast<const FLOAT*>(&out_val);
                    top[top_off + j]               = out_val_as_scalar[j];
#if DO_SCALE
                    const FLOAT* prv_scale_as_scalar = reinterpret_cast<const FLOAT*>(&prv_scale);
                    scale[scale_off + j]             = prv_scale_as_scalar[j];
#endif
                }
            }
            else
            {
                FLOAT* top_offset                        = top + top_off;
                *reinterpret_cast<ReadType*>(top_offset) = out_val;

#if DO_SCALE
                FLOAT* scale_offset                        = scale + top_off;
                *reinterpret_cast<ReadType*>(scale_offset) = prv_scale;
#endif
            }
        }
    }

    for(; c_i < N_INPUTS + PAD; c_i++, c_o++)
    {
        accum = accum - bot_in2[0];
        for(int i = 0; i < KERNEL_SIZE - 1; i++)
        {
            bot_in2[i] = bot_in2[i + 1];
            bot_in[i]  = bot_in[i + 1];
        }

        top_off = batch * TOP_BATCH_STRIDE + c_o * TOP_CHANNEL_STRIDE + (pix_id * READ_UNIT);
#if DO_SCALE
        scale_off = batch * SCALE_BATCH_STRIDE + c_o * SCALE_CHANNEL_STRIDE + (pix_id * READ_UNIT);
#endif
        const ReadType prv_scale     = (ReadType(K) + accum * ReadType(alphaoverarea));
        const ReadType prv_scale_log = miopen::log<ReadType>(prv_scale);
        const ReadType exp_scale     = miopen::exp<ReadType>(ReadType(-beta) * prv_scale_log);
        const ReadType prv_out       = bot_in[PRE_PAD];
        const ReadType out_val       = prv_out * exp_scale;
        if(!low_channel_count || c_o < N_INPUTS)
        {
            // if the last one
            if(C1x1_PIXLEFT > 0 && (pix_id == MAP_SZ_4 - 1))
            {
                for(int j = 0; j < C1x1_PIXLEFT; ++j)
                {
                    const FLOAT* out_val_as_scalar = reinterpret_cast<const FLOAT*>(&out_val);
                    top[top_off + j]               = out_val_as_scalar[j];
#if DO_SCALE
                    const FLOAT* prv_scale_as_scalar = reinterpret_cast<const FLOAT*>(&prv_scale);
                    scale[scale_off + j]             = prv_scale_as_scalar[j];
#endif
                }
            }
            else
            {
                FLOAT* top_offset                        = top + top_off;
                *reinterpret_cast<ReadType*>(top_offset) = out_val;
#if DO_SCALE
                FLOAT* scale_offset                        = scale + top_off;
                *reinterpret_cast<ReadType*>(scale_offset) = prv_scale;
#endif
            }
        }
    }
}
