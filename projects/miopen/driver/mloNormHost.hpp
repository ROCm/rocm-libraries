// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MLO_NORMHOST_H_
#define MLO_NORMHOST_H_

#include <cmath>

#include <common_utils/ford.hpp>

////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////

#ifndef MLO_LRN_WITHIN_CHANNEL
#define MLO_LRN_WITHIN_CHANNEL 0
#define MLO_LRN_ACROSS_CHANNELS 1
#endif

template <typename Tgpu_ /* the data type used in GPU computations (usually half) */,
          typename Tcheck_ /* the data type used in CPU checkings (usually double) */>
int mloLRNForwardRunHost(bool do_scale,
                         int norm_region,
                         int pad,
                         int local_area,
                         Tcheck_ alphaoverarea,
                         Tcheck_ alpha,
                         Tcheck_ beta,
                         Tcheck_ K,
                         int n_batches,
                         int n_outputs,
                         int n_inputs,
                         int bot_height,
                         int bot_width,
                         int bot_stride,
                         int bot_channel_stride,
                         int bot_batch_stride,
                         int top_height,
                         int top_width,
                         int top_v_stride,
                         int top_v_channel_stride,
                         int top_v_batch_stride,
                         int scale_v_stride,
                         int scale_v_channel_stride,
                         int scale_v_batch_stride,
                         const Tgpu_* bot_ptr,
                         Tcheck_* scale_v_ptr,
                         Tcheck_* top_v_ptr,
                         bool multi_threaded)
{
    int ret = 0;
    if(local_area < 1 + pad)
    {
        std::cout << "ERROR: Lrn kernel size is insufficient." << std::endl;
        return -1;
    }

    const int n_inputs_plus_pad              = n_inputs + pad;
    const int local_area_minus_one_minus_pad = local_area - 1 - pad;
    const int bot_height_plus_pad            = bot_height + pad;
    const int bot_width_plus_pad             = bot_width + pad;
    const size_t min_grain                   = multi_threaded ? 8 : n_batches;

    if(norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        miopen::par_for(n_batches, min_grain, [&](int b) {
            const int b_by_bot_batch_stride     = b * bot_batch_stride;
            const int b_by_scale_v_batch_stride = b * scale_v_batch_stride;
            const int b_by_top_v_batch_stride   = b * top_v_batch_stride;

            for(int j = 0; j < top_height; ++j)
            {
                const int j_by_bot_stride        = j * bot_stride;
                const int j_by_scale_v_stride    = j * scale_v_stride;
                const int j_by_top_v_stride      = j * top_v_stride;
                const int bot_ptr_base_idx_j     = b_by_bot_batch_stride + j_by_bot_stride;
                const int scale_v_ptr_base_idx_j = b_by_scale_v_batch_stride + j_by_scale_v_stride;
                const int top_v_ptr_base_idx_j   = b_by_top_v_batch_stride + j_by_top_v_stride;

                for(int i = 0; i < top_width; ++i)
                {
                    const int bot_ptr_base_idx_i     = bot_ptr_base_idx_j + i;
                    const int scale_v_ptr_base_idx_i = scale_v_ptr_base_idx_j + i;
                    const int top_v_ptr_base_idx_i   = top_v_ptr_base_idx_j + i;

                    // c-emulator
                    Tcheck_ accum_scale = Tcheck_{0};
                    int head            = 0;
                    Tcheck_ bot_val;
                    while(head < pad)
                    {
                        bot_val = (head < n_inputs)
                                      ? static_cast<Tcheck_>(
                                            bot_ptr[bot_ptr_base_idx_i + head * bot_channel_stride])
                                      : static_cast<Tcheck_>(0);
                        accum_scale += bot_val * bot_val;
                        ++head;
                    }
                    // until we reach size, nothing needs to be subtracted
                    while(head < local_area)
                    {
                        bot_val = (head < n_inputs)
                                      ? static_cast<Tcheck_>(
                                            bot_ptr[bot_ptr_base_idx_i + head * bot_channel_stride])
                                      : static_cast<Tcheck_>(0);
                        accum_scale += bot_val * bot_val;
                        const Tcheck_ scale      = K + accum_scale * alphaoverarea;
                        const int head_minus_pad = head - pad;
                        if(head_minus_pad >= 0 && head_minus_pad < n_outputs && do_scale)
                        {
                            scale_v_ptr[scale_v_ptr_base_idx_i +
                                        head_minus_pad * scale_v_channel_stride] = scale;
                        }
                        bot_val =
                            (head_minus_pad >= 0 && head_minus_pad < n_inputs)
                                ? static_cast<Tcheck_>(bot_ptr[bot_ptr_base_idx_i +
                                                               head_minus_pad * bot_channel_stride])
                                : static_cast<Tcheck_>(0);
                        Tcheck_ s     = pow(scale, -beta);
                        Tcheck_ c_val = bot_val * s;
                        if(head_minus_pad >= 0 && head_minus_pad < n_outputs)
                        {
                            top_v_ptr[top_v_ptr_base_idx_i +
                                      head_minus_pad * top_v_channel_stride] = c_val;
                        }
                        ++head;
                    }
                    // both add and subtract
                    while(head < n_inputs)
                    {
                        bot_val = static_cast<Tcheck_>(
                            bot_ptr[bot_ptr_base_idx_i + head * bot_channel_stride]);
                        accum_scale += bot_val * bot_val;
                        const int head_minus_local_area = head - local_area;
                        bot_val                         = (head_minus_local_area >= 0)
                                                              ? static_cast<Tcheck_>(
                                            bot_ptr[bot_ptr_base_idx_i +
                                                    head_minus_local_area * bot_channel_stride])
                                                              : static_cast<Tcheck_>(0);
                        accum_scale -= bot_val * bot_val;
                        const Tcheck_ scale      = K + accum_scale * alphaoverarea;
                        const int head_minus_pad = head - pad;
                        if(head_minus_pad >= 0 && do_scale)
                        {
                            scale_v_ptr[scale_v_ptr_base_idx_i +
                                        head_minus_pad * scale_v_channel_stride] = scale;
                        }
                        Tcheck_ s = pow(scale, -beta);
                        bot_val =
                            (head_minus_pad >= 0)
                                ? static_cast<Tcheck_>(bot_ptr[bot_ptr_base_idx_i +
                                                               head_minus_pad * bot_channel_stride])
                                : static_cast<Tcheck_>(0);
                        Tcheck_ c_val = bot_val * s;
                        if(head_minus_pad >= 0)
                        {
                            top_v_ptr[top_v_ptr_base_idx_i +
                                      head_minus_pad * top_v_channel_stride] = c_val;
                        }
                        ++head;
                    }
                    // subtract only
                    while(head < n_inputs_plus_pad)
                    {
                        const int head_minus_local_area = head - local_area;
                        bot_val = (head_minus_local_area >= 0 && head_minus_local_area < n_inputs)
                                      ? static_cast<Tcheck_>(
                                            bot_ptr[bot_ptr_base_idx_i +
                                                    head_minus_local_area * bot_channel_stride])
                                      : static_cast<Tcheck_>(0);
                        accum_scale -= bot_val * bot_val;
                        const Tcheck_ scale      = K + accum_scale * alphaoverarea;
                        const int head_minus_pad = head - pad;
                        if(head_minus_pad >= 0 && head_minus_pad < n_outputs && do_scale)
                        {
                            scale_v_ptr[scale_v_ptr_base_idx_i +
                                        head_minus_pad * scale_v_channel_stride] = scale;
                        }
                        bot_val =
                            (head_minus_pad >= 0 && head_minus_pad < n_inputs)
                                ? static_cast<Tcheck_>(bot_ptr[bot_ptr_base_idx_i +
                                                               head_minus_pad * bot_channel_stride])
                                : static_cast<Tcheck_>(0);
                        const Tcheck_ s     = pow(scale, -beta);
                        const Tcheck_ c_val = bot_val * s;
                        if(head_minus_pad >= 0 && head_minus_pad < n_outputs)
                        {
                            top_v_ptr[top_v_ptr_base_idx_i +
                                      head_minus_pad * top_v_channel_stride] = c_val;
                        }
                        ++head;
                    }
                } // for (int i = 0; i < top_width; i++)
            } // for (int j = 0; j < top_height; j++)
        }); // miopen::par_for
    }
    else
    {
        miopen::par_for(n_batches, min_grain, [&](int b) {
            const int b_by_bot_batch_stride     = b * bot_batch_stride;
            const int b_by_scale_v_batch_stride = b * scale_v_batch_stride;
            const int b_by_top_v_batch_stride   = b * top_v_batch_stride;

            for(int o = 0; o < n_outputs; ++o)
            {
                const int b_by_bot_batch_stride_plus_o_by_bot_channel_stride =
                    b_by_bot_batch_stride + o * bot_channel_stride;
                const int b_by_scale_v_batch_stride_plus_o_by_scale_v_channel_stride =
                    b_by_scale_v_batch_stride + o * scale_v_channel_stride;
                const int b_by_top_v_batch_stride_plus_o_by_top_v_channel_stride =
                    b_by_top_v_batch_stride + o * top_v_channel_stride;

                for(int j = 0; j < top_height; ++j)
                {
                    const int bot_ptr_base_idx_j =
                        b_by_bot_batch_stride_plus_o_by_bot_channel_stride + j * bot_stride;
                    const int scale_v_ptr_base_idx_j =
                        b_by_scale_v_batch_stride_plus_o_by_scale_v_channel_stride +
                        j * scale_v_stride;
                    const int top_v_ptr_base_idx_j =
                        b_by_top_v_batch_stride_plus_o_by_top_v_channel_stride + j * top_v_stride;

                    for(int i = 0; i < top_width; ++i)
                    {
                        // c-emulator
                        Tcheck_ scale     = static_cast<Tcheck_>(0);
                        int hstart        = j - local_area_minus_one_minus_pad;
                        int wstart        = i - local_area_minus_one_minus_pad;
                        int hend          = std::min(hstart + local_area, bot_height_plus_pad);
                        int wend          = std::min(wstart + local_area, bot_width_plus_pad);
                        int adj_area_size = (hend - hstart) * (wend - wstart);
                        hstart            = std::max(hstart, 0);
                        wstart            = std::max(wstart, 0);
                        hend              = std::min(hend, bot_height);
                        wend              = std::min(wend, bot_width);
                        Tcheck_ accum     = static_cast<Tcheck_>(0);
                        for(int h = hstart; h < hend; ++h)
                        {
                            const int bot_ptr_base_idx_h =
                                b_by_bot_batch_stride_plus_o_by_bot_channel_stride + h * bot_stride;

                            for(int w = wstart; w < wend; ++w)
                            {
                                Tcheck_ bot_val =
                                    static_cast<Tcheck_>(bot_ptr[bot_ptr_base_idx_h + w]);
                                accum += bot_val * bot_val;
                            }
                        }

                        scale = K + accum * (alpha / adj_area_size);

                        if(do_scale)
                        {
                            scale_v_ptr[scale_v_ptr_base_idx_j + i] = scale;
                        }

                        const Tcheck_ s = pow(scale, -beta);
                        const Tcheck_ bot_val =
                            static_cast<Tcheck_>(bot_ptr[bot_ptr_base_idx_j + i]);
                        const Tcheck_ c_val = bot_val * s;

                        top_v_ptr[top_v_ptr_base_idx_j + i] = c_val;
                    } // for (int i = 0; i < top_width; i++)
                } // for (int j = 0; j < top_height; j++)
            } // for (int o = 0; o < outputs; o++)
        }); // miopen::par_for
    } // (norm_region == ACROSS_CHANNELS)

    return ret;
}

template <typename Tgpu_ /* the data type used in GPU computations (usually half) */,
          typename Tcheck_ /* the data type used in CPU checkings (usually double) */>
int mloLRNBackwardRunHost(int norm_region,
                          int pad,
                          int local_area,
                          Tcheck_ /*alphaoverarea*/,
                          Tcheck_ alpha,
                          Tcheck_ beta,
                          Tcheck_ /*K*/,
                          int n_batches,
                          int /*n_outputs*/,
                          int n_inputs,
                          int bot_height,
                          int bot_width,
                          int bot_stride,
                          int bot_channel_stride,
                          int bot_batch_stride,
                          int bot_df_v_stride,
                          int bot_df_v_channel_stride,
                          int bot_df_v_batch_stride,
                          int top_height,
                          int top_width,
                          int top_stride,
                          int top_channel_stride,
                          int top_batch_stride,
                          int top_df_stride,
                          int top_df_channel_stride,
                          int top_df_batch_stride,
                          int scale_stride,
                          int scale_channel_stride,
                          int scale_batch_stride,
                          const Tgpu_* top_ptr,
                          const Tgpu_* top_df_ptr,
                          const Tgpu_* scale_ptr,
                          const Tgpu_* bot_ptr,
                          Tcheck_* bot_df_v_ptr,
                          bool multi_threaded)
{
    int ret                     = 0;
    const Tcheck_ negative_beta = -beta;
    const int pre_pad           = local_area - 1 - pad;
    if(pre_pad < 0)
    {
        std::cout << "ERROR: Lrn kernel size is insufficient." << std::endl;
        return -1;
    }

    const size_t min_grain = multi_threaded ? 8 : n_batches;

    // Precompute constant values used by the subsequent loops
    const Tcheck_ double_alpha_beta   = static_cast<Tcheck_>(2.) * alpha * beta;
    const int top_height_plus_pre_pad = top_height + pre_pad;
    const int top_width_plus_pre_pad  = top_width + pre_pad;
    const int n_inputs_plus_pre_pad   = n_inputs + pre_pad;

    if(norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        const Tcheck_ ratio_dta_bwd = double_alpha_beta / static_cast<Tcheck_>(local_area);

        miopen::par_for(n_batches, min_grain, [&](int b) {
            // Precompute constant values used by the subsequent loops
            const int b_by_top_df_batch_stride   = b * top_df_batch_stride;
            const int b_by_top_batch_stride      = b * top_batch_stride;
            const int b_by_scale_batch_stride    = b * scale_batch_stride;
            const int b_by_bot_df_v_batch_stride = b * bot_df_v_batch_stride;
            const int b_by_bot_batch_stride      = b * bot_batch_stride;

            for(int j = 0; j < bot_height; ++j)
            {
                // Precompute constant values used by the subsequent loops
                const int top_df_ptr_base_idx_j = b_by_top_df_batch_stride + j * top_df_stride;
                const int top_ptr_base_idx_j    = b_by_top_batch_stride + j * top_stride;
                const int scale_ptr_base_idx_j  = b_by_scale_batch_stride + j * scale_stride;
                const int bot_df_v_ptr_base_idx_j =
                    b_by_bot_df_v_batch_stride + j * bot_df_v_stride;
                const int bot_ptr_base_idx_j = b_by_bot_batch_stride + j * bot_stride;

                for(int i = 0; i < bot_width; ++i)
                {
                    // c-emulator
                    int head            = 0;
                    Tcheck_ accum_ratio = static_cast<Tcheck_>(0);

                    // Precompute constant values used by the subsequent loops
                    const int top_df_ptr_base_idx_i   = top_df_ptr_base_idx_j + i;
                    const int top_ptr_base_idx_i      = top_ptr_base_idx_j + i;
                    const int scale_ptr_base_idx_i    = scale_ptr_base_idx_j + i;
                    const int bot_df_v_ptr_base_idx_i = bot_df_v_ptr_base_idx_j + i;
                    const int bot_ptr_base_idx_i      = bot_ptr_base_idx_j + i;

                    // accumulate values
                    while(head < pre_pad)
                    {
                        if(head < n_inputs)
                        {
                            Tcheck_ adder =
                                (static_cast<Tcheck_>(top_df_ptr[top_df_ptr_base_idx_i +
                                                                 head * top_df_channel_stride]) *
                                 static_cast<Tcheck_>(
                                     top_ptr[top_ptr_base_idx_i + head * top_channel_stride])) /
                                static_cast<Tcheck_>(
                                    scale_ptr[scale_ptr_base_idx_i + head * scale_channel_stride]);

                            accum_ratio += adder;
                        }

                        ++head;
                    }

                    // until we reach size, nothing needs to be subtracted
                    while(head < local_area)
                    {
                        if(head < n_inputs)
                        {
                            Tcheck_ adder =
                                (static_cast<Tcheck_>(top_df_ptr[top_df_ptr_base_idx_i +
                                                                 head * top_df_channel_stride]) *
                                 static_cast<Tcheck_>(
                                     top_ptr[top_ptr_base_idx_i + head * top_channel_stride])) /
                                static_cast<Tcheck_>(
                                    scale_ptr[scale_ptr_base_idx_i + head * scale_channel_stride]);

                            accum_ratio += adder;
                        }

                        const int head_minus_pre_pad = head - pre_pad;

                        if(head_minus_pre_pad >= 0 && head_minus_pre_pad < n_inputs)
                        {
                            bot_df_v_ptr[bot_df_v_ptr_base_idx_i +
                                         head_minus_pre_pad * bot_df_v_channel_stride] =
                                static_cast<Tcheck_>(
                                    top_df_ptr[top_df_ptr_base_idx_i +
                                               head_minus_pre_pad * top_df_channel_stride]) *
                                    pow(static_cast<Tcheck_>(
                                            scale_ptr[scale_ptr_base_idx_i +
                                                      head_minus_pre_pad * scale_channel_stride]),
                                        negative_beta) -
                                ratio_dta_bwd *
                                    static_cast<Tcheck_>(
                                        bot_ptr[bot_ptr_base_idx_i +
                                                head_minus_pre_pad * bot_channel_stride]) *
                                    accum_ratio;
                        }
                        ++head;
                    }

                    // both add and subtract
                    while(head < n_inputs)
                    {

                        Tcheck_ adder =
                            static_cast<Tcheck_>(
                                top_df_ptr[top_df_ptr_base_idx_i + head * top_df_channel_stride]) *
                            static_cast<Tcheck_>(
                                top_ptr[top_ptr_base_idx_i + head * top_channel_stride]) /
                            static_cast<Tcheck_>(
                                scale_ptr[scale_ptr_base_idx_i + head * scale_channel_stride]);

                        accum_ratio += adder;

                        const int head_minus_local_area = head - local_area;

                        if(head_minus_local_area >= 0)
                        {
                            Tcheck_ subs =
                                (static_cast<Tcheck_>(
                                     top_df_ptr[top_df_ptr_base_idx_i +
                                                head_minus_local_area * top_df_channel_stride]) *
                                 static_cast<Tcheck_>(
                                     top_ptr[top_ptr_base_idx_i +
                                             head_minus_local_area * top_channel_stride])) /
                                static_cast<Tcheck_>(
                                    scale_ptr[scale_ptr_base_idx_i +
                                              head_minus_local_area * scale_channel_stride]);

                            accum_ratio -= subs;
                        }

                        const int head_minus_pre_pad = head - pre_pad;

                        if(head_minus_pre_pad >= 0)
                        {
                            bot_df_v_ptr[bot_df_v_ptr_base_idx_i +
                                         head_minus_pre_pad * bot_df_v_channel_stride] =
                                static_cast<Tcheck_>(
                                    top_df_ptr[top_df_ptr_base_idx_i +
                                               head_minus_pre_pad * top_df_channel_stride]) *
                                    pow(static_cast<Tcheck_>(
                                            scale_ptr[scale_ptr_base_idx_i +
                                                      head_minus_pre_pad * scale_channel_stride]),
                                        negative_beta) -
                                ratio_dta_bwd *
                                    static_cast<Tcheck_>(
                                        bot_ptr[bot_ptr_base_idx_i +
                                                head_minus_pre_pad * bot_channel_stride]) *
                                    accum_ratio;
                        }

                        ++head;
                    }

                    // subtract only
                    while(head < n_inputs_plus_pre_pad)
                    {
                        const int head_minus_local_area = head - local_area;

                        if(head_minus_local_area >= 0 && head_minus_local_area < n_inputs)
                        {
                            Tcheck_ subs =
                                (static_cast<Tcheck_>(
                                     top_df_ptr[top_df_ptr_base_idx_i +
                                                head_minus_local_area * top_df_channel_stride]) *
                                 static_cast<Tcheck_>(
                                     top_ptr[top_ptr_base_idx_i +
                                             head_minus_local_area * top_channel_stride])) /
                                static_cast<Tcheck_>(
                                    scale_ptr[scale_ptr_base_idx_i +
                                              head_minus_local_area * scale_channel_stride]);

                            accum_ratio -= subs;
                        }

                        const int head_minus_pre_pad = head - pre_pad;

                        if(head_minus_pre_pad >= 0 && head_minus_pre_pad < n_inputs)
                        {
                            bot_df_v_ptr[bot_df_v_ptr_base_idx_i +
                                         head_minus_pre_pad * bot_df_v_channel_stride] =
                                static_cast<Tcheck_>(
                                    top_df_ptr[top_df_ptr_base_idx_i +
                                               head_minus_pre_pad * top_df_channel_stride]) *
                                    pow(static_cast<Tcheck_>(
                                            scale_ptr[scale_ptr_base_idx_i +
                                                      head_minus_pre_pad * scale_channel_stride]),
                                        negative_beta) -
                                ratio_dta_bwd *
                                    static_cast<Tcheck_>(
                                        bot_ptr[bot_ptr_base_idx_i +
                                                head_minus_pre_pad * bot_channel_stride]) *
                                    accum_ratio;
                        }

                        ++head;
                    }
                } // for (int i = 0; i < bot_width; i++)
            } // for (int j = 0; j < bot_height; j++)
        }); // miopen::par_for
    } // if (norm_region == MLO_LRN_ACROSS_CHANNELS)
    else
    {
        miopen::par_for(n_batches, min_grain, [&](int b) {
            // Precompute constant values used by the subsequent loops
            const int b_by_top_df_batch_stride   = b * top_df_batch_stride;
            const int b_by_top_batch_stride      = b * top_batch_stride;
            const int b_by_scale_batch_stride    = b * scale_batch_stride;
            const int b_by_bot_df_v_batch_stride = b * bot_df_v_batch_stride;
            const int b_by_bot_batch_stride      = b * bot_batch_stride;

            for(int o = 0; o < n_inputs; ++o)
            {
                // Precompute constant values used by the subsequent loops
                const int top_df_ptr_base_idx_o =
                    b_by_top_df_batch_stride + o * top_df_channel_stride;
                const int top_ptr_base_idx_o   = b_by_top_batch_stride + o * top_channel_stride;
                const int scale_ptr_base_idx_o = b_by_scale_batch_stride + o * scale_channel_stride;
                const int bot_df_v_ptr_base_idx_o =
                    b_by_bot_df_v_batch_stride + o * bot_df_v_channel_stride;
                const int bot_ptr_base_idx_o = b_by_bot_batch_stride + o * bot_channel_stride;

                for(int j = 0; j < bot_height; ++j)
                {
                    // Precompute constant values used by the subsequent loops
                    const int bot_df_v_ptr_base_idx_j =
                        bot_df_v_ptr_base_idx_o + j * bot_df_v_stride;
                    const int top_df_ptr_base_idx_j = top_df_ptr_base_idx_o + j * top_df_stride;
                    const int scale_ptr_base_idx_j  = scale_ptr_base_idx_o + j * scale_stride;
                    const int bot_ptr_base_idx_j    = bot_ptr_base_idx_o + j * bot_stride;

                    for(int i = 0; i < bot_width; ++i)
                    {
                        Tcheck_ accum_ratio = static_cast<Tcheck_>(0);

                        int hstart        = j - pad;
                        int wstart        = i - pad;
                        int hend          = std::min(hstart + local_area, top_height_plus_pre_pad);
                        int wend          = std::min(wstart + local_area, top_width_plus_pre_pad);
                        int adj_area_size = (hend - hstart) * (wend - wstart);
                        hstart            = std::max(hstart, 0);
                        wstart            = std::max(wstart, 0);
                        hend              = std::min(hend, top_height);
                        wend              = std::min(wend, top_width);
                        for(int h = hstart; h < hend; ++h)
                        {
                            // Precompute constant values used by the subsequent loops
                            const int top_df_ptr_base_idx_h =
                                top_df_ptr_base_idx_o + h * top_df_stride;
                            const int top_ptr_base_idx_h = top_ptr_base_idx_o + h * top_stride;
                            const int scale_ptr_base_idx_h =
                                scale_ptr_base_idx_o + h * scale_stride;

                            for(int w = wstart; w < wend; ++w)
                            {
                                const Tcheck_ adder =
                                    static_cast<Tcheck_>(top_df_ptr[top_df_ptr_base_idx_h + w]) *
                                    static_cast<Tcheck_>(top_ptr[top_ptr_base_idx_h + w]) /
                                    static_cast<Tcheck_>(scale_ptr[scale_ptr_base_idx_h + w]);

                                accum_ratio += adder;
                            }
                        }

                        const Tcheck_ ratio_dta_bwd =
                            double_alpha_beta / static_cast<Tcheck_>(adj_area_size);

                        bot_df_v_ptr[bot_df_v_ptr_base_idx_j + i] =
                            static_cast<Tcheck_>(top_df_ptr[top_df_ptr_base_idx_j + i]) *
                                pow(static_cast<Tcheck_>(scale_ptr[scale_ptr_base_idx_j + i]),
                                    negative_beta) -
                            ratio_dta_bwd * static_cast<Tcheck_>(bot_ptr[bot_ptr_base_idx_j + i]) *
                                accum_ratio;
                    } // for(int i = 0; i < bot_width; ++i)
                } // for(int j = 0; j < bot_height; ++j)
            } // for(int o = 0; o < n_inputs; ++o)
        }); // miopen::par_for
    } // if (norm_region == MLO_LRN_ACROSS_CHANNELS)

    return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
