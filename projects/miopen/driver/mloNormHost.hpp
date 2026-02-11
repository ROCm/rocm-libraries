// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MLO_NORMHOST_H_
#define MLO_NORMHOST_H_

#include <cmath>
#include <iomanip>
#include <typeinfo>

#include <miopen/ford.hpp>

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
    const auto t0 = std::chrono::high_resolution_clock::now();
    int ret = 0;
    if(local_area < 1 + pad)
    {
        std::cout << "ERROR: Lrn kernel size is insufficient." << std::endl;
        std::ofstream("/data/Dev/LRN-" + std::string{typeid(Tgpu_).name()} + "-" + (multi_threaded ? "MT" : "ST") + "-Forward.csv", std::ofstream::app) << "ERROR: Lrn kernel size is insufficient." << std::endl;
        return -1;
    }

    const size_t min_grain = multi_threaded ? 8 : n_batches;

    if(norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        miopen::par_for(n_batches, min_grain, [&](int b) {
            for(int j = 0; j < top_height; j++)
            {
                for(int i = 0; i < top_width; i++)
                {
                    // c-emulator
                    Tcheck_ accum_scale = Tcheck_{0};
                    int head            = 0;
                    Tcheck_ bot_val;
                    while(head < pad)
                    {
                        bot_val = (head < n_inputs)
                                      ? static_cast<Tcheck_>(
                                            bot_ptr[b * bot_batch_stride +
                                                    head * bot_channel_stride + j * bot_stride + i])
                                      : static_cast<Tcheck_>(0);
                        accum_scale += bot_val * bot_val;
                        ++head;
                    }
                    // until we reach size, nothing needs to be subtracted
                    while(head < local_area)
                    {
                        bot_val = (head < n_inputs)
                                      ? static_cast<Tcheck_>(
                                            bot_ptr[b * bot_batch_stride +
                                                    head * bot_channel_stride + j * bot_stride + i])
                                      : static_cast<Tcheck_>(0);
                        accum_scale += bot_val * bot_val;
                        Tcheck_ scale = K + accum_scale * alphaoverarea;
                        if((head - pad) >= 0 && (head - pad) < n_outputs && do_scale)
                        {
                            scale_v_ptr[b * scale_v_batch_stride +
                                        (head - pad) * scale_v_channel_stride + j * scale_v_stride +
                                        i] = scale;
                        }
                        bot_val =
                            ((head - pad) >= 0 && (head - pad) < n_inputs)
                                ? static_cast<Tcheck_>(bot_ptr[b * bot_batch_stride +
                                                               (head - pad) * bot_channel_stride +
                                                               j * bot_stride + i])
                                : static_cast<Tcheck_>(0);
                        Tcheck_ s     = pow(scale, -beta);
                        Tcheck_ c_val = bot_val * s;
                        if((head - pad) >= 0 && (head - pad) < n_outputs)
                        {
                            top_v_ptr[b * top_v_batch_stride + (head - pad) * top_v_channel_stride +
                                      j * top_v_stride + i] = c_val;
                        }
                        ++head;
                    }
                    // both add and subtract
                    while(head < n_inputs)
                    {
                        bot_val = static_cast<Tcheck_>(
                            bot_ptr[b * bot_batch_stride + head * bot_channel_stride +
                                    j * bot_stride + i]);
                        accum_scale += bot_val * bot_val;
                        bot_val = ((head - local_area) >= 0)
                                      ? static_cast<Tcheck_>(
                                            bot_ptr[b * bot_batch_stride +
                                                    (head - local_area) * bot_channel_stride +
                                                    j * bot_stride + i])
                                      : static_cast<Tcheck_>(0);
                        accum_scale -= bot_val * bot_val;
                        Tcheck_ scale = K + accum_scale * alphaoverarea;
                        if((head - pad) >= 0 && do_scale)
                        {
                            scale_v_ptr[b * scale_v_batch_stride +
                                        (head - pad) * scale_v_channel_stride + j * scale_v_stride +
                                        i] = scale;
                        }
                        Tcheck_ s = pow(scale, -beta);
                        bot_val =
                            ((head - pad) >= 0)
                                ? static_cast<Tcheck_>(bot_ptr[b * bot_batch_stride +
                                                               (head - pad) * bot_channel_stride +
                                                               j * bot_stride + i])
                                : static_cast<Tcheck_>(0);
                        Tcheck_ c_val = bot_val * s;
                        if((head - pad) >= 0)
                        {
                            top_v_ptr[b * top_v_batch_stride + (head - pad) * top_v_channel_stride +
                                      j * top_v_stride + i] = c_val;
                        }
                        ++head;
                    }
                    // subtract only
                    while(head < n_inputs + pad)
                    {
                        bot_val = ((head - local_area) >= 0 && (head - local_area) < n_inputs)
                                      ? static_cast<Tcheck_>(
                                            bot_ptr[b * bot_batch_stride +
                                                    (head - local_area) * bot_channel_stride +
                                                    j * bot_stride + i])
                                      : static_cast<Tcheck_>(0);
                        accum_scale -= bot_val * bot_val;
                        Tcheck_ scale = K + accum_scale * alphaoverarea;
                        if((head - pad) >= 0 && (head - pad) < n_outputs && do_scale)
                        {
                            scale_v_ptr[b * scale_v_batch_stride +
                                        (head - pad) * scale_v_channel_stride + j * scale_v_stride +
                                        i] = scale;
                        }
                        bot_val =
                            ((head - pad) >= 0 && (head - pad) < n_inputs)
                                ? static_cast<Tcheck_>(bot_ptr[b * bot_batch_stride +
                                                               (head - pad) * bot_channel_stride +
                                                               j * bot_stride + i])
                                : static_cast<Tcheck_>(0);
                        Tcheck_ s     = pow(scale, -beta);
                        Tcheck_ c_val = bot_val * s;
                        if((head - pad) >= 0 && (head - pad) < n_outputs)
                        {
                            top_v_ptr[b * top_v_batch_stride + (head - pad) * top_v_channel_stride +
                                      j * top_v_stride + i] = c_val;
                        }
                        ++head;
                    }

                } // for (int i = 0; i < top_width; i++)
            } // for (int j = 0; j < top_height; j++)
        }); // miopen::par_for(n_batches, min_grain, [&](int b) {
    }
    else
    {
        miopen::par_for(n_batches, min_grain, [&](int b) {
            for(int o = 0; o < n_outputs; o++)
            {
                for(int j = 0; j < top_height; j++)
                {
                    for(int i = 0; i < top_width; i++)
                    {
                        // c-emulator
                        Tcheck_ scale     = static_cast<Tcheck_>(0);
                        int hstart        = j - (local_area - 1 - pad);
                        int wstart        = i - (local_area - 1 - pad);
                        int hend          = std::min(hstart + local_area, bot_height + pad);
                        int wend          = std::min(wstart + local_area, bot_width + pad);
                        int adj_area_size = (hend - hstart) * (wend - wstart);
                        hstart            = std::max(hstart, 0);
                        wstart            = std::max(wstart, 0);
                        hend              = std::min(hend, bot_height);
                        wend              = std::min(wend, bot_width);
                        Tcheck_ accum     = static_cast<Tcheck_>(0);
                        for(int h = hstart; h < hend; ++h)
                        {
                            for(int w = wstart; w < wend; ++w)
                            {

                                Tcheck_ bot_val = static_cast<Tcheck_>(
                                    bot_ptr[b * bot_batch_stride + o * bot_channel_stride +
                                            h * bot_stride + w]);
                                accum += bot_val * bot_val;
                            }
                        }

                        alphaoverarea = alpha / adj_area_size;
                        scale         = K + accum * alphaoverarea;
                        if(do_scale)
                        {
                            scale_v_ptr[b * scale_v_batch_stride + o * scale_v_channel_stride +
                                        j * scale_v_stride + i] = scale;
                        }

                        Tcheck_ s       = pow(scale, -beta);
                        Tcheck_ bot_val = static_cast<Tcheck_>(
                            bot_ptr[b * bot_batch_stride + o * bot_channel_stride + j * bot_stride +
                                    i]);
                        Tcheck_ c_val = bot_val * s;

                        top_v_ptr[b * top_v_batch_stride + o * top_v_channel_stride +
                                  j * top_v_stride + i] = c_val;

                    } // for (int i = 0; i < top_width; i++)
                } // for (int j = 0; j < top_height; j++)
            } // for (int o = 0; o < outputs; o++)
        }); // miopen::par_for(n_batches, min_grain, [&](int b) {
    } // (norm_region == ACROSS_CHANNELS)
    
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::ofstream("/data/Dev/LRN-" + std::string{typeid(Tgpu_).name()} + "-" + (multi_threaded ? "MT" : "ST") + "-Forward.csv", std::ofstream::app) << ns << std::endl;

    return (ret);
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
    const auto t0 = std::chrono::high_resolution_clock::now();
    int ret                     = 0;
    const Tcheck_ negative_beta = -beta;
    const int pre_pad           = local_area - 1 - pad;
    if(pre_pad < 0)
    {
        std::cout << "ERROR: Lrn kernel size is insufficient." << std::endl;
        std::ofstream("/data/Dev/LRN-" + std::string{typeid(Tgpu_).name()} + "-" + (multi_threaded ? "MT" : "ST") + "-Backward.csv", std::ofstream::app) << "ERROR: Lrn kernel size is insufficient." << std::endl;
        return -1;
    }

    const size_t min_grain = multi_threaded ? 8 : n_batches;

    if(norm_region == MLO_LRN_ACROSS_CHANNELS)
    {
        const Tcheck_ ratio_dta_bwd =
            static_cast<Tcheck_>(2.) * alpha * beta / static_cast<Tcheck_>(local_area);

        miopen::par_for(n_batches, min_grain, [&](int b) {
            for(int j = 0; j < bot_height; j++)
            {
                for(int i = 0; i < bot_width; i++)
                {

                    // c-emulator
                    int head            = 0;
                    Tcheck_ accum_ratio = static_cast<Tcheck_>(0);

                    // accumulate values
                    while(head < pre_pad)
                    {
                        if(head < n_inputs)
                        {
                            const Tcheck_ adder =
                                (static_cast<Tcheck_>(top_df_ptr[b * top_df_batch_stride +
                                                                 head * top_df_channel_stride +
                                                                 j * top_df_stride + i]) *
                                 static_cast<Tcheck_>(
                                     top_ptr[b * top_batch_stride + head * top_channel_stride +
                                             j * top_stride + i])) /
                                static_cast<Tcheck_>(
                                    scale_ptr[b * scale_batch_stride + head * scale_channel_stride +
                                              j * scale_stride + i]);

                            accum_ratio += adder;
                        }

                        ++head;
                    }

                    // until we reach size, nothing needs to be subtracted
                    while(head < local_area)
                    {

                        if(head < n_inputs)
                        {
                            const Tcheck_ adder =
                                (static_cast<Tcheck_>(top_df_ptr[b * top_df_batch_stride +
                                                                 head * top_df_channel_stride +
                                                                 j * top_df_stride + i]) *
                                 static_cast<Tcheck_>(
                                     top_ptr[b * top_batch_stride + head * top_channel_stride +
                                             j * top_stride + i])) /
                                static_cast<Tcheck_>(
                                    scale_ptr[b * scale_batch_stride + head * scale_channel_stride +
                                              j * scale_stride + i]);

                            accum_ratio += adder;
                        }

                        if(head - pre_pad >= 0 && head - pre_pad < n_inputs)
                        {
                            bot_df_v_ptr[b * bot_df_v_batch_stride +
                                         (head - pre_pad) * bot_df_v_channel_stride +
                                         j * bot_df_v_stride + i] =
                                static_cast<Tcheck_>(
                                    top_df_ptr[b * top_df_batch_stride +
                                               (head - pre_pad) * top_df_channel_stride +
                                               j * top_df_stride + i]) *
                                    pow(static_cast<Tcheck_>(
                                            scale_ptr[b * scale_batch_stride +
                                                      (head - pre_pad) * scale_channel_stride +
                                                      j * scale_stride + i]),
                                        negative_beta) -
                                ratio_dta_bwd *
                                    static_cast<Tcheck_>(
                                        bot_ptr[b * bot_batch_stride +
                                                (head - pre_pad) * bot_channel_stride +
                                                j * bot_stride + i]) *
                                    accum_ratio;
                        }
                        ++head;
                    }

                    // both add and subtract
                    while(head < n_inputs)
                    {

                        const Tcheck_ adder =
                            static_cast<Tcheck_>(
                                top_df_ptr[b * top_df_batch_stride + head * top_df_channel_stride +
                                           j * top_df_stride + i]) *
                            static_cast<Tcheck_>(
                                top_ptr[b * top_batch_stride + head * top_channel_stride +
                                        j * top_stride + i]) /
                            static_cast<Tcheck_>(
                                scale_ptr[b * scale_batch_stride + head * scale_channel_stride +
                                          j * scale_stride + i]);

                        accum_ratio += adder;

                        if(head - local_area >= 0)
                        {
                            const Tcheck_ subs =
                                (static_cast<Tcheck_>(
                                     top_df_ptr[b * top_df_batch_stride +
                                                (head - local_area) * top_df_channel_stride +
                                                j * top_df_stride + i]) *
                                 static_cast<Tcheck_>(
                                     top_ptr[b * top_batch_stride +
                                             (head - local_area) * top_channel_stride +
                                             j * top_stride + i])) /
                                static_cast<Tcheck_>(
                                    scale_ptr[b * scale_batch_stride +
                                              (head - local_area) * scale_channel_stride +
                                              j * scale_stride + i]);

                            accum_ratio -= subs;
                        }
                        if(head - pre_pad >= 0)
                        {
                            bot_df_v_ptr[b * bot_df_v_batch_stride +
                                         (head - pre_pad) * bot_df_v_channel_stride +
                                         j * bot_df_v_stride + i] =
                                static_cast<Tcheck_>(
                                    top_df_ptr[b * top_df_batch_stride +
                                               (head - pre_pad) * top_df_channel_stride +
                                               j * top_df_stride + i]) *
                                    pow(static_cast<Tcheck_>(
                                            scale_ptr[b * scale_batch_stride +
                                                      (head - pre_pad) * scale_channel_stride +
                                                      j * scale_stride + i]),
                                        negative_beta) -
                                ratio_dta_bwd *
                                    static_cast<Tcheck_>(
                                        bot_ptr[b * bot_batch_stride +
                                                (head - pre_pad) * bot_channel_stride +
                                                j * bot_stride + i]) *
                                    accum_ratio;
                        }

                        ++head;
                    }
                    // subtract only
                    while(head < n_inputs + pre_pad)
                    {
                        if(head - local_area >= 0 && head - local_area < n_inputs)
                        {
                            const Tcheck_ subs =
                                (static_cast<Tcheck_>(
                                     top_df_ptr[b * top_df_batch_stride +
                                                (head - local_area) * top_df_channel_stride +
                                                j * top_df_stride + i]) *
                                 static_cast<Tcheck_>(
                                     top_ptr[b * top_batch_stride +
                                             (head - local_area) * top_channel_stride +
                                             j * top_stride + i])) /
                                static_cast<Tcheck_>(
                                    scale_ptr[b * scale_batch_stride +
                                              (head - local_area) * scale_channel_stride +
                                              j * scale_stride + i]);

                            accum_ratio -= subs;
                        }
                        if(head - pre_pad >= 0 && head - pre_pad < n_inputs)
                        {
                            bot_df_v_ptr[b * bot_df_v_batch_stride +
                                         (head - pre_pad) * bot_df_v_channel_stride +
                                         j * bot_df_v_stride + i] =
                                static_cast<Tcheck_>(
                                    top_df_ptr[b * top_df_batch_stride +
                                               (head - pre_pad) * top_df_channel_stride +
                                               j * top_df_stride + i]) *
                                    pow(static_cast<Tcheck_>(
                                            scale_ptr[b * scale_batch_stride +
                                                      (head - pre_pad) * scale_channel_stride +
                                                      j * scale_stride + i]),
                                        negative_beta) -
                                ratio_dta_bwd *
                                    static_cast<Tcheck_>(
                                        bot_ptr[b * bot_batch_stride +
                                                (head - pre_pad) * bot_channel_stride +
                                                j * bot_stride + i]) *
                                    accum_ratio;
                        }

                        ++head;
                    }

                } // for (int i = 0; i < bot_width; i++)
            } // for (int j = 0; j < bot_height; j++)
        }); // miopen::par_for(n_batches, min_grain, [&](int b) {
    } // if (norm_region == MLO_LRN_ACROSS_CHANNELS)
    else
    {
        miopen::par_for(n_batches, min_grain, [&](int b) {
            for(int o = 0; o < n_inputs; o++)
            {
                for(int j = 0; j < bot_height; j++)
                {
                    for(int i = 0; i < bot_width; i++)
                    {
                        Tcheck_ accum_ratio = static_cast<Tcheck_>(0);

                        int hstart        = j - pad;
                        int wstart        = i - pad;
                        int hend          = std::min(hstart + local_area, top_height + pre_pad);
                        int wend          = std::min(wstart + local_area, top_width + pre_pad);
                        int adj_area_size = (hend - hstart) * (wend - wstart);
                        hstart            = std::max(hstart, 0);
                        wstart            = std::max(wstart, 0);
                        hend              = std::min(hend, top_height);
                        wend              = std::min(wend, top_width);
                        for(int h = hstart; h < hend; ++h)
                        {
                            for(int w = wstart; w < wend; ++w)
                            {
                                const Tcheck_ adder =
                                    static_cast<Tcheck_>(top_df_ptr[b * top_df_batch_stride +
                                                                    o * top_df_channel_stride +
                                                                    h * top_df_stride + w]) *
                                    static_cast<Tcheck_>(
                                        top_ptr[b * top_batch_stride + o * top_channel_stride +
                                                h * top_stride + w]) /
                                    static_cast<Tcheck_>(
                                        scale_ptr[b * scale_batch_stride +
                                                  o * scale_channel_stride + h * scale_stride + w]);

                                accum_ratio += adder;
                            }
                        }

                        const Tcheck_ ratio_dta_bwd = static_cast<Tcheck_>(2.) * alpha * beta /
                                                      static_cast<Tcheck_>(adj_area_size);

                        bot_df_v_ptr[b * bot_df_v_batch_stride + o * bot_df_v_channel_stride +
                                     j * bot_df_v_stride + i] =
                            static_cast<Tcheck_>(
                                top_df_ptr[b * top_df_batch_stride + o * top_df_channel_stride +
                                           j * top_df_stride + i]) *
                                pow(static_cast<Tcheck_>(
                                        scale_ptr[b * scale_batch_stride +
                                                  o * scale_channel_stride + j * scale_stride + i]),
                                    negative_beta) -
                            ratio_dta_bwd *
                                static_cast<Tcheck_>(
                                    bot_ptr[b * bot_batch_stride + o * bot_channel_stride +
                                            j * bot_stride + i]) *
                                accum_ratio;
                    } // for(int i = 0; i < bot_width; i++)
                } // for(int j = 0; j < bot_height; j++)
            } // for(int o = 0; o < n_inputs; o++)
        }); // miopen::par_for(n_batches, min_grain, [&](int b) {
    } // if (norm_region == MLO_LRN_ACROSS_CHANNELS)
    
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::ofstream("/data/Dev/LRN-" + std::string{typeid(Tgpu_).name()} + "-" + (multi_threaded ? "MT" : "ST") + "-Backward.csv", std::ofstream::app) << ns << std::endl;

    return (ret);
}

#endif
