/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#ifndef MLO_CONVHOST_H_
#define MLO_CONVHOST_H_

#include <miopen/tensor.hpp>

#include <cmath>
#include <iostream>

#include "calcerr.hpp"

template <typename Tgpu_ /* the data type used in GPU computations (usually half) */,
          typename Tcheck_ /* the data type used in CPU checkings (usually double) */>
bool mloVerify(const miopenTensorDescriptor_t& cpu_,
               const miopenTensorDescriptor_t& gpu_,
               const Tcheck_* c_ptr,
               const Tgpu_* g_ptr,
               float ulps_tolerance,
               Tcheck_ diff_tolerance,
               double rms_tolerance,
               bool check_ulps,
               double& report_err)
{
    const auto& cpu = miopen::deref(cpu_);
    const auto& gpu = miopen::deref(gpu_);

    const auto spatial_dim = cpu.GetLengths().size() - 2;

    size_t n_batchs, n_channels, depth, height, width;
    size_t c_batch_stride, c_channel_stride, c_depth_stride, c_height_stride, c_width_stride;
    size_t g_batch_stride, g_channel_stride, g_depth_stride, g_height_stride, g_width_stride;

    std::tie(n_batchs, n_channels, depth, height, width) =
        miopen::GetNCDHW(spatial_dim, cpu.GetLengths());
    std::tie(c_batch_stride, c_channel_stride, c_depth_stride, c_height_stride, c_width_stride) =
        miopen::GetNCDHW(spatial_dim, cpu.GetStrides());
    std::tie(g_batch_stride, g_channel_stride, g_depth_stride, g_height_stride, g_width_stride) =
        miopen::GetNCDHW(spatial_dim, gpu.GetStrides());

    bool match          = true;
    double rms_accum    = 0.0;
    Tcheck_ worst_c_val = static_cast<Tcheck_>(0);
    Tcheck_ worst_g_val = static_cast<Tcheck_>(0);
    Tcheck_ worst_diff  = static_cast<Tcheck_>(0);
    size_t worst_b = 0, worst_c = 0, worst_i = 0, worst_j = 0, worst_k = 0;

    for(size_t b = 0; b < n_batchs; ++b)
    {
        for(size_t c = 0; c < n_channels; ++c)
        {
            for(size_t k = 0; k < depth; ++k)
            {
                for(size_t j = 0; j < height; ++j)
                {
                    for(size_t i = 0; i < width; ++i)
                    {
                        Tcheck_ c_val =
                            c_ptr[b * c_batch_stride + c * c_channel_stride + k * c_depth_stride +
                                  j * c_height_stride + i * c_width_stride];
                        Tcheck_ g_val = static_cast<Tcheck_>(
                            g_ptr[b * g_batch_stride + c * g_channel_stride + k * g_depth_stride +
                                  j * g_height_stride + i * g_width_stride]);

                        Tcheck_ diff = std::abs(c_val - g_val);
                        rms_accum += diff * diff;
                        // Register worst (max) abs error and its position.
                        // This info will be used to show additional diagnostics,
                        // but only if sgr_accum is too big.
                        if(diff > worst_diff)
                        {
                            worst_diff  = diff;
                            worst_c_val = c_val;
                            worst_g_val = g_val;
                            worst_b     = b;
                            worst_c     = c;
                            worst_i     = i;
                            worst_j     = j;
                            worst_k     = k;
                        }
                    }
                }
            }
        }
    }

    const double rms = std::sqrt(
        rms_accum / (static_cast<double>(n_batchs * n_channels * depth * height * width)));
    report_err = rms;

    if(rms > rms_tolerance || std::isnan(rms) || !std::isfinite(rms))
    {
        match = false;

        std::cout << "RMS too big: " << rms << ". Max diff: " << worst_diff << " at {" << worst_b
                  << ',' << worst_c << ',';
        if(spatial_dim == 3)
            std::cout << worst_k << ',';
        std::cout << worst_j << ',' << worst_i << "}, cpu_v = " << worst_c_val
                  << " vs gpu_v = " << worst_g_val << std::endl;
    }

    if(check_ulps)
    {
        static int n_logged = 0;
        for(size_t b = 0; b < n_batchs && match; ++b)
        {
            for(size_t c = 0; c < n_channels && match; ++c)
            {
                for(size_t k = 0; k < depth && match; ++k)
                {
                    for(size_t j = 0; j < height && match; ++j)
                    {
                        for(size_t i = 0; i < width && match; ++i)
                        {
                            auto c_val =
                                static_cast<Tgpu_>(c_ptr[b * c_batch_stride + c * c_channel_stride +
                                                         k * c_depth_stride + j * c_height_stride +
                                                         i * c_width_stride]);
                            auto g_val =
                                static_cast<Tgpu_>(g_ptr[b * g_batch_stride + c * g_channel_stride +
                                                         k * g_depth_stride + j * g_height_stride +
                                                         i * g_width_stride]);

                            const auto diff = std::abs(c_val - g_val);
                            const auto ulps = ApproxUlps(c_val, g_val);
                            const bool check_failed =
                                (diff > diff_tolerance && ulps > ulps_tolerance) //
                                || std::isnan(c_val)                             //
                                || std::isnan(g_val)                             //
                                || !std::isfinite(c_val)                         //
                                || !std::isfinite(g_val);

                            if(check_failed)
                                match = false;

                            if(check_failed)
                            {
                                if(!(n_logged >= 10))
                                {
                                    std::cout << "ULPs: " << ulps;
                                    if(check_failed)
                                        std::cout << " is too large (> " << ulps_tolerance << ")";
                                    std::cout << " at {" << b << ',' << c << ',';
                                    if(spatial_dim == 3)
                                        std::cout << k << ',';
                                    std::cout << j << ',' << i << "}, cpu_val = " << c_val
                                              << ", gpu_val = " << g_val << " (diff = " << diff
                                              << ')' << std::endl;
                                    ++n_logged;
                                    if(n_logged >= 10)
                                        std::cout << "(too many lines logged, truncating output...)"
                                                  << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return match;
}

template <typename Tgpu_ /* the data type used in GPU computations (usually half) */,
          typename Tcheck_ /* the data type used in CPU checkings (usually double) */>
bool mloVerify_mt(const miopenTensorDescriptor_t& cpu_,
                  const miopenTensorDescriptor_t& gpu_,
                  const Tcheck_* c_ptr,
                  const Tgpu_* g_ptr,
                  float ulps_tolerance,
                  Tcheck_ diff_tolerance,
                  double rms_tolerance,
                  bool check_ulps,
                  double& report_err)
{
    const auto& cpu = miopen::deref(cpu_);
    const auto& gpu = miopen::deref(gpu_);

    const auto spatial_dim = cpu.GetLengths().size() - 2;

    size_t n_batchs, n_channels, depth, height, width;
    size_t c_batch_stride, c_channel_stride, c_depth_stride, c_height_stride, c_width_stride;
    size_t g_batch_stride, g_channel_stride, g_depth_stride, g_height_stride, g_width_stride;

    std::tie(n_batchs, n_channels, depth, height, width) =
        miopen::GetNCDHW(spatial_dim, cpu.GetLengths());
    std::tie(c_batch_stride, c_channel_stride, c_depth_stride, c_height_stride, c_width_stride) =
        miopen::GetNCDHW(spatial_dim, cpu.GetStrides());
    std::tie(g_batch_stride, g_channel_stride, g_depth_stride, g_height_stride, g_width_stride) =
        miopen::GetNCDHW(spatial_dim, gpu.GetStrides());

    double rms_accum    = 0.0;
    Tcheck_ worst_c_val = static_cast<Tcheck_>(0);
    Tcheck_ worst_g_val = static_cast<Tcheck_>(0);
    Tcheck_ worst_diff  = static_cast<Tcheck_>(0);
    size_t worst_b = 0, worst_c = 0, worst_i = 0, worst_j = 0, worst_k = 0;

    std::mutex mt;
    miopen::par_ford(n_batchs, n_channels)([&](int b, int c) {
        double rms_local          = 0.0;
        Tcheck_ worst_c_val_local = static_cast<Tcheck_>(0);
        Tcheck_ worst_g_val_local = static_cast<Tcheck_>(0);
        Tcheck_ worst_diff_local  = static_cast<Tcheck_>(0);
        size_t worst_b_local = 0, worst_c_local = 0, worst_i_local = 0, worst_j_local = 0,
               worst_k_local = 0;
        for(size_t k = 0; k < depth; ++k)
        {
            for(size_t j = 0; j < height; ++j)
            {
                for(size_t i = 0; i < width; ++i)
                {
                    Tcheck_ c_val =
                        c_ptr[b * c_batch_stride + c * c_channel_stride + k * c_depth_stride +
                              j * c_height_stride + i * c_width_stride];
                    Tcheck_ g_val = static_cast<Tcheck_>(
                        g_ptr[b * g_batch_stride + c * g_channel_stride + k * g_depth_stride +
                              j * g_height_stride + i * g_width_stride]);

                    Tcheck_ diff = std::abs(c_val - g_val);
                    rms_local += diff * diff;
                    // Register worst (max) abs error and its position.
                    // This info will be used to show additional diagnostics,
                    // but only if sgr_accum is too big.
                    if(diff > worst_diff_local)
                    {
                        worst_diff_local  = diff;
                        worst_c_val_local = c_val;
                        worst_g_val_local = g_val;
                        worst_b_local     = b;
                        worst_c_local     = c;
                        worst_i_local     = i;
                        worst_j_local     = j;
                        worst_k_local     = k;
                    }
                }
            }
        }
        std::lock_guard<std::mutex> lock(mt);
        rms_accum += rms_local;
        if(worst_diff_local > worst_diff)
        {
            worst_diff  = worst_diff_local;
            worst_c_val = worst_c_val_local;
            worst_g_val = worst_g_val_local;
            worst_b     = worst_b_local;
            worst_c     = worst_c_local;
            worst_i     = worst_i_local;
            worst_j     = worst_j_local;
            worst_k     = worst_k_local;
        }
    });

    const double rms = std::sqrt(
        rms_accum / (static_cast<double>(n_batchs * n_channels * depth * height * width)));
    report_err = rms;

    if(rms > rms_tolerance || std::isnan(rms) || !std::isfinite(rms))
    {

        std::cout << "RMS too big: " << rms << ". Max diff: " << worst_diff << " at {" << worst_b
                  << ',' << worst_c << ',';
        if(spatial_dim == 3)
            std::cout << worst_k << ',';
        std::cout << worst_j << ',' << worst_i << "}, cpu_v = " << worst_c_val
                  << " vs gpu_v = " << worst_g_val << std::endl;

        return false;
    }

    if(check_ulps)
    {
        std::atomic_bool match = true;
        miopen::par_ford(n_batchs, n_channels)([&](int b, int c) {
            for(size_t k = 0; k < depth && match; ++k)
            {
                for(size_t j = 0; j < height && match; ++j)
                {
                    for(size_t i = 0; i < width && match; ++i)
                    {
                        auto c_val = static_cast<Tgpu_>(
                            c_ptr[b * c_batch_stride + c * c_channel_stride + k * c_depth_stride +
                                  j * c_height_stride + i * c_width_stride]);
                        auto g_val = static_cast<Tgpu_>(
                            g_ptr[b * g_batch_stride + c * g_channel_stride + k * g_depth_stride +
                                  j * g_height_stride + i * g_width_stride]);

                        const auto diff = std::abs(c_val - g_val);
                        const auto ulps = ApproxUlps(c_val, g_val);
                        const bool check_failed =
                            (diff > diff_tolerance && ulps > ulps_tolerance) //
                            || std::isnan(c_val)                             //
                            || std::isnan(g_val)                             //
                            || !std::isfinite(c_val)                         //
                            || !std::isfinite(g_val);

                        if(check_failed)
                        {
                            match = false;
                            std::cout << "ULPs: " << ulps << " is too large (> " << ulps_tolerance
                                      << ")" << " at {" << b << ',' << c << ',';
                            if(spatial_dim == 3)
                                std::cout << k << ',';
                            std::cout << j << ',' << i << "}, cpu_val = " << c_val
                                      << ", gpu_val = " << g_val << " (diff = " << diff << ')'
                                      << std::endl;
                        }
                    }
                }
            }
        });
        return match;
    }
    return true;
}

#endif
