// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

template <typename OutDataType>
struct VerificationInfo
{
    bool do_verification                         = false;
    ck_tile::DeviceMem* p_out_dev                = nullptr;
    ck_tile::HostTensor<OutDataType>* p_out_host = nullptr;
    const OutDataType* p_out_ref                 = nullptr;
    std::size_t output_size                      = 0;
};

enum class VerifyStatus
{
    kSkipped,
    kPass,
    kFail
};

struct KernelRunResult
{
    float time_ms              = std::numeric_limits<float>::max();
    bool is_supported          = false;
    VerifyStatus verify_status = VerifyStatus::kSkipped;
    std::string config_name;
    float tflops     = 0.0f;
    float gb_per_sec = 0.0f;
};

using DepthwiseInvokerResult = std::tuple<float, std::string, VerifyStatus, ck_tile::index_t>;

// TODO: replace with ck_tile::check_err when only best instance is verified
template <typename OutDataType>
bool verify_gpu_result(const OutDataType* p_gpu,
                       const OutDataType* p_cpu,
                       std::size_t size,
                       bool print_errors = false,
                       double rtol       = 1e-3,
                       double atol       = 1e-3)
{
    std::size_t error_count        = 0;
    double max_err                 = 0.0;
    int printed_errors             = 0;
    constexpr int max_print_errors = 4;

    for(std::size_t i = 0; i < size; ++i)
    {
        double gpu_val = static_cast<double>(p_gpu[i]);
        double cpu_val = static_cast<double>(p_cpu[i]);
        double diff    = std::abs(gpu_val - cpu_val);
        double ref_val = std::abs(cpu_val);

        if(diff > max_err)
        {
            max_err = diff;
        }

        if(diff > atol + rtol * ref_val)
        {
            error_count++;
            if(print_errors && printed_errors < max_print_errors)
            {
                std::cout << "\tout[" << i << "] != ref[" << i
                          << "]: " << static_cast<float>(p_gpu[i])
                          << " != " << static_cast<float>(p_cpu[i]) << std::endl;
                printed_errors++;
            }
        }
    }

    if(error_count > 0 && print_errors)
    {
        double error_pct = 100.0 * static_cast<double>(error_count) / static_cast<double>(size);
        std::cout << "max err: " << std::setprecision(6) << max_err
                  << ", number of errors: " << error_count << ", " << std::fixed
                  << std::setprecision(5) << error_pct << "% wrong values" << std::endl;
    }

    return error_count == 0;
}

// CPU reference for depthwise convolution forward (C=K=1 per group).
// Layout: Input [G,N,1,Hi,Wi], Weight [G,1,1,Y,X], Output [G,N,1,Ho,Wo]
template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
void depthwise_conv_fwd_cpu_reference(const InDataType* p_in,
                                      const WeiDataType* p_wei,
                                      OutDataType* p_out,
                                      const ck_tile::conv::ConvParam& conv_param)
{
    const auto G          = static_cast<ck_tile::index_t>(conv_param.G_);
    const auto N          = static_cast<ck_tile::index_t>(conv_param.N_);
    const auto Hi         = static_cast<ck_tile::index_t>(conv_param.input_spatial_lengths_[0]);
    const auto Wi         = static_cast<ck_tile::index_t>(conv_param.input_spatial_lengths_[1]);
    const auto Ho         = static_cast<ck_tile::index_t>(conv_param.output_spatial_lengths_[0]);
    const auto Wo         = static_cast<ck_tile::index_t>(conv_param.output_spatial_lengths_[1]);
    const auto Y          = static_cast<ck_tile::index_t>(conv_param.filter_spatial_lengths_[0]);
    const auto X          = static_cast<ck_tile::index_t>(conv_param.filter_spatial_lengths_[1]);
    const auto stride_h   = static_cast<ck_tile::index_t>(conv_param.conv_filter_strides_[0]);
    const auto stride_w   = static_cast<ck_tile::index_t>(conv_param.conv_filter_strides_[1]);
    const auto dilation_h = static_cast<ck_tile::index_t>(conv_param.conv_filter_dilations_[0]);
    const auto dilation_w = static_cast<ck_tile::index_t>(conv_param.conv_filter_dilations_[1]);
    const auto pad_h      = static_cast<ck_tile::index_t>(conv_param.input_left_pads_[0]);
    const auto pad_w      = static_cast<ck_tile::index_t>(conv_param.input_left_pads_[1]);

    const ck_tile::long_index_t in_g_stride = static_cast<ck_tile::long_index_t>(Hi) * Wi;
    const ck_tile::long_index_t in_n_stride = static_cast<ck_tile::long_index_t>(G) * Hi * Wi;
    const ck_tile::long_index_t in_h_stride = Wi;
    const ck_tile::long_index_t in_w_stride = 1;

    const ck_tile::long_index_t wei_g_stride = static_cast<ck_tile::long_index_t>(Y) * X;
    const ck_tile::long_index_t wei_y_stride = X;
    const ck_tile::long_index_t wei_x_stride = 1;

    const ck_tile::long_index_t out_g_stride = static_cast<ck_tile::long_index_t>(Ho) * Wo;
    const ck_tile::long_index_t out_n_stride = static_cast<ck_tile::long_index_t>(G) * Ho * Wo;
    const ck_tile::long_index_t out_h_stride = Wo;
    const ck_tile::long_index_t out_w_stride = 1;

    for(ck_tile::index_t g = 0; g < G; ++g)
    {
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            for(ck_tile::index_t ho = 0; ho < Ho; ++ho)
            {
                for(ck_tile::index_t wo = 0; wo < Wo; ++wo)
                {
                    AccDataType acc = static_cast<AccDataType>(0);

                    for(ck_tile::index_t y = 0; y < Y; ++y)
                    {
                        for(ck_tile::index_t x = 0; x < X; ++x)
                        {
                            ck_tile::index_t hi = ho * stride_h + y * dilation_h - pad_h;
                            ck_tile::index_t wi = wo * stride_w + x * dilation_w - pad_w;

                            if(hi >= 0 && hi < Hi && wi >= 0 && wi < Wi)
                            {
                                ck_tile::long_index_t in_idx =
                                    static_cast<ck_tile::long_index_t>(g) * in_g_stride +
                                    static_cast<ck_tile::long_index_t>(n) * in_n_stride +
                                    static_cast<ck_tile::long_index_t>(hi) * in_h_stride +
                                    static_cast<ck_tile::long_index_t>(wi) * in_w_stride;
                                ck_tile::long_index_t wei_idx =
                                    static_cast<ck_tile::long_index_t>(g) * wei_g_stride +
                                    static_cast<ck_tile::long_index_t>(y) * wei_y_stride +
                                    static_cast<ck_tile::long_index_t>(x) * wei_x_stride;

                                acc += static_cast<AccDataType>(p_in[in_idx]) *
                                       static_cast<AccDataType>(p_wei[wei_idx]);
                            }
                        }
                    }

                    ck_tile::long_index_t out_idx =
                        static_cast<ck_tile::long_index_t>(g) * out_g_stride +
                        static_cast<ck_tile::long_index_t>(n) * out_n_stride +
                        static_cast<ck_tile::long_index_t>(ho) * out_h_stride +
                        static_cast<ck_tile::long_index_t>(wo) * out_w_stride;
                    p_out[out_idx] = static_cast<OutDataType>(acc);
                }
            }
        }
    }
}
