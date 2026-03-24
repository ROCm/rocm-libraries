// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/depthwise_conv_fwd_kernel.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/depthwise_conv_fwd_traits.hpp"
#include "ck_tile/ops/grouped_convolution/pipeline/depthwise_conv_fwd_pipeline.hpp"

#include "ck/library/utility/convolution_parameter.hpp"

namespace ck {
namespace profiler {

namespace depthwise_detail {

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

    const ck_tile::long_index_t wei_g_stride = static_cast<ck_tile::long_index_t>(Y) * X;

    const ck_tile::long_index_t out_g_stride = static_cast<ck_tile::long_index_t>(Ho) * Wo;
    const ck_tile::long_index_t out_n_stride = static_cast<ck_tile::long_index_t>(G) * Ho * Wo;

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
                                    static_cast<ck_tile::long_index_t>(hi) * Wi +
                                    static_cast<ck_tile::long_index_t>(wi);
                                ck_tile::long_index_t wei_idx =
                                    static_cast<ck_tile::long_index_t>(g) * wei_g_stride +
                                    static_cast<ck_tile::long_index_t>(y) * X +
                                    static_cast<ck_tile::long_index_t>(x);

                                acc += static_cast<AccDataType>(p_in[in_idx]) *
                                       static_cast<AccDataType>(p_wei[wei_idx]);
                            }
                        }
                    }

                    ck_tile::long_index_t out_idx =
                        static_cast<ck_tile::long_index_t>(g) * out_g_stride +
                        static_cast<ck_tile::long_index_t>(n) * out_n_stride +
                        static_cast<ck_tile::long_index_t>(ho) * Wo +
                        static_cast<ck_tile::long_index_t>(wo);
                    p_out[out_idx] = static_cast<OutDataType>(acc);
                }
            }
        }
    }
}

template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutDataType,
          ck_tile::index_t TileH,
          ck_tile::index_t TileW,
          ck_tile::index_t FilterSize,
          ck_tile::index_t DilationH,
          ck_tile::index_t DilationW,
          ck_tile::index_t StrideH,
          ck_tile::index_t StrideW,
          ck_tile::index_t PadH,
          ck_tile::index_t PadW,
          ck_tile::index_t NBatch,
          ck_tile::index_t SubTileH,
          ck_tile::index_t SubTileW,
          ck_tile::index_t InVecSize,
          ck_tile::index_t OutVecSize>
KernelRunResult try_instance(const ck_tile::DepthwiseConvFwdHostArgs& args,
                             const ck_tile::stream_config& s,
                             const VerificationInfo<OutDataType>& verify_info,
                             const std::size_t flop,
                             const std::size_t num_byte,
                             bool& first_error_printed)
{
    KernelRunResult result;

    using Traits = ck_tile::DepthwiseConvFwdTraits<InDataType,
                                                   WeiDataType,
                                                   AccDataType,
                                                   OutDataType,
                                                   64,
                                                   TileH,
                                                   TileW,
                                                   FilterSize,
                                                   FilterSize,
                                                   StrideH,
                                                   StrideW,
                                                   DilationH,
                                                   DilationW,
                                                   PadH,
                                                   PadW,
                                                   NBatch,
                                                   SubTileH,
                                                   SubTileW,
                                                   InVecSize,
                                                   OutVecSize>;

    using Pipeline = ck_tile::DepthwiseConvFwdPipeline<Traits>;
    using Kernel   = ck_tile::DepthwiseConvFwdKernel<Traits, Pipeline>;

    result.config_name = Kernel::GetName();

    if(!Kernel::IsSupportedArgument(args))
    {
        std::cout << result.config_name << " does not support this problem" << std::endl;
        return result;
    }

    const auto kargs = Kernel::MakeKernelArgs(args);

    const auto grids  = Kernel::GridSize(static_cast<ck_tile::index_t>(args.G_),
                                        static_cast<ck_tile::index_t>(args.N_));
    const auto blocks = Kernel::BlockSize();

    const float time_ms =
        ck_tile::launch_kernel(s, ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));

    result.is_supported = true;
    result.time_ms      = time_ms;

    if(flop > 0 && time_ms > 0)
    {
        result.tflops     = static_cast<float>(flop) / 1.E9 / time_ms;
        result.gb_per_sec = static_cast<float>(num_byte) / 1.E6 / time_ms;
    }

    if(verify_info.do_verification)
    {
        verify_info.p_out_dev->FromDevice(verify_info.p_out_host->data());
        const bool print_errors = (s.log_level_ > 0) && !first_error_printed;

        const bool pass      = verify_gpu_result<OutDataType>(verify_info.p_out_host->data(),
                                                         verify_info.p_out_ref,
                                                         verify_info.output_size,
                                                         print_errors);
        result.verify_status = pass ? VerifyStatus::kPass : VerifyStatus::kFail;

        if(!pass && print_errors)
        {
            first_error_printed = true;
        }
    }

    std::cout << "Perf: " << std::setw(10) << time_ms << " ms, " << result.tflops << " TFlops, "
              << result.gb_per_sec << " GB/s, " << result.config_name << std::endl;

    return result;
}

template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
std::tuple<float, std::string, VerifyStatus, ck_tile::index_t, ck_tile::index_t, ck_tile::index_t>
run_all_instances(const ck_tile::DepthwiseConvFwdHostArgs& args,
                  const ck_tile::stream_config& s,
                  const VerificationInfo<OutDataType>& verify_info,
                  const std::size_t flop,
                  const std::size_t num_byte)
{
    float best_time = std::numeric_limits<float>::max();
    std::string best_config;
    VerifyStatus best_verify_status    = VerifyStatus::kSkipped;
    ck_tile::index_t best_instance_idx = -1;
    ck_tile::index_t instance_count    = 0;
    ck_tile::index_t valid_count       = 0;
    bool first_error_printed           = false;

    auto process_result = [&](const KernelRunResult& result) {
        if(result.is_supported)
        {
            valid_count++;
            if(result.time_ms < best_time)
            {
                best_time          = result.time_ms;
                best_config        = result.config_name;
                best_verify_status = result.verify_status;
                best_instance_idx  = instance_count;
            }
        }
        instance_count++;
    };

#define TRY_INSTANCE(                                                                \
    TileH, TileW, Filter, StrH, StrW, PadH, PadW, NBatch, SubH, SubW, InVec, OutVec) \
    process_result(                                                                  \
        try_instance<InDataType,                                                     \
                     WeiDataType,                                                    \
                     AccDataType,                                                    \
                     OutDataType,                                                    \
                     TileH,                                                          \
                     TileW,                                                          \
                     Filter,                                                         \
                     1,                                                              \
                     1,                                                              \
                     StrH,                                                           \
                     StrW,                                                           \
                     PadH,                                                           \
                     PadW,                                                           \
                     NBatch,                                                         \
                     SubH,                                                           \
                     SubW,                                                           \
                     InVec,                                                          \
                     OutVec>(args, s, verify_info, flop, num_byte, first_error_printed))

    // FilterSize = 3, Pad = 1
    TRY_INSTANCE(8, 8, 3, 1, 1, 1, 1, 8, 2, 2, 2, 2);
    TRY_INSTANCE(16, 16, 3, 1, 1, 1, 1, 8, 1, 4, 8, 8);
    TRY_INSTANCE(16, 16, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2);
    TRY_INSTANCE(28, 28, 3, 1, 1, 1, 1, 1, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 3, 1, 1, 1, 1, 1, 4, 4, 8, 8);

    TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 2, 1, 4, 8, 8);
    TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 1, 1, 4, 8, 8);
    TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 1, 2, 2, 8, 8);
    TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 1, 2, 2, 2, 2);
    TRY_INSTANCE(14, 28, 3, 2, 2, 1, 1, 1, 2, 4, 8, 8);
    TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 2, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 1, 4, 4, 4, 4);
    TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 1, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 1, 2, 8, 8, 8);

    // FilterSize = 5, Pad = 2
    TRY_INSTANCE(8, 8, 5, 1, 1, 2, 2, 1, 1, 1, 1, 1);
    TRY_INSTANCE(8, 8, 5, 1, 1, 2, 2, 8, 2, 2, 2, 2);
    TRY_INSTANCE(16, 16, 5, 1, 1, 2, 2, 1, 1, 4, 8, 8);
    TRY_INSTANCE(16, 16, 5, 1, 1, 2, 2, 8, 1, 4, 8, 8);
    TRY_INSTANCE(28, 28, 5, 1, 1, 2, 2, 8, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 5, 1, 1, 2, 2, 4, 4, 4, 8, 8);

    TRY_INSTANCE(8, 8, 5, 2, 2, 2, 2, 4, 2, 2, 2, 2);
    TRY_INSTANCE(8, 8, 5, 2, 2, 2, 2, 1, 2, 2, 2, 2);
    TRY_INSTANCE(16, 16, 5, 2, 2, 2, 2, 1, 1, 4, 8, 8);
    TRY_INSTANCE(16, 16, 5, 2, 2, 2, 2, 1, 2, 2, 8, 8);
    TRY_INSTANCE(14, 28, 5, 2, 2, 2, 2, 2, 2, 4, 8, 8);
    TRY_INSTANCE(16, 32, 5, 2, 2, 2, 2, 4, 1, 8, 8, 8);
    TRY_INSTANCE(32, 32, 5, 2, 2, 2, 2, 1, 4, 4, 4, 4);
    TRY_INSTANCE(32, 32, 5, 2, 2, 2, 2, 1, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 5, 2, 2, 2, 2, 1, 2, 8, 8, 8);

    // FilterSize = 7, Pad = 3
    TRY_INSTANCE(8, 8, 7, 1, 1, 3, 3, 1, 1, 1, 1, 1);
    TRY_INSTANCE(8, 8, 7, 1, 1, 3, 3, 8, 2, 2, 2, 2);
    TRY_INSTANCE(16, 16, 7, 1, 1, 3, 3, 1, 1, 4, 8, 8);
    TRY_INSTANCE(16, 16, 7, 1, 1, 3, 3, 8, 1, 4, 8, 8);
    TRY_INSTANCE(28, 28, 7, 1, 1, 3, 3, 1, 4, 4, 8, 8);
    TRY_INSTANCE(28, 28, 7, 1, 1, 3, 3, 8, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 7, 1, 1, 3, 3, 1, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 7, 1, 1, 3, 3, 4, 4, 4, 8, 8);

    TRY_INSTANCE(8, 8, 7, 2, 2, 3, 3, 4, 2, 2, 2, 2);
    TRY_INSTANCE(16, 16, 7, 2, 2, 3, 3, 2, 1, 4, 8, 8);
    TRY_INSTANCE(14, 28, 7, 2, 2, 3, 3, 2, 2, 4, 8, 8);
    TRY_INSTANCE(16, 32, 7, 2, 2, 3, 3, 4, 1, 8, 8, 8);
    TRY_INSTANCE(32, 32, 7, 2, 2, 3, 3, 2, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 7, 2, 2, 3, 3, 1, 4, 4, 8, 8);

    // FilterSize = 9, Pad = 4
    TRY_INSTANCE(8, 8, 9, 1, 1, 4, 4, 1, 1, 1, 1, 1);
    TRY_INSTANCE(8, 8, 9, 1, 1, 4, 4, 8, 2, 2, 2, 2);
    TRY_INSTANCE(16, 16, 9, 1, 1, 4, 4, 1, 1, 4, 8, 8);
    TRY_INSTANCE(16, 16, 9, 1, 1, 4, 4, 8, 1, 4, 8, 8);
    TRY_INSTANCE(28, 28, 9, 1, 1, 4, 4, 1, 4, 4, 8, 8);
    TRY_INSTANCE(28, 28, 9, 1, 1, 4, 4, 8, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 9, 1, 1, 4, 4, 1, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 9, 1, 1, 4, 4, 4, 4, 4, 8, 8);

    TRY_INSTANCE(8, 8, 9, 2, 2, 4, 4, 4, 2, 2, 2, 2);
    TRY_INSTANCE(16, 16, 9, 2, 2, 4, 4, 2, 1, 4, 8, 8);
    TRY_INSTANCE(14, 28, 9, 2, 2, 4, 4, 2, 2, 4, 8, 8);
    TRY_INSTANCE(16, 32, 9, 2, 2, 4, 4, 4, 1, 8, 8, 8);
    TRY_INSTANCE(32, 32, 9, 2, 2, 4, 4, 2, 4, 4, 8, 8);
    TRY_INSTANCE(32, 32, 9, 2, 2, 4, 4, 1, 4, 4, 8, 8);

#undef TRY_INSTANCE

    if(valid_count == 0)
    {
        return {0.0f, "", VerifyStatus::kSkipped, -1, 0, instance_count};
    }

    return {
        best_time, best_config, best_verify_status, best_instance_idx, valid_count, instance_count};
}

} // namespace depthwise_detail

} // namespace profiler
} // namespace ck
