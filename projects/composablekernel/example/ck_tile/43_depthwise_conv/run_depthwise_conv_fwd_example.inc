// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
int run_depthwise_conv_fwd_example_prec(ck_tile::ArgParser& arg_parser)
{
    const int verbosity                = arg_parser.get_int("v");
    const bool do_verification         = (verbosity > 0);
    const bool verbose                 = (verbosity >= 2);
    const int n_warmup                 = arg_parser.get_int("warmup");
    const int n_repeat                 = arg_parser.get_int("repeat");
    const ck_tile::index_t init_method = arg_parser.get_int("init");

    const ck_tile::index_t G          = arg_parser.get_int("G");
    const ck_tile::index_t N          = arg_parser.get_int("N");
    const ck_tile::index_t H          = arg_parser.get_int("H");
    const ck_tile::index_t W          = arg_parser.get_int("W");
    const ck_tile::index_t Y          = arg_parser.get_int("Y");
    const ck_tile::index_t X          = arg_parser.get_int("X");
    const ck_tile::index_t stride_h   = arg_parser.get_int("stride_h");
    const ck_tile::index_t stride_w   = arg_parser.get_int("stride_w");
    const ck_tile::index_t dilation_h = arg_parser.get_int("dilation_h");
    const ck_tile::index_t dilation_w = arg_parser.get_int("dilation_w");
    const ck_tile::index_t pad_h      = arg_parser.get_int("pad_h");
    const ck_tile::index_t pad_w      = arg_parser.get_int("pad_w");

    constexpr ck_tile::index_t C = 1;
    constexpr ck_tile::index_t K = 1;

    ck_tile::conv::ConvParam conv_param{
        2, // num_dim_spatial
        G,
        N,
        K,
        C,
        {Y, X}, // filter_spatial_lengths
        {H, W}, // input_spatial_lengths
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {pad_h, pad_w}, // input_left_pads
        {pad_h, pad_w}  // input_right_pads; TODO: support asymmetric padding
    };

    const ck_tile::index_t Ho =
        static_cast<ck_tile::index_t>(conv_param.output_spatial_lengths_[0]);
    const ck_tile::index_t Wo =
        static_cast<ck_tile::index_t>(conv_param.output_spatial_lengths_[1]);

    std::string prec_str;
    if constexpr(std::is_same_v<InDataType, ck_tile::fp16_t>)
        prec_str = "fp16";
    else if constexpr(std::is_same_v<InDataType, ck_tile::bf16_t>)
        prec_str = "bf16";
    else
        prec_str = "fp32";

    // TODO: derive strides from tensor descriptor; widen stride type to long_index_t
    // Strides for GNCHW layout: {g, n, c, h, w}
    const std::array<ck_tile::index_t, 5> in_strides  = {C * H * W, G * C * H * W, 1, W * C, C};
    const std::array<ck_tile::index_t, 5> wei_strides = {C * Y * X, G * C * Y * X, 1, X, 1};
    const std::array<ck_tile::index_t, 5> out_strides = {
        K * Ho * Wo, G * K * Ho * Wo, 1, Wo * K, K};

    std::cout << "./bin/tile_example_depthwise_conv_fwd" << " -G=" << G << " -N=" << N
              << " -H=" << H << " -W=" << W << " -Y=" << Y << " -X=" << X
              << " -stride_h=" << stride_h << " -stride_w=" << stride_w
              << " -dilation_h=" << dilation_h << " -dilation_w=" << dilation_w
              << " -pad_h=" << pad_h << " -pad_w=" << pad_w << " -prec=" << prec_str
              << " -v=" << verbosity << std::endl;

    if(do_verification)
    {
        std::cout << "Run Depthwise Conv Fwd kernel" << std::endl;
        std::cout << "in: G=" << G << " N=" << N << " C=" << C << " H=" << H << " W=" << W
                  << std::endl;
        std::cout << "wei: G=" << G << " K=" << K << " C=" << C << " Y=" << Y << " X=" << X
                  << std::endl;
        std::cout << "out: G=" << G << " N=" << N << " K=" << K << " Ho=" << Ho << " Wo=" << Wo
                  << std::endl;
    }

    const std::size_t input_size  = static_cast<std::size_t>(G) * N * C * H * W;
    const std::size_t weight_size = static_cast<std::size_t>(G) * K * C * Y * X;
    const std::size_t output_size = static_cast<std::size_t>(G) * N * K * Ho * Wo;

    // TODO: use HostTensor with tensor descriptor instead of flat 1D allocation
    ck_tile::HostTensor<InDataType> in_host({input_size});
    ck_tile::HostTensor<WeiDataType> wei_host({weight_size});
    ck_tile::HostTensor<OutDataType> out_host({output_size});
    ck_tile::HostTensor<OutDataType> out_ref(do_verification ? std::vector<std::size_t>{output_size}
                                                             : std::vector<std::size_t>{0});

    switch(init_method)
    {
    case 0:
        ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(in_host);
        ck_tile::FillUniformDistribution<WeiDataType>{-5.f, 5.f}(wei_host);
        break;
    case 1:
        ck_tile::FillUniformDistributionIntegerValue<InDataType>{-5.f, 5.f}(in_host);
        ck_tile::FillUniformDistributionIntegerValue<WeiDataType>{-5.f, 5.f}(wei_host);
        break;
    case 2:
        ck_tile::FillUniformDistribution<InDataType>{1.f, 1.f}(in_host);
        ck_tile::FillUniformDistribution<WeiDataType>{1.f, 1.f}(wei_host);
        break;
    default:
        in_host.SetZero();
        wei_host.SetZero();
        break;
    }

    if(do_verification)
    {
        depthwise_conv_fwd_cpu_reference<InDataType, WeiDataType, AccDataType, OutDataType>(
            in_host.data(), wei_host.data(), out_ref.data(), conv_param);
    }

    ck_tile::DeviceMem in_dev(in_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem wei_dev(wei_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_dev(out_host.get_element_space_size_in_bytes());

    in_dev.ToDevice(in_host.data());
    wei_dev.ToDevice(wei_host.data());
    out_dev.SetZero();

    ck_tile::DepthwiseConvFwdHostArgs args(conv_param,
                                           in_dev.GetDeviceBuffer(),
                                           wei_dev.GetDeviceBuffer(),
                                           out_dev.GetDeviceBuffer(),
                                           in_strides,
                                           wei_strides,
                                           out_strides);

    const int log_level = (verbosity >= 2) ? (verbosity - 1) : 0;
    ck_tile::stream_config stream_cfg{nullptr, true, log_level, n_warmup, n_repeat, true};

    const std::size_t flop     = conv_param.GetFlops();
    const std::size_t num_byte = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    VerificationInfo<OutDataType> verify_info{do_verification,
                                              &out_dev,
                                              &out_host,
                                              do_verification ? out_ref.data() : nullptr,
                                              output_size};

    auto [ave_time, best_config, best_verify_status, best_instance_idx] = DepthwiseConvFwdInvoker::
        run_all_instances<InDataType, WeiDataType, AccDataType, OutDataType>(
            args, stream_cfg, verify_info, flop, num_byte);

    const float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    const float gb_per_sec = static_cast<float>(num_byte) / 1.E6 / ave_time;
    const float gflops     = tflops * 1000.0f;

    if(verbose)
    {
        std::cout << "\n=== Best Performance ===" << std::endl;
        std::cout << "[Instance " << best_instance_idx << "] " << std::fixed << std::setprecision(6)
                  << ave_time << " ms, " << std::setprecision(5) << tflops << " TFlops, "
                  << std::setprecision(3) << gb_per_sec << " GB/s";
        if(best_verify_status != VerifyStatus::kSkipped)
        {
            std::cout << " [" << (best_verify_status == VerifyStatus::kPass ? "PASS" : "FAIL")
                      << "]";
        }
        std::cout << ", " << best_config << std::endl;
    }

    const std::size_t bytes_read =
        input_size * sizeof(InDataType) + weight_size * sizeof(WeiDataType);
    const std::size_t bytes_written = output_size * sizeof(OutDataType);

    const std::string conv_name =
        "fwd-conv" + std::to_string(Y) + "x" + std::to_string(X) + "u" + std::to_string(stride_h);

    // stats format: c=G, k=G for depthwise (C=K=1 per group)
    std::cout << "stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
                 "GB/s, timeMs"
              << std::endl;
    std::cout << "stats: " << conv_name << ", " << N << ", " << G << ", " << Ho << ", " << Wo
              << ", " << X << ", " << Y << ", " << G << ", " << flop << ", " << bytes_read << ", "
              << bytes_written << ", " << std::fixed << std::setprecision(0) << gflops << ", "
              << std::setprecision(0) << gb_per_sec << ", " << std::setprecision(6) << ave_time
              << std::endl;

    if(best_verify_status == VerifyStatus::kPass)
    {
        std::cout << "Forward Convolution Verifies OK against CPU reference (instance="
                  << best_instance_idx << ")" << std::endl;
    }
    else if(best_verify_status == VerifyStatus::kFail)
    {
        std::cout << "Forward Convolution FAILED (instance=" << best_instance_idx << ")"
                  << std::endl;
    }

    return best_verify_status != VerifyStatus::kFail;
}
