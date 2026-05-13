// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "gtest/gtest.h"

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/fill.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/ref/naive_grouped_conv_fwd_gpu.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_data_gpu.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_conv_host_args.hpp"

/// Templated integration test harness for direct convolution kernel structs.
///
/// KernelTraits must provide:
///   template <int ConfigIdx> using FwdKernel = ...;      // Fprop kernel struct
///   template <int ConfigIdx> using BwdDataKernel = ...;  // Dgrad kernel struct
///
/// Each kernel struct must provide MakeKernelArgs(), IsSupportedArgument(), Run().
///
/// RunFprop and RunDgrad are templated on the config index. If the specified
/// config is not supported for the given problem, the test fails.
template <typename KernelTraits, typename ElementT = ck_tile::half_t>
class DirectConvGrouped4cFp16TestHarness : public ::testing::Test
{
    protected:
    using HalfT = ElementT;

    public:
    template <int ConfigIdx>
    bool RunFprop(int N,
                  int H,
                  int W,
                  int groups,
                  int c_per_group,
                  int k_per_group,
                  int kh,
                  int kw,
                  int pad_h,
                  int pad_w)
    {
        using namespace ck_tile;
        using Kernel = typename KernelTraits::template FwdKernel<ConfigIdx>;

        // ConvParam takes K and C per group
        conv::ConvParam param(
            2,                                         // num_dim_spatial
            groups,                                    // group_count
            N,                                         // n_batch
            k_per_group,                               // n_out_channels (per group)
            c_per_group,                               // n_in_channels (per group)
            std::vector<index_t>{kh, kw},              // filter lengths
            std::vector<index_t>{H, W},                // input lengths
            std::vector<index_t>{1, 1},                // strides
            std::vector<index_t>{1, 1},                // dilations
            std::vector<index_t>{pad_h, pad_w},        // left pads
            std::vector<index_t>{pad_h, pad_w});       // right pads

        int C_total = groups * c_per_group;
        int K_total = groups * k_per_group;
        int Ho      = static_cast<int>(param.output_spatial_lengths_[0]);
        int Wo      = static_cast<int>(param.output_spatial_lengths_[1]);

        std::size_t in_size  = static_cast<std::size_t>(N * H * W * C_total);
        std::size_t wei_size = static_cast<std::size_t>(K_total * kh * kw * c_per_group);
        std::size_t out_size = static_cast<std::size_t>(N * Ho * Wo * K_total);

        // Fill host buffers
        HostTensor<HalfT> t_in({in_size});
        HostTensor<HalfT> t_wei({wei_size});
        FillUniformDistribution<HalfT>{-1.f, 1.f}(t_in);
        FillUniformDistribution<HalfT>{-1.f, 1.f}(t_wei);

        // Allocate device memory
        DeviceMem d_in(in_size * sizeof(HalfT));
        DeviceMem d_wei(wei_size * sizeof(HalfT));
        DeviceMem d_out(out_size * sizeof(HalfT));
        DeviceMem d_ref_out(out_size * sizeof(HalfT));

        d_in.ToDevice(t_in.data());
        d_wei.ToDevice(t_wei.data());
        d_out.SetZero();
        d_ref_out.SetZero();

        // GPU reference
        naive_grouped_conv_fwd<2>(
            static_cast<const HalfT*>(d_in.GetDeviceBuffer()),
            static_cast<const HalfT*>(d_wei.GetDeviceBuffer()),
            static_cast<HalfT*>(d_ref_out.GetDeviceBuffer()),
            groups,
            N,
            k_per_group,
            c_per_group,
            {static_cast<long_index_t>(H), static_cast<long_index_t>(W)},
            {static_cast<long_index_t>(kh), static_cast<long_index_t>(kw)},
            {static_cast<long_index_t>(Ho), static_cast<long_index_t>(Wo)},
            {1, 1},
            {1, 1},
            {static_cast<long_index_t>(pad_h), static_cast<long_index_t>(pad_w)});

        // Build kernel args and check support
        GroupedConvFwdHostArgs<> host_args(param,
                                          d_in.GetDeviceBuffer(),
                                          d_wei.GetDeviceBuffer(),
                                          {},
                                          d_out.GetDeviceBuffer(),
                                          1);
        auto kargs = Kernel::MakeKernelArgs(host_args);

        if(!Kernel::IsSupportedArgument(kargs))
            return false;

        Kernel kernel;
        stream_config s_conf{nullptr, false, 0, 0, 0};
        auto [supported, avg_time, name] = kernel.Run(kargs, s_conf);
        hip_check_error(hipDeviceSynchronize());

        if(!supported)
            return false;

        // Compare
        std::vector<HalfT> h_result(out_size);
        std::vector<HalfT> h_ref(out_size);
        d_out.FromDevice(h_result.data());
        d_ref_out.FromDevice(h_ref.data());

        // TODO: Fix the tolerance. BF16 should have more or less the same tolerance as FP16.
        constexpr double rtol = std::is_same_v<ElementT, ck_tile::bfloat16_t> ? 5e-2 : 1e-2;
        constexpr double atol = std::is_same_v<ElementT, ck_tile::bfloat16_t> ? 5e-2 : 1e-2;
        return check_err(h_result, h_ref, "Error: Fprop incorrect results!", rtol, atol);
    }

    template <int ConfigIdx>
    bool RunDgrad(int N,
                  int H,
                  int W,
                  int groups,
                  int c_per_group,
                  int k_per_group,
                  int kh,
                  int kw,
                  int pad_h,
                  int pad_w)
    {
        using namespace ck_tile;
        using Kernel = typename KernelTraits::template BwdDataKernel<ConfigIdx>;

        conv::ConvParam param(
            2,
            groups,
            N,
            k_per_group,
            c_per_group,
            std::vector<index_t>{kh, kw},
            std::vector<index_t>{H, W},
            std::vector<index_t>{1, 1},
            std::vector<index_t>{1, 1},
            std::vector<index_t>{pad_h, pad_w},
            std::vector<index_t>{pad_h, pad_w});

        int C_total = groups * c_per_group;
        int K_total = groups * k_per_group;
        int Ho      = static_cast<int>(param.output_spatial_lengths_[0]);
        int Wo      = static_cast<int>(param.output_spatial_lengths_[1]);

        std::size_t in_size  = static_cast<std::size_t>(N * H * W * C_total);
        std::size_t wei_size = static_cast<std::size_t>(K_total * kh * kw * c_per_group);
        std::size_t out_size = static_cast<std::size_t>(N * Ho * Wo * K_total);

        // Fill output gradient and weights
        HostTensor<HalfT> t_out_grad({out_size});
        HostTensor<HalfT> t_wei({wei_size});
        FillUniformDistribution<HalfT>{-1.f, 1.f}(t_out_grad);
        FillUniformDistribution<HalfT>{-1.f, 1.f}(t_wei);

        // Allocate device memory
        DeviceMem d_out_grad(out_size * sizeof(HalfT));
        DeviceMem d_wei(wei_size * sizeof(HalfT));
        DeviceMem d_in_grad(in_size * sizeof(HalfT));
        DeviceMem d_ref_in_grad(in_size * sizeof(HalfT));

        d_out_grad.ToDevice(t_out_grad.data());
        d_wei.ToDevice(t_wei.data());
        d_in_grad.SetZero();
        d_ref_in_grad.SetZero();

        // GPU reference
        naive_grouped_conv_bwd_data<2>(
            static_cast<HalfT*>(d_ref_in_grad.GetDeviceBuffer()),
            static_cast<const HalfT*>(d_wei.GetDeviceBuffer()),
            static_cast<const HalfT*>(d_out_grad.GetDeviceBuffer()),
            groups,
            N,
            k_per_group,
            c_per_group,
            {static_cast<long_index_t>(H), static_cast<long_index_t>(W)},
            {static_cast<long_index_t>(kh), static_cast<long_index_t>(kw)},
            {static_cast<long_index_t>(Ho), static_cast<long_index_t>(Wo)},
            {1, 1},
            {1, 1},
            {static_cast<long_index_t>(pad_h), static_cast<long_index_t>(pad_w)});

        // Build kernel args and check support
        GroupedConvBwdDataHostArgs host_args(param,
                                            d_in_grad.GetDeviceBuffer(),
                                            d_wei.GetDeviceBuffer(),
                                            {},
                                            d_out_grad.GetDeviceBuffer(),
                                            1);
        auto kargs = Kernel::MakeKernelArgs(host_args);

        if(!Kernel::IsSupportedArgument(kargs))
            return false;

        Kernel kernel;
        stream_config s_conf{nullptr, false, 0, 0, 0};
        auto [supported, avg_time, name] = kernel.Run(kargs, s_conf);
        hip_check_error(hipDeviceSynchronize());

        if(!supported)
            return false;

        // Compare
        std::vector<HalfT> h_result(in_size);
        std::vector<HalfT> h_ref(in_size);
        d_in_grad.FromDevice(h_result.data());
        d_ref_in_grad.FromDevice(h_ref.data());

        // TODO: Fix the tolerance. BF16 should have more or less the same tolerance as FP16.
        constexpr double rtol = std::is_same_v<ElementT, ck_tile::bfloat16_t> ? 5e-2 : 1e-2;
        constexpr double atol = std::is_same_v<ElementT, ck_tile::bfloat16_t> ? 5e-2 : 1e-2;
        return check_err(h_result, h_ref, "Error: Dgrad incorrect results!", rtol, atol);
    }
};
