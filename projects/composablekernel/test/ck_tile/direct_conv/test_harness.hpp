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
#include "ck/library/utility/gpu_verification.hpp"

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
class DirectConvGroupedTestHarness : public ::testing::Test
{
    protected:
    using HalfT = ElementT;

    // ck::profiler::gpu_verify dispatches on the ck:: numeric types. The ck_tile
    // element types share the same underlying representation (_Float16 / __bf16),
    // so map ElementT to its ck:: equivalent for the device-side comparison.
    using VerifyT = std::conditional_t<std::is_same_v<ElementT, ck_tile::bfloat16_t>,
                                       ck::bhalf_t,
                                       ck::half_t>;

    // Compare two device buffers on the GPU. Only a small result struct is copied
    // back to the host, avoiding a full device-to-host transfer of both tensors.
    static bool GpuCompare(const void* d_result,
                           const void* d_ref,
                           std::size_t size,
                           const char* msg)
    {
        // BF16 needs wider tolerance because MFMA multiplies at BF16 precision
        // (7-bit mantissa) while the GPU reference promotes to fp32 first.
        // Tolerance values match profiler/common.hpp get_rtol/get_atol<ck::bhalf_t>.
        constexpr float rtol = std::is_same_v<ElementT, ck_tile::bfloat16_t> ? 5e-2f : 1e-2f;
        constexpr float atol = std::is_same_v<ElementT, ck_tile::bfloat16_t> ? 5e-2f : 1e-2f;

        auto result = ck::profiler::gpu_verify<VerifyT>(d_result, d_ref, rtol, atol, size);
        if(result.error_count != 0)
        {
            std::cerr << msg << std::endl;
            result.print_error_summary();
            return false;
        }
        return true;
    }

#ifdef CK_TILE_TEST_NO_DGRAD
    // Helper kept void so GTEST_SKIP() (which expands to a void `return`) is
    // valid; marks the currently running test as skipped.
    static void SkipDgradUnsupported()
    {
        GTEST_SKIP() << "Dgrad (backward-data) direct conv uses ds_read_b64_tr_b16, "
                        "a CDNA4 (gfx950) only transpose read; skipped on this arch.";
    }
#endif

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

        // Compare on the GPU (only a small result struct is copied back to host).
        return GpuCompare(d_out.GetDeviceBuffer(),
                          d_ref_out.GetDeviceBuffer(),
                          out_size,
                          "Error: Fprop incorrect results!");
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
#ifdef CK_TILE_TEST_NO_DGRAD
        // Backward-data direct conv is gfx950-only (CDNA4 transpose read). Skip
        // these tests on architectures that define CK_TILE_TEST_NO_DGRAD.
        SkipDgradUnsupported();
        return true;
#else
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

        // Compare on the GPU (only a small result struct is copied back to host).
        return GpuCompare(d_in_grad.GetDeviceBuffer(),
                          d_ref_in_grad.GetDeviceBuffer(),
                          in_size,
                          "Error: Dgrad incorrect results!");
#endif // CK_TILE_TEST_NO_DGRAD
    }
};
