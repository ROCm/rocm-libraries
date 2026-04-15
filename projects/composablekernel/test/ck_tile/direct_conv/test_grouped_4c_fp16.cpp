// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_fp16_kernel.hpp"

using namespace ck_tile;
using namespace ck_tile::direct_conv;

class DirectConvGrouped4cFp16Test : public ::testing::Test
{
    protected:
    using HalfT = ck_tile::half_t;

    bool RunTest(Direction direction, int N, int H, int W, int groups,
                 int c_per_group, int k_per_group, int kh, int kw,
                 int pad_h, int pad_w)
    {
        // Setup Conv2dParams
        Conv2dParams par;
        par.direction = direction;
        par.n = N;
        par.h = H;
        par.w = W;
        par.c = groups * c_per_group;
        par.k = groups * k_per_group;
        par.kh = kh;
        par.kw = kw;
        par.pad_h = pad_h;
        par.pad_w = pad_w;
        par.stride_h = 1;
        par.stride_w = 1;
        par.dilation_h = 1;
        par.dilation_w = 1;
        par.groups = groups;
        par.in_type = DataType::fp16;
        par.wei_type = DataType::fp16;
        par.out_type = DataType::fp16;
        par.order = TensorOrder::NHWC;
        par.compute_output_size();

        if(!par.is_valid())
            return false;

        auto variant = grouped_4c::make_variant();
        if(!variant.is_applicable(par))
            return false;

        // Find first valid config
        int config_idx = -1;
        for(int i = 0; i < variant.num_configs; i++)
        {
            if(variant.config_is_compatible(par, i))
            {
                config_idx = i;
                break;
            }
        }
        if(config_idx < 0)
            return false;

        auto lp = variant.get_launch_params(config_idx, par);

        int Ho = par.p;
        int Wo = par.q;
        int C_total = par.c;
        int K_total = par.k;

        // Tensor sizes in elements (all in flat NHWGC / GKYXC / NHWGK layout)
        std::size_t in_size  = static_cast<std::size_t>(N * H * W * C_total);
        std::size_t wei_size = static_cast<std::size_t>(K_total * kh * kw * c_per_group);
        std::size_t out_size = static_cast<std::size_t>(N * Ho * Wo * K_total);

        // Fill host buffers with random data
        std::vector<HalfT> h_in(in_size);
        std::vector<HalfT> h_wei(wei_size);
        HostTensor<HalfT> t_in({in_size});
        HostTensor<HalfT> t_wei({wei_size});
        FillUniformDistribution<HalfT>{-1.f, 1.f}(t_in);
        FillUniformDistribution<HalfT>{-1.f, 1.f}(t_wei);
        std::copy(t_in.begin(), t_in.end(), h_in.begin());
        std::copy(t_wei.begin(), t_wei.end(), h_wei.begin());

        if(direction == Direction::Dgrad)
        {
            // For Dgrad:
            //   "input" to the kernel = output gradient, shape [N][Ho][Wo][K_total]
            //   "output" from the kernel = input gradient, shape [N][H][W][C_total]
            // We fill "output gradient" with random data (reuse h_in buffer concept but
            // we need out_size elements for the gradient input to the kernel).
            std::vector<HalfT> h_out_grad(out_size);
            HostTensor<HalfT> t_out_grad({out_size});
            FillUniformDistribution<HalfT>{-1.f, 1.f}(t_out_grad);
            std::copy(t_out_grad.begin(), t_out_grad.end(), h_out_grad.begin());

            // Allocate device memory
            DeviceMem d_out_grad(out_size * sizeof(HalfT));
            DeviceMem d_wei(wei_size * sizeof(HalfT));
            DeviceMem d_in_grad(in_size * sizeof(HalfT));
            DeviceMem d_ref_in_grad(in_size * sizeof(HalfT));

            d_out_grad.ToDevice(h_out_grad.data());
            d_wei.ToDevice(h_wei.data());
            d_in_grad.SetZero();
            d_ref_in_grad.SetZero();

            // Run GPU reference: naive_grouped_conv_bwd_data
            naive_grouped_conv_bwd_data<2>(
                static_cast<HalfT*>(d_ref_in_grad.GetDeviceBuffer()),
                static_cast<const HalfT*>(d_wei.GetDeviceBuffer()),
                static_cast<const HalfT*>(d_out_grad.GetDeviceBuffer()),
                groups, N, k_per_group, c_per_group,
                {(long_index_t)H, (long_index_t)W},
                {(long_index_t)kh, (long_index_t)kw},
                {(long_index_t)Ho, (long_index_t)Wo},
                {(long_index_t)par.stride_h, (long_index_t)par.stride_w},
                {(long_index_t)par.dilation_h, (long_index_t)par.dilation_w},
                {(long_index_t)pad_h, (long_index_t)pad_w});

            // Run streaming kernel
            hipStream_t stream = nullptr;
            variant.launch(config_idx, lp, par,
                           d_out_grad.GetDeviceBuffer(), d_wei.GetDeviceBuffer(),
                           d_in_grad.GetDeviceBuffer(), nullptr, stream);
            hip_check_error(hipDeviceSynchronize());

            // Copy results to host and compare
            std::vector<HalfT> h_result(in_size);
            std::vector<HalfT> h_ref(in_size);
            d_in_grad.FromDevice(h_result.data());
            d_ref_in_grad.FromDevice(h_ref.data());

            return check_err(h_result, h_ref,
                             "Error: Dgrad incorrect results!", 1e-2, 1e-2);
        }
        else
        {
            // Fprop
            DeviceMem d_in(in_size * sizeof(HalfT));
            DeviceMem d_wei_dev(wei_size * sizeof(HalfT));
            DeviceMem d_out(out_size * sizeof(HalfT));
            DeviceMem d_ref_out(out_size * sizeof(HalfT));

            d_in.ToDevice(h_in.data());
            d_wei_dev.ToDevice(h_wei.data());
            d_out.SetZero();
            d_ref_out.SetZero();

            // Run GPU reference: naive_grouped_conv_fwd
            naive_grouped_conv_fwd<2>(
                static_cast<const HalfT*>(d_in.GetDeviceBuffer()),
                static_cast<const HalfT*>(d_wei_dev.GetDeviceBuffer()),
                static_cast<HalfT*>(d_ref_out.GetDeviceBuffer()),
                groups, N, k_per_group, c_per_group,
                {(long_index_t)H, (long_index_t)W},
                {(long_index_t)kh, (long_index_t)kw},
                {(long_index_t)Ho, (long_index_t)Wo},
                {(long_index_t)par.stride_h, (long_index_t)par.stride_w},
                {(long_index_t)par.dilation_h, (long_index_t)par.dilation_w},
                {(long_index_t)pad_h, (long_index_t)pad_w});

            // Run streaming kernel
            hipStream_t stream = nullptr;
            variant.launch(config_idx, lp, par,
                           d_in.GetDeviceBuffer(), d_wei_dev.GetDeviceBuffer(),
                           d_out.GetDeviceBuffer(), nullptr, stream);
            hip_check_error(hipDeviceSynchronize());

            // Copy results to host and compare
            std::vector<HalfT> h_result(out_size);
            std::vector<HalfT> h_ref(out_size);
            d_out.FromDevice(h_result.data());
            d_ref_out.FromDevice(h_ref.data());

            return check_err(h_result, h_ref,
                             "Error: Fprop incorrect results!", 1e-2, 1e-2);
        }
    }
};

// Minimum groups = 16 (block_groups = waves_c64 * 16, min waves_c64 = 1)
// Groups must be a multiple of 16 for waves_c64=1, or 32 for waves_c64=2.

TEST_F(DirectConvGrouped4cFp16Test, Fprop_SmallSmoke)
{
    ASSERT_TRUE(RunTest(Direction::Fprop, 1, 8, 8, 16, 4, 4, 3, 3, 1, 1));
}

TEST_F(DirectConvGrouped4cFp16Test, Fprop_NoPadding)
{
    ASSERT_TRUE(RunTest(Direction::Fprop, 1, 8, 8, 16, 4, 4, 3, 3, 0, 0));
}

TEST_F(DirectConvGrouped4cFp16Test, Fprop_LargerBatch)
{
    ASSERT_TRUE(RunTest(Direction::Fprop, 4, 16, 16, 32, 4, 4, 3, 3, 1, 1));
}

TEST_F(DirectConvGrouped4cFp16Test, Fprop_ManyGroups)
{
    ASSERT_TRUE(RunTest(Direction::Fprop, 1, 8, 8, 64, 4, 4, 3, 3, 1, 1));
}

TEST_F(DirectConvGrouped4cFp16Test, Dgrad_SmallSmoke)
{
    ASSERT_TRUE(RunTest(Direction::Dgrad, 1, 8, 8, 16, 4, 4, 3, 3, 1, 1));
}

TEST_F(DirectConvGrouped4cFp16Test, Dgrad_NoPadding)
{
    ASSERT_TRUE(RunTest(Direction::Dgrad, 1, 8, 8, 16, 4, 4, 3, 3, 0, 0));
}

TEST_F(DirectConvGrouped4cFp16Test, Dgrad_LargerBatch)
{
    ASSERT_TRUE(RunTest(Direction::Dgrad, 4, 16, 16, 32, 4, 4, 3, 3, 1, 1));
}
