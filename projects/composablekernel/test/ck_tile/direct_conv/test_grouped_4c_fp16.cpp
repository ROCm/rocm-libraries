// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/fill.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/reference/reference_grouped_conv_fwd.hpp"
#include "ck_tile/host/reference/reference_grouped_conv_bwd_data.hpp"

#include "ck_tile/ops/direct_convolution/kernel/grouped_4c_fp16_kernel.hpp"

using namespace ck_tile;
using namespace ck_tile::direct_conv;

class DirectConvGrouped4cFp16Test : public ::testing::Test
{
    protected:
    using HalfT = ck_tile::half_t;

    // Repack from HostTensor(G,N,C,H,W) to flat [N][H][W][G*C] buffer
    static void repack_gnchw_to_nhwgc(const HostTensor<HalfT>& src, HalfT* dst)
    {
        auto G = src.get_lengths()[0];
        auto N = src.get_lengths()[1];
        auto C = src.get_lengths()[2];
        auto H = src.get_lengths()[3];
        auto W = src.get_lengths()[4];
        auto GC = G * C;
        for(std::size_t n = 0; n < N; n++)
            for(std::size_t h = 0; h < H; h++)
                for(std::size_t w = 0; w < W; w++)
                    for(std::size_t g = 0; g < G; g++)
                        for(std::size_t c = 0; c < C; c++)
                            dst[((n * H + h) * W + w) * GC + g * C + c] = src(g, n, c, h, w);
    }

    // Repack from HostTensor(G,K,C,Y,X) to flat [G*K][Y][X][C] buffer (KYXC)
    static void repack_gkcyx_to_kyxc(const HostTensor<HalfT>& src, HalfT* dst)
    {
        auto G = src.get_lengths()[0];
        auto K = src.get_lengths()[1];
        auto C = src.get_lengths()[2];
        auto Y = src.get_lengths()[3];
        auto X = src.get_lengths()[4];
        for(std::size_t g = 0; g < G; g++)
            for(std::size_t k = 0; k < K; k++)
                for(std::size_t y = 0; y < Y; y++)
                    for(std::size_t x = 0; x < X; x++)
                        for(std::size_t c = 0; c < C; c++)
                            dst[((g * K + k) * Y + y) * X * C + x * C + c] = src(g, k, c, y, x);
    }

    // Repack from flat [N][H][W][G*K] buffer to HostTensor(G,N,K,H,W)
    static void repack_nhwgk_to_gnkhw(const HalfT* src, HostTensor<HalfT>& dst)
    {
        auto G = dst.get_lengths()[0];
        auto N = dst.get_lengths()[1];
        auto K = dst.get_lengths()[2];
        auto H = dst.get_lengths()[3];
        auto W = dst.get_lengths()[4];
        auto GK = G * K;
        for(std::size_t n = 0; n < N; n++)
            for(std::size_t h = 0; h < H; h++)
                for(std::size_t w = 0; w < W; w++)
                    for(std::size_t g = 0; g < G; g++)
                        for(std::size_t k = 0; k < K; k++)
                            dst(g, n, k, h, w) = src[((n * H + h) * W + w) * GK + g * K + k];
    }

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

        // Create reference tensors in grouped layout [G, N, C_per_group, H, W]
        HostTensor<HalfT> ref_input({(std::size_t)groups, (std::size_t)N, (std::size_t)c_per_group,
                                      (std::size_t)H, (std::size_t)W});
        HostTensor<HalfT> ref_weight({(std::size_t)groups, (std::size_t)k_per_group,
                                       (std::size_t)c_per_group, (std::size_t)kh, (std::size_t)kw});
        HostTensor<HalfT> ref_output({(std::size_t)groups, (std::size_t)N, (std::size_t)k_per_group,
                                       (std::size_t)Ho, (std::size_t)Wo});

        FillUniformDistribution<HalfT>{-1.f, 1.f}(ref_input);
        FillUniformDistribution<HalfT>{-1.f, 1.f}(ref_weight);

        // Flat buffers for kernel (NHWC layout)
        int C_total = par.c;
        int K_total = par.k;
        std::size_t in_size = (std::size_t)N * H * W * C_total;
        std::size_t wei_size = (std::size_t)K_total * kh * kw * c_per_group;
        std::size_t out_size = (std::size_t)N * Ho * Wo * K_total;

        std::vector<HalfT> flat_in(in_size);
        std::vector<HalfT> flat_wei(wei_size);
        std::vector<HalfT> flat_out(out_size, HalfT(0));

        repack_gnchw_to_nhwgc(ref_input, flat_in.data());
        repack_gkcyx_to_kyxc(ref_weight, flat_wei.data());

        if(direction == Direction::Dgrad)
        {
            // For Dgrad, "in" to the kernel is the output gradient (ref_output),
            // and "out" from the kernel is the input gradient.
            // We need to fill ref_output (the output gradient) and repack it as "in".
            FillUniformDistribution<HalfT>{-1.f, 1.f}(ref_output);
            // Repack ref_output as the kernel's "in" tensor: [N][Ho][Wo][G*K] -> flat
            std::vector<HalfT> flat_dgrad_in(out_size);
            // Repack ref_output (G,N,K,Ho,Wo) -> flat [N][Ho][Wo][G*K]
            for(std::size_t n = 0; n < (std::size_t)N; n++)
                for(std::size_t ho = 0; ho < (std::size_t)Ho; ho++)
                    for(std::size_t wo = 0; wo < (std::size_t)Wo; wo++)
                        for(std::size_t g = 0; g < (std::size_t)groups; g++)
                            for(std::size_t k = 0; k < (std::size_t)k_per_group; k++)
                                flat_dgrad_in[((n * Ho + ho) * Wo + wo) * K_total + g * k_per_group + k] =
                                    ref_output(g, n, k, ho, wo);

            DeviceMem d_in(flat_dgrad_in.size() * sizeof(HalfT));
            DeviceMem d_wei(flat_wei.size() * sizeof(HalfT));
            DeviceMem d_out(in_size * sizeof(HalfT));
            d_in.ToDevice(flat_dgrad_in.data());
            d_wei.ToDevice(flat_wei.data());
            d_out.SetZero();

            hipStream_t stream = nullptr;
            variant.launch(config_idx, lp, par,
                           d_in.GetDeviceBuffer(), d_wei.GetDeviceBuffer(),
                           d_out.GetDeviceBuffer(), nullptr, stream);
            hipDeviceSynchronize();

            std::vector<HalfT> flat_result(in_size);
            d_out.FromDevice(flat_result.data());

            // Repack kernel output [N][H][W][G*C] -> ref layout (G,N,C,H,W)
            HostTensor<HalfT> kernel_result({(std::size_t)groups, (std::size_t)N,
                                              (std::size_t)c_per_group, (std::size_t)H, (std::size_t)W});
            repack_nhwgk_to_gnkhw(flat_result.data(), kernel_result);

            // Run reference bwd_data
            HostTensor<HalfT> ref_input_grad({(std::size_t)groups, (std::size_t)N,
                                               (std::size_t)c_per_group, (std::size_t)H, (std::size_t)W});
            reference_grouped_conv_bwd_data<2>(
                ref_input_grad, ref_weight, ref_output,
                {1, 1}, {1, 1}, {(long_index_t)pad_h, (long_index_t)pad_w},
                {(long_index_t)pad_h, (long_index_t)pad_w});

            return check_err(kernel_result, ref_input_grad,
                             "Error: Dgrad incorrect results!", 1e-2, 1e-2);
        }
        else
        {
            // Fprop
            DeviceMem d_in(flat_in.size() * sizeof(HalfT));
            DeviceMem d_wei(flat_wei.size() * sizeof(HalfT));
            DeviceMem d_out(flat_out.size() * sizeof(HalfT));
            d_in.ToDevice(flat_in.data());
            d_wei.ToDevice(flat_wei.data());
            d_out.SetZero();

            hipStream_t stream = nullptr;
            variant.launch(config_idx, lp, par,
                           d_in.GetDeviceBuffer(), d_wei.GetDeviceBuffer(),
                           d_out.GetDeviceBuffer(), nullptr, stream);
            hipDeviceSynchronize();

            d_out.FromDevice(flat_out.data());

            // Repack kernel output to grouped layout for comparison
            HostTensor<HalfT> kernel_result({(std::size_t)groups, (std::size_t)N,
                                              (std::size_t)k_per_group, (std::size_t)Ho, (std::size_t)Wo});
            repack_nhwgk_to_gnkhw(flat_out.data(), kernel_result);

            // Run reference forward
            ref_output.SetZero();
            reference_grouped_conv_fwd<2>(
                ref_input, ref_weight, ref_output,
                {1, 1}, {1, 1}, {(long_index_t)pad_h, (long_index_t)pad_w},
                {(long_index_t)pad_h, (long_index_t)pad_w});

            return check_err(kernel_result, ref_output,
                             "Error: Fprop incorrect results!", 1e-2, 1e-2);
        }
    }
};

TEST_F(DirectConvGrouped4cFp16Test, Fprop_SmallSmoke)
{
    ASSERT_TRUE(RunTest(Direction::Fprop, 1, 8, 8, 4, 4, 4, 3, 3, 1, 1));
}

TEST_F(DirectConvGrouped4cFp16Test, Fprop_NoPadding)
{
    ASSERT_TRUE(RunTest(Direction::Fprop, 1, 8, 8, 4, 4, 4, 3, 3, 0, 0));
}

TEST_F(DirectConvGrouped4cFp16Test, Fprop_LargerBatch)
{
    ASSERT_TRUE(RunTest(Direction::Fprop, 4, 16, 16, 8, 4, 4, 3, 3, 1, 1));
}

TEST_F(DirectConvGrouped4cFp16Test, Fprop_ManyGroups)
{
    ASSERT_TRUE(RunTest(Direction::Fprop, 1, 8, 8, 32, 4, 4, 3, 3, 1, 1));
}

TEST_F(DirectConvGrouped4cFp16Test, Dgrad_SmallSmoke)
{
    ASSERT_TRUE(RunTest(Direction::Dgrad, 1, 8, 8, 4, 4, 4, 3, 3, 1, 1));
}

TEST_F(DirectConvGrouped4cFp16Test, Dgrad_NoPadding)
{
    ASSERT_TRUE(RunTest(Direction::Dgrad, 1, 8, 8, 4, 4, 4, 3, 3, 0, 0));
}

TEST_F(DirectConvGrouped4cFp16Test, Dgrad_LargerBatch)
{
    ASSERT_TRUE(RunTest(Direction::Dgrad, 4, 16, 16, 8, 4, 4, 3, 3, 1, 1));
}
