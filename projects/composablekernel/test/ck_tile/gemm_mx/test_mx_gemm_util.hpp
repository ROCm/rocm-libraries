// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "test_mx_gemm_config.hpp"
#include "test_mx_gemm_instance.hpp"

template <typename Layout>
static constexpr auto is_row_major(Layout)
{
    return ck_tile::bool_constant<
        std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol_mx(ck_tile::index_t K, float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(K);
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value, K);
    return ck_tile::make_tuple(rtol, atol);
}

template <typename GemmConfig>
struct TestMXGemmArchTraits
{
    template <typename dtype>
    static auto preShuffleWeight(ck_tile::HostTensor<dtype>& src)
    {
        constexpr ck_tile::index_t NLane = GemmConfig::N_Warp_Tile;
        auto src_lengths                 = src.get_lengths();
        const int K                      = src_lengths[0];
        const int N                      = src_lengths[1];
        constexpr int packed_size        = ck_tile::numeric_traits<dtype>::PackedSize;
        const int KPack = std::is_same_v<dtype, ck_tile::pk_fp6x16_t> ? 32 : 16 * packed_size;

        const int KLane = ck_tile::get_warp_size() / NLane;
        const int K0    = K / (KLane * KPack);

        ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor(
            {static_cast<std::size_t>(N * K)}, {static_cast<std::size_t>(1)}));

        for(int n = 0; n < N; ++n)
        {
            for(int k = 0; k < K; k += packed_size)
            {
                const int n0 = n / NLane;
                const int n1 = n % NLane;

                const int k0    = k / (KLane * KPack);
                const int tempk = k % (KLane * KPack);
                const int k1    = tempk / KPack;
                const int k2    = tempk % KPack;

                const int outputIndex = n0 * KPack * NLane * KLane * K0 +
                                        k0 * KPack * NLane * KLane + k1 * KPack * NLane +
                                        n1 * KPack + k2;

                shuffled(outputIndex) = src(k, n);
            }
        }

        return shuffled;
    }

    template <bool KLast, typename dtype>
    static auto preShuffleScale(ck_tile::HostTensor<dtype>& src)
    {
        auto src_lengths = src.get_lengths();
        const auto MN    = KLast ? src_lengths[0] : src_lengths[1];
        const auto K     = KLast ? src_lengths[1] : src_lengths[0];

        constexpr std::size_t MNXdlPack   = 2;
        constexpr std::size_t KXdlPack    = 2;
        constexpr std::size_t XdlMNThread = GemmConfig::N_Warp_Tile;
        constexpr std::size_t XdlKThread  = ck_tile::get_warp_size() / XdlMNThread;

        const auto MNPadded = ck_tile::integer_least_multiple(MN, XdlMNThread * MNXdlPack);
        ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor(
            {static_cast<std::size_t>(MNPadded * K)}, {static_cast<std::size_t>(1)}));

        const std::size_t K0 = K / KXdlPack / XdlKThread;

        for(std::size_t n = 0; n < static_cast<std::size_t>(MNPadded); ++n)
        {
            for(std::size_t k = 0; k < static_cast<std::size_t>(K); ++k)
            {
                const auto n0    = n / (XdlMNThread * MNXdlPack);
                const auto tempn = n % (XdlMNThread * MNXdlPack);
                const auto n1    = tempn % XdlMNThread;
                const auto n2    = tempn / XdlMNThread;

                const auto k0    = k / (XdlKThread * KXdlPack);
                const auto tempk = k % (XdlKThread * KXdlPack);
                const auto k1    = tempk % XdlKThread;
                const auto k2    = tempk / XdlKThread;

                const auto outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                                         k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                                         k1 * MNXdlPack * KXdlPack * XdlMNThread +
                                         n1 * MNXdlPack * KXdlPack + k2 * MNXdlPack + n2;

                if constexpr(KLast)
                    shuffled(outputIndex) = n < static_cast<std::size_t>(MN) ? src(n, k) : dtype{};
                else
                    shuffled(outputIndex) = n < static_cast<std::size_t>(MN) ? src(k, n) : dtype{};
            }
        }

        return shuffled;
    }
};

template <typename ADataType,
          typename BDataType,
          typename GemmConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout>
class TestMxGemmUtil : public ::testing::Test
{
    protected:
    using AccDataType = float;
    using CDataType   = ck_tile::fp16_t;
    using ScaleType   = ck_tile::e8m0_t;
    using ScaleM      = ck_tile::MXScalePointer<ScaleType, 1, 32>;
    using ScaleN      = ck_tile::MXScalePointer<ScaleType, 1, 32>;

    void Run(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K, int seed = 1234)
    {
        const ck_tile::index_t scale_k_size = K / 32;
        const ck_tile::index_t stride_A =
            ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
        const ck_tile::index_t stride_B =
            ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
        const ck_tile::index_t stride_C =
            ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));
        const ck_tile::index_t stride_scale_a =
            ck_tile::get_default_stride(M, scale_k_size, 0, is_row_major(ALayout{}));
        const ck_tile::index_t stride_scale_b =
            ck_tile::get_default_stride(scale_k_size, N, 0, is_row_major(BLayout{}));

        ck_tile::HostTensor<ADataType> a_host(
            ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_host(
            ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
        ck_tile::HostTensor<CDataType> c_host(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        ck_tile::HostTensor<ScaleType> scale_a_host(ck_tile::host_tensor_descriptor(
            M, scale_k_size, stride_scale_a, is_row_major(ALayout{})));
        ck_tile::HostTensor<ScaleType> scale_b_host(ck_tile::host_tensor_descriptor(
            scale_k_size, N, stride_scale_b, is_row_major(BLayout{})));

        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_host);
        ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_a_host);
        ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_b_host);

        const auto b_host_for_device = [&]() {
            if constexpr(GemmConfig::Preshuffle)
                return TestMXGemmArchTraits<GemmConfig>::preShuffleWeight(b_host);
            else
                return b_host;
        }();

        const auto scale_a_host_for_device = [&]() {
            if constexpr(GemmConfig::Preshuffle)
                return TestMXGemmArchTraits<GemmConfig>::template preShuffleScale<true>(
                    scale_a_host);
            else
                return scale_a_host;
        }();

        const auto scale_b_host_for_device = [&]() {
            if constexpr(GemmConfig::Preshuffle)
                return TestMXGemmArchTraits<GemmConfig>::template preShuffleScale<false>(
                    scale_b_host);
            else
                return scale_b_host;
        }();

        ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_dev_buf(b_host_for_device.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_dev_buf(c_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem scale_a_dev_buf(
            scale_a_host_for_device.get_element_space_size_in_bytes());
        ck_tile::DeviceMem scale_b_dev_buf(
            scale_b_host_for_device.get_element_space_size_in_bytes());

        a_dev_buf.ToDevice(a_host.data());
        b_dev_buf.ToDevice(b_host_for_device.data());
        c_dev_buf.SetZero();
        scale_a_dev_buf.ToDevice(scale_a_host_for_device.data());
        scale_b_dev_buf.ToDevice(scale_b_host_for_device.data());

        ScaleM scale_m(reinterpret_cast<ScaleType*>(scale_a_dev_buf.GetDeviceBuffer()));
        ScaleN scale_n(reinterpret_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()));

        MXGemmHostArgs<ScaleM, ScaleN> args(a_dev_buf.GetDeviceBuffer(),
                                            b_dev_buf.GetDeviceBuffer(),
                                            c_dev_buf.GetDeviceBuffer(),
                                            1,
                                            M,
                                            N,
                                            K,
                                            stride_A,
                                            stride_B,
                                            stride_C,
                                            scale_m,
                                            scale_n);

        if constexpr(!GemmConfig::Preshuffle)
        {
            mx_gemm_calc<GemmConfig,
                         ADataType,
                         BDataType,
                         AccDataType,
                         CDataType,
                         ALayout,
                         BLayout,
                         CLayout,
                         ScaleM,
                         ScaleN,
                         true,
                         false>(args,
                                ck_tile::stream_config{nullptr, true, 1, 0, 1, true, true, 50});
        }
        else
        {
            using GemmShape = ck_tile::TileGemmShape<
                ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
                ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
                ck_tile::sequence<GemmConfig::M_Warp_Tile,
                                  GemmConfig::N_Warp_Tile,
                                  GemmConfig::K_Warp_Tile>>;
            using TilePartitioner =
                ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                           GemmConfig::TileParitionerGroupNum,
                                                           GemmConfig::TileParitionerM01>;

            const ck_tile::index_t k_pad =
                (K + GemmConfig::K_Tile - 1) / GemmConfig::K_Tile * GemmConfig::K_Tile;
            const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(k_pad);
            const bool has_hot_loop =
                ck_tile::BaseMXGemmPipelineAGmemBGmemCRegV1::BlockHasHotloop(num_loop);
            const ck_tile::TailNumber tail_num =
                ck_tile::BaseMXGemmPipelineAGmemBGmemCRegV1::GetBlockLoopTailNum(num_loop);

            ck_tile::BaseMXGemmPipelineAGmemBGmemCRegV1::template TailHandler<true>(
                [&](auto has_hot_loop_, auto tail_num_) {
                    constexpr auto has_hot_loop_v = has_hot_loop_.value;
                    constexpr auto tail_num_v     = tail_num_.value;

                    return mx_gemm_calc<GemmConfig,
                                        ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        CLayout,
                                        ScaleM,
                                        ScaleN,
                                        true,
                                        false,
                                        has_hot_loop_v,
                                        tail_num_v>(
                        args, ck_tile::stream_config{nullptr, true, 1, 0, 1, true, true, 50});
                },
                has_hot_loop,
                tail_num);
        }

        c_dev_buf.FromDevice(c_host.data());

        ck_tile::HostTensor<CDataType> c_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_ref.SetZero();
        ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, AccDataType, CDataType>(
            a_host, b_host, c_ref, scale_a_host, scale_b_host);

        const float max_accumulated_value = ck_tile::type_convert<float>(c_ref.max());
        const auto rtol_atol = calculate_rtol_atol_mx<ADataType, BDataType, AccDataType, CDataType>(
            K, max_accumulated_value);
        const double rtol = rtol_atol.at(ck_tile::number<0>{});
        const double atol = rtol_atol.at(ck_tile::number<1>{});

        bool pass = ck_tile::check_err(c_host, c_ref, "MX GEMM: Incorrect results!", rtol, atol);

        EXPECT_TRUE(pass);
    }
};
