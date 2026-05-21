// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile::test::warp_gemm {

template <typename A,
          typename B,
          typename Acc,
          index_t M,
          index_t N,
          index_t K,
          bool TransposeC,
          bool SwizzleA              = false,
          bool UseStructuredSparsity = false,
          WGAttrNumAccessEnum NA     = WGAttrNumAccessEnum::Single,
          bool UseScale              = true>
struct WGDispCase
{
    using AType                              = A;
    using BType                              = B;
    using AccType                            = Acc;
    static constexpr index_t MPerWave        = M;
    static constexpr index_t NPerWave        = N;
    static constexpr index_t KPerWave        = K;
    static constexpr bool kTransposeC        = TransposeC;
    static constexpr bool kSwizzleA          = SwizzleA;
    static constexpr bool kUSS               = UseStructuredSparsity;
    static constexpr WGAttrNumAccessEnum kNA = NA;
    static constexpr bool kUseScale          = UseScale;
};

template <typename Case>
struct WarpGemmKernel
{
    static constexpr int kBlockSize = 64;
    __device__ void operator()(void* A, void* B, void* C, void* ScaleA, void* ScaleB) const
    {
        using WarpGemm = ck_tile::WarpGemmDispatcher<typename Case::AType,
                                                     typename Case::BType,
                                                     typename Case::AccType,
                                                     Case::MPerWave,
                                                     Case::NPerWave,
                                                     Case::KPerWave,
                                                     Case::kTransposeC,
                                                     Case::kSwizzleA,
                                                     Case::kUSS,
                                                     Case::kNA>;

        const auto a_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            static_cast<typename Case::AType*>(A),
            ck_tile::make_tuple(Case::MPerWave, Case::KPerWave),
            ck_tile::make_tuple(Case::KPerWave, ck_tile::number<1>{}),
            ck_tile::number<Case::KPerWave>{},
            ck_tile::number<1>{});
        const auto b_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            static_cast<typename Case::BType*>(B),
            ck_tile::make_tuple(Case::NPerWave, Case::KPerWave),
            ck_tile::make_tuple(Case::KPerWave, ck_tile::number<1>{}),
            ck_tile::number<Case::KPerWave>{},
            ck_tile::number<1>{});
        const auto c_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            static_cast<typename Case::AccType*>(C),
            ck_tile::make_tuple(Case::MPerWave, Case::NPerWave),
            ck_tile::make_tuple(Case::NPerWave, ck_tile::number<1>{}),
            ck_tile::number<Case::NPerWave>{},
            ck_tile::number<1>{});

        using AWarpTensor = typename WarpGemm::AWarpTensor;
        using BWarpTensor = typename WarpGemm::BWarpTensor;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        constexpr auto a_len = AWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto b_len = BWarpTensor::get_tile_distribution().get_lengths();
        constexpr auto c_len = CWarpTensor::get_tile_distribution().get_lengths();

        auto a_win = ck_tile::make_tile_window(
            a_view, a_len, ck_tile::make_multi_index(0, 0), AWarpTensor::get_tile_distribution());
        auto b_win = ck_tile::make_tile_window(
            b_view, b_len, ck_tile::make_multi_index(0, 0), BWarpTensor::get_tile_distribution());
        auto c_win = ck_tile::make_tile_window(
            c_view, c_len, ck_tile::make_multi_index(0, 0), CWarpTensor::get_tile_distribution());

        AWarpTensor a_tile;
        BWarpTensor b_tile;
        ck_tile::load_tile(a_tile, a_win);
        ck_tile::load_tile(b_tile, b_win);

        const auto c_tile = [&]() {
            if constexpr(Case::kUseScale)
            {
                const auto scale_a =
                    static_cast<int32_t>(static_cast<ck_tile::e8m0_t*>(ScaleA)[0].get());
                const auto scale_b =
                    static_cast<int32_t>(static_cast<ck_tile::e8m0_t*>(ScaleB)[0].get());
                return WarpGemm{}.template operator()<OpSelA<0>, OpSelB<0>>(
                    a_tile, b_tile, scale_a, scale_b);
            }
            else
            {
                ck_tile::ignore = ScaleA;
                ck_tile::ignore = ScaleB;
                return WarpGemm{}.template operator()<OpSelA<0>, OpSelB<0>>(a_tile, b_tile);
            }
        }();

        ck_tile::store_tile(c_win, c_tile);
    }
};

template <typename Case>
void RunWarpGemmCase(const ck_tile::HostTensor<typename Case::AType>& A,
                     const ck_tile::HostTensor<typename Case::BType>& B,
                     const ck_tile::HostTensor<ck_tile::e8m0_t>& ScaleA,
                     const ck_tile::HostTensor<ck_tile::e8m0_t>& ScaleB,
                     ck_tile::HostTensor<typename Case::AccType>& C)
{
    ck_tile::DeviceMem Ad(A), Bd(B), Cd(C), SAd(ScaleA), SBd(ScaleB);
    dim3 grid(1), block{64};

    (void)ck_tile::launch_kernel(ck_tile::stream_config{nullptr, true, 0, 0, 1},
                                 ck_tile::make_kernel(WarpGemmKernel<Case>{},
                                                      grid,
                                                      block,
                                                      0,
                                                      Ad.GetDeviceBuffer(),
                                                      Bd.GetDeviceBuffer(),
                                                      Cd.GetDeviceBuffer(),
                                                      SAd.GetDeviceBuffer(),
                                                      SBd.GetDeviceBuffer()));

    Cd.FromDevice(C.mData.data());
}

template <typename Case>
void RunCompareDispatcherAndReference()
{
    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType;

    constexpr index_t M = Case::MPerWave;
    constexpr index_t N = Case::NPerWave;
    constexpr index_t K = Case::KPerWave;

    const auto ScaleA = ck_tile::e8m0_t{2.f};
    const auto ScaleB = ck_tile::e8m0_t{4.f};

    ck_tile::HostTensor<AType> A({M, K});
    ck_tile::HostTensor<BType> B({N, K});
    ck_tile::HostTensor<CType> C({M, N});
    ck_tile::HostTensor<ck_tile::e8m0_t> sA({M, 1});
    ck_tile::HostTensor<ck_tile::e8m0_t> sB({N, 1});

    ck_tile::FillUniformDistribution<AType>{-5.f, 5.f}(A);
    ck_tile::FillUniformDistribution<BType>{-5.f, 5.f}(B);
    C.SetZero();
    ck_tile::FillConstant<ck_tile::e8m0_t>{ScaleA}(sA);
    ck_tile::FillConstant<ck_tile::e8m0_t>{ScaleB}(sB);

    RunWarpGemmCase<Case>(A, B, sA, sB, C);

    ck_tile::HostTensor<CType> C_ref({M, N});
    C_ref.SetZero();

    if constexpr(Case::kUseScale)
    {
        ck_tile::reference_mx_gemm<AType, BType, ck_tile::e8m0_t, ck_tile::e8m0_t, CType, CType>(
            A, B.transpose(), C_ref, sA, sB.transpose());
    }
    else
    {
        ck_tile::reference_gemm<AType, BType, CType, CType>(A, B.transpose(), C_ref);
    }

    EXPECT_TRUE(ck_tile::check_err(C, C_ref, "Warp gemm result error."));
}

} // namespace ck_tile::test::warp_gemm
