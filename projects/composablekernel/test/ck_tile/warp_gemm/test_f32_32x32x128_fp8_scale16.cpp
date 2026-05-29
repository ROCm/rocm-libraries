// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Unit test for a 32x32x128 block-scaled GEMM built from the 16x16x128 scale16
// warp gemm (V_WMMA_SCALE16_F32_16X16X128_F8F6F4), modeled after the MX GEMM
// pipelines (BlockGemmARegBRegCRegV1 / mx_flatmm pipeline): all A/B sub-tiles and
// their per-block K-scales for the 2x2 block-level loop are preloaded into
// registers in advance, then consumed by the warp-gemm loop. OpSelA/OpSelB are
// threaded through the warp-gemm call as the pipeline does; for a 16x16 tile the
// hardware A/B sub-block index is 0, so per-M-block / per-N-block scale selection
// is realized by indexing the preloaded per-block scales (each int64 carries that
// block's 8 e8m0 K-scales).

#include <gtest/gtest.h>
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

using namespace ck_tile;

template <index_t NumScales>
CK_TILE_DEVICE static constexpr auto MakeScaleDistribution()
{
    return make_static_tile_distribution(
        tile_distribution_encoding<sequence<2>,
                                   tuple<sequence<16>, sequence<NumScales>>,
                                   tuple<sequence<0, 1>>,
                                   tuple<sequence<0, 0>>,
                                   sequence<2>,
                                   sequence<0>>{});
}

// Pipeline-style kernel: preload all scales for the 2x2 block in advance, then
// run a 2x2 block-level loop over the 16x16x128 scale16 warp gemm. Each (mIter,
// nIter) sub-tile consumes its own independent per-M/per-N-block K-scales.
template <typename AType,
          typename BType,
          typename CType,
          index_t M,
          index_t N,
          index_t K,
          bool TransposeC>
struct WarpGemmScale16BlockLoopKernel
{
    static constexpr int kBlockSize      = 32;
    static constexpr index_t ScaleBlockK = 16;
    static constexpr index_t NumScales   = K / ScaleBlockK;
    static constexpr index_t MPerWarp    = 16;
    static constexpr index_t NPerWarp    = 16;
    static constexpr index_t MIter       = M / MPerWarp;
    static constexpr index_t NIter       = N / NPerWarp;

    __device__ void operator()(void* A, void* B, void* C, void* ScaleA, void* ScaleB) const
    {
        const auto a_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<AType*>(A),
                                                               make_tuple(M, K),
                                                               make_tuple(K, number<1>{}),
                                                               number<K>{},
                                                               number<1>{});
        const auto b_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<BType*>(B),
                                                               make_tuple(N, K),
                                                               make_tuple(K, number<1>{}),
                                                               number<K>{},
                                                               number<1>{});
        const auto c_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<CType*>(C),
                                                               make_tuple(M, N),
                                                               make_tuple(N, number<1>{}),
                                                               number<N>{},
                                                               number<1>{});
        const auto sa_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<e8m0_t*>(ScaleA),
                                                               make_tuple(M, NumScales),
                                                               make_tuple(NumScales, number<1>{}),
                                                               number<NumScales>{},
                                                               number<1>{});
        const auto sb_view =
            make_naive_tensor_view<address_space_enum::global>(static_cast<e8m0_t*>(ScaleB),
                                                               make_tuple(N, NumScales),
                                                               make_tuple(NumScales, number<1>{}),
                                                               number<NumScales>{},
                                                               number<1>{});

        using WarpGemm = WarpGemmDispatcher<AType,
                                            BType,
                                            float,
                                            MPerWarp,
                                            NPerWarp,
                                            K,
                                            TransposeC,
                                            false,
                                            false,
                                            WGAttrNumAccessEnum::Default,
                                            WGAttrNumAccessEnum::Default,
                                            scale16_tag>;

        constexpr auto a_dstr     = typename WarpGemm::AWarpDstr{};
        constexpr auto b_dstr     = typename WarpGemm::BWarpDstr{};
        constexpr auto c_dstr     = typename WarpGemm::CWarpDstr{};
        constexpr auto scale_dstr = MakeScaleDistribution<NumScales>();

        // ---- Preload phase ----
        // Stage every A/B sub-tile and its per-block K-scales into registers before
        // the compute loop, mirroring how the MX GEMM pipelines pre-stage scales
        // (each int64 holds one 16-row block's 8 e8m0 K-scales).
        statically_indexed_array<decltype(load_tile(make_tile_window(
                                     a_view,
                                     make_tuple(number<MPerWarp>{}, number<K>{}),
                                     make_multi_index(0, 0),
                                     a_dstr))),
                                 MIter>
            a_tiles;
        statically_indexed_array<int64_t, MIter> scale_a;
        static_for<0, MIter, 1>{}([&](auto mIter) {
            auto a_win  = make_tile_window(a_view,
                                          make_tuple(number<MPerWarp>{}, number<K>{}),
                                          make_multi_index(mIter.value * MPerWarp, 0),
                                          a_dstr);
            auto sa_win = make_tile_window(sa_view,
                                           make_tuple(number<MPerWarp>{}, number<NumScales>{}),
                                           make_multi_index(mIter.value * MPerWarp, 0),
                                           scale_dstr);
            a_tiles(mIter)   = load_tile(a_win);
            auto sa_tile     = load_tile(sa_win);
            scale_a(mIter)   = bit_cast<int64_t>(
                sa_tile.get_thread_buffer()
                    .template get_as<ext_vector_t<e8m0_t, NumScales>>()[number<0>{}]);
        });

        statically_indexed_array<decltype(load_tile(make_tile_window(
                                     b_view,
                                     make_tuple(number<NPerWarp>{}, number<K>{}),
                                     make_multi_index(0, 0),
                                     b_dstr))),
                                 NIter>
            b_tiles;
        statically_indexed_array<int64_t, NIter> scale_b;
        static_for<0, NIter, 1>{}([&](auto nIter) {
            auto b_win  = make_tile_window(b_view,
                                          make_tuple(number<NPerWarp>{}, number<K>{}),
                                          make_multi_index(nIter.value * NPerWarp, 0),
                                          b_dstr);
            auto sb_win = make_tile_window(sb_view,
                                           make_tuple(number<NPerWarp>{}, number<NumScales>{}),
                                           make_multi_index(nIter.value * NPerWarp, 0),
                                           scale_dstr);
            b_tiles(nIter)   = load_tile(b_win);
            auto sb_tile     = load_tile(sb_win);
            scale_b(nIter)   = bit_cast<int64_t>(
                sb_tile.get_thread_buffer()
                    .template get_as<ext_vector_t<e8m0_t, NumScales>>()[number<0>{}]);
        });

        // ---- Compute phase ----
        // 2x2 block-level loop, mimicking BlockGemmARegBRegCRegV1. All scales are
        // already register-resident. The proper per-M/per-N-block scale set is
        // selected by indexing the preloaded scale_a / scale_b. OpSelA/OpSelB (the
        // hardware A/B sub-block index) are 0 for a 16x16 tile, i.e. the warp-gemm
        // default -- the int64 scale itself fully specifies the block's K-scales.
        static_for<0, MIter, 1>{}([&](auto mIter) {
            static_for<0, NIter, 1>{}([&](auto nIter) {
                auto c_win = make_tile_window(
                    c_view,
                    make_tuple(number<MPerWarp>{}, number<NPerWarp>{}),
                    make_multi_index(mIter.value * MPerWarp, nIter.value * NPerWarp),
                    c_dstr);

                auto c_tile =
                    WarpGemm{}(a_tiles(mIter), b_tiles(nIter), scale_a(mIter), scale_b(nIter));
                store_tile(c_win, c_tile);
            });
        });
    }
};

template <typename A, typename B, typename Acc, index_t M, index_t N, index_t K>
struct WGDispCase
{
    using AType                       = A;
    using BType                       = B;
    using AccType                     = Acc;
    static constexpr index_t MPerWave = M;
    static constexpr index_t NPerWave = N;
    static constexpr index_t KPerWave = K;
};

template <typename Case, bool TransposeC>
static void RunTest(const HostTensor<typename Case::AType>& A,
                    const HostTensor<typename Case::BType>& B,
                    const HostTensor<e8m0_t>& ScaleA,
                    const HostTensor<e8m0_t>& ScaleB,
                    HostTensor<typename Case::AccType>& C)
{
    DeviceMem Ad(A), Bd(B), Cd(C), SAd(ScaleA), SBd(ScaleB);
    dim3 grid(1), block{32};

    using K = WarpGemmScale16BlockLoopKernel<typename Case::AType,
                                             typename Case::BType,
                                             typename Case::AccType,
                                             Case::MPerWave,
                                             Case::NPerWave,
                                             Case::KPerWave,
                                             TransposeC>;

    (void)launch_kernel(stream_config{nullptr, true, 0, 0, 1},
                        make_kernel(K{},
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

using WGDispatcherTypesList = ::testing::Types<WGDispCase<fp8_t, fp8_t, float, 32, 32, 128>,
                                               WGDispCase<bf8_t, bf8_t, float, 32, 32, 128>,
                                               WGDispCase<fp8_t, bf8_t, float, 32, 32, 128>,
                                               WGDispCase<bf8_t, fp8_t, float, 32, 32, 128>>;

template <typename Case>
class WGScale16Test : public ::testing::Test
{
};

TYPED_TEST_SUITE(WGScale16Test, WGDispatcherTypesList);

TYPED_TEST(WGScale16Test, Scale16_32x32x128_UniformScale)
{
    using Case  = TypeParam;
    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType;

    constexpr index_t M         = Case::MPerWave;
    constexpr index_t N         = Case::NPerWave;
    constexpr index_t K         = Case::KPerWave;
    constexpr index_t NumScales = K / 16;

    HostTensor<AType> A({M, K});
    HostTensor<BType> B({N, K});
    HostTensor<CType> C({M, N});
    HostTensor<e8m0_t> sA({M, NumScales});
    HostTensor<e8m0_t> sB({N, NumScales});

    FillUniformDistribution<AType>{-5.f, 5.f}(A);
    FillUniformDistribution<BType>{-5.f, 5.f}(B);
    C.SetZero();
    FillConstant<e8m0_t>{e8m0_t{2.f}}(sA);
    FillConstant<e8m0_t>{e8m0_t{4.f}}(sB);

    RunTest<Case, false>(A, B, sA, sB, C);

    HostTensor<CType> C_ref({M, N});
    C_ref.SetZero();
    reference_mx_gemm<AType, BType, e8m0_t, e8m0_t, CType, CType>(
        A, B.transpose(), C_ref, sA, sB.transpose());

    EXPECT_TRUE(check_err(C, C_ref, "Scale16 32x32x128 uniform scale error."));
}

TYPED_TEST(WGScale16Test, Scale16_32x32x128_RandomDataRandomScales)
{
    using Case  = TypeParam;
    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType;

    constexpr index_t M         = Case::MPerWave;
    constexpr index_t N         = Case::NPerWave;
    constexpr index_t K         = Case::KPerWave;
    constexpr index_t NumScales = K / 16;

    HostTensor<AType> A({M, K});
    HostTensor<BType> B({N, K});
    HostTensor<CType> C({M, N});
    HostTensor<e8m0_t> sA({M, NumScales});
    HostTensor<e8m0_t> sB({N, NumScales});

    FillUniformDistribution<AType>{-5.f, 5.f, 42}(A);
    FillUniformDistribution<BType>{-5.f, 5.f, 137}(B);
    C.SetZero();

    {
        constexpr int bias = ck_tile::numeric_traits<e8m0_t>::bias;
        std::mt19937 gen(9999);
        std::uniform_int_distribution<int> dist(bias - 4, bias + 2);
        for(auto& s : sA.mData)
            s = e8m0_t(static_cast<typename e8m0_t::type>(dist(gen)));
        for(auto& s : sB.mData)
            s = e8m0_t(static_cast<typename e8m0_t::type>(dist(gen)));
    }

    RunTest<Case, false>(A, B, sA, sB, C);

    HostTensor<CType> C_ref({M, N});
    C_ref.SetZero();
    reference_mx_gemm<AType, BType, e8m0_t, e8m0_t, CType, CType>(
        A, B.transpose(), C_ref, sA, sB.transpose());

    const float max_acc = *std::max_element(C_ref.mData.begin(), C_ref.mData.end());
    const auto rtol     = ck_tile::get_relative_threshold<AType, CType, CType>(K);
    const auto atol     = ck_tile::get_absolute_threshold<AType, CType, CType>(max_acc, K);
    EXPECT_TRUE(
        check_err(C, C_ref, "Scale16 32x32x128 random data + random scales error.", rtol, atol));
}

TYPED_TEST(WGScale16Test, Scale16_32x32x128_TransposeC)
{
    using Case  = TypeParam;
    using AType = typename Case::AType;
    using BType = typename Case::BType;
    using CType = typename Case::AccType;

    constexpr index_t M         = Case::MPerWave;
    constexpr index_t N         = Case::NPerWave;
    constexpr index_t K         = Case::KPerWave;
    constexpr index_t NumScales = K / 16;

    HostTensor<AType> A({M, K});
    HostTensor<BType> B({N, K});
    HostTensor<CType> C({M, N});
    HostTensor<e8m0_t> sA({M, NumScales});
    HostTensor<e8m0_t> sB({N, NumScales});

    FillUniformDistribution<AType>{-5.f, 5.f, 77}(A);
    FillUniformDistribution<BType>{-5.f, 5.f, 88}(B);
    C.SetZero();

    {
        constexpr int bias = ck_tile::numeric_traits<e8m0_t>::bias;
        std::mt19937 gen(5555);
        std::uniform_int_distribution<int> dist(bias - 4, bias + 2);
        for(auto& s : sA.mData)
            s = e8m0_t(static_cast<typename e8m0_t::type>(dist(gen)));
        for(auto& s : sB.mData)
            s = e8m0_t(static_cast<typename e8m0_t::type>(dist(gen)));
    }

    RunTest<Case, true>(A, B, sA, sB, C);

    HostTensor<CType> C_ref({M, N});
    C_ref.SetZero();
    reference_mx_gemm<AType, BType, e8m0_t, e8m0_t, CType, CType>(
        A, B.transpose(), C_ref, sA, sB.transpose());
    auto C_ref_T = C_ref.transpose();

    const float max_acc = *std::max_element(C_ref.mData.begin(), C_ref.mData.end());
    const auto rtol     = ck_tile::get_relative_threshold<AType, CType, CType>(K);
    const auto atol     = ck_tile::get_absolute_threshold<AType, CType, CType>(max_acc, K);
    EXPECT_TRUE(check_err(C, C_ref_T, "Scale16 32x32x128 TransposeC error.", rtol, atol));
}
