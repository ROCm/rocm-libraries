// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipelines.hpp"

#include "test_swiglu_fixture.hpp"

#include <gtest/gtest.h>
#include <cstddef>

template <typename Params_>
struct TestCkTileSwiGLU : public ::testing::Test
{
    using Params = Params_;

    using ALayout = Params::ALayout;
    using BLayout = Params::BLayout;
    using CLayout = Params::CLayout;

    constexpr static bool a_is_row_major =
        std::is_same<ALayout, ck_tile::tensor_layout::gemm::RowMajor>{};
    constexpr static bool b_is_row_major =
        std::is_same<BLayout, ck_tile::tensor_layout::gemm::RowMajor>{};
    constexpr static bool c_is_row_major =
        std::is_same<CLayout, ck_tile::tensor_layout::gemm::RowMajor>{};

    template <bool PadM, bool PadN, bool PadK>
    void RunWithPadding(std::size_t M, std::size_t N, std::size_t K)
    {
        ck_tile::with_padding<Params, PadM, PadN, PadK>().run_test({
            .m              = M,
            .n              = N,
            .k              = K,
            .a_is_row_major = a_is_row_major,
            .b_is_row_major = b_is_row_major,
            .c_is_row_major = c_is_row_major,
        });
    }

    template <typename ActMul>
    void RunWithActMul(std::size_t M, std::size_t N, std::size_t K)
    {
        ck_tile::with_act_mul<Params, ActMul>().run_test({
            .m              = M,
            .n              = N,
            .k              = K,
            .a_is_row_major = a_is_row_major,
            .b_is_row_major = b_is_row_major,
            .c_is_row_major = c_is_row_major,
        });
    }

    void Run(std::size_t M, std::size_t N, std::size_t K)
    {
        Params{}.run_test({
            .m              = M,
            .n              = N,
            .k              = K,
            .a_is_row_major = a_is_row_major,
            .b_is_row_major = b_is_row_major,
            .c_is_row_major = c_is_row_major,
        });
    }
};

namespace ck_tile {
struct MyActMulOp
{
    // SwishAndMul can give inf's for big matrices.
    // So for testing we use this operator instead.

    template <typename T>
    CK_TILE_HOST_DEVICE auto operator()(const T& x0, const T& x1) const -> T
    {
        return ck_tile::tanh(x0) * ck_tile::tanh(x1);
    }

    template <typename T>
    CK_TILE_HOST_DEVICE auto operator()(T& y, const T& x0, const T& x1) const -> void
    {
        y = this->operator()(x0, x1);
    }
};

using Intra = constant<GemmPipelineScheduler::Intrawave>;
using Inter = constant<GemmPipelineScheduler::Interwave>;
using R     = ck_tile::tensor_layout::gemm::RowMajor;
using C     = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename TIn, typename TOut, typename TAcc>
using dtypes = tuple<TIn, TIn, TOut, TAcc>;

template <typename ALayout, typename BLayout, typename CLayout>
using layouts = tuple<ALayout, BLayout, CLayout>;

template <bool PadM, bool PadN, bool PadK>
using padding = utils::boolseq<PadM, PadN, PadK>;

template <typename DTypes, typename Layouts>
using Test_Intra_V3 = decltype(TestParams{
    .dtypes       = DTypes{},
    .layouts      = Layouts{},
    .padding      = utils::boolseq<true, true, true>{},
    .act_mul      = MyActMulOp{},
    .pipeline     = constant<GemmPipeline::COMPUTE_V3>{},
    .scheduler    = Intra{},
    .block_per_cu = constant<1>{},
    .tile         = sequence<256, 256, 64>{},
    .warp         = sequence<2, 2, 1>{},
    .warp_tile    = sequence<32, 32, 16>{},
});

template <typename DTypes, typename Layouts>
using Test_Intra_V4 = decltype(TestParams{
    .dtypes       = DTypes{},
    .layouts      = Layouts{},
    .padding      = utils::boolseq<true, true, true>{},
    .act_mul      = MyActMulOp{},
    .pipeline     = constant<GemmPipeline::COMPUTE_V4>{},
    .scheduler    = Intra{},
    .block_per_cu = constant<1>{},
    .tile         = sequence<256, 256, 32>{},
    .warp         = sequence<2, 2, 1>{},
    .warp_tile    = sequence<32, 32, 16>{},
});

template <typename DTypes, typename Layouts>
using Test_Intra_V6 = decltype(TestParams{
    .dtypes       = DTypes{},
    .layouts      = Layouts{},
    .padding      = utils::boolseq<true, true, true>{},
    .act_mul      = MyActMulOp{},
    .pipeline     = constant<GemmPipeline::COMPUTE_V6>{},
    .scheduler    = Intra{},
    .block_per_cu = constant<1>{},
    .tile         = sequence<256, 256, 64>{},
    .warp         = sequence<2, 2, 1>{},
    .warp_tile    = sequence<32, 32, 16>{},
});

using KernelTypes = ::testing::Types<
    /* V3 */
    Test_Intra_V3<dtypes<F16, F16, F32>, layouts<R, R, R>>
    // Test_Intra_V3<dtypes<F16, F16, F32>, layouts<R, C, R>>,
    // Test_Intra_V3<dtypes<F16, F16, F32>, layouts<C, R, R>>,
    // Test_Intra_V3<dtypes<F16, F16, F32>, layouts<C, C, R>>,

    // Test_Intra_V3<dtypes<F8, F16, F32>, layouts<R, R, R>>,
    // Test_Intra_V3<dtypes<F8, F16, F32>, layouts<R, C, R>>,
    // Test_Intra_V3<dtypes<F8, F16, F32>, layouts<C, R, R>>,
    // Test_Intra_V3<dtypes<F8, F16, F32>, layouts<C, C, R>>,

    // Test_Intra_V3<dtypes<BF16, F16, F32>, layouts<R, R, R>>,
    // Test_Intra_V3<dtypes<BF16, F16, F32>, layouts<R, C, R>>,
    // Test_Intra_V3<dtypes<BF16, F16, F32>, layouts<C, R, R>>,
    // Test_Intra_V3<dtypes<BF16, F16, F32>, layouts<C, C, R>>,

    // /* V4 */
    // Test_Intra_V4<dtypes<F16, F16, F32>, layouts<R, R, R>>,
    // Test_Intra_V4<dtypes<F16, F16, F32>, layouts<R, C, R>>,
    // Test_Intra_V4<dtypes<F16, F16, F32>, layouts<C, R, R>>,
    // Test_Intra_V4<dtypes<F16, F16, F32>, layouts<C, C, R>>,

    // Test_Intra_V4<dtypes<BF16, F16, F32>, layouts<R, R, R>>,
    // Test_Intra_V4<dtypes<BF16, F16, F32>, layouts<R, C, R>>,
    // Test_Intra_V4<dtypes<BF16, F16, F32>, layouts<C, R, R>>,
    // Test_Intra_V4<dtypes<BF16, F16, F32>, layouts<C, C, R>>,

    // Test_Intra_V4<dtypes<F8, F16, F32>, layouts<R, R, R>>,
    // Test_Intra_V4<dtypes<F8, F16, F32>, layouts<R, C, R>>,
    // Test_Intra_V4<dtypes<F8, F16, F32>, layouts<C, R, R>>,
    // Test_Intra_V4<dtypes<F8, F16, F32>, layouts<C, C, R>>,

    // /* V6 */
    // Test_Intra_V6<dtypes<F16, F16, F32>, layouts<R, R, R>>
    // Test_Intra_V6<dtypes<F16, F16, F32>, layouts<R, C, R>>,
    // Test_Intra_V6<dtypes<F16, F16, F32>, layouts<C, R, R>>,
    // Test_Intra_V6<dtypes<F16, F16, F32>, layouts<C, C, R>>,

    // Test_Intra_V6<dtypes<BF16, F16, F32>, layouts<R, R, R>>,
    // Test_Intra_V6<dtypes<BF16, F16, F32>, layouts<R, C, R>>,
    // Test_Intra_V6<dtypes<BF16, F16, F32>, layouts<C, R, R>>,
    // Test_Intra_V6<dtypes<BF16, F16, F32>, layouts<C, C, R>>,

    // Test_Intra_V6<dtypes<F8, F16, F32>, layouts<R, R, R>>,
    // Test_Intra_V6<dtypes<F8, F16, F32>, layouts<R, C, R>>,
    // Test_Intra_V6<dtypes<F8, F16, F32>, layouts<C, R, R>>,
    // Test_Intra_V6<dtypes<F8, F16, F32>, layouts<C, C, R>>
    >;
} // namespace ck_tile

TYPED_TEST_SUITE(TestCkTileSwiGLU, ck_tile::KernelTypes);

TYPED_TEST(TestCkTileSwiGLU, Regular)
{
    std::vector<std::tuple<size_t, size_t, size_t>> sizes{
        {512, 512, 512},
    };

    for(auto [m, n, k] : sizes)
    {
        this->Run(m, n, k);
    }
}

TYPED_TEST(TestCkTileSwiGLU, SingleTile)
{
    constexpr auto M_Tile = TestFixture::Params::Tile::template get<0>();
    constexpr auto N_Tile = TestFixture::Params::Tile::template get<1>();
    constexpr auto K_Tile = TestFixture::Params::Tile::template get<2>();

    std::vector<std::tuple<size_t, size_t, size_t>> sizes{
        {M_Tile, N_Tile, K_Tile},
    };

    for(auto [m, n, k] : sizes)
    {
        this->Run(m, n, k);
    }
}

TYPED_TEST(TestCkTileSwiGLU, MultipleTiles)
{
    constexpr auto M_Tile = TestFixture::Params::Tile::template get<0>();
    constexpr auto N_Tile = TestFixture::Params::Tile::template get<1>();
    constexpr auto K_Tile = TestFixture::Params::Tile::template get<2>();

    std::vector<std::tuple<size_t, size_t, size_t>> sizes{
        {M_Tile * 2, N_Tile * 2, K_Tile * 2},
    };

    for(auto [m, n, k] : sizes)
    {
        this->Run(m, n, k);
    }
}

TYPED_TEST(TestCkTileSwiGLU, NonSquare)
{
    constexpr auto M_Tile = TestFixture::Params::Tile::template get<0>();
    constexpr auto N_Tile = TestFixture::Params::Tile::template get<1>();
    constexpr auto K_Tile = TestFixture::Params::Tile::template get<2>();

    std::vector<std::tuple<size_t, size_t, size_t>> sizes{
        {M_Tile * 2, N_Tile, K_Tile},
        {M_Tile, N_Tile * 2, K_Tile},
        {M_Tile, N_Tile, K_Tile * 2},
    };

    for(auto [m, n, k] : sizes)
    {
        this->Run(m, n, k);
    }
}

TYPED_TEST(TestCkTileSwiGLU, LargeMatrix)
{
    std::vector<std::tuple<size_t, size_t, size_t>> sizes{
        {2048, 2048, 2048},
    };

    for(auto [m, n, k] : sizes)
    {
        this->template RunWithActMul<ck_tile::MyActMulOp>(m, n, k);
    }
}

TYPED_TEST(TestCkTileSwiGLU, PaddK)
{
    std::vector<std::tuple<size_t, size_t, size_t>> sizes{
        {256, 256, 432},
    };

    for(auto [m, n, k] : sizes)
    {
        this->Run(m, n, k);
    }
}

TYPED_TEST(TestCkTileSwiGLU, NotSupportedArgument)
{
    std::vector<std::tuple<size_t, size_t, size_t>> sizes{
        {256, 267, 513},
    };

    for(auto [m, n, k] : sizes)
    {
        EXPECT_THROW((this->template RunWithPadding<false, false, false>(m, n, k)),
                     std::runtime_error);
    }
}
