// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"

#include "mx_wmma_op.hpp"

using ck::bf6_t;
using ck::bf8_t;
using ck::e4m3_scale_t;
using ck::e5m3_scale_t;
using ck::e8m0_bexp_t;
using ck::f4_t;
using ck::f6_t;
using ck::f8_t;
using ck::type_convert;

/**
 * @brief Run the test for the given WMMA scale instruction
 *
 * @param init - selects initialization algorithm for A and B tensors
 */
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AType,
          typename BType,
          typename CType,
          typename AScaleType,
          typename BScaleType,
          ck::WMMA_SCALE wmma>
bool run_mx_wmma_test(ck::index_t init)
{
    static_assert((wmma == ck::WMMA_SCALE::SCALE_F32_16x16x128 ||
                   wmma == ck::WMMA_SCALE::SCALE16_F32_16x16x128),
                  "Only SCALE_F32_16x16x128 and SCALE16_F32_16x16x128 are supported");

    using AccType = float; // only F32 instructions supported

    // WMMA scale instruction parameters
    ck::mfma_type<static_cast<ck::MfmaInstr>(wmma)> wmma_instr;
    constexpr auto BLOCK_M = wmma_instr.m_per_blk;
    constexpr auto BLOCK_N = wmma_instr.n_per_blk;
    constexpr auto BLOCK_K = wmma_instr.num_input_blks * wmma_instr.k_per_blk;
    constexpr auto BLOCK_X = wmma_instr.scale_blk_size; // scaling vector size

    const auto mx_wmma_kernel = ck::matmul<AType,
                                           BType,
                                           AScaleType,
                                           BScaleType,
                                           CType,
                                           AccType,
                                           BLOCK_M,
                                           BLOCK_N,
                                           BLOCK_K,
                                           BLOCK_X,
                                           ALayout,
                                           BLayout,
                                           CLayout>;

    bool pass = true;

    pass = ck::mx_wmma_test::TestMXWMMA<decltype(mx_wmma_kernel),
                                        AType,
                                        BType,
                                        AScaleType,
                                        BScaleType,
                                        CType,
                                        ALayout,
                                        BLayout,
                                        CLayout,
                                        BLOCK_M,
                                        BLOCK_N,
                                        BLOCK_K,
                                        BLOCK_X>{}(mx_wmma_kernel, init);

    return pass;
}

const ck::index_t common_init = -1;

// test FP8@FP8 with e8m0 scale and 32 block size
TEST(MXWMMA, MXFP8WMMA16x16x128_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP8@FP8 with e8m0 scale and 16 block size
TEST(MXWMMA, MXFP8WMMA16x16x128_SCALE16_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test BF8@BF8 with e8m0 scale and 32 block size
TEST(MXWMMA, MXBF8WMMA16x16x128_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf8_t,
                                 bf8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test BF8@BF8 with e8m0 scale and 16 block size
TEST(MXWMMA, MXBF8WMMA16x16x128_SCALE16_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf8_t,
                                 bf8_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP6@FP6 with e8m0 scale and 32 block size
TEST(MXWMMA, MXFP6WMMA16x16x128_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f6_t,
                                 f6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP6@FP6 with e8m0 scale and 16 block size
TEST(MXWMMA, MXFP6WMMA16x16x128_SCALE16_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f6_t,
                                 f6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test BF6@BF6 with e8m0 scale and 32 block size
TEST(MXWMMA, MXBF6WMMA16x16x128_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf6_t,
                                 bf6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test BF6@BF6 with e8m0 scale and 16 block size
TEST(MXWMMA, MXBF6WMMA16x16x128_SCALE16_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 bf6_t,
                                 bf6_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e8m0 scale and 32 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e8m0 scale and 16 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e4m3 scale and 32 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_E4M3)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e4m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e4m3 scale and 16 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E4M3)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e4m3_scale_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e5m3 scale and 32 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_E5M3)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e5m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e5m3 scale and 16 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E5M3)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e5m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e4m3 and e5m3 scales and 32 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_E4M3_E5M3)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e4m3 and e5m3 scales and 16 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E4M3_E5M3)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e4m3_scale_t,
                                 e5m3_scale_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e5m3 and e4m3 scales and 32 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_E5M3_E4M3)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e5m3_scale_t,
                                 e4m3_scale_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP4@FP4 with e5m3 and e4m3 scales and 16 block size
TEST(MXWMMA, MXFP4WMMA16x16x128_SCALE16_E5M3_E4M3)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f4_t,
                                 f4_t,
                                 float,
                                 e5m3_scale_t,
                                 e4m3_scale_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP8@FP4 with e8m0 scale and 32 block size
TEST(MXWMMA, MXFP8FP4WMMA16x16x128_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}

// test FP8@FP4 with e8m0 scale and 16 block size
TEST(MXWMMA, MXFP8FP4WMMA16x16x128_SCALE16_E8M0)
{
    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    auto pass = run_mx_wmma_test<ALayout,
                                 BLayout,
                                 CLayout,
                                 f8_t,
                                 f4_t,
                                 float,
                                 e8m0_bexp_t,
                                 e8m0_bexp_t,
                                 ck::WMMA_SCALE::SCALE16_F32_16x16x128>(common_init);
    EXPECT_TRUE(pass);
}
