// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompAsyncPPWmma
    : public TestCkTileGemmPipelineWmmaBase<T, class TestCkTileGemmPipelineCompAsyncPPWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompAsyncPPWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompAsyncPPWmma, KernelTypesCompAsyncPPWmma);

#include "test_gemm_pipeline_ut_cases.inc"

// The last K tile is only partly in bounds; the range-checked async loads zero-fill the rest.
TYPED_TEST(TEST_SUITE_NAME, RaggedK)
{
    constexpr int M = 256;
    constexpr int N = 256;

    for(int K : {40, 504})
        this->Run(M, N, K);
}

// Without padding the async loads are unchecked; exact tile multiples, odd and even tile counts.
TYPED_TEST(TEST_SUITE_NAME, Unpadded)
{
    constexpr int M = 4 * TestFixture::M_Tile;
    constexpr int N = 2 * TestFixture::N_Tile;

    for(int k_tiles : {1, 2, 3, 8})
        this->template Run<false, false, false>(M, N, k_tiles * TestFixture::K_Tile);
}

// An unpadded instance cannot load a partial tile in bounds, so it rejects M/N/K tails.
// All types here are RCR: the kernel's own layout checks already reject the N tail (row-major
// C) and the K tail (row-major A), so only the M tail is decided by the pipeline's
// IsSupportedArgument hook. The N and K cases are kept as regression coverage.
TYPED_TEST(TEST_SUITE_NAME, UnpaddedRejectsTail)
{
    constexpr int M = 2 * TestFixture::M_Tile;
    constexpr int N = 2 * TestFixture::N_Tile;
    constexpr int K = 4 * TestFixture::K_Tile;

    EXPECT_THROW((this->template Run<false, false, false>(M + 8, N, K)), std::runtime_error);
    EXPECT_THROW((this->template Run<false, false, false>(M, N + 8, K)), std::runtime_error);
    EXPECT_THROW((this->template Run<false, false, false>(M, N, K + 8)), std::runtime_error);
}

#undef TEST_SUITE_NAME
