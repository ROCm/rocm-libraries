// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Cases shared by the TDM multi-D GEMM tests (test_gemm_multi_d_tdm_wmma.cpp and
// test_gemm_multi_d_tdm_run_ahead_wmma.cpp), run for every type in the including file's
// KernelTypes.

// Aligned shapes.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_256x512x256)
{
    EXPECT_TRUE(this->Run(256, 512, 256, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_512x256x256)
{
    EXPECT_TRUE(this->Run(512, 256, 256, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_512x512x256)
{
    EXPECT_TRUE(this->Run(512, 512, 256, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_64x64x32)
{
    EXPECT_TRUE(this->Run(64, 64, 32, 1));
}

// Unaligned shapes. They depend on the plain TDM GEMM handling the same N (the E row pitch is not
// a multiple of 8 elements); filter with --gtest_filter=-*Unaligned* if that is not yet the case.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdmUnaligned_253x381x256)
{
    EXPECT_TRUE(this->Run(253, 381, 256, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdmUnaligned_255x257x96)
{
    EXPECT_TRUE(this->Run(255, 257, 96, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdmUnaligned_65x63x32)
{
    EXPECT_TRUE(this->Run(65, 63, 32, 1));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdmUnaligned_1x64x32)
{
    EXPECT_TRUE(this->Run(1, 64, 32, 1));
}

// Non-packed D0 and E row pitches.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_StrideD0_256x512x256)
{
    constexpr int M = 256;
    constexpr int N = 512;
    constexpr int K = 256;
    EXPECT_TRUE(this->Run(M, N, K, 1, 0, 0, N + 8, 0, 0));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_StrideE_256x512x256)
{
    constexpr int M = 256;
    constexpr int N = 512;
    constexpr int K = 256;
    EXPECT_TRUE(this->Run(M, N, K, 1, 0, 0, 0, 0, N + 16));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_StrideD0E_512x256x256)
{
    constexpr int M = 512;
    constexpr int N = 256;
    constexpr int K = 256;
    EXPECT_TRUE(this->Run(M, N, K, 1, 0, 0, N + 8, 0, N + 16));
}

// Ramp D0 / identity D1: every (m, n) sees a distinct D0 value and D1 is the identity of the
// elementwise op, so E depends on D0 alone. Most elements must differ from the GEMM without Ds
// and E must not be all zero, so a dropped or misplaced D0 fusion cannot pass.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_RampD0IdentityD1_256x256x128)
{
    EXPECT_TRUE(this->Run(256, 256, 128, 1, 0, 0, 0, 0, 0, MultiDFill::RampD0IdentityD1, true));
}

// Host-side argument check on gfx125: the TDM kernel is accepted for k_batch = 1.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_KBatch1Supported)
{
    EXPECT_TRUE(this->IsTdmSupported(256, 256, 256, 1));
}

// TDM requires exactly one split: its store overwrites E and cannot accumulate split-K.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_InvalidKBatchNotSupported)
{
    for(int k_batch : {0, -1, 2})
    {
        SCOPED_TRACE(k_batch);
        EXPECT_FALSE(this->IsTdmSupported(256, 256, 256, k_batch));
    }
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_NonPositiveKNotSupported)
{
    for(int K : {0, -1})
    {
        SCOPED_TRACE(K);
        EXPECT_FALSE(this->IsTdmSupported(256, 256, K, 1));
    }
}

// A stride-0 D0 (one N-vector added to every row, i.e. a bias) whose value depends on the column
// alone. Every configuration must place each column's value correctly: through the generic D path,
// and through the row-broadcast path for the configurations that enable it.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_Stride0RampD0_256x256x128)
{
    EXPECT_TRUE(
        this->Run(256, 256, 128, 1, 0, 0, 0, 0, 0, MultiDFill::RampD0IdentityD1, true, true));
}

// A = 0: E is the D fusion alone, checked with the tight tolerance of a small reference, so a
// D0 value added to the wrong column fails in every E type. With MultiplyMultiply E is 0.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdmUnaligned_ZeroAStride0RampD0_253x381x256)
{
    constexpr bool is_add =
        !std::is_same_v<typename TestFixture::CDElementWiseFn, MultiplyMultiply>;
    EXPECT_TRUE(this->Run(
        253, 381, 256, 1, 0, 0, 0, 0, 0, MultiDFill::RampD0IdentityD1, is_add, true, true));
}

TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdmUnaligned_Stride0D0_253x381x256)
{
    EXPECT_TRUE(this->Run(253, 381, 256, 1, 0, 0, 0, 0, 0, MultiDFill::Uniform, false, true));
}

// Host-side argument check: a stride-0 D0 is always accepted; a row-broadcast D0 must be passed
// with stride 0, since the epilogue reads only its first row.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_D0StrideSupported)
{
    EXPECT_TRUE(this->IsTdmSupported(256, 256, 256, 1, 0));
    EXPECT_EQ(this->IsTdmSupported(256, 256, 256, 1, 256), !TestFixture::kD0RowBroadcast);
}

// Same for D1: the mask is generic over the D index, so a row-broadcast D1 must be passed with
// stride 0 and a full D1 is accepted with either stride.
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDTdm_D1StrideSupported)
{
    EXPECT_TRUE(this->IsTdmSupported(256, 256, 256, 1, -1, 0));
    EXPECT_EQ(this->IsTdmSupported(256, 256, 256, 1, -1, 256), !TestFixture::kD1RowBroadcast);
}
