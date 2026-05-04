// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Utils.hpp>

using namespace TensileLite;

// ============================================================================
// MX Scale Padding Tests
//
// Verify that setMXScaleA/B pads scale tensor dimensions so kernels that
// process data in fixed-size blocks always have valid scale entries:
//   - Bound dimension (K): ceil(K/mxBlock) rounded up to multiple of 8
//     (covers K in 256-element blocks: 8 scale entries * 32 data/scale = 256)
//   - Free dimension (M or N): rounded up to multiple of 32
//     (covers M/N in 32-element blocks)
//   - Batch dimension: unchanged
// ============================================================================

// Helper: create a ContractionProblemGemm with given M, N, K and set MX scales
static ContractionProblemGemm makeMXProblem(size_t M,
                                            size_t N,
                                            size_t K,
                                            int    mxBlock,
                                            size_t batch  = 1,
                                            bool   transA = true,
                                            bool   transB = false)
{
    auto problem = ContractionProblemGemm::GEMM_Strides(
        transA,
        transB,
        rocisa::DataType::Float4,
        rocisa::DataType::Float4,
        rocisa::DataType::BFloat16,
        rocisa::DataType::BFloat16,
        M, N, K, batch,
        transA ? K : M,
        transA ? K * M : M * K,
        transB ? N : K,
        transB ? N * K : K * N,
        M, M * N,
        M, M * N,
        0.0);

    problem.setMXScaleA(mxBlock);
    problem.setMXScaleB(mxBlock);
    return problem;
}

// Params: M, N, K, expectedPaddedScaleK, expectedPaddedM, expectedPaddedN
class MXScalePaddingTest
    : public ::testing::TestWithParam<
          std::tuple<size_t, size_t, size_t, size_t, size_t, size_t>>
{
};

TEST_P(MXScalePaddingTest, ScaleDimensionsPaddedCorrectly)
{
    auto [M, N, K, expectedScaleK, expectedM, expectedN] = GetParam();
    const int mxBlock = 32;

    auto problem = makeMXProblem(M, N, K, mxBlock);

    auto const& sa = problem.mxsa().sizes();
    auto const& sb = problem.mxsb().sizes();

    // transA=true: scaleA = {K/mxBlock_padded, M_padded, batch}
    EXPECT_EQ(sa[0], expectedScaleK);
    EXPECT_EQ(sa[1], expectedM);
    EXPECT_EQ(sa[2], 1u);

    // transB=false: scaleB = {K/mxBlock_padded, N_padded, batch}
    EXPECT_EQ(sb[0], expectedScaleK);
    EXPECT_EQ(sb[1], expectedN);
    EXPECT_EQ(sb[2], 1u);

    // Alignment invariants
    EXPECT_EQ(sa[0] % 8, 0u);
    EXPECT_EQ(sa[1] % 32, 0u);
    EXPECT_EQ(sb[0] % 8, 0u);
    EXPECT_EQ(sb[1] % 32, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    ScalePadding,
    MXScalePaddingTest,
    ::testing::Values(
        //        M,    N,    K,  scaleK, padM, padN
        // Aligned: no padding needed
        std::make_tuple(128u, 256u, 256u,  8u, 128u, 256u),
        std::make_tuple(64u,  128u, 512u, 16u,  64u, 128u),
        // Unaligned M only
        std::make_tuple(80u,   64u, 256u,  8u,  96u,  64u),
        std::make_tuple(1u,    64u, 256u,  8u,  32u,  64u),
        // Unaligned N only
        std::make_tuple(128u,  50u, 256u,  8u, 128u,  64u),
        // Unaligned K only (ceil(300/32)=10 -> 16)
        std::make_tuple(128u, 128u, 300u, 16u, 128u, 128u),
        // K=32: ceil(32/32)=1 -> 8
        std::make_tuple(128u, 128u,  32u,  8u, 128u, 128u),
        // All unaligned: M=80->96, N=50->64, K=300->scaleK=16
        std::make_tuple(80u,   50u, 300u, 16u,  96u,  64u),
        // Non-multiple-of-32 M,N (multiples of 16): M=16->32, N=48->64
        std::make_tuple(16u,   48u, 256u,  8u,  32u,  64u),
        // M=48->64, N=16->32
        std::make_tuple(48u,   16u, 256u,  8u,  64u,  32u),
        // M=48->64, N=48->64 (both non-mult-of-32, mult-of-16)
        std::make_tuple(48u,   48u, 256u,  8u,  64u,  64u),
        // M=112->128, N=176->192 (larger mult-of-16 values)
        std::make_tuple(112u, 176u, 256u,  8u, 128u, 192u),
        // M=240->256, N=112->128 (just below 32-boundary)
        std::make_tuple(240u, 112u, 512u, 16u, 256u, 128u),
        // Odd M,N
        std::make_tuple(17u,   33u, 256u,  8u,  32u,  64u),
        std::make_tuple(33u,   17u, 256u,  8u,  64u,  32u),
        std::make_tuple(63u,   63u, 256u,  8u,  64u,  64u),
        std::make_tuple(97u,  129u, 256u,  8u, 128u, 160u),
        std::make_tuple(3u,     5u, 256u,  8u,  32u,  32u),
        // Even non-multiple-of-16: M=10->32, N=34->64
        std::make_tuple(10u,   34u, 256u,  8u,  32u,  64u),
        // Even non-multiple-of-16: M=50->64, N=100->128
        std::make_tuple(50u,  100u, 256u,  8u,  64u, 128u),
        // Even non-multiple-of-16: M=66->96, N=2->32
        std::make_tuple(66u,    2u, 256u,  8u,  96u,  32u),
        // Mixed odd M, even non-mult-of-16 N
        std::make_tuple(7u,    26u, 256u,  8u,  32u,  32u),
        std::make_tuple(99u,   50u, 256u,  8u, 128u,  64u)
    )
);

// Batch dimension must not be padded; strides and total size must be consistent
TEST(MXScalePadding, BatchAndStridesCorrect)
{
    auto problem = makeMXProblem(80, 50, 300, 32, /*batch=*/3);

    auto const& sa = problem.mxsa().sizes();
    auto const& sb = problem.mxsb().sizes();

    // Batch unchanged
    EXPECT_EQ(sa[2], 3u);
    EXPECT_EQ(sb[2], 3u);

    // Free dims still padded
    EXPECT_EQ(sa[1], 96u);  // M=80 -> 96
    EXPECT_EQ(sb[1], 64u);  // N=50 -> 64

    // Column-major strides
    auto const& saStr = problem.mxsa().strides();
    EXPECT_EQ(saStr[0], 1u);
    EXPECT_EQ(saStr[1], sa[0]);
    EXPECT_EQ(saStr[2], sa[0] * sa[1]);

    // totalAllocatedElements includes padding
    EXPECT_EQ(problem.mxsa().totalAllocatedElements(), sa[0] * sa[1] * sa[2]);
    EXPECT_GT(problem.mxsa().totalAllocatedElements(), (size_t)(300 / 32) * 80 * 3);
}
