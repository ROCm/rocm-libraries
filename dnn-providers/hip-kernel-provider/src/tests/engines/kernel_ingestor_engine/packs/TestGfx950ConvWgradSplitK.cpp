// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvWgradGeometry.hpp"

/**
 * @file TestGfx950ConvWgradSplitK.cpp
 * @brief Pins gfx950_conv_wgrad's runtime split-K degree selection
 *        (selectSplitKDegreeWgrad, wgradAtomicEpilogueAvailable,
 *        wgradAtomicStoreVecDefault) against the Python it ports
 *        (rocke.helpers.split_k.select_split_k_wgrad and
 *        kernels.common.conv_implicit_gemm_wgrad.wgrad_atomic_epilogue_available /
 *        wgrad_atomic_store_vec).
 *
 * implicitGemmPlan() reads the KMD's split_k field as a build-time capability flag only
 * (whether the binary even contains the atomic cshuffle epilogue) and computes the actual
 * launch degree from the real problem shape through these functions. A regression here does
 * not fault -- it silently launches with the wrong ks_count -- and nothing else in the suite
 * exercises the formula with real numbers, so every expected value below is worked out by
 * hand from the ported formula and written as a literal, not re-derived from the code under
 * test.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using gfx950_conv_wgrad::Geometry;
using gfx950_conv_wgrad::Problem;

/// A minimal Problem/Geometry pair for the split-K formulas, which only read
/// problem.{y,x,n,c,groups} and geometry.{cpg,kpg,ho,wo} -- never hi/wi/strides/pads/
/// dilations/tensorBytes, so those are left at harmless placeholder values. Built field by
/// field (not an aggregate literal): this target compiles as C++17, which does not have
/// designated initializers.
Problem makeProblem(int64_t n, int64_t c, int64_t y, int64_t x, int64_t groups)
{
    Problem problem{};
    problem.n = n;
    problem.c = c;
    problem.k = c;
    problem.hi = 1;
    problem.wi = 1;
    problem.y = y;
    problem.x = x;
    problem.strideH = 1;
    problem.strideW = 1;
    problem.padH = 0;
    problem.padW = 0;
    problem.dilationH = 1;
    problem.dilationW = 1;
    problem.groups = groups;
    return problem;
}

Geometry makeGeometry(int64_t ho, int64_t wo, int64_t cpg, int64_t kpg)
{
    Geometry geometry{};
    geometry.ho = ho;
    geometry.wo = wo;
    geometry.cpg = cpg;
    geometry.kpg = kpg;
    geometry.tensorBytes = {0, 0, 0};
    return geometry;
}

} // namespace

// =============================================================================
// wgradAtomicStoreVecDefault / wgradAtomicEpilogueAvailable -- wgrad_atomic_store_vec and
// wgrad_atomic_epilogue_available (conv_implicit_gemm_wgrad.py), vector_size_c=None always
// (the resolver never asks with a concrete instance width, only the default policy).
// =============================================================================

TEST(TestGfx950ConvWgradSplitK, StoreVecDefaultPicksTheWidestDivisorOfTheChannelCount)
{
    // _vec(n, dt) for dt != "fp32": widest of {8,4,2,1} dividing n.
    const auto problem = makeProblem(/*n=*/1, /*c=*/24, /*y=*/1, /*x=*/1, /*groups=*/1);
    EXPECT_EQ(gfx950_conv_wgrad::wgradAtomicStoreVecDefault(
                  problem, makeGeometry(1, 1, /*cpg=*/24, /*kpg=*/1)),
              8); // 24 % 8 == 0
    EXPECT_EQ(gfx950_conv_wgrad::wgradAtomicStoreVecDefault(
                  makeProblem(1, 12, 1, 1, 1), makeGeometry(1, 1, 12, 1)),
              4); // 12 % 8 != 0, 12 % 4 == 0
    EXPECT_EQ(gfx950_conv_wgrad::wgradAtomicStoreVecDefault(
                  makeProblem(1, 6, 1, 1, 1), makeGeometry(1, 1, 6, 1)),
              2); // 6 % 4 != 0, 6 % 2 == 0
    EXPECT_EQ(gfx950_conv_wgrad::wgradAtomicStoreVecDefault(
                  makeProblem(1, 3, 1, 1, 1), makeGeometry(1, 1, 3, 1)),
              1); // 3 is odd: only 1 divides it
}

TEST(TestGfx950ConvWgradSplitK, StoreVecDefaultReadsCpgNotCWhenGrouped)
{
    // groups > 1: the channel count is cpg (per-group), not the problem's total C.
    const auto problem = makeProblem(/*n=*/1, /*c=*/256, /*y=*/1, /*x=*/1, /*groups=*/32);
    // cpg = 256 / 32 = 8, so store_vec = 8 even though the total C (256) would also give 8 --
    // use a cpg the total C would NOT give to prove cpg is what is actually read.
    EXPECT_EQ(gfx950_conv_wgrad::wgradAtomicStoreVecDefault(
                  problem, makeGeometry(1, 1, /*cpg=*/3, /*kpg=*/1)),
              1); // cpg=3 is odd, even though C=256 is not
}

TEST(TestGfx950ConvWgradSplitK, AtomicEpilogueAvailableRequiresAnEvenRowLengthAndStoreVec)
{
    // wg_N = Y * X * cpg = 3 * 3 * 8 = 72 (even); store_vec(cpg=8) = 8 (even) -> available.
    EXPECT_TRUE(gfx950_conv_wgrad::wgradAtomicEpilogueAvailable(
        makeProblem(1, 8, 3, 3, 1), makeGeometry(1, 1, /*cpg=*/8, /*kpg=*/1)));
}

TEST(TestGfx950ConvWgradSplitK, AtomicEpilogueAvailableFalseWhenTheChannelCountIsOdd)
{
    // cpg=3 -> store_vec=1 (odd), regardless of wg_N's own parity (wg_N = 3*3*3 = 27, also
    // odd here, but store_vec is the decisive failure either way).
    EXPECT_FALSE(gfx950_conv_wgrad::wgradAtomicEpilogueAvailable(
        makeProblem(1, 3, 3, 3, 1), makeGeometry(1, 1, /*cpg=*/3, /*kpg=*/1)));
}

// =============================================================================
// selectSplitKDegreeWgrad -- select_split_k_wgrad's two CK-formula branches (split_k.py),
// gfx950 constants inlined: GFX950_NUM_CUS=256, GFX950_NUM_XCDS=8,
// WGRAD_ASSUMED_WAVES_PER_CU=2, WGRAD_TARGET_WAVES_PER_CU=16, MAX_GRID_DIM_Z unused here
// (that clamp is the caller's job in implicitGemmPlan, not this function's).
// =============================================================================

TEST(TestGfx950ConvWgradSplitK, UngroupedDegreeIsCapacityOverBaseGrid)
{
    // base_grid = ceil(256/128) * ceil(2304/128) = 2 * 18 = 36.
    // max_capacity = WAVES_PER_CU(2) * NUM_CUS(256) = 512.
    // split_k = floor(512 / 36) = 14, not capped by wg_K (25088 here).
    EXPECT_EQ(gfx950_conv_wgrad::selectSplitKDegreeWgrad(/*wgM=*/256,
                                                          /*wgN=*/2304,
                                                          /*wgK=*/25088,
                                                          /*tileM=*/128,
                                                          /*tileN=*/128,
                                                          /*groups=*/1,
                                                          /*blockSize=*/256),
              14);
}

TEST(TestGfx950ConvWgradSplitK, UngroupedDegreeIsClampedToWgK)
{
    // base_grid = ceil(128/128) * ceil(128/128) = 1, so floor(512/1) = 512 before the wg_K
    // clamp -- a GEMM reduction this shallow cannot be split past its own extent.
    EXPECT_EQ(gfx950_conv_wgrad::selectSplitKDegreeWgrad(
                  /*wgM=*/128, /*wgN=*/128, /*wgK=*/10, /*tileM=*/128, /*tileN=*/128,
                  /*groups=*/1, /*blockSize=*/256),
              10);
}

TEST(TestGfx950ConvWgradSplitK, UngroupedDegreeFloorsToOneWhenTheGridAlreadyFillsCapacity)
{
    // base_grid = ceil(1200/128) * ceil(1200/128) = 10 * 10 = 100 tiles in one dim product...
    // pick a base_grid that already exceeds max_capacity (512): 30 * 30 = 900.
    EXPECT_EQ(gfx950_conv_wgrad::selectSplitKDegreeWgrad(
                  /*wgM=*/3840, /*wgN=*/3840, /*wgK=*/100000, /*tileM=*/128, /*tileN=*/128,
                  /*groups=*/1, /*blockSize=*/256),
              1); // ceil(3840/128) = 30, base_grid = 900, floor(512/900) = 0 -> max(1, 0) = 1
}

TEST(TestGfx950ConvWgradSplitK, GroupedDegreeSnapsDownToAnXcdCountMultiple)
{
    // base_grid = ceil(128/128) * ceil(384/128) = 1 * 3 = 3.
    // waves_per_block = blockSize(256) / 64 = 4; blocks_per_cu = max(1, 16/4) = 4;
    // capacity = 4 * 256 = 1024. grid_no_split = base_grid(3) * groups(32) = 96.
    // split_k = floor(1024/96) = 10. cap = min(wg_K, 65535/32=2047) = 2047 (wg_K=100000 here).
    // step = NUM_XCDS(8) / gcd(base_grid=3, 8)=1 -> step = 8.
    // snapped = (10 / 8) * 8 = 8 (not zero, <= cap) -> split_k = 8.
    EXPECT_EQ(gfx950_conv_wgrad::selectSplitKDegreeWgrad(
                  /*wgM=*/128, /*wgN=*/384, /*wgK=*/100000, /*tileM=*/128, /*tileN=*/128,
                  /*groups=*/32, /*blockSize=*/256),
              8);
}

TEST(TestGfx950ConvWgradSplitK, GroupedDegreeIsClampedByTheGridDimZBudgetBeforeSnapping)
{
    // Same shape as above but a tiny wg_K (5) forces the pre-snap cap down to
    // min(wg_K=5, 65535/32=2047) = 5, below the unclamped split_k (10) and below the XCD
    // step (8) too -- the snap must not round back up past the cap.
    EXPECT_EQ(gfx950_conv_wgrad::selectSplitKDegreeWgrad(
                  /*wgM=*/128, /*wgN=*/384, /*wgK=*/5, /*tileM=*/128, /*tileN=*/128,
                  /*groups=*/32, /*blockSize=*/256),
              5);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
