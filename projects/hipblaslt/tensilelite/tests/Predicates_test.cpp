/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>

#include <Tensile/ContractionProblemPredicates.hpp>

TEST(Predicates, ArithmeticIntensity)
{
    using namespace TensileLite;

    ContractionProblemGemm a = ContractionProblemGemm::GEMM(
        false, true, 1000, 1500, 500, 2000, 2000, 2000, 3.0, false, 1); // 88.4
    ContractionProblemGemm b = ContractionProblemGemm::GEMM(
        false, true, 500, 1000, 1000, 2000, 2000, 2000, 0.0, false, 5); // 125
    ContractionProblemGemm c = ContractionProblemGemm::GEMM(
        false, true, 2000, 100, 2000, 2000, 2000, 2000, 1.0, false, 10); // 43.5
    ContractionProblemGemm d = ContractionProblemGemm::GEMM(
        false, true, 2000, 2000, 450, 2000, 2000, 2000, 2.0, false, 1); // 92.04

    auto pg1 = std::make_shared<Predicates::Contraction::AIGreaterThanEqual>(100);
    auto pg2 = std::make_shared<Predicates::Contraction::AIGreaterThanEqual>(75);
    auto pl1 = std::make_shared<Predicates::Contraction::AILessThanEqual>(100);
    auto pl2 = std::make_shared<Predicates::Contraction::AILessThanEqual>(75);

    EXPECT_EQ(false, (*pg1)(a));
    EXPECT_EQ(true, (*pg2)(a));
    EXPECT_EQ(true, (*pl1)(a));
    EXPECT_EQ(false, (*pl2)(a));

    EXPECT_EQ(true, (*pg1)(b));
    EXPECT_EQ(true, (*pg2)(b));
    EXPECT_EQ(false, (*pl1)(b));
    EXPECT_EQ(false, (*pl2)(b));

    EXPECT_EQ(false, (*pg1)(c));
    EXPECT_EQ(false, (*pg2)(c));
    EXPECT_EQ(true, (*pl1)(c));
    EXPECT_EQ(true, (*pl2)(c));

    EXPECT_EQ(false, (*pg1)(d));
    EXPECT_EQ(true, (*pg2)(d));
    EXPECT_EQ(true, (*pl1)(d));
    EXPECT_EQ(false, (*pl2)(d));
}

// ----------------------------------------------------------------------------
// WorkgroupMappingXCCCheck: CPU-only tests with injected cuCount (ROCM-2963).
// These use the test-only constructor so we don't need a GPU. See
// docs/solution-selection-unit-test-pattern.md.
// ----------------------------------------------------------------------------

TEST(Predicates, WorkgroupMappingXCCCheck_38CU_XCC4_Fails)
{
    using namespace TensileLite;
    // 38 % 4 != 0 -> predicate must reject (would have caught ROCM-2963).
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{4, -1}, 38u);
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    EXPECT_FALSE((*pred)(problem)) << "38 CUs with XCC=4 should fail (38 % 4 != 0)";
}

TEST(Predicates, WorkgroupMappingXCCCheck_38CU_XCC1_Passes)
{
    using namespace TensileLite;
    // 38 % 1 == 0 -> predicate must accept (fix for ROCM-2963).
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{1, -1}, 38u);
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    EXPECT_TRUE((*pred)(problem)) << "38 CUs with XCC=1 should pass (38 % 1 == 0)";
}

TEST(Predicates, WorkgroupMappingXCCCheck_80CU_XCC4_Passes)
{
    using namespace TensileLite;
    // 80 % 4 == 0 -> predicate must accept.
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{4, -1}, 80u);
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    EXPECT_TRUE((*pred)(problem)) << "80 CUs with XCC=4 should pass (80 % 4 == 0)";
}

TEST(Predicates, WorkgroupMappingXCCCheck_XCCMinus1_AlwaysPasses)
{
    using namespace TensileLite;
    // value[0] == -1 means no check.
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{-1, -1}, 38u);
    auto problem = ContractionProblemGemm::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.0, false, 1);
    EXPECT_TRUE((*pred)(problem)) << "XCC=-1 should always pass";
}

TEST(Predicates, WorkgroupMappingXCCCheck_FallbackTreatsXCCAs1)
{
    using namespace TensileLite;
    // When problem is cu-fallback, effective XCC is 1 so 38 % 1 == 0 -> pass.
    auto pred = std::make_shared<Predicates::Contraction::WorkgroupMappingXCCCheck>(
        std::array<int, 2>{4, -1}, 38u);
    auto problem = ContractionProblemGemm::GEMM(false, false, 1024, 1024, 1024, 1024, 1024, 1024,
                                                 1.0, false, 1);
    problem.setParams().setFallbackStatus(true);
    EXPECT_TRUE((*pred)(problem)) << "With fallback status, effective XCC=1 so 38 % 1 == 0";
}
// ----------------------------------------------------------------------------
// StreamKWorkgroupNumberCheck: grid-dimension-overflow guard for Stream-K
// solutions (ROCM-31290).
//
// WorkgroupNumberCheck (above) bounds the tile-scaled grid at
// MAX_WORKGROUP_NUMBER (2^24) but is skipped for Stream-K solutions, because
// Stream-K's normal grid is CU-scaled, not tile-scaled. But
// resolveStreamKSettings()/getSKGridImpl() (ContractionSolution.cpp) has
// several launch-time fallbacks (most notably the tree-fixup 24-bit
// bounds guard) that hand Stream-K a one-workgroup-per-tile grid instead,
// same shape as the grid WorkgroupNumberCheck already bounds. Without an
// equivalent check, that fallback grid can itself overflow the 32-bit
// work-item count used to launch the kernel: numWorkItems is computed as
// workGroupSize * numWorkGroups in size_t, then narrowed to the
// `unsigned int` globalWorkSize parameters of hipExtModuleLaunchKernel.
//
// Measured directly against that API on gfx950 (MI350X) at 256 threads,
// the thread count of the MacroTile 16x16 Stream-K-static (SK3) kernels
// this shape selects:
//
//   tiles         work items    narrowed   observed
//   2^24 - 1      2^32 - 256    unchanged  correct
//   2^24          2^32          0          launch rejected, invalid argument
//   2^24 + 16     2^32 + 4096   4096       16 workgroups ran, 16,777,216
//                                          missing, no error reported
//
// The last row is this bug. Note that 2^24 tiles is *not* safe at 256
// threads, so the bound below is 2^24 - 1, not 2^24. At 64 threads the
// same tile counts are all fine (confirmed: 2^24 + 16 workgroups ran
// complete), which is why the limit tracks the work-item product rather
// than the tile count alone.
//
// Both regimes are visible through the library as well. Sweeping every
// algorithm the heuristic offers for a bf16 NN shape with N = 256, K = 48
// and ldd = M, against the 10.1.0a20260822 tuned library: at M = 16,777,216
// (tiles exactly 2^24) 25 of 568 fail the launch and none corrupts, while at
// M = 19,398,656 (past 2^24) 39 of 545 return success with part of D never
// written and none reports an error.
// ----------------------------------------------------------------------------

// The next two tests bracket the limit exactly, one tile apart. They use
// N == MacroTile1 so the N factor of the tile product is 1 and the tile count
// is just ceil(M/MacroTile0); with the N=256 geometry used elsewhere in this
// file the tile count is always a multiple of 16 and cannot land on an
// arbitrary boundary value.
TEST(Predicates, StreamKWorkgroupNumberCheck_AtLargestSafeTileCount_Accepted)
{
    using namespace TensileLite;
    constexpr int    macroTile0 = 16;
    constexpr int    macroTile1 = 16;
    constexpr size_t tiles      = (size_t(1) << 24) - 1;
    constexpr size_t m          = tiles * macroTile0;
    constexpr size_t n          = macroTile1; // exactly one tile in N
    static_assert(tiles * 256 < (size_t(1) << 32),
                  "largest tile count whose 256-thread work-item product fits in uint32");

    auto problem = ContractionProblemGemm::GEMM(
        false, false, m, n, /*k=*/48, m, /*ldb=*/48, m, 0.0, false, /*batchSize=*/1);
    auto pred = std::make_shared<Predicates::Contraction::StreamKWorkgroupNumberCheck>(
        std::array<int, 2>{macroTile0, macroTile1});
    EXPECT_TRUE((*pred)(problem))
        << "M=" << m << " N=" << n << ": tiles == " << tiles
        << " gives 2^32 - 256 work items at 256 threads, which still fits in 32 bits "
           "and must be accepted.";
}

TEST(Predicates, StreamKWorkgroupNumberCheck_OneTilePastLargestSafeCount_Rejected)
{
    using namespace TensileLite;
    // One tile more than the test above: 2^24 tiles is exactly 2^32 work
    // items at 256 threads, which narrows to a zero-sized grid. HIP rejects
    // that with an invalid-argument error rather than dropping stores, but it
    // is still a shape dispatch must not select. An earlier revision of this
    // guard compared `<= 2^24` and wrongly admitted it.
    constexpr int    macroTile0 = 16;
    constexpr int    macroTile1 = 16;
    constexpr size_t tiles      = size_t(1) << 24;
    constexpr size_t m          = tiles * macroTile0;
    constexpr size_t n          = macroTile1; // exactly one tile in N
    static_assert(tiles * 256 == (size_t(1) << 32), "must land on exactly 2^32 work items");

    auto problem = ContractionProblemGemm::GEMM(
        false, false, m, n, /*k=*/48, m, /*ldb=*/48, m, 0.0, false, /*batchSize=*/1);
    auto pred = std::make_shared<Predicates::Contraction::StreamKWorkgroupNumberCheck>(
        std::array<int, 2>{macroTile0, macroTile1});
    EXPECT_FALSE((*pred)(problem))
        << "M=" << m << " N=" << n << ": tiles == " << tiles
        << " gives exactly 2^32 work items at 256 threads, which narrows to a "
           "zero-sized grid and fails to launch.";
}

TEST(Predicates, StreamKWorkgroupNumberCheck_AtHardwareConfirmedBoundaryShape_Rejected)
{
    using namespace TensileLite;
    // The shape the hardware investigation used: MacroTile 16x16 at N=256, so
    // tiles == M. M == 2^24 is the smallest tile count in this geometry that
    // reaches 2^32 work items at 256 threads.
    constexpr int    macroTile0 = 16;
    constexpr int    macroTile1 = 16;
    constexpr size_t m          = 16777216; // 2^24
    constexpr size_t n          = 256;
    static_assert(m / macroTile0 * (n / macroTile1) == (size_t(1) << 24),
                  "tiles == M for this geometry");

    auto problem = ContractionProblemGemm::GEMM(
        false, false, m, n, /*k=*/48, m, /*ldb=*/48, m, 0.0, false, /*batchSize=*/1);
    auto pred = std::make_shared<Predicates::Contraction::StreamKWorkgroupNumberCheck>(
        std::array<int, 2>{macroTile0, macroTile1});
    EXPECT_FALSE((*pred)(problem))
        << "M=" << m << " N=" << n
        << ": a realistic 8.6 GiB shape that reaches exactly 2^32 work items.";
}

TEST(Predicates, StreamKWorkgroupNumberCheck_JustPastBoundary_Rejected_ROCM31290)
{
    using namespace TensileLite;
    // Same tile geometry as above, but one confirmed-broken shape past the
    // boundary: tiles == 16,781,312 > 2^24.
    constexpr int    macroTile0 = 16;
    constexpr int    macroTile1 = 16;
    constexpr size_t m          = 16781312;
    constexpr size_t n          = 256;
    static_assert(m > 16777216, "must exceed 2^24 to exercise the guard");

    auto problem = ContractionProblemGemm::GEMM(
        false, false, m, n, /*k=*/48, m, /*ldb=*/48, m, 0.0, false, /*batchSize=*/1);
    auto pred = std::make_shared<Predicates::Contraction::StreamKWorkgroupNumberCheck>(
        std::array<int, 2>{macroTile0, macroTile1});
    EXPECT_FALSE((*pred)(problem))
        << "M=" << m << " N=" << n << ": tiles == " << m
        << " > 2^24; confirmed on gfx950 hardware to silently drop most of D when "
           "a Stream-K launch-time fallback uses one workgroup per tile.";
}

TEST(Predicates, StreamKWorkgroupNumberCheck_OrdinaryProblem_Accepted)
{
    using namespace TensileLite;
    // Sanity check: an ordinary, small problem must still be accepted.
    auto problem = ContractionProblemGemm::GEMM(
        false, false, 1024, 1024, 1024, 1024, 1024, 1024, 0.0, false, /*batchSize=*/1);
    auto pred = std::make_shared<Predicates::Contraction::StreamKWorkgroupNumberCheck>(
        std::array<int, 2>{128, 128});
    EXPECT_TRUE((*pred)(problem));
}

TEST(Predicates, StreamKWorkgroupNumberCheck_BatchMultiplierCounted)
{
    using namespace TensileLite;
    // A batch count large enough to push tiles past 2^24 on its own must
    // also be rejected: tiles == ceil(M/MT0) * ceil(N/MT1) * batchSize.
    constexpr int    macroTile0 = 16;
    constexpr int    macroTile1 = 16;
    constexpr size_t m          = 256;
    constexpr size_t n          = 256;
    constexpr size_t batchSize  = 20000000; // (256/16) * (256/16) * 20e6 > 2^24
    static_assert((m / macroTile0) * (n / macroTile1) * batchSize > 16777216,
                  "batch multiplier must push tiles past 2^24");

    auto problem = ContractionProblemGemm::GEMM(
        false, false, m, n, /*k=*/48, m, /*ldb=*/48, m, 0.0, false, batchSize);
    auto pred = std::make_shared<Predicates::Contraction::StreamKWorkgroupNumberCheck>(
        std::array<int, 2>{macroTile0, macroTile1});
    EXPECT_FALSE((*pred)(problem));
}

TEST(Predicates, StreamKWorkgroupNumberCheck_EveryBatchIndexCounted)
{
    using namespace TensileLite;
    // getNumTiles() multiplies every batch index (ContractionProblem.cpp,
    // m_batchIndices loop), so the predicate has to as well. Taking only
    // batchSize(0), the way WorkgroupNumberCheck and the
    // LeadingFree*SizesGreaterOrEqual predicates do, undercounts a
    // multi-batch contraction and admits a grid that then wraps.
    //
    // Sized so the two answers land on opposite sides of the bound:
    // 16x16 tiles of a 256x256 output is 256 tiles per batch, so all batch
    // indices give 256 * 4096 * 64 == 2^26 (rejected), while batchSize(0)
    // alone gives 256 * 4096 == 2^20 (accepted).
    constexpr int macroTile0 = 16;
    constexpr int macroTile1 = 16;

    // bound = l; A = i,l,k,m; B = l,j,k,m; C = D = i,j,k,m. i and j are free,
    // k and m are both batch. GEMM() above can only build one batch index.
    const std::string   id    = "Contraction_l_Ailkm_Bljkm_Cijkm_Dijkm";
    std::vector<size_t> sizes = {/*i=*/256, /*j=*/256, /*k=*/4096, /*m=*/64, /*l=*/48};

    auto problem = ContractionProblemGemm::FromIndexSizes(id,
                                                          sizes,
                                                          rocisa::DataType::Float,
                                                          {},
                                                          rocisa::DataType::Float,
                                                          {},
                                                          rocisa::DataType::Float,
                                                          {},
                                                          rocisa::DataType::Float,
                                                          {},
                                                          0.0);

    ASSERT_EQ(problem.batchIndices().size(), 2u)
        << "the fixture must actually carry two batch indices";
    ASSERT_EQ(problem.batchSize(0) * problem.batchSize(1), 4096u * 64u);

    auto pred = std::make_shared<Predicates::Contraction::StreamKWorkgroupNumberCheck>(
        std::array<int, 2>{macroTile0, macroTile1});
    EXPECT_FALSE((*pred)(problem))
        << "tiles across both batch indices is 2^26, past the bound; counting only "
           "batchSize(0) would give 2^20 and wrongly admit this shape";
}

TEST(Predicates, StreamKWorkgroupNumberCheck_NonRepresentableDimension_Rejected)
{
    using namespace TensileLite;
    // A float ceil() loses precision once a dimension exceeds 2^24: float
    // cannot represent 16777217 exactly, so it rounds to 16777216, making
    // ceil(16777217/16) come out to 1048576 instead of the correct 1048577.
    // That silently drops tiles from 16,777,232 to exactly 16,777,216 == 2^24,
    // flipping this predicate from reject to accept. Integer ceiling division
    // must not repeat that mistake.
    constexpr int    macroTile0 = 16;
    constexpr int    macroTile1 = 16;
    constexpr size_t m          = 16777217; // 2^24 + 1, not exactly representable in float
    constexpr size_t n          = 256;
    static_assert((m + macroTile0 - 1) / macroTile0 * (n / macroTile1) > 16777216,
                  "true integer tile count must exceed 2^24");

    auto problem = ContractionProblemGemm::GEMM(
        false, false, m, n, /*k=*/48, m, /*ldb=*/48, m, 0.0, false, /*batchSize=*/1);
    auto pred = std::make_shared<Predicates::Contraction::StreamKWorkgroupNumberCheck>(
        std::array<int, 2>{macroTile0, macroTile1});
    EXPECT_FALSE((*pred)(problem))
        << "M=" << m << " N=" << n << ": true tiles == 16,777,232 > 2^24, but a "
           "float-precision bug would round this down to exactly 2^24 and wrongly accept it.";
}
