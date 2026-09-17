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
// BufferStoreOffsetLimitCheck: silent-store-drop guard for BufferStore=True
// solutions (ROCM-31016 / https://github.com/ROCm/hipBLASLt/issues/2299).
//
// The SRD base is re-based per workgroup along the N dimension (see
// computeStoreSrdStart in KernelWriterAssembly.py), so the worst-case store
// byte offset from a given workgroup's SRD base is bounded by one
// MacroTile1's worth of columns, not the full D extent (min(MacroTile1,
// size[1]) is the correct quantity to check), confirmed by exhaustive
// hardware sweeps on gfx950 (MI350X): large-M/modest-N BufferStore=True
// solutions with a big D extent but a small MacroTile1 ran correctly at
// every tested size up to 16 GiB. (A large-M shape *can* still corrupt
// output, but through a different mechanism entirely, a Stream-K
// grid-dimension overflow; see StreamKWorkgroupNumberCheck below.)
//
// The threshold here mirrors KernelWriterAssembly.py's BufferOOB num_records
// sentinel (0xfffff000, ~4 GiB - 4 KiB).
// ----------------------------------------------------------------------------

TEST(Predicates, BufferStoreOffsetLimitCheck_OrdinaryProblem_StillAccepted)
{
    using namespace TensileLite;
    // Negative/sanity check: an ordinary, well-within-range problem must
    // still be accepted by the corrected predicate.
    auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                         false,
                                                         rocisa::DataType::BFloat16,
                                                         rocisa::DataType::BFloat16,
                                                         rocisa::DataType::BFloat16,
                                                         rocisa::DataType::BFloat16,
                                                         1024,
                                                         1024,
                                                         1024,
                                                         /*batchSize=*/1,
                                                         /*lda=*/1024,
                                                         /*aStride=*/-1,
                                                         /*ldb=*/1024,
                                                         /*bStride=*/-1,
                                                         /*ldc=*/1024,
                                                         /*cStride=*/-1,
                                                         /*ldd=*/1024,
                                                         /*dStride=*/-1,
                                                         /*beta=*/0.0);
    auto pred = std::make_shared<Predicates::Contraction::BufferStoreOffsetLimitCheck>(64);
    EXPECT_TRUE((*pred)(problem));
}

TEST(Predicates, BufferStoreOffsetLimitCheck_JustUnderCeiling_Accepted)
{
    using namespace TensileLite;
    // Boundary check: a D just under the BufferOOB sentinel must still pass.
    constexpr size_t n = 256;
    constexpr size_t m = (0xfffff000ull / 2 / n) - 1; // comfortably under the line
    auto             problem = ContractionProblemGemm::GEMM_Strides(false,
                                                         false,
                                                         rocisa::DataType::BFloat16,
                                                         rocisa::DataType::BFloat16,
                                                         rocisa::DataType::BFloat16,
                                                         rocisa::DataType::BFloat16,
                                                         m,
                                                         n,
                                                         /*k=*/48,
                                                         /*batchSize=*/1,
                                                         /*lda=*/m,
                                                         /*aStride=*/-1,
                                                         /*ldb=*/48,
                                                         /*bStride=*/-1,
                                                         /*ldc=*/m,
                                                         /*cStride=*/-1,
                                                         /*ldd=*/m,
                                                         /*dStride=*/-1,
                                                         /*beta=*/0.0);
    // macroTile1 == n so min(macroTile1, n) == n and the predicate actually
    // checks the full extent this test claims to exercise.
    auto pred = std::make_shared<Predicates::Contraction::BufferStoreOffsetLimitCheck>(n);
    EXPECT_TRUE((*pred)(problem));
}

TEST(Predicates, BufferStoreOffsetLimitCheck_BetweenSentinelAndTwoPow32_Rejected)
{
    using namespace TensileLite;
    // Isolates the threshold change (0xfffff000 vs. the old 2^32 constant)
    // from the min()-cap fix: macroTile1 == n here, so the old formula's
    // min(macroTile1, n) always equals n and computes the same full extent
    // the fixed formula does. A shape whose extent falls strictly between
    // the two constants is therefore only rejected by the corrected
    // threshold, not by the extent-formula fix on its own.
    constexpr size_t macroTile1 = 256;
    constexpr size_t n          = 256;
    constexpr size_t m          = 8388601;
    static_assert(m * n * 2 > 0xfffff000ull,
                  "extent must exceed the real hardware sentinel");
    static_assert(m * n * 2 < 4294967296ull,
                  "extent must stay under the old, looser 2^32 constant, or this "
                  "test would not isolate the threshold change");

    auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                         false,
                                                         rocisa::DataType::BFloat16,
                                                         rocisa::DataType::BFloat16,
                                                         rocisa::DataType::BFloat16,
                                                         rocisa::DataType::BFloat16,
                                                         m,
                                                         n,
                                                         /*k=*/48,
                                                         /*batchSize=*/1,
                                                         /*lda=*/m,
                                                         /*aStride=*/-1,
                                                         /*ldb=*/48,
                                                         /*bStride=*/-1,
                                                         /*ldc=*/m,
                                                         /*cStride=*/-1,
                                                         /*ldd=*/m,
                                                         /*dStride=*/-1,
                                                         /*beta=*/0.0);
    auto pred = std::make_shared<Predicates::Contraction::BufferStoreOffsetLimitCheck>(macroTile1);
    EXPECT_FALSE((*pred)(problem))
        << "M=" << m << " N=" << n << " bf16: true D extent falls between the "
           "0xfffff000 sentinel and 2^32, so the corrected threshold (not just "
           "the extent-formula fix) must reject this shape.";
}

// ----------------------------------------------------------------------------
// StreamKWorkgroupNumberCheck: grid-dimension-overflow guard for Stream-K
// solutions (ROCM-31016 / https://github.com/ROCm/hipBLASLt/issues/2299).
//
// WorkgroupNumberCheck (above) bounds the tile-scaled grid at
// MAX_WORKGROUP_NUMBER (2^24) but is skipped for Stream-K solutions, because
// Stream-K's normal grid is CU-scaled, not tile-scaled. But
// resolveStreamKSettings()/getSKGridImpl() (ContractionSolution.cpp) has
// several launch-time fallbacks (most notably the tree-fixup 24-bit
// bounds guard) that hand Stream-K a one-workgroup-per-tile grid instead,
// same shape as the grid WorkgroupNumberCheck already bounds. Without an
// equivalent check, that fallback grid can itself overflow the 32-bit
// work-item count used to launch the kernel (workGroupSize * numWorkGroups),
// wrapping to a much smaller-than-intended grid and leaving most of D
// unwritten.
//
// Confirmed on real gfx950 (MI350X) hardware with a MacroTile 16x16
// Stream-K-static (SK3) kernel at N=256 (so tiles == M): every solution
// hitting the tree-bounds fallback ran correctly up to and including
// M=16,777,216 (tiles == 2^24 exactly) and silently dropped most of D just
// above it (first confirmed-broken shape: M=16,781,312).
// ----------------------------------------------------------------------------

TEST(Predicates, StreamKWorkgroupNumberCheck_AtExactBoundary_Accepted)
{
    using namespace TensileLite;
    // MacroTile0=16, MacroTile1=16, N=256 -> tiles == M. Confirmed-safe on
    // gfx950 hardware: tiles == 2^24 exactly still runs correctly.
    constexpr int    macroTile0 = 16;
    constexpr int    macroTile1 = 16;
    constexpr size_t m          = 16777216; // 2^24
    constexpr size_t n          = 256;
    static_assert(m == 16777216, "tiles == m when n / macroTile1 == 16");

    auto problem = ContractionProblemGemm::GEMM(
        false, false, m, n, /*k=*/48, m, /*ldb=*/48, m, 0.0, false, /*batchSize=*/1);
    auto pred = std::make_shared<Predicates::Contraction::StreamKWorkgroupNumberCheck>(
        std::array<int, 2>{macroTile0, macroTile1});
    EXPECT_TRUE((*pred)(problem))
        << "M=" << m << " N=" << n << ": tiles == 2^24 exactly is confirmed safe on "
           "gfx950 hardware and must still be accepted.";
}

TEST(Predicates, StreamKWorkgroupNumberCheck_JustPastBoundary_Rejected_ROCM31016)
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
