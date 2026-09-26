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
// BufferStoreOffsetLimitCheck bounds the byte range a kernel's output stores
// can reach, so that solution selection never offers a shape the kernel would
// only partly write. The bound has to be the BufferOOB sentinel that
// KernelWriterAssembly.py programs as the store descriptor's num_records,
// 0xfffff000. A larger bound admits shapes whose stores the hardware discards
// past that offset, without raising a fault.
//
// Regression coverage for ROCM-31016, reported as
// https://github.com/ROCm/hipBLASLt/issues/2299.
//
// The M values below come from hardware rather than from arithmetic. Measured
// on an MI350X (gfx950) at N = 256, K = 48 in bfloat16: at M = 8,388,607, 22 of
// the 64 algorithms the heuristic offered returned success while leaving part
// of D unwritten, and every one of them had MacroTile1 = 256, which for this
// shape makes one workgroup's store extent span the whole tensor. M = 8,388,599
// is the largest M in the same geometry where all 64 ran clean.
// ----------------------------------------------------------------------------

namespace
{
    // Column-major bf16 D of M x N with ldd == M, which is the layout the
    // reported defect used.
    TensileLite::ContractionProblemGemm bf16ColumnMajorD(size_t m, size_t n, size_t k = 48)
    {
        using namespace TensileLite;
        return ContractionProblemGemm::GEMM_Strides(false,
                                                    false,
                                                    rocisa::DataType::BFloat16,
                                                    rocisa::DataType::BFloat16,
                                                    rocisa::DataType::BFloat16,
                                                    rocisa::DataType::BFloat16,
                                                    m,
                                                    n,
                                                    k,
                                                    /*batchSize=*/1,
                                                    /*lda=*/m,
                                                    /*aStride=*/-1,
                                                    /*ldb=*/k,
                                                    /*bStride=*/-1,
                                                    /*ldc=*/m,
                                                    /*cStride=*/-1,
                                                    /*ldd=*/m,
                                                    /*dStride=*/-1,
                                                    /*beta=*/0.0);
    }
}

TEST(Predicates, BufferStoreOffsetLimitCheck_PastSentinel_Rejected)
{
    using namespace TensileLite;
    // The shape that reproduced the defect. Its store extent lands between the
    // sentinel the kernel enforces and the 2^32 the predicate used to compare
    // against, so only the corrected threshold rejects it.
    constexpr size_t m          = 8388607;
    constexpr size_t n          = 256;
    constexpr size_t macroTile1 = 256;
    static_assert(m * n * 2 > 0xfffff000ull, "must exceed the sentinel the kernel enforces");
    static_assert(m * n * 2 < 4294967296ull, "must stay under the threshold the predicate used");

    auto pred = std::make_shared<Predicates::Contraction::BufferStoreOffsetLimitCheck>(macroTile1);
    EXPECT_FALSE((*pred)(bf16ColumnMajorD(m, n)))
        << "M=" << m << " N=" << n << " bf16: the store extent is " << (size_t)m * n * 2
        << " bytes, past the 4294963200-byte sentinel, so 22 of 64 offered algorithms left "
           "part of D unwritten on gfx950.";
}

TEST(Predicates, BufferStoreOffsetLimitCheck_BelowSentinel_Accepted)
{
    using namespace TensileLite;
    // The largest shape in this geometry that the kernel writes in full, and
    // the control the hardware measurement used: all 64 offered algorithms ran
    // clean here. Rejecting it would cost solutions for no reason, so this
    // holds the accepting side of the boundary one shape away from the test
    // above.
    constexpr size_t m          = 8388599;
    constexpr size_t n          = 256;
    constexpr size_t macroTile1 = 256;
    static_assert(m * n * 2 < 0xfffff000ull, "must stay within the sentinel");

    auto pred = std::make_shared<Predicates::Contraction::BufferStoreOffsetLimitCheck>(macroTile1);
    EXPECT_TRUE((*pred)(bf16ColumnMajorD(m, n)))
        << "M=" << m << " N=" << n << " bf16: extent " << (size_t)m * n * 2
        << " bytes fits within the sentinel and must remain selectable.";
}

TEST(Predicates, BufferStoreOffsetLimitCheck_ExactlyAtSentinel_RejectedConservatively)
{
    using namespace TensileLite;
    // One shape sits exactly on the sentinel: its last store begins at
    // extent - 2 and so is still in range, but the comparison is strict and
    // rejects it. Losing a single extent value is worth not having to reason
    // about how wide the final store of a tile is, which varies with the
    // vector width a solution uses. Recorded here so the choice is visible
    // rather than looking like an off-by-one.
    constexpr size_t n          = 256;
    constexpr size_t macroTile1 = 256;
    constexpr size_t m          = 0xfffff000ull / (n * 2);
    static_assert(m * n * 2 == 0xfffff000ull, "extent lands exactly on the sentinel");

    auto pred = std::make_shared<Predicates::Contraction::BufferStoreOffsetLimitCheck>(macroTile1);
    EXPECT_FALSE((*pred)(bf16ColumnMajorD(m, n)));
}

TEST(Predicates, BufferStoreOffsetLimitCheck_ReportedShape_Accepted)
{
    using namespace TensileLite;
    // The shape in the original report. It was rejected by the hardware on
    // ROCm 7.2.x, where the kernel's sentinel was 0x80000000, and runs
    // correctly now that the sentinel is 0xfffff000. It must stay selectable:
    // rejecting it would turn a fixed case back into a lost solution.
    auto pred = std::make_shared<Predicates::Contraction::BufferStoreOffsetLimitCheck>(256);
    EXPECT_TRUE((*pred)(bf16ColumnMajorD(5326464, 256)));
}

TEST(Predicates, BufferStoreOffsetLimitCheck_NarrowOutputUsesItsOwnWidth)
{
    using namespace TensileLite;
    // A workgroup re-bases along N, so the extent is one MacroTile1 of columns.
    // When N is smaller than MacroTile1 the tensor supplies the width, and a
    // tall output with only 8 columns stays selectable even though multiplying
    // by MacroTile1 would put it past the sentinel.
    constexpr size_t n          = 8;
    constexpr size_t macroTile1 = 256;
    constexpr size_t m          = 100000000;
    static_assert(m * n * 2 < 0xfffff000ull, "the real extent fits");
    static_assert(m * macroTile1 * 2 > 0xfffff000ull, "ignoring min() would not");

    auto pred = std::make_shared<Predicates::Contraction::BufferStoreOffsetLimitCheck>(macroTile1);
    EXPECT_TRUE((*pred)(bf16ColumnMajorD(m, n)));
}
