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

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include <Tensile/ContractionProblemPredicates.hpp>
#ifndef TENSILE_YAML
#include <Tensile/ContractionTaskPredicates.hpp>
#include <Tensile/Serialization/ContractionPredicates.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#endif

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

// ----------------------------------------------------------------------------
// StrideAMultiple / StrideBMultiple: strides are in elements.
// ----------------------------------------------------------------------------

// Non-transposed GEMM, so lda and ldb are the leading dimensions under test.
static TensileLite::ContractionProblemGemm strideProblem(size_t lda, size_t ldb)
{
    using namespace TensileLite;
    return ContractionProblemGemm::GEMM_Strides(false,
                                                false,
                                                rocisa::DataType::Float8,
                                                rocisa::DataType::Float8,
                                                rocisa::DataType::BFloat16,
                                                rocisa::DataType::BFloat16,
                                                /*m=*/64,
                                                /*n=*/64,
                                                /*k=*/64,
                                                /*batchSize=*/1,
                                                lda,
                                                /*aStride=*/-1,
                                                ldb,
                                                /*bStride=*/-1,
                                                /*ldc=*/64,
                                                /*cStride=*/-1,
                                                /*ldd=*/64,
                                                /*dStride=*/-1,
                                                /*beta=*/0.0);
}

struct StrideAOperand
{
    using Predicate                    = TensileLite::Predicates::Contraction::StrideAMultiple;
    static constexpr char const* name = "StrideAMultiple";

    static TensileLite::ContractionProblemGemm problem(size_t ld)
    {
        return strideProblem(ld, 64);
    }
};

struct StrideBOperand
{
    using Predicate                    = TensileLite::Predicates::Contraction::StrideBMultiple;
    static constexpr char const* name = "StrideBMultiple";

    static TensileLite::ContractionProblemGemm problem(size_t ld)
    {
        return strideProblem(64, ld);
    }
};

template <typename Operand>
class StrideMultipleTest : public ::testing::Test
{
};

using StrideOperands = ::testing::Types<StrideAOperand, StrideBOperand>;
TYPED_TEST_SUITE(StrideMultipleTest, StrideOperands);

TYPED_TEST(StrideMultipleTest, LeadingStrideMustBeANonZeroMultiple)
{
    using Predicate = typename TypeParam::Predicate;
    EXPECT_EQ(Predicate::Type(), TypeParam::name);

    Predicate multipleOfFour(1, 4);
    for(size_t ld : {4, 8, 5120})
        EXPECT_TRUE(multipleOfFour(TypeParam::problem(ld))) << "ld=" << ld;
    for(size_t ld : {4098, 5121})
        EXPECT_FALSE(multipleOfFour(TypeParam::problem(ld))) << "ld=" << ld;

    // A broadcast stride would reach a kernel that takes it as a byte kernarg.
    EXPECT_FALSE(multipleOfFour(TypeParam::problem(0)));
    EXPECT_FALSE(Predicate(1, 0)(TypeParam::problem(5120)));
    // Index 0 is the unit stride.
    EXPECT_TRUE(Predicate(0, 1)(TypeParam::problem(5121)));
    EXPECT_FALSE(Predicate(0, 4)(TypeParam::problem(5120)));
}

TYPED_TEST(StrideMultipleTest, OutOfRangeIndexIsRejectedAndReported)
{
    using Predicate = typename TypeParam::Predicate;
    Predicate outOfRange(7, 4);

    EXPECT_FALSE(outOfRange(TypeParam::problem(5120)));

    std::ostringstream stream;
    EXPECT_FALSE(outOfRange.debugEval(TypeParam::problem(5120), stream));
    EXPECT_NE(stream.str().find("out of range"), std::string::npos) << stream.str();
}

// ----------------------------------------------------------------------------
// K predicates of the skinny MXFP8 kernels: SizeGreaterThan {3: 0} rejects an empty K, and
// the K multiple is 256 (BoundSizeMultiple), or 1024 for the K16 kernel (SizeMultiple {3: 1024}).
// ----------------------------------------------------------------------------

static TensileLite::ContractionProblemGemm kProblem(size_t k, size_t m = 64, size_t n = 2)
{
    using namespace TensileLite;
    return ContractionProblemGemm::GEMM_Strides(false,
                                                false,
                                                rocisa::DataType::Float8,
                                                rocisa::DataType::Float8,
                                                rocisa::DataType::BFloat16,
                                                rocisa::DataType::BFloat16,
                                                m,
                                                n,
                                                k,
                                                /*batchSize=*/1,
                                                /*lda=*/k,
                                                /*aStride=*/-1,
                                                /*ldb=*/k,
                                                /*bStride=*/-1,
                                                /*ldc=*/m,
                                                /*cStride=*/-1,
                                                /*ldd=*/m,
                                                /*dStride=*/-1,
                                                /*beta=*/0.0);
}

TEST(Predicates, KPredicatesOfTheSkinnyKernels)
{
    using namespace TensileLite;
    using namespace TensileLite::Predicates::Contraction;

    SizeGreaterThan notEmpty(3, 0);
    BoundSizeMultiple whole256(-1, 256);
    SizeMultiple      multiple1024(3, 1024);
    // A zero divisor would otherwise raise SIGFPE.
    SizeMultiple multipleOfZero(3, 0);

    struct PredicateRow
    {
        char const*                  name;
        TensileLite::Predicates::Predicate<ContractionProblemGemm> const* predicate;
        std::vector<size_t>          accepted;
        std::vector<size_t>          rejected;
    };

    std::vector<PredicateRow> const rows = {
        {"SizeGreaterThan{3,0}", &notEmpty, {1, 256, 1280}, {0}},
        {"BoundSizeMultiple{-1,256}", &whole256, {256, 512, 1024, 1280, 25600}, {1, 255, 257, 5121}},
        {"SizeMultiple{3,1024}", &multiple1024, {1024, 2048, 5120, 25600}, {256, 1280, 1536, 5121}},
        {"SizeMultiple{3,0}", &multipleOfZero, {}, {256}},
    };
    for(PredicateRow const& row : rows)
    {
        for(size_t k : row.accepted)
            EXPECT_TRUE((*row.predicate)(kProblem(k))) << row.name << " k=" << k;
        for(size_t k : row.rejected)
        {
            EXPECT_FALSE((*row.predicate)(kProblem(k))) << row.name << " k=" << k;
            std::ostringstream stream;
            EXPECT_FALSE(row.predicate->debugEval(kProblem(k), stream)) << row.name << " k=" << k;
        }
    }
}

// The wide cap {0: 1048577, 1: 1048577} bounds m and n independently and leaves k alone.
TEST(Predicates, SizeLessThanBoundsEachIndexIndependently)
{
    using namespace TensileLite;
    using TensileLite::Predicates::Contraction::SizeLessThan;
    constexpr size_t cap = 1048577;
    SizeLessThan     capM(0, cap);
    SizeLessThan     capN(1, cap);

    EXPECT_TRUE(capM(kProblem(256, cap - 1, 2)));
    EXPECT_FALSE(capM(kProblem(256, cap, 2)));
    EXPECT_TRUE(capM(kProblem(256, 2, cap)));
    EXPECT_TRUE(capN(kProblem(256, 2, cap - 1)));
    EXPECT_FALSE(capN(kProblem(256, 2, cap)));
    EXPECT_TRUE(capN(kProblem(256, cap, 2)));

    // K (index 3) is not bounded by the caps on indices 0 and 1.
    EXPECT_TRUE(capM(kProblem(10 * cap)));
    EXPECT_TRUE(capN(kProblem(10 * cap)));
    EXPECT_FALSE(SizeLessThan(3, cap)(kProblem(cap)));
}

// The problem-type predicates Python emits for a solution without bias, activation or
// ScaleAlphaVec (UseBias 0, Activation None, UseScaleAlphaVec 0) reject problems using them.
TEST(Predicates, ProblemTypeWithoutBiasActivationOrScaleAlphaVecRejectsThem)
{
    using namespace TensileLite;
    using namespace TensileLite::Predicates::Contraction;

    UseBiasCheck          noBias(0);
    ActivationCheck       noActivation(ActivationType::None);
    UseScaleAlphaVecCheck noScaleAlphaVec(0);
    EXPECT_EQ(UseBiasCheck::Type(), "UseBias");
    EXPECT_EQ(ActivationCheck::Type(), "Activation");
    EXPECT_EQ(UseScaleAlphaVecCheck::Type(), "UseScaleAlphaVec");

    ContractionProblemGemm plain = kProblem(256);
    EXPECT_TRUE(noBias(plain));
    EXPECT_TRUE(noActivation(plain));
    EXPECT_TRUE(noScaleAlphaVec(plain));

    ContractionProblemGemm withBias = kProblem(256);
    withBias.setUseBias(1);
    EXPECT_FALSE(noBias(withBias));
    EXPECT_TRUE(noActivation(withBias));
    EXPECT_TRUE(noScaleAlphaVec(withBias));

    ContractionProblemGemm withActivation = kProblem(256);
    withActivation.setActivationType(ActivationType::Relu);
    EXPECT_TRUE(noBias(withActivation));
    EXPECT_FALSE(noActivation(withActivation));
    EXPECT_TRUE(noScaleAlphaVec(withActivation));

    ContractionProblemGemm withScaleAlphaVec = kProblem(256);
    withScaleAlphaVec.setUseScaleAlphaVec(1);
    EXPECT_TRUE(noBias(withScaleAlphaVec));
    EXPECT_TRUE(noActivation(withScaleAlphaVec));
    EXPECT_FALSE(noScaleAlphaVec(withScaleAlphaVec));

    // A solution that supports them accepts them.
    EXPECT_TRUE(UseBiasCheck(1)(withBias));
    EXPECT_TRUE(ActivationCheck(ActivationType::Relu)(withActivation));
    EXPECT_TRUE(UseScaleAlphaVecCheck(1)(withScaleAlphaVec));
}

// ----------------------------------------------------------------------------
// TensorBytesLessThan: the skinny kernels compute each tensor's 32-bit buffer extent
// from (rows - 1) * ld * elementBytes + extent * elementBytes, leading strides included.
// ----------------------------------------------------------------------------

static constexpr uint64_t bytesLimit = uint64_t(1) << 32;

// TN with 1-byte A/B and 2-byte C/D, so each tensor's bytes are easy to place.
static TensileLite::ContractionProblemGemm tensorBytesProblem(
    size_t m, size_t n, size_t k, size_t lda, size_t ldb, size_t ldc, size_t ldd)
{
    using namespace TensileLite;
    return ContractionProblemGemm::GEMM_Strides(true,
                                                false,
                                                rocisa::DataType::Float8,
                                                rocisa::DataType::Float8,
                                                rocisa::DataType::BFloat16,
                                                rocisa::DataType::BFloat16,
                                                m,
                                                n,
                                                k,
                                                /*batchSize=*/1,
                                                lda,
                                                /*aStride=*/-1,
                                                ldb,
                                                /*bStride=*/-1,
                                                ldc,
                                                /*cStride=*/-1,
                                                ldd,
                                                /*dStride=*/-1,
                                                /*beta=*/0.0);
}

namespace
{
    struct TensorBytesCase
    {
        char     tensor;
        size_t   m, n, k;
        size_t   ld;
        uint64_t bytes;
        bool     accepted;
    };

    struct TensorBytesProblem
    {
        TensileLite::ContractionProblemGemm problem;
        uint64_t                            formulaBytes;
        uint64_t                            descriptorBytes;
    };
}

// Overrides the leading stride of the named tensor and returns its byte count from the
// formula, independently of the descriptor.
static TensorBytesProblem problemFor(TensorBytesCase const& c)
{
    using namespace TensileLite;
    uint64_t const cdBytes = ((c.n - 1) * c.ld + c.m) * 2;
    switch(c.tensor)
    {
    case 'A': {
        ContractionProblemGemm problem = tensorBytesProblem(c.m, c.n, c.k, c.ld, c.k, c.m, c.m);
        return {problem, (c.m - 1) * c.ld + c.k, problem.a().totalAllocatedBytes()};
    }
    case 'B': {
        ContractionProblemGemm problem = tensorBytesProblem(c.m, c.n, c.k, c.k, c.ld, c.m, c.m);
        return {problem, (c.n - 1) * c.ld + c.k, problem.b().totalAllocatedBytes()};
    }
    case 'C': {
        ContractionProblemGemm problem = tensorBytesProblem(c.m, c.n, c.k, c.k, c.k, c.ld, c.m);
        return {problem, cdBytes, problem.c().totalAllocatedBytes()};
    }
    default: {
        ContractionProblemGemm problem = tensorBytesProblem(c.m, c.n, c.k, c.k, c.k, c.m, c.ld);
        return {problem, cdBytes, problem.d().totalAllocatedBytes()};
    }
    }
}

TEST(Predicates, TensorBytesLessThanIsExactAtTwoToThe32)
{
    using namespace TensileLite;
    Predicates::Contraction::TensorBytesLessThan pred(bytesLimit);
    EXPECT_EQ(Predicates::Contraction::TensorBytesLessThan::Type(), "TensorBytesLessThan");

    // A and B use 4096-byte rows, unpadded and padded; C and D pad the leading dimension of
    // a tiny tensor up to the limit.
    std::vector<TensorBytesCase> const cases = {
        {'A', 1048575, 2, 4096, 4096, bytesLimit - 4096, true},
        {'A', 1048576, 2, 4096, 4096, bytesLimit, false},
        {'A', 524288, 2, 4096, 8192, bytesLimit - 4096, true},
        {'A', 524288, 2, 4096, 8196, bytesLimit + 524287 * 4 - 4096, false},
        {'B', 2, 1048575, 4096, 4096, bytesLimit - 4096, true},
        {'B', 2, 1048576, 4096, 4096, bytesLimit, false},
        {'B', 2, 524288, 4096, 8192, bytesLimit - 4096, true},
        {'B', 2, 524288, 4096, 8196, bytesLimit + 524287 * 4 - 4096, false},
        {'C', 4, 2, 64, (size_t(1) << 31) - 8, bytesLimit - 8, true},
        {'C', 4, 2, 64, (size_t(1) << 31) - 4, bytesLimit, false},
        {'D', 4, 2, 64, (size_t(1) << 31) - 8, bytesLimit - 8, true},
        {'D', 4, 2, 64, (size_t(1) << 31) - 4, bytesLimit, false},
    };
    for(TensorBytesCase const& c : cases)
    {
        SCOPED_TRACE(std::string("tensor=") + c.tensor + " m=" + std::to_string(c.m)
                     + " n=" + std::to_string(c.n) + " ld=" + std::to_string(c.ld));
        TensorBytesProblem built = problemFor(c);

        EXPECT_EQ(built.formulaBytes, c.bytes);
        EXPECT_EQ(built.descriptorBytes, c.bytes);
        EXPECT_EQ(pred(built.problem), c.accepted);

        std::ostringstream stream;
        EXPECT_EQ(pred.debugEval(built.problem, stream), c.accepted);
    }
}

// A problem without a C tensor has no C extent to bound. The same problem with a C over the
// limit proves the skip is what lets the empty-C case through.
TEST(Predicates, TensorBytesLessThanSkipsAnEmptyC)
{
    using namespace TensileLite;
    Predicates::Contraction::TensorBytesLessThan pred(bytesLimit);

    // A, B and D are small; only C reaches the limit.
    ContractionProblemGemm bigC = tensorBytesProblem(4, 2, 64, 64, 64, (size_t(1) << 31) - 4, 4);
    ASSERT_EQ(bigC.c().totalAllocatedBytes(), bytesLimit);
    EXPECT_FALSE(pred(bigC));

    ContractionProblemGemm noC
        = ContractionProblemGemm::GEMM(true,
                                       false,
                                       bigC.a(),
                                       TensorOps(),
                                       bigC.b(),
                                       TensorOps(),
                                       TensorDescriptor("c"),
                                       TensorOps(),
                                       bigC.d(),
                                       TensorOps(),
                                       0.0);
    ASSERT_TRUE(noC.c().empty());
    EXPECT_TRUE(pred(noC));
    EXPECT_FALSE(Predicates::Contraction::TensorBytesLessThan(64)(noC));
}

#ifndef TENSILE_YAML
// The predicate must survive the library round trip with a value that needs 33 bits.
TEST(Predicates, TensorBytesLessThanDeserializesAFullWidthLimit)
{
    using namespace TensileLite;
    msgpack::sbuffer                 packed;
    msgpack::packer<msgpack::sbuffer> packer(packed);
    packer.pack_map(2);
    packer.pack(std::string("type"));
    packer.pack(std::string("TensorBytesLessThan"));
    packer.pack(std::string("value"));
    packer.pack(uint64_t(1) << 32);

    msgpack::object_handle unpacked = msgpack::unpack(packed.data(), packed.size());
    Serialization::MessagePackInput reader(unpacked.get());
    std::shared_ptr<Predicates::Predicate<ContractionProblemGemm>> pred;
    reader.input(pred);
    ASSERT_TRUE(reader.error.empty()) << reader.error.front();
    ASSERT_NE(pred, nullptr);

    ContractionProblemGemm over = tensorBytesProblem(1048576, 2, 4096, 4096, 4096, 1048576, 1048576);
    ContractionProblemGemm under = tensorBytesProblem(1048575, 2, 4096, 4096, 4096, 1048575, 1048575);
    ASSERT_EQ(over.a().totalAllocatedBytes(), bytesLimit);
    ASSERT_EQ(under.a().totalAllocatedBytes(), bytesLimit - 4096);
    EXPECT_FALSE((*pred)(over));
    EXPECT_TRUE((*pred)(under));
}
#endif
