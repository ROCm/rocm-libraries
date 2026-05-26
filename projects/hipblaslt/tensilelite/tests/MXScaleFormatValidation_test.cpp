// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/MXScaleFormatValidation.hpp>

#include <array>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace TensileLite;

// ============================================================================
// MX Scale Format Validation Tests
//
// gfx1250's v_wmma_scale_f32_16x16x128_f8f6f4 only accepts a fixed set of
// (matrix-A class, A-scale fmt, matrix-B class, B-scale fmt) tuples. The
// AMDGPU assembler does not enforce these joint constraints (see
// ROCm/llvm-project#2634), so the host has to. These tests exercise the
// validator surface in MXScaleFormatValidation.hpp and the
// ContractionProblemGemm hooks that call it.
//
// Enum mapping in tensilelite vocabulary:
//   - rocisa::DataType::Float8  matrix -> FP8   (also "E4M3" scale byte)
//   - rocisa::DataType::BFloat8 matrix -> BF8
//   - rocisa::DataType::Float6           -> FP6
//   - rocisa::DataType::BFloat6          -> BF6
//   - rocisa::DataType::Float4           -> FP4
//   - rocisa::DataType::E8               -> UE8M0 scale ("E8")
//   - rocisa::DataType::E5M3             -> E5M3 scale
//   - rocisa::DataType::Float8           -> E4M3 scale (OCP FP8 byte)
// ============================================================================

namespace
{
    // Shorthands so the table reads like the spec.
    constexpr auto FP8  = rocisa::DataType::Float8;
    constexpr auto BF8  = rocisa::DataType::BFloat8;
    constexpr auto FP6  = rocisa::DataType::Float6;
    constexpr auto BF6  = rocisa::DataType::BFloat6;
    constexpr auto FP4  = rocisa::DataType::Float4;
    constexpr auto E8   = rocisa::DataType::E8;
    constexpr auto E5M3 = rocisa::DataType::E5M3;
    constexpr auto E4M3 = rocisa::DataType::Float8; // OCP FP8 byte used as MX scale
    constexpr auto NONE = rocisa::DataType::None;

    // Build a ContractionProblemGemm with the requested A/B matrix dtypes.
    // Geometry mirrors tests/MXScalePadding_test.cpp::makeMXProblem - tile
    // and pad alignment match the kernel's expectations so the scale-tensor
    // creation in setMXScale*() succeeds.
    ContractionProblemGemm makeProblem(rocisa::DataType aType,
                                       rocisa::DataType bType,
                                       size_t           M = 128,
                                       size_t           N = 128,
                                       size_t           K = 256,
                                       bool             transA = true,
                                       bool             transB = false)
    {
        return ContractionProblemGemm::GEMM_Strides(
            transA,
            transB,
            aType,
            bType,
            rocisa::DataType::BFloat16,
            rocisa::DataType::BFloat16,
            M, N, K, /*batch=*/1,
            transA ? K : M,
            transA ? K * M : M * K,
            transB ? N : K,
            transB ? N * K : K * N,
            M, M * N,
            M, M * N,
            0.0);
    }
} // namespace

// ---------------------------------------------------------------------------
// Per-side rule: isValidMXScaleFormatForDataType
// ---------------------------------------------------------------------------

class MXPerSideValidTest
    : public ::testing::TestWithParam<std::tuple<rocisa::DataType, rocisa::DataType, bool>>
{
};

TEST_P(MXPerSideValidTest, MatchesSpec)
{
    auto [matrixDt, scaleDt, expected] = GetParam();
    EXPECT_EQ(isValidMXScaleFormatForDataType(matrixDt, scaleDt), expected);
}

INSTANTIATE_TEST_SUITE_P(
    PerSide,
    MXPerSideValidTest,
    ::testing::Values(
        // FP8 / BF8 / FP6 / BF6: only E8 is legal.
        std::make_tuple(FP8, E8,   true),
        std::make_tuple(FP8, E5M3, false),
        std::make_tuple(FP8, E4M3, false),
        std::make_tuple(FP8, BF8,  false),
        std::make_tuple(BF8, E8,   true),
        std::make_tuple(BF8, E5M3, false),
        std::make_tuple(BF8, E4M3, false),
        std::make_tuple(BF8, BF8,  false),
        std::make_tuple(FP6, E8,   true),
        std::make_tuple(FP6, E5M3, false),
        std::make_tuple(FP6, E4M3, false),
        std::make_tuple(FP6, BF8,  false),
        std::make_tuple(BF6, E8,   true),
        std::make_tuple(BF6, E5M3, false),
        std::make_tuple(BF6, E4M3, false),
        std::make_tuple(BF6, BF8,  false),
        // FP4: E8 / E5M3 / E4M3 all OK, BF8 / Half not.
        std::make_tuple(FP4, E8,   true),
        std::make_tuple(FP4, E5M3, true),
        std::make_tuple(FP4, E4M3, true),
        std::make_tuple(FP4, BF8,  false),
        std::make_tuple(FP4, rocisa::DataType::Half, false),
        // Non-MX matrix classes: the per-side rule does not apply, so the
        // function returns true regardless of the scale dtype.
        std::make_tuple(rocisa::DataType::Float,    E8,   true),
        std::make_tuple(rocisa::DataType::Float,    E5M3, true),
        std::make_tuple(rocisa::DataType::Half,     E8,   true),
        std::make_tuple(rocisa::DataType::Half,     BF8,  true),
        std::make_tuple(rocisa::DataType::BFloat16, E8,   true),
        std::make_tuple(rocisa::DataType::BFloat16, E5M3, true),
        std::make_tuple(NONE,                       NONE, true)
    )
);

// ---------------------------------------------------------------------------
// isMXMatrixDataType / isFP4MatrixDataType classification
// ---------------------------------------------------------------------------

TEST(MXClassification, MXMatrixClass)
{
    EXPECT_TRUE(isMXMatrixDataType(FP8));
    EXPECT_TRUE(isMXMatrixDataType(BF8));
    EXPECT_TRUE(isMXMatrixDataType(FP6));
    EXPECT_TRUE(isMXMatrixDataType(BF6));
    EXPECT_TRUE(isMXMatrixDataType(FP4));

    EXPECT_FALSE(isMXMatrixDataType(rocisa::DataType::Float));
    EXPECT_FALSE(isMXMatrixDataType(rocisa::DataType::Half));
    EXPECT_FALSE(isMXMatrixDataType(rocisa::DataType::BFloat16));
    EXPECT_FALSE(isMXMatrixDataType(NONE));
}

TEST(MXClassification, FP4MatrixClass)
{
    EXPECT_TRUE(isFP4MatrixDataType(FP4));
    EXPECT_FALSE(isFP4MatrixDataType(FP8));
    EXPECT_FALSE(isFP4MatrixDataType(BF8));
    EXPECT_FALSE(isFP4MatrixDataType(FP6));
    EXPECT_FALSE(isFP4MatrixDataType(BF6));
    EXPECT_FALSE(isFP4MatrixDataType(rocisa::DataType::Float));
}

// ---------------------------------------------------------------------------
// Joint rule: isValidMXScaleFormatCombination
//
// This parametrized table covers the full set of 43 valid combinations
// implied by the spec plus a curated set of invalid sample tuples.
//
// Valid layout:
//   - 4x4 = 16 (non-FP4 x non-FP4, all forced to E8/E8)
//   - 4 x 3 = 12 (non-FP4 A x FP4 B, A forced E8, B chooses E8/E5M3/E4M3)
//   - 3 x 4 = 12 (FP4 A x non-FP4 B, B forced E8, A chooses E8/E5M3/E4M3)
//   - 3 (FP4 x FP4, matching scales)
//   --------- 43
// ---------------------------------------------------------------------------

class MXCombinationTest : public ::testing::TestWithParam<
                              std::tuple<rocisa::DataType,
                                         rocisa::DataType,
                                         rocisa::DataType,
                                         rocisa::DataType,
                                         bool>>
{
};

TEST_P(MXCombinationTest, MatchesSpec)
{
    auto [aType, aScale, bType, bScale, expected] = GetParam();
    EXPECT_EQ(isValidMXScaleFormatCombination(aType, aScale, bType, bScale),
              expected)
        << "Combination "
        << formatMXScaleFormatCombination(aType, aScale, bType, bScale);
}

// ---- 43 valid combinations -------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    Valid_NonFP4_x_NonFP4,
    MXCombinationTest,
    ::testing::Values(
        // 4 x 4 = 16 non-FP4 / non-FP4 pairs, all scales forced to E8.
        std::make_tuple(FP8, E8, FP8, E8, true),
        std::make_tuple(FP8, E8, BF8, E8, true),
        std::make_tuple(FP8, E8, FP6, E8, true),
        std::make_tuple(FP8, E8, BF6, E8, true),
        std::make_tuple(BF8, E8, FP8, E8, true),
        std::make_tuple(BF8, E8, BF8, E8, true),
        std::make_tuple(BF8, E8, FP6, E8, true),
        std::make_tuple(BF8, E8, BF6, E8, true),
        std::make_tuple(FP6, E8, FP8, E8, true),
        std::make_tuple(FP6, E8, BF8, E8, true),
        std::make_tuple(FP6, E8, FP6, E8, true),
        std::make_tuple(FP6, E8, BF6, E8, true),
        std::make_tuple(BF6, E8, FP8, E8, true),
        std::make_tuple(BF6, E8, BF8, E8, true),
        std::make_tuple(BF6, E8, FP6, E8, true),
        std::make_tuple(BF6, E8, BF6, E8, true)
    )
);

INSTANTIATE_TEST_SUITE_P(
    Valid_NonFP4_x_FP4,
    MXCombinationTest,
    ::testing::Values(
        // 4 (A in {FP8,BF8,FP6,BF6}) x 3 (B scale in {E8,E5M3,E4M3}) = 12.
        std::make_tuple(FP8, E8, FP4, E8,   true),
        std::make_tuple(FP8, E8, FP4, E5M3, true),
        std::make_tuple(FP8, E8, FP4, E4M3, true),
        std::make_tuple(BF8, E8, FP4, E8,   true),
        std::make_tuple(BF8, E8, FP4, E5M3, true),
        std::make_tuple(BF8, E8, FP4, E4M3, true),
        std::make_tuple(FP6, E8, FP4, E8,   true),
        std::make_tuple(FP6, E8, FP4, E5M3, true),
        std::make_tuple(FP6, E8, FP4, E4M3, true),
        std::make_tuple(BF6, E8, FP4, E8,   true),
        std::make_tuple(BF6, E8, FP4, E5M3, true),
        std::make_tuple(BF6, E8, FP4, E4M3, true)
    )
);

INSTANTIATE_TEST_SUITE_P(
    Valid_FP4_x_NonFP4,
    MXCombinationTest,
    ::testing::Values(
        // 3 (A scale in {E8,E5M3,E4M3}) x 4 (B in {FP8,BF8,FP6,BF6}) = 12.
        std::make_tuple(FP4, E8,   FP8, E8, true),
        std::make_tuple(FP4, E5M3, FP8, E8, true),
        std::make_tuple(FP4, E4M3, FP8, E8, true),
        std::make_tuple(FP4, E8,   BF8, E8, true),
        std::make_tuple(FP4, E5M3, BF8, E8, true),
        std::make_tuple(FP4, E4M3, BF8, E8, true),
        std::make_tuple(FP4, E8,   FP6, E8, true),
        std::make_tuple(FP4, E5M3, FP6, E8, true),
        std::make_tuple(FP4, E4M3, FP6, E8, true),
        std::make_tuple(FP4, E8,   BF6, E8, true),
        std::make_tuple(FP4, E5M3, BF6, E8, true),
        std::make_tuple(FP4, E4M3, BF6, E8, true)
    )
);

INSTANTIATE_TEST_SUITE_P(
    Valid_FP4_x_FP4_MatchingScales,
    MXCombinationTest,
    ::testing::Values(
        // 3 FP4xFP4 tuples - scales must match.
        std::make_tuple(FP4, E8,   FP4, E8,   true),
        std::make_tuple(FP4, E5M3, FP4, E5M3, true),
        std::make_tuple(FP4, E4M3, FP4, E4M3, true)
    )
);

// ---- Invalid samples -------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    Invalid_NonFP4_BadAScale,
    MXCombinationTest,
    ::testing::Values(
        // Any A in {FP8,BF8,FP6,BF6} with a non-E8 scale is invalid even if
        // the B side is fine.
        std::make_tuple(FP8, E5M3, FP8, E8, false),
        std::make_tuple(FP8, E4M3, FP6, E8, false),
        std::make_tuple(BF8, E5M3, FP4, E8, false),
        std::make_tuple(BF8, E4M3, FP4, E5M3, false),
        std::make_tuple(FP6, E5M3, FP6, E8, false),
        std::make_tuple(FP6, E4M3, BF6, E8, false),
        std::make_tuple(BF6, E5M3, FP8, E8, false),
        std::make_tuple(BF6, E4M3, FP4, E4M3, false)
    )
);

INSTANTIATE_TEST_SUITE_P(
    Invalid_NonFP4_BadBScale,
    MXCombinationTest,
    ::testing::Values(
        // A side good, B side bad.
        std::make_tuple(FP8, E8, FP8, E5M3, false),
        std::make_tuple(FP8, E8, BF8, E4M3, false),
        std::make_tuple(FP6, E8, FP6, E5M3, false),
        std::make_tuple(BF6, E8, BF8, E4M3, false),
        std::make_tuple(FP4, E8, FP6, E5M3, false),
        std::make_tuple(FP4, E5M3, FP8, E5M3, false), // B is FP8 -> needs E8
        std::make_tuple(FP4, E4M3, BF8, E4M3, false)
    )
);

INSTANTIATE_TEST_SUITE_P(
    Invalid_FP4_x_FP4_MismatchedScales,
    MXCombinationTest,
    ::testing::Values(
        std::make_tuple(FP4, E8,   FP4, E5M3, false),
        std::make_tuple(FP4, E8,   FP4, E4M3, false),
        std::make_tuple(FP4, E5M3, FP4, E8,   false),
        std::make_tuple(FP4, E5M3, FP4, E4M3, false),
        std::make_tuple(FP4, E4M3, FP4, E8,   false),
        std::make_tuple(FP4, E4M3, FP4, E5M3, false)
    )
);

// ---------------------------------------------------------------------------
// Mixed-class spot tests called out by the task spec.
// ---------------------------------------------------------------------------

TEST(MXMixedClass, ExpectedValidCombos)
{
    EXPECT_TRUE(isValidMXScaleFormatCombination(FP8, E8,   FP4, E8));
    EXPECT_TRUE(isValidMXScaleFormatCombination(FP6, E8,   FP4, E4M3));
    EXPECT_TRUE(isValidMXScaleFormatCombination(FP4, E5M3, FP6, E8));
    EXPECT_TRUE(isValidMXScaleFormatCombination(FP4, E4M3, FP8, E8));
}

TEST(MXMixedClass, ExpectedInvalidCombo)
{
    // B is FP8 -> needs E8.
    EXPECT_FALSE(isValidMXScaleFormatCombination(FP4, E5M3, FP8, E5M3));
}

// ---------------------------------------------------------------------------
// Error-string contract: empty on valid, mentions bug on invalid.
// ---------------------------------------------------------------------------

TEST(MXErrorString, EmptyOnValid)
{
    EXPECT_TRUE(mxScaleFormatCombinationError(FP8, E8, FP8, E8).empty());
    EXPECT_TRUE(mxScaleFormatCombinationError(FP4, E5M3, FP4, E5M3).empty());
    EXPECT_TRUE(mxScaleFormatCombinationError(FP4, E4M3, FP6, E8).empty());
}

TEST(MXErrorString, NonEmptyAndDescribesProblem)
{
    auto err = mxScaleFormatCombinationError(FP4, E8, FP4, E5M3);
    EXPECT_FALSE(err.empty());
    // The error must reference the bug tracker so log readers know why.
    EXPECT_NE(err.find("2634"), std::string::npos) << "err=" << err;
    // ...and it should describe what was rejected.
    EXPECT_NE(err.find("FP4"), std::string::npos) << "err=" << err;
    EXPECT_NE(err.find("FP4 x FP4"), std::string::npos) << "err=" << err;
}

TEST(MXErrorString, MentionsBadSideLabels)
{
    auto err = mxScaleFormatCombinationError(FP8, E5M3, FP6, E8);
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("A"), std::string::npos);
    EXPECT_NE(err.find("FP8"), std::string::npos);
    EXPECT_NE(err.find("E5M3"), std::string::npos);
}

TEST(MXErrorString, FormatLooksLikeTuple)
{
    auto s = formatMXScaleFormatCombination(FP4, E5M3, FP8, E8);
    EXPECT_NE(s.find("A=FP4"), std::string::npos)    << s;
    EXPECT_NE(s.find("AScale=E5M3"), std::string::npos) << s;
    EXPECT_NE(s.find("B=FP8"), std::string::npos)    << s;
    EXPECT_NE(s.find("BScale=E8"), std::string::npos)   << s;
}

// ---------------------------------------------------------------------------
// End-to-end integration: validateMXScaleFormats / isValidMXScaleFormats on
// a real ContractionProblemGemm.
// ---------------------------------------------------------------------------

TEST(MXProblemIntegration, FP4xFP4_MatchingScalesSucceeds)
{
    auto problem = makeProblem(FP4, FP4);
    EXPECT_NO_THROW(problem.setMXScaleA(E8, 32));
    EXPECT_NO_THROW(problem.setMXScaleB(E8, 32));
    EXPECT_TRUE(problem.isValidMXScaleFormats());
    EXPECT_NO_THROW(problem.validateMXScaleFormats());
}

TEST(MXProblemIntegration, FP4xFP4_MismatchedScalesThrows)
{
    auto problem = makeProblem(FP4, FP4);
    // Setting A by itself is fine (B has no MX yet).
    EXPECT_NO_THROW(problem.setMXScaleA(E8, 32));
    // Setting B with a mismatched scale triggers the joint-rule throw.
    EXPECT_THROW(problem.setMXScaleB(E5M3, 32), std::runtime_error);
}

TEST(MXProblemIntegration, FP4xFP4_AllMatchingTriples)
{
    for(auto scale : {E8, E5M3, E4M3})
    {
        auto problem = makeProblem(FP4, FP4);
        EXPECT_NO_THROW(problem.setMXScaleA(scale, 32));
        EXPECT_NO_THROW(problem.setMXScaleB(scale, 32));
        EXPECT_TRUE(problem.isValidMXScaleFormats());
    }
}

TEST(MXProblemIntegration, FP4xFP4_AllMismatchedPairsThrow)
{
    const std::array<rocisa::DataType, 3> scales{E8, E5M3, E4M3};
    for(auto a : scales)
    {
        for(auto b : scales)
        {
            if(a == b)
                continue;
            auto problem = makeProblem(FP4, FP4);
            EXPECT_NO_THROW(problem.setMXScaleA(a, 32));
            EXPECT_THROW(problem.setMXScaleB(b, 32), std::runtime_error)
                << "A=" << static_cast<int>(a) << " B=" << static_cast<int>(b);
        }
    }
}

TEST(MXProblemIntegration, ThrowMessageMentionsSideLabels)
{
    auto problem = makeProblem(FP4, FP4);
    problem.setMXScaleA(E8, 32);
    try
    {
        problem.setMXScaleB(E5M3, 32);
        FAIL() << "Expected std::runtime_error";
    }
    catch(std::runtime_error const& e)
    {
        std::string msg = e.what();
        EXPECT_NE(msg.find("A=FP4"), std::string::npos)        << msg;
        EXPECT_NE(msg.find("BScale=E5M3"), std::string::npos) << msg;
        EXPECT_NE(msg.find("2634"), std::string::npos)         << msg;
    }
}

TEST(MXProblemIntegration, IsValidReturnsFalseInsteadOfThrowing)
{
    // setMXScale*() mutates state before validating, so a thrown call leaves
    // the problem in the invalid state. The non-throwing accessor must
    // report that.
    auto problem = makeProblem(FP4, FP4);
    EXPECT_NO_THROW(problem.setMXScaleA(E8, 32));
    EXPECT_TRUE(problem.isValidMXScaleFormats());

    try
    {
        problem.setMXScaleB(E5M3, 32);
        FAIL() << "Expected std::runtime_error";
    }
    catch(std::runtime_error const&)
    {
        // State is now (FP4,E8,FP4,E5M3) - mismatched, per the joint rule.
        EXPECT_FALSE(problem.isValidMXScaleFormats());
        EXPECT_THROW(problem.validateMXScaleFormats(), std::runtime_error);
    }
}

// One-sided MX on the A side: B has no MX scale so the FP4 x FP4 joint rule
// must not trigger, regardless of what A's scale is (as long as it is
// per-side valid for FP4).
TEST(MXProblemIntegration, OnlyA_FP4_AcceptsAllLegalAScales)
{
    for(auto scaleA : {E8, E5M3, E4M3})
    {
        auto problem = makeProblem(FP4, rocisa::DataType::BFloat16);
        EXPECT_NO_THROW(problem.setMXScaleA(scaleA, 32))
            << "scaleA=" << static_cast<int>(scaleA);
        EXPECT_TRUE(problem.isValidMXScaleFormats());
    }
}

TEST(MXProblemIntegration, OnlyA_FP8_AcceptsE8)
{
    auto problem = makeProblem(FP8, rocisa::DataType::BFloat16);
    EXPECT_NO_THROW(problem.setMXScaleA(E8, 32));
    EXPECT_TRUE(problem.isValidMXScaleFormats());
}

TEST(MXProblemIntegration, OnlyA_FP8_RejectsE5M3)
{
    auto problem = makeProblem(FP8, rocisa::DataType::BFloat16);
    EXPECT_THROW(problem.setMXScaleA(E5M3, 32), std::runtime_error);
}

// Symmetric "B only" one-sided cases.
TEST(MXProblemIntegration, OnlyB_FP4_AcceptsAllLegalBScales)
{
    for(auto scaleB : {E8, E5M3, E4M3})
    {
        auto problem = makeProblem(rocisa::DataType::BFloat16, FP4);
        EXPECT_NO_THROW(problem.setMXScaleB(scaleB, 32))
            << "scaleB=" << static_cast<int>(scaleB);
        EXPECT_TRUE(problem.isValidMXScaleFormats());
    }
}

TEST(MXProblemIntegration, OnlyB_FP8_AcceptsE8)
{
    auto problem = makeProblem(rocisa::DataType::BFloat16, FP8);
    EXPECT_NO_THROW(problem.setMXScaleB(E8, 32));
    EXPECT_TRUE(problem.isValidMXScaleFormats());
}

TEST(MXProblemIntegration, OnlyB_FP6_RejectsE4M3)
{
    auto problem = makeProblem(rocisa::DataType::BFloat16, FP6);
    EXPECT_THROW(problem.setMXScaleB(E4M3, 32), std::runtime_error);
}

// Mixed-class problem (FP8 x FP4) built end-to-end.
TEST(MXProblemIntegration, MixedFP8FP4_Succeeds)
{
    auto problem = makeProblem(FP8, FP4);
    EXPECT_NO_THROW(problem.setMXScaleA(E8, 32));
    EXPECT_NO_THROW(problem.setMXScaleB(E5M3, 32));
    EXPECT_TRUE(problem.isValidMXScaleFormats());
}

TEST(MXProblemIntegration, MixedFP4FP8_RequiresE8OnB)
{
    auto problem = makeProblem(FP4, FP8);
    EXPECT_NO_THROW(problem.setMXScaleA(E5M3, 32));
    EXPECT_THROW(problem.setMXScaleB(E5M3, 32), std::runtime_error);
}

// No-MX problem (mxBlock == 0 on both sides) must always pass validation.
TEST(MXProblemIntegration, NoMxScalingIsAlwaysValid)
{
    auto problem = makeProblem(rocisa::DataType::BFloat16,
                               rocisa::DataType::BFloat16);
    EXPECT_TRUE(problem.isValidMXScaleFormats());
    EXPECT_NO_THROW(problem.validateMXScaleFormats());
}
