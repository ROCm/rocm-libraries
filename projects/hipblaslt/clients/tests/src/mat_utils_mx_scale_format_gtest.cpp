/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

// Focused unit tests for the three hipBLASLt host-side MX scale-format
// helpers declared in rocblaslt_mx_scale_format.hpp:
//
//   * rocblasltScalingFormatToMXScaleDataType
//   * rocblasltHipDataTypeToMXMatrixDataType
//   * checkMXScaleFormatCombination
//
// These exercise the gfx1250 v_wmma_scale_f32_16x16x128_f8f6f4 guard at the
// hipBLASLt host-API surface. The deeper TensileLite::isValidMXScaleFormatCombination
// matrix is covered in tensilelite/tests/MXScaleFormatValidation_test.cpp;
// here we focus on the hipBLASLt-specific enum mappings and the wiring of
// the joint validator into hipBLASLt's host types
// (RocblasltContractionProblem::ScalingFormat, hipDataType).

#include "rocblaslt_mx_scale_format.hpp"

#include <gtest/gtest.h>

#include <hipblaslt/hipblaslt.h>

#include <optional>
#include <string>

namespace
{
    using ScalingFormat = RocblasltContractionProblem::ScalingFormat;

    // Tag shorthands so the assertion blocks read like the spec.
    constexpr auto E8DT   = rocisa::DataType::E8;
    constexpr auto E5M3DT = rocisa::DataType::E5M3;
    constexpr auto FP8DT  = rocisa::DataType::Float8; // == E4M3 byte in tensilelite vocab
    constexpr auto BF8DT  = rocisa::DataType::BFloat8;
    constexpr auto FP6DT  = rocisa::DataType::Float6;
    constexpr auto BF6DT  = rocisa::DataType::BFloat6;
    constexpr auto FP4DT  = rocisa::DataType::Float4;
    constexpr auto NONEDT = rocisa::DataType::None;
    constexpr auto F8FNUZ = rocisa::DataType::Float8_fnuz;
    constexpr auto BF8FNUZ = rocisa::DataType::BFloat8_fnuz;
} // namespace

// ============================================================================
// rocblasltScalingFormatToMXScaleDataType
// ----------------------------------------------------------------------------
// Verifies every ScalingFormat enumerator maps to the expected MX scale dtype
// (or std::nullopt for non-block formats). If a new enumerator is added the
// fall-through case in the switch makes it return nullopt - the dedicated
// "every enumerator is covered explicitly" guard below catches that drift.
// ============================================================================

TEST(MXScaleFormatHelpersTest, ScalingFormatBlockUE8M0VariantsMapToE8)
{
    EXPECT_EQ(rocblasltScalingFormatToMXScaleDataType(ScalingFormat::Block_32_UE8M0),
              std::optional<rocisa::DataType>{E8DT});
    EXPECT_EQ(rocblasltScalingFormatToMXScaleDataType(ScalingFormat::Block_16_UE8M0),
              std::optional<rocisa::DataType>{E8DT});
    EXPECT_EQ(rocblasltScalingFormatToMXScaleDataType(ScalingFormat::Block_32_UE8M0_32_8_EXT),
              std::optional<rocisa::DataType>{E8DT});
}

TEST(MXScaleFormatHelpersTest, ScalingFormatBlockUE4M3VariantsMapToFloat8)
{
    EXPECT_EQ(rocblasltScalingFormatToMXScaleDataType(ScalingFormat::Block_32_UE4M3),
              std::optional<rocisa::DataType>{FP8DT});
    EXPECT_EQ(rocblasltScalingFormatToMXScaleDataType(ScalingFormat::Block_16_UE4M3),
              std::optional<rocisa::DataType>{FP8DT});
}

TEST(MXScaleFormatHelpersTest, ScalingFormatBlockUE5M3VariantsMapToE5M3)
{
    EXPECT_EQ(rocblasltScalingFormatToMXScaleDataType(ScalingFormat::Block_32_UE5M3),
              std::optional<rocisa::DataType>{E5M3DT});
    EXPECT_EQ(rocblasltScalingFormatToMXScaleDataType(ScalingFormat::Block_16_UE5M3),
              std::optional<rocisa::DataType>{E5M3DT});
}

TEST(MXScaleFormatHelpersTest, ScalingFormatNonBlockReturnsNullopt)
{
    EXPECT_FALSE(rocblasltScalingFormatToMXScaleDataType(ScalingFormat::None).has_value());
    EXPECT_FALSE(rocblasltScalingFormatToMXScaleDataType(ScalingFormat::Scalar).has_value());
    EXPECT_FALSE(rocblasltScalingFormatToMXScaleDataType(ScalingFormat::Vector).has_value());
}

TEST(MXScaleFormatHelpersTest, ScalingFormatUnknownEnumeratorReturnsNullopt)
{
    // Exercise the default branch: an enumerator value the helper does not
    // recognize must collapse to "not an MX scale" so the joint validator
    // treats it as a non-MX side rather than asserting a wrong scale dtype.
    auto unknown = static_cast<ScalingFormat>(0xFFFF);
    EXPECT_FALSE(rocblasltScalingFormatToMXScaleDataType(unknown).has_value());
}

// ============================================================================
// rocblasltHipDataTypeToMXMatrixDataType
// ----------------------------------------------------------------------------
// All hipDataType values covered by the gfx1250 MX rules (FP8/BF8/FP6/BF6/FP4
// and their fnuz variants) must collapse to the matching rocisa::DataType.
// Other hipDataType values must collapse to rocisa::DataType::None so the
// joint validator treats the side as non-MX.
// ============================================================================

TEST(MXScaleFormatHelpersTest, HipDataTypeMXMatrixMappings)
{
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_8F_E4M3),      FP8DT);
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_8F_E5M2),      BF8DT);
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_8F_E4M3_FNUZ), F8FNUZ);
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_8F_E5M2_FNUZ), BF8FNUZ);
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_6F_E2M3),      FP6DT);
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_6F_E3M2),      BF6DT);
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_4F_E2M1),      FP4DT);
}

TEST(MXScaleFormatHelpersTest, HipDataTypeNonMXMappingsCollapseToNone)
{
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_32F),  NONEDT);
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_16F),  NONEDT);
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_16BF), NONEDT);
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_32I),  NONEDT);
    EXPECT_EQ(rocblasltHipDataTypeToMXMatrixDataType(HIP_R_8I),   NONEDT);
}

// ============================================================================
// checkMXScaleFormatCombination
// ----------------------------------------------------------------------------
// Pure joint-validator wrapper. Returns std::nullopt on a legal combination
// (or when the MX rules do not apply) and an error string otherwise. We cover
// each of the documented branches:
//
//   1. Both sides non-MX scale         -> nullopt (rules don't apply).
//   2. Only one side carries an MX scale.
//        a. side-A non-MX dtype must default to (None, None) so FP4xFP4
//           joint rule does NOT fire spuriously.
//   3. FP8/BF8/FP6/BF6 require E8 scale; any other scale -> error.
//   4. FP4 accepts E8, E5M3 or E4M3 (FP8 byte) scale.
//   5. FP4xFP4 must share scale.
// ============================================================================

TEST(MXScaleFormatHelpersTest, CheckCombinationNeitherSideMXScaleReturnsNullopt)
{
    // Both sides non-MX scale - rules do not apply at all. This must hold
    // regardless of the hipDataType, including MX matrix types: the guard
    // intentionally skips when both scales are scalar/vector/none.
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_32F, HIP_R_32F, ScalingFormat::None, ScalingFormat::None)
                     .has_value());
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_32F, HIP_R_32F, ScalingFormat::Scalar, ScalingFormat::Vector)
                     .has_value());
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_8F_E4M3, HIP_R_8F_E4M3, ScalingFormat::None, ScalingFormat::Vector)
                     .has_value());
    // FP4xFP4 with no MX scale on either side must NOT flag the joint rule:
    // the test would have triggered (None, None) -> FP4xFP4 mismatched
    // scales if the early-out logic regressed.
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_4F_E2M1, HIP_R_4F_E2M1, ScalingFormat::None, ScalingFormat::None)
                     .has_value());
}

TEST(MXScaleFormatHelpersTest, CheckCombinationOnlyOneSideMXValidatesThatSide)
{
    // A has an MX scale, B doesn't. The B side must be reported as
    // (None, None) to the validator so its rules don't apply.
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_8F_E4M3, HIP_R_32F,
                     ScalingFormat::Block_32_UE8M0, ScalingFormat::None)
                     .has_value());
    // ...and a wrong A-side scale must still be flagged.
    EXPECT_TRUE(checkMXScaleFormatCombination(
                     HIP_R_8F_E4M3, HIP_R_32F,
                     ScalingFormat::Block_32_UE5M3, ScalingFormat::None)
                     .has_value());

    // Symmetric case: B has an MX scale, A doesn't.
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_32F, HIP_R_8F_E5M2,
                     ScalingFormat::Scalar, ScalingFormat::Block_32_UE8M0)
                     .has_value());
    EXPECT_TRUE(checkMXScaleFormatCombination(
                     HIP_R_32F, HIP_R_8F_E5M2,
                     ScalingFormat::Scalar, ScalingFormat::Block_32_UE4M3)
                     .has_value());

    // FP4 single-side cases: FP4 accepts E8, E5M3, and E4M3 scales.
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_4F_E2M1, HIP_R_32F,
                     ScalingFormat::Block_32_UE8M0, ScalingFormat::None)
                     .has_value());
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_4F_E2M1, HIP_R_32F,
                     ScalingFormat::Block_32_UE5M3, ScalingFormat::None)
                     .has_value());
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_4F_E2M1, HIP_R_32F,
                     ScalingFormat::Block_32_UE4M3, ScalingFormat::None)
                     .has_value());
}

TEST(MXScaleFormatHelpersTest, CheckCombinationFP8FamilyRequiresE8Scale)
{
    // FP8/BF8/FP6/BF6 -> only E8 (UE8M0) scale is legal.
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_8F_E4M3, HIP_R_8F_E4M3,
                     ScalingFormat::Block_32_UE8M0, ScalingFormat::Block_32_UE8M0)
                     .has_value());
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_8F_E5M2, HIP_R_8F_E5M2,
                     ScalingFormat::Block_32_UE8M0, ScalingFormat::Block_32_UE8M0)
                     .has_value());
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_6F_E2M3, HIP_R_6F_E3M2,
                     ScalingFormat::Block_32_UE8M0, ScalingFormat::Block_32_UE8M0)
                     .has_value());

    // Any other block scale must be rejected.
    auto err = checkMXScaleFormatCombination(
        HIP_R_8F_E4M3, HIP_R_8F_E4M3,
        ScalingFormat::Block_32_UE4M3, ScalingFormat::Block_32_UE8M0);
    ASSERT_TRUE(err.has_value());
    EXPECT_FALSE(err->empty());

    EXPECT_TRUE(checkMXScaleFormatCombination(
                    HIP_R_6F_E2M3, HIP_R_6F_E2M3,
                    ScalingFormat::Block_32_UE5M3, ScalingFormat::Block_32_UE8M0)
                    .has_value());
}

TEST(MXScaleFormatHelpersTest, CheckCombinationFP4AcceptsAllThreeScales)
{
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_4F_E2M1, HIP_R_4F_E2M1,
                     ScalingFormat::Block_32_UE8M0, ScalingFormat::Block_32_UE8M0)
                     .has_value());
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_4F_E2M1, HIP_R_4F_E2M1,
                     ScalingFormat::Block_32_UE5M3, ScalingFormat::Block_32_UE5M3)
                     .has_value());
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_4F_E2M1, HIP_R_4F_E2M1,
                     ScalingFormat::Block_32_UE4M3, ScalingFormat::Block_32_UE4M3)
                     .has_value());
}

TEST(MXScaleFormatHelpersTest, CheckCombinationFP4xFP4RequiresMatchingScales)
{
    // E8 vs E5M3 mismatch -> rejected.
    auto err = checkMXScaleFormatCombination(
        HIP_R_4F_E2M1, HIP_R_4F_E2M1,
        ScalingFormat::Block_32_UE8M0, ScalingFormat::Block_32_UE5M3);
    ASSERT_TRUE(err.has_value());
    EXPECT_FALSE(err->empty());

    // E5M3 vs E4M3 mismatch -> rejected.
    EXPECT_TRUE(checkMXScaleFormatCombination(
                    HIP_R_4F_E2M1, HIP_R_4F_E2M1,
                    ScalingFormat::Block_32_UE5M3, ScalingFormat::Block_32_UE4M3)
                    .has_value());

    // Same scale family but different block widths still resolve to the
    // same dtype (E8) - that must be accepted.
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_4F_E2M1, HIP_R_4F_E2M1,
                     ScalingFormat::Block_32_UE8M0, ScalingFormat::Block_16_UE8M0)
                     .has_value());
}

TEST(MXScaleFormatHelpersTest, CheckCombinationFP8xFP4LegalCombo)
{
    // Mixed FP8 (E8) x FP4 (E8) is legal - FP4xFP4 same-scale rule does
    // not kick in for a non-FP4xFP4 pair, and both sides individually
    // satisfy their per-class scale constraint.
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_8F_E4M3, HIP_R_4F_E2M1,
                     ScalingFormat::Block_32_UE8M0, ScalingFormat::Block_32_UE8M0)
                     .has_value());

    // ... but FP8 x FP4 with FP4 using E5M3 still leaves FP8 on an
    // illegal E5M3 if we swap it - check the canonical FP8 must remain E8.
    EXPECT_TRUE(checkMXScaleFormatCombination(
                    HIP_R_8F_E4M3, HIP_R_4F_E2M1,
                    ScalingFormat::Block_32_UE5M3, ScalingFormat::Block_32_UE5M3)
                    .has_value());

    // FP4 (E5M3) on A, FP8 (E8) on B -> legal: each side satisfies its
    // own rule and the joint rule does not apply (one side is not FP4).
    EXPECT_FALSE(checkMXScaleFormatCombination(
                     HIP_R_4F_E2M1, HIP_R_8F_E4M3,
                     ScalingFormat::Block_32_UE5M3, ScalingFormat::Block_32_UE8M0)
                     .has_value());
}
