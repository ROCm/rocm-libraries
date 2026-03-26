// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Helpers.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <limits>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::helpers;
using namespace hipdnn_data_sdk::types;

TEST(TestCpuFpReferenceMiopenRmsValidation, NegativeToleranceThrows)
{
    EXPECT_THROW(const CpuFpReferenceMiopenRmsValidation<float> refValidation(-1e-5f),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceMiopenRmsValidation, NaNToleranceThrows)
{
    EXPECT_THROW(const CpuFpReferenceMiopenRmsValidation<float> refValidation(
                     std::numeric_limits<float>::quiet_NaN()),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceMiopenRmsValidation, InfToleranceThrows)
{
    EXPECT_THROW(const CpuFpReferenceMiopenRmsValidation<float> refValidation(
                     std::numeric_limits<float>::infinity()),
                 std::invalid_argument);
}

// Test MIOpen-specific RMS calculation behavior
TEST(TestCpuFpReferenceMiopenRmsValidation, MiopenRmsCalculation)
{
    // Test that RMS error is calculated correctly
    const CpuFpReferenceMiopenRmsValidation<double> refValidation(0.1);

    Tensor<double> tensor1({4});
    Tensor<double> tensor2({4});

    // Set up test data
    tensor1.setHostValue(1.0, 0);
    tensor1.setHostValue(2.0, 1);
    tensor1.setHostValue(3.0, 2);
    tensor1.setHostValue(4.0, 3);

    tensor2.setHostValue(1.1, 0);
    tensor2.setHostValue(2.1, 1);
    tensor2.setHostValue(3.1, 2);
    tensor2.setHostValue(4.1, 3);

    // Expected RMS calculation:
    // Square differences: 0.01, 0.01, 0.01, 0.01
    // Sum of square differences: 0.04
    // sqrt(0.04) = 0.2
    // Max magnitude in either buffer: 4.1
    // Element count: 4
    // Relative RMS error = 0.2 / (sqrt(4) * 4.1) = 0.2 / (2 * 4.1) = 0.0244

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2)); // 0.0244 < 0.1

    // Now with tighter tolerance it should fail
    const CpuFpReferenceMiopenRmsValidation<double> refValidationTight(0.02);
    EXPECT_FALSE(refValidationTight.allClose(tensor1, tensor2)); // 0.0244 > 0.02
}

// ============================================================================
// ITensor allClose Tests - Basic Usage
// ============================================================================

TEST(TestCpuFpReferenceMiopenRmsValidationITensorBfp16, BasicUsage)
{
    const CpuFpReferenceMiopenRmsValidation<bfloat16> validator;

    Tensor<bfloat16> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0_bf);

    Tensor<bfloat16> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0_bf);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp16, BasicUsage)
{
    const CpuFpReferenceMiopenRmsValidation<half> validator;

    Tensor<half> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0_h);

    Tensor<half> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0_h);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp32, BasicUsage)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp64, BasicUsage)
{
    const CpuFpReferenceMiopenRmsValidation<double> validator;

    Tensor<double> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0);

    Tensor<double> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

// ============================================================================
// ITensor allClose Tests - Non-Matching Values
// ============================================================================

TEST(TestCpuFpReferenceMiopenRmsValidationITensorBfp16, NotComparable)
{
    const CpuFpReferenceMiopenRmsValidation<bfloat16> validator;

    Tensor<bfloat16> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0_bf);

    Tensor<bfloat16> tensor2({2, 3, 4});
    tensor2.fillWithValue(2.0_bf);

    EXPECT_FALSE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp16, NotComparable)
{
    const CpuFpReferenceMiopenRmsValidation<half> validator;

    Tensor<half> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0_h);

    Tensor<half> tensor2({2, 3, 4});
    tensor2.fillWithValue(2.0_h);

    EXPECT_FALSE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp32, NotComparable)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({2, 3, 4});
    tensor2.fillWithValue(2.0f);

    EXPECT_FALSE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp64, NotComparable)
{
    const CpuFpReferenceMiopenRmsValidation<double> validator;

    Tensor<double> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0);

    Tensor<double> tensor2({2, 3, 4});
    tensor2.fillWithValue(2.0);

    EXPECT_FALSE(validator.allClose(tensor1, tensor2));
}

// ============================================================================
// ITensor allClose Tests - Different Layouts
// ============================================================================

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, NchwLayout)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4, 5}, TensorLayout::NCHW);
    tensor1.fillWithValue(1.5f);

    Tensor<float> tensor2({2, 3, 4, 5}, TensorLayout::NCHW);
    tensor2.fillWithValue(1.5f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, NhwcLayout)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4, 5}, TensorLayout::NHWC);
    tensor1.fillWithValue(1.5f);

    Tensor<float> tensor2({2, 3, 4, 5}, TensorLayout::NHWC);
    tensor2.fillWithValue(1.5f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, DifferentLayoutsSameLogicalValues)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    // Create two tensors with different layouts but same logical values
    Tensor<float> tensorNchw({2, 3, 4, 5}, TensorLayout::NCHW);
    Tensor<float> tensorNhwc({2, 3, 4, 5}, TensorLayout::NHWC);

    // Fill both with same pattern of values
    for(int n = 0; n < 2; ++n)
    {
        for(int c = 0; c < 3; ++c)
        {
            for(int h = 0; h < 4; ++h)
            {
                for(int w = 0; w < 5; ++w)
                {
                    auto value = static_cast<float>((n * 60) + (c * 20) + (h * 5) + w);
                    tensorNchw.setHostValue(value, n, c, h, w);
                    tensorNhwc.setHostValue(value, n, c, h, w);
                }
            }
        }
    }

    EXPECT_TRUE(validator.allClose(tensorNchw, tensorNhwc));
}

// ============================================================================
// ITensor allClose Tests - Different Dimensions
// ============================================================================

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, OneDimensional)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({100});
    tensor1.fillWithValue(2.5f);

    Tensor<float> tensor2({100});
    tensor2.fillWithValue(2.5f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, ThreeDimensional)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({3, 4, 5});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({3, 4, 5});
    tensor2.fillWithValue(1.0f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, FourDimensional)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4, 5});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({2, 3, 4, 5});
    tensor2.fillWithValue(1.0f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, FiveDimensional)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 2, 3, 4, 5}, TensorLayout::NCDHW);
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({2, 2, 3, 4, 5}, TensorLayout::NCDHW);
    tensor2.fillWithValue(1.0f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

// ============================================================================
// ITensor allClose Tests - Edge Cases
// ============================================================================

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, DifferentSizeTensors)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({2, 3, 5});
    tensor2.fillWithValue(1.0f);

    EXPECT_FALSE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, SparseTensor)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    // Create sparse tensors with custom strides
    const std::vector<int64_t> dims = {2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    // Fill with same pattern
    for(int i = 0; i < 2; ++i)
    {
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 0; k < 2; ++k)
            {
                auto value = static_cast<float>((i * 4) + (j * 2) + k);
                tensor1.setHostValue(value, i, j, k);
                tensor2.setHostValue(value, i, j, k);
            }
        }
    }

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, SingleElement)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({1});
    tensor1.fillWithValue(42.0f);

    Tensor<float> tensor2({1});
    tensor2.fillWithValue(42.0f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

// ============================================================================
// ITensor allClose Tests - Tolerance Tests
// ============================================================================

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, ToleranceComparison)
{
    const CpuFpReferenceMiopenRmsValidation<double> validatorLowTolerance(1e-7);
    const CpuFpReferenceMiopenRmsValidation<double> validatorHighTolerance(1e-3);

    Tensor<double> tensor1({10, 10});
    Tensor<double> tensor2({10, 10});

    tensor1.fillWithValue(1.0);
    tensor2.fillWithValue(1.0001);

    // High tolerance should pass
    EXPECT_TRUE(validatorHighTolerance.allClose(tensor1, tensor2));

    // Low tolerance should fail
    EXPECT_FALSE(validatorLowTolerance.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, DefaultTolerance)
{
    const CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({1});
    Tensor<float> tensor2({1});

    tensor1.setHostValue(1.0f, 0);
    tensor2.setHostValue(1.0f + std::numeric_limits<float>::epsilon(), 0);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, TensorSameElementCountDifferentDims)
{
    const CpuFpReferenceMiopenRmsValidation<float> refValidation;

    Tensor<float> tensor1({2, 50}); // 100 elements
    Tensor<float> tensor2({10, 10}); // 100 elements
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    // Should return false because dimensions don't match
    // even though element counts are the same
    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

/* ======== NaN/Inf detection tests (TYPED_TEST across fp types) ======== */

template <typename T>
class CpuFpReferenceMiopenRmsValidationNanInf : public ::testing::Test
{
};

using RmsFpValidationTypes = ::testing::Types<float, double, half, bfloat16>;
TYPED_TEST_SUITE(CpuFpReferenceMiopenRmsValidationNanInf, RmsFpValidationTypes, );

TYPED_TEST(CpuFpReferenceMiopenRmsValidationNanInf, FailsWhenReferenceHasNaN)
{
    const CpuFpReferenceMiopenRmsValidation<TypeParam> refValidation(TypeParam(1.0f));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceMiopenRmsValidationNanInf, FailsWhenImplementationHasNaN)
{
    const CpuFpReferenceMiopenRmsValidation<TypeParam> refValidation(TypeParam(1.0f));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor2.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceMiopenRmsValidationNanInf, FailsWhenBothHaveNaN)
{
    const CpuFpReferenceMiopenRmsValidation<TypeParam> refValidation(TypeParam(1.0f));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillWithSentinelValue();
    tensor2.fillWithSentinelValue();

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceMiopenRmsValidationNanInf, FailsWhenReferenceHasInf)
{
    const CpuFpReferenceMiopenRmsValidation<TypeParam> refValidation(TypeParam(1.0f));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceMiopenRmsValidationNanInf, FailsWhenImplementationHasNegativeInf)
{
    const CpuFpReferenceMiopenRmsValidation<TypeParam> refValidation(TypeParam(1.0f));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor2.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceMiopenRmsValidationNanInf, FailsWhenBothHaveInf)
{
    const CpuFpReferenceMiopenRmsValidation<TypeParam> refValidation(TypeParam(1.0f));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);
    tensor2.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceMiopenRmsValidationNanInf, PassesForFiniteValues)
{
    const CpuFpReferenceMiopenRmsValidation<TypeParam> refValidation(TypeParam(1.0f));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}
