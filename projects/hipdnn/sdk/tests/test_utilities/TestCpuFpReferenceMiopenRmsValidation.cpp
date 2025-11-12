// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Helpers.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>
#include <limits>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;
using namespace helpers;

TEST(TestCpuFpReferenceMiopenRmsValidation, NegativeToleranceThrows)
{
    EXPECT_THROW(CpuFpReferenceMiopenRmsValidation<float> refValidation(-1e-5f),
                 std::invalid_argument);
}

// Test MIOpen-specific RMS calculation behavior
TEST(TestCpuFpReferenceMiopenRmsValidation, MiopenRmsCalculation)
{
    // Test that RMS error is calculated correctly
    CpuFpReferenceMiopenRmsValidation<double> refValidation(0.1);

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
    CpuFpReferenceMiopenRmsValidation<double> refValidationTight(0.02);
    EXPECT_FALSE(refValidationTight.allClose(tensor1, tensor2)); // 0.0244 > 0.02
}

// ============================================================================
// ITensor allClose Tests - Basic Usage
// ============================================================================

TEST(TestCpuFpReferenceMiopenRmsValidationITensorBfp16, BasicUsage)
{
    CpuFpReferenceMiopenRmsValidation<hip_bfloat16> validator;

    Tensor<hip_bfloat16> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0_bf);

    Tensor<hip_bfloat16> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0_bf);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp16, BasicUsage)
{
    CpuFpReferenceMiopenRmsValidation<half> validator;

    Tensor<half> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0_h);

    Tensor<half> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0_h);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp32, BasicUsage)
{
    CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp64, BasicUsage)
{
    CpuFpReferenceMiopenRmsValidation<double> validator;

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
    CpuFpReferenceMiopenRmsValidation<hip_bfloat16> validator;

    Tensor<hip_bfloat16> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0_bf);

    Tensor<hip_bfloat16> tensor2({2, 3, 4});
    tensor2.fillWithValue(2.0_bf);

    EXPECT_FALSE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp16, NotComparable)
{
    CpuFpReferenceMiopenRmsValidation<half> validator;

    Tensor<half> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0_h);

    Tensor<half> tensor2({2, 3, 4});
    tensor2.fillWithValue(2.0_h);

    EXPECT_FALSE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp32, NotComparable)
{
    CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({2, 3, 4});
    tensor2.fillWithValue(2.0f);

    EXPECT_FALSE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensorFp64, NotComparable)
{
    CpuFpReferenceMiopenRmsValidation<double> validator;

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
    CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4, 5}, TensorLayout::NCHW);
    tensor1.fillWithValue(1.5f);

    Tensor<float> tensor2({2, 3, 4, 5}, TensorLayout::NCHW);
    tensor2.fillWithValue(1.5f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, NhwcLayout)
{
    CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4, 5}, TensorLayout::NHWC);
    tensor1.fillWithValue(1.5f);

    Tensor<float> tensor2({2, 3, 4, 5}, TensorLayout::NHWC);
    tensor2.fillWithValue(1.5f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, DifferentLayoutsSameLogicalValues)
{
    CpuFpReferenceMiopenRmsValidation<float> validator;

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
    CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({100});
    tensor1.fillWithValue(2.5f);

    Tensor<float> tensor2({100});
    tensor2.fillWithValue(2.5f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, ThreeDimensional)
{
    CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({3, 4, 5});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({3, 4, 5});
    tensor2.fillWithValue(1.0f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, FourDimensional)
{
    CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4, 5});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({2, 3, 4, 5});
    tensor2.fillWithValue(1.0f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, FiveDimensional)
{
    CpuFpReferenceMiopenRmsValidation<float> validator;

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
    CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({2, 3, 5});
    tensor2.fillWithValue(1.0f);

    EXPECT_FALSE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, SparseTensor)
{
    CpuFpReferenceMiopenRmsValidation<float> validator;

    // Create sparse tensors with custom strides
    std::vector<int64_t> dims = {2, 2, 2};
    std::vector<int64_t> strides = {2, 4, 8};

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
    CpuFpReferenceMiopenRmsValidation<float> validator;

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
    CpuFpReferenceMiopenRmsValidation<double> validatorLowTolerance(1e-7);
    CpuFpReferenceMiopenRmsValidation<double> validatorHighTolerance(1e-3);

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
    CpuFpReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({1});
    Tensor<float> tensor2({1});

    tensor1.setHostValue(1.0f, 0);
    tensor2.setHostValue(1.0f + std::numeric_limits<float>::epsilon(), 0);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceMiopenRmsValidationITensor, TensorSameElementCountDifferentDims)
{
    CpuFpReferenceMiopenRmsValidation<float> refValidation;

    Tensor<float> tensor1({2, 50}); // 100 elements
    Tensor<float> tensor2({10, 10}); // 100 elements
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    // Should return false because dimensions don't match
    // even though element counts are the same
    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}
