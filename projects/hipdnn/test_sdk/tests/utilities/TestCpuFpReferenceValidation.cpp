// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Helpers.hpp"

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::helpers;
using namespace hipdnn_data_sdk::types;

namespace
{
template <typename T>
void makeTensorsEqual(T& tensor1, T& tensor2)
{
    iterateAlongDimensions(tensor1.dims(), [&](const std::vector<int64_t>& indices) {
        tensor2.setHostValue(tensor1.getHostValue(indices), indices);
    });
}
} // namespace

/* ======== CpuFpReferenceValidation tests ======== */

TEST(TestCpuFpReferenceValidation, NegativeToleranceThrows)
{
    EXPECT_THROW(const CpuFpReferenceValidation<float> refValidation(-1e-5f),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceValidationFp32, BasicTensorUsage)
{
    CpuFpReferenceValidation<float> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<float> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    Tensor<float> tensor2(dims);
    makeTensorsEqual<Tensor<float>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp32, TensorsToleranceDifferent)
{
    CpuFpReferenceValidation<float> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<float> tensor1(dims);
    Tensor<float> tensor2(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    makeTensorsEqual<Tensor<float>>(tensor1, tensor2);
    std::vector<int64_t> const indices
        = {2, 5}; //index 25 because strides are [10, 1] so 10*2 + 1*5 = 25
    tensor2.setHostValue(1000, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

// Additional BasicTensorUsage tests for other data types
TEST(TestCpuFpReferenceValidationBfp16, BasicTensorUsage)
{
    CpuFpReferenceValidation<bfloat16> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<bfloat16> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    Tensor<bfloat16> tensor2(dims);
    makeTensorsEqual<Tensor<bfloat16>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp16, BasicTensorUsage)
{
    CpuFpReferenceValidation<half> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<half> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    Tensor<half> tensor2(dims);
    makeTensorsEqual<Tensor<half>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp64, BasicTensorUsage)
{
    CpuFpReferenceValidation<double> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<double> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0, 1.0);
    Tensor<double> tensor2(dims);
    makeTensorsEqual<Tensor<double>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

// TensorNotComparable tests
TEST(TestCpuFpReferenceValidationBfp16, TensorNotComparable)
{
    CpuFpReferenceValidation<bfloat16> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<bfloat16> tensor1(dims);
    Tensor<bfloat16> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(2.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp16, TensorNotComparable)
{
    CpuFpReferenceValidation<half> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<half> tensor1(dims);
    Tensor<half> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(2.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp32, TensorNotComparable)
{
    CpuFpReferenceValidation<float> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<float> tensor1(dims);
    Tensor<float> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(2.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp64, TensorNotComparable)
{
    CpuFpReferenceValidation<double> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<double> tensor1(dims);
    Tensor<double> tensor2(dims);
    tensor1.fillTensorWithValue(1.0);
    tensor2.fillTensorWithValue(2.0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

// Tolerance tests
TEST(TestCpuFpReferenceValidation, TensorToleranceComparison)
{
    CpuFpReferenceValidation<double> const refValidationLowTolerance(1e-7f, 1e-7f);
    CpuFpReferenceValidation<double> const refValidationHighTolerance(1e-5f, 1e-5f);
    std::vector<int64_t> const dims = {10, 10};

    Tensor<double> tensor1(dims);
    Tensor<double> tensor2(dims);
    tensor1.fillTensorWithValue(1.0);
    tensor2.fillTensorWithValue(1.000001f);

    EXPECT_TRUE(refValidationHighTolerance.allClose(tensor1, tensor2));
    EXPECT_FALSE(refValidationLowTolerance.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidation, TensorDefaultTolerance)
{
    CpuFpReferenceValidation<float> const refValidation;
    std::vector<int64_t> const dims = {1};

    Tensor<float> tensor1(dims);
    Tensor<float> tensor2(dims);
    tensor1.setHostValue(1.0f, 0);
    tensor2.setHostValue(1.0f + std::numeric_limits<float>::epsilon(), 0);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

// Edge case: different element counts
TEST(TestCpuFpReferenceValidation, TensorDifferentElementCounts)
{
    CpuFpReferenceValidation<float> const refValidation;

    Tensor<float> tensor1({10, 10});
    Tensor<float> tensor2({5, 5});
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationStrided, StridedTensorEqual)
{
    CpuFpReferenceValidation<float> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    // Fill with same values
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        auto value = static_cast<float>((indices[0] * 1000) + (indices[1] * 100) + (indices[2] * 10)
                                        + indices[3]);
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value, indices);
    });

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidation, StridedTensorNotEqual)
{
    CpuFpReferenceValidation<float> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    // Fill tensor1
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        auto value = static_cast<float>((indices[0] * 1000) + (indices[1] * 100) + (indices[2] * 10)
                                        + indices[3]);
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value, indices);
    });

    // Change one element in tensor2
    std::vector<int64_t> const indices = {1, 1, 1, 1};
    tensor2.setHostValue(9999.0f, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidation, StridedTensorAllZeros)
{
    CpuFpReferenceValidation<float> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    tensor1.fillTensorWithValue(0.0f);
    tensor2.fillTensorWithValue(0.0f);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidation, StridedTensorDifferentStrides)
{
    CpuFpReferenceValidation<float> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides1 = {2, 4, 8, 16};
    std::vector<int64_t> const strides2 = {8, 4, 2, 1}; // Different stride order

    Tensor<float> tensor1(dims, strides1);
    Tensor<float> tensor2(dims, strides2);

    // Set same logical values despite different memory layouts
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        auto value = static_cast<float>(indices[0] + indices[1] + indices[2] + indices[3]);
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value, indices);
    });

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidation, StridedTensorWithTolerance)
{
    float const customTolerance = 1e-5f;
    CpuFpReferenceValidation<float> const refValidation(customTolerance, customTolerance);
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        float const value = 1.0f;
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value + 5e-6f, indices); // Within tolerance
    });

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidation, StridedTensorFirstElementDiffers)
{
    CpuFpReferenceValidation<float> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    // Change first element
    std::vector<int64_t> const indices = {0, 0, 0, 0};
    tensor2.setHostValue(2.0f, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidation, StridedTensorLastElementDiffers)
{
    CpuFpReferenceValidation<float> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    // Change last element
    std::vector<int64_t> const indices = {1, 1, 1, 1};
    tensor2.setHostValue(2.0f, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidation, TensorSameElementCountDifferentDims)
{
    CpuFpReferenceValidation<float> const refValidation;

    Tensor<float> tensor1({2, 50}); // 100 elements
    Tensor<float> tensor2({10, 10}); // 100 elements
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    // Should return false because dimensions don't match
    // even though element counts are the same
    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

/* ================================================= */

/* ======== CpuIntReferenceValidation tests ======== */

TEST(TestCpuIntReferenceValidationInt32, BasicTensorUsage)
{
    CpuIntReferenceValidation<int32_t> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<int32_t> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-25, 25);
    Tensor<int32_t> tensor2(dims);
    makeTensorsEqual<Tensor<int32_t>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidationInt8, BasicTensorUsage)
{
    CpuIntReferenceValidation<int8_t> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<int8_t> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-128, 127);
    Tensor<int8_t> tensor2(dims);
    makeTensorsEqual<Tensor<int8_t>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidationUint8, BasicTensorUsage)
{
    CpuIntReferenceValidation<uint8_t> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<uint8_t> tensor1(dims);
    tensor1.fillTensorWithRandomValues(0, 256);
    Tensor<uint8_t> tensor2(dims);
    makeTensorsEqual<Tensor<uint8_t>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

// TensorNotComparable tests
TEST(TestCpuIntReferenceValidationInt32, TensorNotComparable)
{
    CpuIntReferenceValidation<int32_t> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<int32_t> tensor1(dims);
    Tensor<int32_t> tensor2(dims);
    tensor1.fillTensorWithValue(-10);
    tensor2.fillTensorWithValue(10);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidationInt8, TensorNotComparable)
{
    CpuIntReferenceValidation<int8_t> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<int8_t> tensor1(dims);
    Tensor<int8_t> tensor2(dims);
    tensor1.fillTensorWithValue(-10);
    tensor2.fillTensorWithValue(10);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidationUint8, TensorNotComparable)
{
    CpuIntReferenceValidation<uint8_t> const refValidation;
    std::vector<int64_t> const dims = {10, 10};

    Tensor<uint8_t> tensor1(dims);
    Tensor<uint8_t> tensor2(dims);
    tensor1.fillTensorWithValue(-10);
    tensor2.fillTensorWithValue(10);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

// Edge case: different element counts
TEST(TestCpuIntReferenceValidation, TensorDifferentElementCounts)
{
    CpuIntReferenceValidation<int32_t> const refValidation;

    Tensor<int32_t> tensor1({10, 10});
    Tensor<int32_t> tensor2({5, 5});
    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidationStrided, StridedTensorEqual)
{
    CpuIntReferenceValidation<int32_t> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<int32_t> tensor1(dims, strides);
    Tensor<int32_t> tensor2(dims, strides);

    // Fill with same values
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        auto value = static_cast<int32_t>((indices[0] * 1000) + (indices[1] * 100)
                                          + (indices[2] * 10) + indices[3]);
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value, indices);
    });

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, StridedTensorNotEqual)
{
    CpuIntReferenceValidation<int32_t> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<int32_t> tensor1(dims, strides);
    Tensor<int32_t> tensor2(dims, strides);

    // Fill tensor1
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        auto value = static_cast<int32_t>((indices[0] * 1000) + (indices[1] * 100)
                                          + (indices[2] * 10) + indices[3]);
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value, indices);
    });

    // Change one element in tensor2
    std::vector<int64_t> const indices = {1, 1, 1, 1};
    tensor2.setHostValue(9999.0f, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, StridedTensorAllZeros)
{
    CpuIntReferenceValidation<int32_t> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<int32_t> tensor1(dims, strides);
    Tensor<int32_t> tensor2(dims, strides);

    tensor1.fillTensorWithValue(0);
    tensor2.fillTensorWithValue(0);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, StridedTensorDifferentStrides)
{
    CpuIntReferenceValidation<int32_t> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides1 = {2, 4, 8, 16};
    std::vector<int64_t> const strides2 = {8, 4, 2, 1}; // Different stride order

    Tensor<int32_t> tensor1(dims, strides1);
    Tensor<int32_t> tensor2(dims, strides2);

    // Set same logical values despite different memory layouts
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        auto value = static_cast<int32_t>(indices[0] + indices[1] + indices[2] + indices[3]);
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value, indices);
    });

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, StridedTensorFirstElementDiffers)
{
    CpuIntReferenceValidation<int32_t> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<int32_t> tensor1(dims, strides);
    Tensor<int32_t> tensor2(dims, strides);

    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    // Change first element
    std::vector<int64_t> const indices = {0, 0, 0, 0};
    tensor2.setHostValue(2, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, StridedTensorLastElementDiffers)
{
    CpuIntReferenceValidation<int32_t> const refValidation;
    std::vector<int64_t> const dims = {2, 2, 2, 2};
    std::vector<int64_t> const strides = {2, 4, 8, 16};

    Tensor<int32_t> tensor1(dims, strides);
    Tensor<int32_t> tensor2(dims, strides);

    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    // Change last element
    std::vector<int64_t> const indices = {1, 1, 1, 1};
    tensor2.setHostValue(2, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, TensorSameElementCountDifferentDims)
{
    CpuIntReferenceValidation<int32_t> const refValidation;

    Tensor<int32_t> tensor1({2, 50}); // 100 elements
    Tensor<int32_t> tensor2({10, 10}); // 100 elements
    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    // Should return false because dimensions don't match
    // even though element counts are the same
    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

/* ================================================= */
