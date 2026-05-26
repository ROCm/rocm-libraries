// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Helpers.hpp"

#include <cmath>
#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::helpers;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

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

TEST(TestCpuReferenceValidation, NegativeToleranceThrows)
{
    EXPECT_THROW(const CpuReferenceValidation<float> refValidation(-1e-5f),
                 std::invalid_argument);
}

TEST(TestCpuReferenceValidation, NaNToleranceThrows)
{
    EXPECT_THROW(const CpuReferenceValidation<float> refValidation(
                     std::numeric_limits<float>::quiet_NaN()),
                 std::invalid_argument);
}

TEST(TestCpuReferenceValidation, InfToleranceThrows)
{
    EXPECT_THROW(const CpuReferenceValidation<float> refValidation(
                     0.0f, std::numeric_limits<float>::infinity()),
                 std::invalid_argument);
}

TEST(TestCpuReferenceValidationFp32, BasicTensorUsage)
{
    const CpuReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<float> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    Tensor<float> tensor2(dims);
    makeTensorsEqual<Tensor<float>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidationFp32, TensorsToleranceDifferent)
{
    const CpuReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<float> tensor1(dims);
    Tensor<float> tensor2(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    makeTensorsEqual<Tensor<float>>(tensor1, tensor2);
    const std::vector<int64_t> indices
        = {2, 5}; //index 25 because strides are [10, 1] so 10*2 + 1*5 = 25
    tensor2.setHostValue(1000, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

// Additional BasicTensorUsage tests for other data types
TEST(TestCpuReferenceValidationBfp16, BasicTensorUsage)
{
    const CpuReferenceValidation<bfloat16> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<bfloat16> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    Tensor<bfloat16> tensor2(dims);
    makeTensorsEqual<Tensor<bfloat16>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidationFp16, BasicTensorUsage)
{
    const CpuReferenceValidation<half> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<half> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    Tensor<half> tensor2(dims);
    makeTensorsEqual<Tensor<half>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidationFp64, BasicTensorUsage)
{
    const CpuReferenceValidation<double> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<double> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0, 1.0);
    Tensor<double> tensor2(dims);
    makeTensorsEqual<Tensor<double>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

// TensorNotComparable tests
TEST(TestCpuReferenceValidationBfp16, TensorNotComparable)
{
    const CpuReferenceValidation<bfloat16> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<bfloat16> tensor1(dims);
    Tensor<bfloat16> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(2.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidationFp16, TensorNotComparable)
{
    const CpuReferenceValidation<half> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<half> tensor1(dims);
    Tensor<half> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(2.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidationFp32, TensorNotComparable)
{
    const CpuReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<float> tensor1(dims);
    Tensor<float> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(2.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidationFp64, TensorNotComparable)
{
    const CpuReferenceValidation<double> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<double> tensor1(dims);
    Tensor<double> tensor2(dims);
    tensor1.fillTensorWithValue(1.0);
    tensor2.fillTensorWithValue(2.0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

// Tolerance tests
TEST(TestCpuReferenceValidation, TensorToleranceComparison)
{
    const CpuReferenceValidation<double> refValidationLowTolerance(1e-7f, 1e-7f);
    const CpuReferenceValidation<double> refValidationHighTolerance(1e-5f, 1e-5f);
    const std::vector<int64_t> dims = {10, 10};

    Tensor<double> tensor1(dims);
    Tensor<double> tensor2(dims);
    tensor1.fillTensorWithValue(1.0);
    tensor2.fillTensorWithValue(1.000001f);

    EXPECT_TRUE(refValidationHighTolerance.allClose(tensor1, tensor2));
    EXPECT_FALSE(refValidationLowTolerance.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidation, TensorDefaultTolerance)
{
    const CpuReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {1};

    Tensor<float> tensor1(dims);
    Tensor<float> tensor2(dims);
    tensor1.setHostValue(1.0f, 0);
    tensor2.setHostValue(1.0f + std::numeric_limits<float>::epsilon(), 0);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

// Edge case: different element counts
TEST(TestCpuReferenceValidation, TensorDifferentElementCounts)
{
    const CpuReferenceValidation<float> refValidation;

    Tensor<float> tensor1({10, 10});
    Tensor<float> tensor2({5, 5});
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidationStrided, StridedTensorEqual)
{
    const CpuReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

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

TEST(TestCpuReferenceValidation, StridedTensorNotEqual)
{
    const CpuReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

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
    const std::vector<int64_t> indices = {1, 1, 1, 1};
    tensor2.setHostValue(9999.0f, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidation, StridedTensorAllZeros)
{
    const CpuReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    tensor1.fillTensorWithValue(0.0f);
    tensor2.fillTensorWithValue(0.0f);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidation, StridedTensorDifferentStrides)
{
    const CpuReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides1 = {2, 4, 8, 16};
    const std::vector<int64_t> strides2 = {8, 4, 2, 1}; // Different stride order

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

TEST(TestCpuReferenceValidation, StridedTensorWithTolerance)
{
    const float customTolerance = 1e-5f;
    const CpuReferenceValidation<float> refValidation(customTolerance, customTolerance);
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        const float value = 1.0f;
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value + 5e-6f, indices); // Within tolerance
    });

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidation, StridedTensorFirstElementDiffers)
{
    const CpuReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    // Change first element
    const std::vector<int64_t> indices = {0, 0, 0, 0};
    tensor2.setHostValue(2.0f, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidation, StridedTensorLastElementDiffers)
{
    const CpuReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    // Change last element
    const std::vector<int64_t> indices = {1, 1, 1, 1};
    tensor2.setHostValue(2.0f, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceValidation, TensorSameElementCountDifferentDims)
{
    const CpuReferenceValidation<float> refValidation;

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
    const CpuIntReferenceValidation<int32_t> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<int32_t> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-25, 25);
    Tensor<int32_t> tensor2(dims);
    makeTensorsEqual<Tensor<int32_t>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidationInt8, BasicTensorUsage)
{
    const CpuIntReferenceValidation<int8_t> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<int8_t> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-128, 127);
    Tensor<int8_t> tensor2(dims);
    makeTensorsEqual<Tensor<int8_t>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidationUint8, BasicTensorUsage)
{
    const CpuIntReferenceValidation<uint8_t> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<uint8_t> tensor1(dims);
    tensor1.fillTensorWithRandomValues(0, 256);
    Tensor<uint8_t> tensor2(dims);
    makeTensorsEqual<Tensor<uint8_t>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

// TensorNotComparable tests
TEST(TestCpuIntReferenceValidationInt32, TensorNotComparable)
{
    const CpuIntReferenceValidation<int32_t> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<int32_t> tensor1(dims);
    Tensor<int32_t> tensor2(dims);
    tensor1.fillTensorWithValue(-10);
    tensor2.fillTensorWithValue(10);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidationInt8, TensorNotComparable)
{
    const CpuIntReferenceValidation<int8_t> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<int8_t> tensor1(dims);
    Tensor<int8_t> tensor2(dims);
    tensor1.fillTensorWithValue(-10);
    tensor2.fillTensorWithValue(10);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidationUint8, TensorNotComparable)
{
    const CpuIntReferenceValidation<uint8_t> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<uint8_t> tensor1(dims);
    Tensor<uint8_t> tensor2(dims);
    tensor1.fillTensorWithValue(-10);
    tensor2.fillTensorWithValue(10);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

// Edge case: different element counts
TEST(TestCpuIntReferenceValidation, TensorDifferentElementCounts)
{
    const CpuIntReferenceValidation<int32_t> refValidation;

    Tensor<int32_t> tensor1({10, 10});
    Tensor<int32_t> tensor2({5, 5});
    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidationStrided, StridedTensorEqual)
{
    const CpuIntReferenceValidation<int32_t> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<int32_t> tensor1(dims, strides);
    Tensor<int32_t> tensor2(dims, strides);

    // Fill with same values
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        auto value = safeTestTypeCast<int32_t>((indices[0] * 1000) + (indices[1] * 100)
                                               + (indices[2] * 10) + indices[3]);
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value, indices);
    });

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, StridedTensorNotEqual)
{
    const CpuIntReferenceValidation<int32_t> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<int32_t> tensor1(dims, strides);
    Tensor<int32_t> tensor2(dims, strides);

    // Fill tensor1
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        auto value = safeTestTypeCast<int32_t>((indices[0] * 1000) + (indices[1] * 100)
                                               + (indices[2] * 10) + indices[3]);
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value, indices);
    });

    // Change one element in tensor2
    const std::vector<int64_t> indices = {1, 1, 1, 1};
    tensor2.setHostValue(9999.0f, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, StridedTensorAllZeros)
{
    const CpuIntReferenceValidation<int32_t> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<int32_t> tensor1(dims, strides);
    Tensor<int32_t> tensor2(dims, strides);

    tensor1.fillTensorWithValue(0);
    tensor2.fillTensorWithValue(0);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, StridedTensorDifferentStrides)
{
    const CpuIntReferenceValidation<int32_t> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides1 = {2, 4, 8, 16};
    const std::vector<int64_t> strides2 = {8, 4, 2, 1}; // Different stride order

    Tensor<int32_t> tensor1(dims, strides1);
    Tensor<int32_t> tensor2(dims, strides2);

    // Set same logical values despite different memory layouts
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        auto value = safeTestTypeCast<int32_t>(indices[0] + indices[1] + indices[2] + indices[3]);
        tensor1.setHostValue(value, indices);
        tensor2.setHostValue(value, indices);
    });

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, StridedTensorFirstElementDiffers)
{
    const CpuIntReferenceValidation<int32_t> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<int32_t> tensor1(dims, strides);
    Tensor<int32_t> tensor2(dims, strides);

    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    // Change first element
    const std::vector<int64_t> indices = {0, 0, 0, 0};
    tensor2.setHostValue(2, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, StridedTensorLastElementDiffers)
{
    const CpuIntReferenceValidation<int32_t> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<int32_t> tensor1(dims, strides);
    Tensor<int32_t> tensor2(dims, strides);

    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    // Change last element
    const std::vector<int64_t> indices = {1, 1, 1, 1};
    tensor2.setHostValue(2, indices);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuIntReferenceValidation, TensorSameElementCountDifferentDims)
{
    const CpuIntReferenceValidation<int32_t> refValidation;

    Tensor<int32_t> tensor1({2, 50}); // 100 elements
    Tensor<int32_t> tensor2({10, 10}); // 100 elements
    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    // Should return false because dimensions don't match
    // even though element counts are the same
    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

/* ================================================= */

/* ======== NaN/Inf detection tests (TYPED_TEST across fp types) ======== */

template <typename T>
class CpuReferenceValidationNanInf : public ::testing::Test
{
};

using ValidationTypes = ::testing::Types<float, double, half, bfloat16>;
TYPED_TEST_SUITE(CpuReferenceValidationNanInf, ValidationTypes, );

TYPED_TEST(CpuReferenceValidationNanInf, FailsWhenReferenceHasNaN)
{
    const CpuReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuReferenceValidationNanInf, FailsWhenImplementationHasNaN)
{
    const CpuReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor2.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuReferenceValidationNanInf, FailsWhenBothHaveNaN)
{
    const CpuReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillWithSentinelValue();
    tensor2.fillWithSentinelValue();

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuReferenceValidationNanInf, FailsWhenReferenceHasInf)
{
    const CpuReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuReferenceValidationNanInf, FailsWhenImplementationHasNegativeInf)
{
    const CpuReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor2.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuReferenceValidationNanInf, FailsWhenBothHaveInf)
{
    const CpuReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);
    tensor2.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuReferenceValidationNanInf, PassesForFiniteValues)
{
    const CpuReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

/* ================================================= */

/* ======== Integer sentinel detection tests (TYPED_TEST across int types) ======== */

template <typename T>
class CpuIntReferenceValidationSentinel : public ::testing::Test
{
};

using IntValidationTypes = ::testing::Types<int32_t, int8_t, uint8_t>;
TYPED_TEST_SUITE(CpuIntReferenceValidationSentinel, IntValidationTypes, );

TYPED_TEST(CpuIntReferenceValidationSentinel, FailsWhenReferenceHasSentinel)
{
    const CpuIntReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::max(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuIntReferenceValidationSentinel, FailsWhenImplementationHasSentinel)
{
    const CpuIntReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    tensor2.setHostValue(std::numeric_limits<TypeParam>::max(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuIntReferenceValidationSentinel, FailsWhenBothHaveSentinel)
{
    const CpuIntReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillWithSentinelValue();
    tensor2.fillWithSentinelValue();

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuIntReferenceValidationSentinel, PassesForNonSentinelValues)
{
    const CpuIntReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1);
    tensor2.fillTensorWithValue(1);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

/* ================================================= */
