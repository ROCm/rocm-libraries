// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Helpers.hpp"

#include <cmath>
#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
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

template <typename T>
float validationEpsilon()
{
    return static_cast<float>(std::numeric_limits<T>::epsilon());
}
} // namespace

/* ======== CpuFpReferenceValidation tests ======== */

TEST(TestCpuFpReferenceValidation, NegativeToleranceThrows)
{
    EXPECT_THROW(const CpuFpReferenceValidation<float> refValidation(-1e-5f),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceValidation, NaNToleranceThrows)
{
    EXPECT_THROW(const CpuFpReferenceValidation<float> refValidation(
                     std::numeric_limits<float>::quiet_NaN()),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceValidation, InfToleranceThrows)
{
    EXPECT_THROW(const CpuFpReferenceValidation<float> refValidation(
                     0.0f, std::numeric_limits<float>::infinity()),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceValidationFp32, BasicTensorUsage)
{
    const CpuFpReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<float> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    Tensor<float> tensor2(dims);
    makeTensorsEqual<Tensor<float>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp32, TensorsToleranceDifferent)
{
    const CpuFpReferenceValidation<float> refValidation;
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
TEST(TestCpuFpReferenceValidationBfp16, BasicTensorUsage)
{
    const CpuFpReferenceValidation<bfloat16> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<bfloat16> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    Tensor<bfloat16> tensor2(dims);
    makeTensorsEqual<Tensor<bfloat16>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp16, BasicTensorUsage)
{
    const CpuFpReferenceValidation<half> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<half> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0f, 1.0f);
    Tensor<half> tensor2(dims);
    makeTensorsEqual<Tensor<half>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp64, BasicTensorUsage)
{
    const CpuFpReferenceValidation<double> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<double> tensor1(dims);
    tensor1.fillTensorWithRandomValues(-1.0, 1.0);
    Tensor<double> tensor2(dims);
    makeTensorsEqual<Tensor<double>>(tensor1, tensor2);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

// TensorNotComparable tests
TEST(TestCpuFpReferenceValidationBfp16, TensorNotComparable)
{
    const CpuFpReferenceValidation<bfloat16> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<bfloat16> tensor1(dims);
    Tensor<bfloat16> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(2.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp16, TensorNotComparable)
{
    const CpuFpReferenceValidation<half> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<half> tensor1(dims);
    Tensor<half> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(2.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp32, TensorNotComparable)
{
    const CpuFpReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<float> tensor1(dims);
    Tensor<float> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(2.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationFp64, TensorNotComparable)
{
    const CpuFpReferenceValidation<double> refValidation;
    const std::vector<int64_t> dims = {10, 10};

    Tensor<double> tensor1(dims);
    Tensor<double> tensor2(dims);
    tensor1.fillTensorWithValue(1.0);
    tensor2.fillTensorWithValue(2.0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

// Tolerance tests
TEST(TestCpuFpReferenceValidation, TensorToleranceComparison)
{
    const CpuFpReferenceValidation<double> refValidationLowTolerance(1e-7f, 1e-7f);
    const CpuFpReferenceValidation<double> refValidationHighTolerance(1e-5f, 1e-5f);
    const std::vector<int64_t> dims = {10, 10};

    Tensor<double> tensor1(dims);
    Tensor<double> tensor2(dims);
    tensor1.fillTensorWithValue(1.0);
    tensor2.fillTensorWithValue(1.000001f);

    EXPECT_TRUE(refValidationHighTolerance.allClose(tensor1, tensor2));
    EXPECT_FALSE(refValidationLowTolerance.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidation, TensorDefaultTolerance)
{
    const CpuFpReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {1};

    Tensor<float> tensor1(dims);
    Tensor<float> tensor2(dims);
    tensor1.setHostValue(1.0f, 0);
    tensor2.setHostValue(1.0f + std::numeric_limits<float>::epsilon(), 0);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

// Edge case: different element counts
TEST(TestCpuFpReferenceValidation, TensorDifferentElementCounts)
{
    const CpuFpReferenceValidation<float> refValidation;

    Tensor<float> tensor1({10, 10});
    Tensor<float> tensor2({5, 5});
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationStrided, StridedTensorEqual)
{
    const CpuFpReferenceValidation<float> refValidation;
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

TEST(TestCpuFpReferenceValidation, StridedTensorNotEqual)
{
    const CpuFpReferenceValidation<float> refValidation;
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

TEST(TestCpuFpReferenceValidation, StridedTensorAllZeros)
{
    const CpuFpReferenceValidation<float> refValidation;
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<float> tensor1(dims, strides);
    Tensor<float> tensor2(dims, strides);

    tensor1.fillTensorWithValue(0.0f);
    tensor2.fillTensorWithValue(0.0f);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidation, StridedTensorDifferentStrides)
{
    const CpuFpReferenceValidation<float> refValidation;
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

TEST(TestCpuFpReferenceValidation, StridedTensorWithTolerance)
{
    const float customTolerance = 1e-5f;
    const CpuFpReferenceValidation<float> refValidation(customTolerance, customTolerance);
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

TEST(TestCpuFpReferenceValidation, StridedTensorFirstElementDiffers)
{
    const CpuFpReferenceValidation<float> refValidation;
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

TEST(TestCpuFpReferenceValidation, StridedTensorLastElementDiffers)
{
    const CpuFpReferenceValidation<float> refValidation;
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

TEST(TestCpuFpReferenceValidation, TensorSameElementCountDifferentDims)
{
    const CpuFpReferenceValidation<float> refValidation;

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
TEST(TestCpuIntReferenceValidationBool, ComparesLegalTrueValues)
{
    const CpuIntReferenceValidation<bool> refValidation;
    Tensor<bool> reference({2});
    Tensor<bool> implementation({2});
    reference.setHostValue(true, 0);
    reference.setHostValue(false, 1);
    implementation.setHostValue(true, 0);
    implementation.setHostValue(false, 1);

    EXPECT_TRUE(refValidation.allClose(reference, implementation));

    implementation.setHostValue(true, 1);
    EXPECT_FALSE(refValidation.allClose(reference, implementation));
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
class CpuFpReferenceValidationNanInf : public ::testing::Test
{
};

using FpValidationTypes = ::testing::Types<float, double, half, bfloat16>;
TYPED_TEST_SUITE(CpuFpReferenceValidationNanInf, FpValidationTypes, );

TYPED_TEST(CpuFpReferenceValidationNanInf, FailsWhenReferenceHasNaN)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationNanInf, FailsWhenImplementationHasNaN)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor2.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationNanInf, FailsWhenBothHaveNaN)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillWithSentinelValue();
    tensor2.fillWithSentinelValue();

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationNanInf, FailsWhenReferenceHasInf)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationNanInf, FailsWhenImplementationHasNegativeInf)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor2.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationNanInf, FailsWhenBothHaveInf)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);
    tensor2.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationNanInf, PassesForFiniteValues)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}

/* ================================================= */

/* ======== Matching infinity opt-in tests (TYPED_TEST across fp types) ======== */

template <typename T>
class CpuFpReferenceValidationMatchingInf : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceValidationMatchingInf, FpValidationTypes, );

TYPED_TEST(CpuFpReferenceValidationMatchingInf, NegativeInfinitiesEqualOnlyWhenOptedIn)
{
    const float epsilon = validationEpsilon<TypeParam>();
    const CpuFpReferenceValidation<TypeParam> strictValidation(epsilon, epsilon);
    const CpuFpReferenceValidation<TypeParam> matchingInfValidation(
        epsilon, epsilon, MatchingInfinities::ACCEPTED);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);
    tensor2.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(strictValidation.allClose(tensor1, tensor2));
    EXPECT_TRUE(matchingInfValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationMatchingInf, PositiveInfinitiesEqualOnlyWhenOptedIn)
{
    const float epsilon = validationEpsilon<TypeParam>();
    const CpuFpReferenceValidation<TypeParam> strictValidation(epsilon, epsilon);
    const CpuFpReferenceValidation<TypeParam> matchingInfValidation(
        epsilon, epsilon, MatchingInfinities::ACCEPTED);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);
    tensor2.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(strictValidation.allClose(tensor1, tensor2));
    EXPECT_TRUE(matchingInfValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationMatchingInf, OppositeSignedInfinitiesAlwaysFail)
{
    const float epsilon = validationEpsilon<TypeParam>();
    const CpuFpReferenceValidation<TypeParam> strictValidation(epsilon, epsilon);
    const CpuFpReferenceValidation<TypeParam> matchingInfValidation(
        epsilon, epsilon, MatchingInfinities::ACCEPTED);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);
    tensor2.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(strictValidation.allClose(tensor1, tensor2));
    EXPECT_FALSE(matchingInfValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationMatchingInf, FiniteVersusInfinityAlwaysFails)
{
    const float epsilon = validationEpsilon<TypeParam>();
    const CpuFpReferenceValidation<TypeParam> strictValidation(epsilon, epsilon);
    const CpuFpReferenceValidation<TypeParam> matchingInfValidation(
        epsilon, epsilon, MatchingInfinities::ACCEPTED);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(strictValidation.allClose(tensor1, tensor2));
    EXPECT_FALSE(matchingInfValidation.allClose(tensor1, tensor2));

    tensor1.setHostValue(safeTestTypeCast<TypeParam>(1.0f), 0, 0);
    tensor2.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(strictValidation.allClose(tensor1, tensor2));
    EXPECT_FALSE(matchingInfValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationMatchingInf, NaNAlwaysFails)
{
    const float epsilon = validationEpsilon<TypeParam>();
    const CpuFpReferenceValidation<TypeParam> matchingInfValidation(
        epsilon, epsilon, MatchingInfinities::ACCEPTED);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    EXPECT_FALSE(matchingInfValidation.allClose(tensor1, tensor2));

    tensor2.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    EXPECT_FALSE(matchingInfValidation.allClose(tensor1, tensor2));

    tensor1.setHostValue(safeTestTypeCast<TypeParam>(1.0f), 0, 0);
    EXPECT_FALSE(matchingInfValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuFpReferenceValidationMatchingInf, FiniteMismatchStillFailsWhenOptedIn)
{
    const float epsilon = validationEpsilon<TypeParam>();
    const CpuFpReferenceValidation<TypeParam> matchingInfValidation(
        epsilon, epsilon, MatchingInfinities::ACCEPTED);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);
    tensor2.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);
    tensor2.setHostValue(safeTestTypeCast<TypeParam>(2.0f), 1, 1);

    EXPECT_FALSE(matchingInfValidation.allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationMatchingInf, FactorySelectsTheRelaxedComparison)
{
    const auto strictValidation
        = createAllCloseValidator(hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    const auto matchingInfValidation = createAllCloseMatchingInfinitiesValidator(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        std::numeric_limits<float>::epsilon(),
        std::numeric_limits<float>::epsilon());
    const std::vector<int64_t> dims = {2, 2};

    Tensor<float> tensor1(dims);
    Tensor<float> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(-std::numeric_limits<float>::infinity(), 0, 0);
    tensor2.setHostValue(-std::numeric_limits<float>::infinity(), 0, 0);

    EXPECT_FALSE(strictValidation->allClose(tensor1, tensor2));
    EXPECT_TRUE(matchingInfValidation->allClose(tensor1, tensor2));
}

TEST(TestCpuFpReferenceValidationMatchingInf, FactoryRejectsIntegerDataTypes)
{
    EXPECT_THROW(createAllCloseMatchingInfinitiesValidator(
                     hipdnn_flatbuffers_sdk::data_objects::DataType::INT32, 0.0f, 0.0f),
                 std::runtime_error);
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

/* ======== Ragged comparison tests (TYPED_TEST across fp types) ======== */

template <typename T>
class CpuFpReferenceValidationRagged : public ::testing::Test
{
};

using RaggedFpValidationTypes = ::testing::Types<float, half, bfloat16>;
TYPED_TEST_SUITE(CpuFpReferenceValidationRagged, RaggedFpValidationTypes, );

TYPED_TEST(CpuFpReferenceValidationRagged, PassesForIdenticalTensors)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    auto reference = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto implementation = createRaggedSdpaTensor<TypeParam>(0.5f);

    EXPECT_TRUE(refValidation.allClose(reference, implementation));
}

TYPED_TEST(CpuFpReferenceValidationRagged, SkipsInBlockNaNReference)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    auto reference = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto implementation = createRaggedSdpaTensor<TypeParam>(0.5f);
    reference.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 1, 1, 2, 1);
    implementation.setHostValue(TypeParam(2.0f), 1, 1, 2, 1);

    EXPECT_TRUE(refValidation.allClose(reference, implementation));
}

TYPED_TEST(CpuFpReferenceValidationRagged, FailsForInBlockDifference)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    auto reference = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto implementation = createRaggedSdpaTensor<TypeParam>(0.5f);
    implementation.setHostValue(TypeParam(2.0f), 1, 1, 2, 1);

    EXPECT_FALSE(refValidation.allClose(reference, implementation));
}

TYPED_TEST(CpuFpReferenceValidationRagged, FailsWhenEveryReferenceValueIsNaN)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    auto reference = createRaggedSdpaTensor<TypeParam>(std::numeric_limits<float>::quiet_NaN());
    auto implementation = createRaggedSdpaTensor<TypeParam>(0.5f);

    EXPECT_FALSE(refValidation.allClose(reference, implementation));
}

TYPED_TEST(CpuFpReferenceValidationRagged, FailsForMismatchedRaggedLayouts)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    auto reference = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto implementation = createRaggedSdpaTensor<TypeParam>(0.5f, {0, 3, 5});

    EXPECT_FALSE(refValidation.allClose(reference, implementation));
}

TYPED_TEST(CpuFpReferenceValidationRagged, FailsForRaggedAgainstDense)
{
    const CpuFpReferenceValidation<TypeParam> refValidation;
    auto ragged = createRaggedSdpaTensor<TypeParam>(0.5f);
    Tensor<TypeParam> dense(RAGGED_SDPA_DIMS);
    dense.fillTensorWithValue(0.5f);

    EXPECT_FALSE(refValidation.allClose(ragged, dense));
    EXPECT_FALSE(refValidation.allClose(dense, ragged));
}

TEST(TestCpuIntReferenceValidationRagged, PassesForIdenticalTensors)
{
    const CpuIntReferenceValidation<int32_t> refValidation;
    auto reference = createRaggedSdpaTensor<int32_t>(7.0f);
    auto implementation = createRaggedSdpaTensor<int32_t>(7.0f);

    EXPECT_TRUE(refValidation.allClose(reference, implementation));
}

TEST(TestCpuIntReferenceValidationRagged, FailsForInBlockDifference)
{
    const CpuIntReferenceValidation<int32_t> refValidation;
    auto reference = createRaggedSdpaTensor<int32_t>(7.0f);
    auto implementation = createRaggedSdpaTensor<int32_t>(7.0f);
    implementation.setHostValue(8, 1, 1, 2, 1);

    EXPECT_FALSE(refValidation.allClose(reference, implementation));
}

TEST(TestCpuIntReferenceValidationRagged, FailsForMismatchedRaggedLayouts)
{
    const CpuIntReferenceValidation<int32_t> refValidation;
    auto reference = createRaggedSdpaTensor<int32_t>(7.0f);
    auto implementation = createRaggedSdpaTensor<int32_t>(7.0f, {0, 3, 5});

    EXPECT_FALSE(refValidation.allClose(reference, implementation));
}

/* ================================================= */

/* ======== Non-finite mask tests (TYPED_TEST across fp types) ======== */

namespace
{
template <typename T>
CpuFpReferenceValidation<T> makeMaskedValidation(float tolerance,
                                                 std::optional<uint64_t> unwrittenSentinelBits
                                                 = std::nullopt)
{
    return CpuFpReferenceValidation<T>(tolerance,
                                       tolerance,
                                       MatchingInfinities::REJECTED,
                                       NonFiniteMask::matching(unwrittenSentinelBits));
}
} // namespace

template <typename T>
class CpuFpReferenceValidationNonFiniteMask : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceValidationNonFiniteMask, FpValidationTypes, );

// Bug caught: a mask that never matches NaN (every NaN-producing output still fails), or one
// that changes the default behavior (NaN against NaN must stay a failure when the mask is off).
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, NaNMatchesNaNOnlyWhenOptedIn)
{
    const float epsilon = validationEpsilon<TypeParam>();
    const CpuFpReferenceValidation<TypeParam> strictValidation(epsilon, epsilon);
    const auto maskedValidation = makeMaskedValidation<TypeParam>(epsilon);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    reference.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    implementation.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);

    EXPECT_FALSE(strictValidation.allClose(reference, implementation));
    EXPECT_TRUE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: a sign or payload comparison on NaN. Any NaN matches any NaN.
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, NaNPayloadAndSignDoNotMatter)
{
    const auto maskedValidation = makeMaskedValidation<TypeParam>(validationEpsilon<TypeParam>());
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    reference.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    implementation.setHostValue(-payloadNanSentinelValue<TypeParam>(), 0, 0);

    EXPECT_TRUE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: +Inf not matching +Inf, or the mask not being consulted for infinities.
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, PositiveInfinitiesMatch)
{
    const float epsilon = validationEpsilon<TypeParam>();
    const CpuFpReferenceValidation<TypeParam> strictValidation(epsilon, epsilon);
    const auto maskedValidation = makeMaskedValidation<TypeParam>(epsilon);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    reference.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);
    implementation.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(strictValidation.allClose(reference, implementation));
    EXPECT_TRUE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: -Inf not matching -Inf (for example a sign check that only handles +Inf).
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, NegativeInfinitiesMatch)
{
    const float epsilon = validationEpsilon<TypeParam>();
    const CpuFpReferenceValidation<TypeParam> strictValidation(epsilon, epsilon);
    const auto maskedValidation = makeMaskedValidation<TypeParam>(epsilon);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    reference.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);
    implementation.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(strictValidation.allClose(reference, implementation));
    EXPECT_TRUE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: NaN treated as a wildcard that matches a finite value in either direction.
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, NaNAgainstFiniteFails)
{
    const auto maskedValidation = makeMaskedValidation<TypeParam>(validationEpsilon<TypeParam>());
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);

    reference.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));

    reference.setHostValue(safeTestTypeCast<TypeParam>(1.0f), 0, 0);
    implementation.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: an infinity accepted where the other side holds a finite value.
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, FiniteAgainstInfinityFails)
{
    const auto maskedValidation = makeMaskedValidation<TypeParam>(validationEpsilon<TypeParam>());
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);

    reference.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));

    reference.setHostValue(safeTestTypeCast<TypeParam>(1.0f), 0, 0);
    implementation.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: infinities compared without their sign, so +Inf matches -Inf.
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, OppositeSignedInfinitiesFail)
{
    const auto maskedValidation = makeMaskedValidation<TypeParam>(validationEpsilon<TypeParam>());
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    reference.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);
    implementation.setHostValue(-std::numeric_limits<TypeParam>::infinity(), 0, 0);

    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: "non-finite matches non-finite" collapsing NaN and Inf into one class.
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, NaNAgainstInfinityFails)
{
    const auto maskedValidation = makeMaskedValidation<TypeParam>(validationEpsilon<TypeParam>());
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);

    reference.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    implementation.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));

    reference.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 0);
    implementation.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: the mask switching tolerance grading off for the whole tensor instead of only
// for the matched non-finite positions. A finite element within atol passes; one outside fails.
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, FiniteElementsAreStillGradedByTolerance)
{
    const auto maskedValidation = makeMaskedValidation<TypeParam>(0.1f);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    reference.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    implementation.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    reference.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 1);
    implementation.setHostValue(std::numeric_limits<TypeParam>::infinity(), 0, 1);

    implementation.setHostValue(safeTestTypeCast<TypeParam>(1.0625f), 1, 1);
    EXPECT_TRUE(maskedValidation.allClose(reference, implementation));

    implementation.setHostValue(safeTestTypeCast<TypeParam>(2.0f), 1, 1);
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: a vacuous pass. A reference with no finite element has nothing graded, and two
// runs that both produced garbage must not be reported as agreeing.
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, ReferenceWithNoFiniteElementFails)
{
    const auto maskedValidation = makeMaskedValidation<TypeParam>(validationEpsilon<TypeParam>());
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(std::numeric_limits<float>::quiet_NaN());
    implementation.fillTensorWithValue(std::numeric_limits<float>::quiet_NaN());
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));

    reference.fillTensorWithValue(std::numeric_limits<float>::infinity());
    implementation.fillTensorWithValue(std::numeric_limits<float>::infinity());
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));

    // One finite element, matched, is enough to grade.
    reference.setHostValue(safeTestTypeCast<TypeParam>(1.0f), 1, 1);
    implementation.setHostValue(safeTestTypeCast<TypeParam>(1.0f), 1, 1);
    EXPECT_TRUE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: an unwritten output element (still holding the harness fill) passing as
// "NaN matches NaN". The sentinel bit pattern fails, an ordinary NaN with another payload
// passes, and with no sentinel configured the validator cannot tell them apart.
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, SentinelBitPatternInImplementationFails)
{
    const auto sentinelBits = payloadNanSentinelBits<TypeParam>();
    ASSERT_TRUE(sentinelBits.has_value());
    const float epsilon = validationEpsilon<TypeParam>();
    const auto detectingValidation = makeMaskedValidation<TypeParam>(epsilon, sentinelBits);
    const auto blindValidation = makeMaskedValidation<TypeParam>(epsilon);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    reference.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);

    implementation.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);
    EXPECT_TRUE(detectingValidation.allClose(reference, implementation));

    implementation.setHostValue(payloadNanSentinelValue<TypeParam>(), 0, 0);
    EXPECT_FALSE(detectingValidation.allClose(reference, implementation));
    EXPECT_TRUE(blindValidation.allClose(reference, implementation));
}

// Bug caught: the sentinel being honored only where the reference is NaN. An unwritten element
// fails wherever it sits, including against a finite reference value.
TYPED_TEST(CpuFpReferenceValidationNonFiniteMask, SentinelFailsAgainstFiniteReferenceToo)
{
    const auto detectingValidation = makeMaskedValidation<TypeParam>(
        validationEpsilon<TypeParam>(), payloadNanSentinelBits<TypeParam>());
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> reference(dims);
    Tensor<TypeParam> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    implementation.setHostValue(payloadNanSentinelValue<TypeParam>(), 1, 0);

    EXPECT_FALSE(detectingValidation.allClose(reference, implementation));
}

/* ======== Payload NaN sentinel helper tests (TYPED_TEST across fp types) ======== */

template <typename T>
class PayloadNanSentinel : public ::testing::Test
{
};

TYPED_TEST_SUITE(PayloadNanSentinel, FpValidationTypes, );

// Bug caught: a sentinel that is not a NaN, not quiet, or identical to the NaN an engine
// computes, so unwritten detection either never fires or flags correct NaN outputs.
TYPED_TEST(PayloadNanSentinel, IsQuietNaNDistinctFromDefaultQuietNaN)
{
    const auto sentinelBits = payloadNanSentinelBits<TypeParam>();
    ASSERT_TRUE(sentinelBits.has_value());

    const auto sentinelValue = payloadNanSentinelValue<TypeParam>();
    const uint64_t defaultBits
        = non_finite_detail::elementBits(std::numeric_limits<TypeParam>::quiet_NaN());

    EXPECT_TRUE(hipdnn_data_sdk::types::isnan(sentinelValue));
    // The quiet bit is part of the default quiet NaN's bits, so a superset of them is quiet.
    EXPECT_EQ(*sentinelBits & defaultBits, defaultBits);
    EXPECT_NE(*sentinelBits, defaultBits);
}

// Bug caught: the value and bits helpers disagreeing, so the harness fills one pattern and
// the validator looks for another.
TYPED_TEST(PayloadNanSentinel, ValueCarriesExactlyTheReportedBits)
{
    const auto sentinelBits = payloadNanSentinelBits<TypeParam>();
    ASSERT_TRUE(sentinelBits.has_value());

    EXPECT_EQ(non_finite_detail::elementBits(payloadNanSentinelValue<TypeParam>()), *sentinelBits);
}

// Bug caught: a payload the element storage cannot hold (it would be truncated or quieted in
// the tensor and never match). The pattern must survive a store into a tensor and a load.
TYPED_TEST(PayloadNanSentinel, SurvivesStoreAndLoadThroughTensor)
{
    const auto sentinelBits = payloadNanSentinelBits<TypeParam>();
    ASSERT_TRUE(sentinelBits.has_value());

    Tensor<TypeParam> tensor({2});
    tensor.fillTensorWithValue(1.0f);
    tensor.setHostValue(payloadNanSentinelValue<TypeParam>(), 0);

    const TypeParam loaded = tensor.getHostValue(0);
    EXPECT_TRUE(hipdnn_data_sdk::types::isnan(loaded));
    EXPECT_EQ(non_finite_detail::elementBits(loaded), *sentinelBits);
    EXPECT_NE(non_finite_detail::elementBits(tensor.getHostValue(1)), *sentinelBits);
}

// Bug caught: the data-type overload drifting from the per-type helper the validators use.
TEST(TestPayloadNanSentinel, DataTypeOverloadMatchesTypedHelper)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    EXPECT_EQ(payloadNanSentinelBits(DataType::FLOAT), payloadNanSentinelBits<float>());
    EXPECT_EQ(payloadNanSentinelBits(DataType::DOUBLE), payloadNanSentinelBits<double>());
    EXPECT_EQ(payloadNanSentinelBits(DataType::HALF), payloadNanSentinelBits<half>());
    EXPECT_EQ(payloadNanSentinelBits(DataType::BFLOAT16), payloadNanSentinelBits<bfloat16>());
    EXPECT_EQ(payloadNanSentinelBits(DataType::FP8_E5M2), payloadNanSentinelBits<fp8_e5m2>());
}

// Bug caught: reporting a payload sentinel for types that have none, which would make the
// harness fill with an ordinary NaN and the validator reject legitimate NaN outputs.
TEST(TestPayloadNanSentinel, TypesWithoutPayloadNaNReportNone)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    EXPECT_FALSE(payloadNanSentinelBits(DataType::FP8_E4M3).has_value());
    EXPECT_FALSE(payloadNanSentinelBits(DataType::INT32).has_value());
    EXPECT_FALSE(payloadNanSentinelBits(DataType::UINT8).has_value());
    EXPECT_FALSE(payloadNanSentinelBits<fp8_e4m3>().has_value());
    EXPECT_FALSE(payloadNanSentinelBits<int32_t>().has_value());
}

/* ======== Non-finite mask: factory and ragged tests ======== */

// Bug caught: the factory dropping the mask or the sentinel on the way to the validator.
TEST(TestCpuFpReferenceValidationNonFiniteMask, FactoryBuildsTheMaskedComparison)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto strictValidation = createAllCloseValidator(DataType::HALF, 0.0f, 0.0f);
    const auto maskedValidation = createAllCloseMatchingNonFiniteValidator(
        DataType::HALF, 0.0f, 0.0f, payloadNanSentinelBits(DataType::HALF));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<half> reference(dims);
    Tensor<half> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    reference.setHostValue(std::numeric_limits<half>::quiet_NaN(), 0, 0);
    implementation.setHostValue(std::numeric_limits<half>::quiet_NaN(), 0, 0);

    EXPECT_FALSE(strictValidation->allClose(reference, implementation));
    EXPECT_TRUE(maskedValidation->allClose(reference, implementation));

    implementation.setHostValue(payloadNanSentinelValue<half>(), 0, 0);
    EXPECT_FALSE(maskedValidation->allClose(reference, implementation));
}

// Bug caught: integer outputs silently accepted, with the mask a no-op, instead of reporting
// that there is no NaN or Inf to match.
TEST(TestCpuFpReferenceValidationNonFiniteMask, FactoryRejectsIntegerDataTypes)
{
    EXPECT_THROW(createAllCloseMatchingNonFiniteValidator(
                     hipdnn_flatbuffers_sdk::data_objects::DataType::INT32, 0.0f, 0.0f),
                 std::runtime_error);
}

// Bug caught: a sentinel wider than the element type, which can never match and would silently
// disable unwritten-output detection (for example a double's bits handed to a half validator).
TEST(TestCpuFpReferenceValidationNonFiniteMask, SentinelWiderThanTheElementTypeThrows)
{
    EXPECT_THROW(createAllCloseMatchingNonFiniteValidator(
                     hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                     0.0f,
                     0.0f,
                     payloadNanSentinelBits<double>()),
                 std::invalid_argument);
}

// Bug caught: the mask overriding the ragged contract. A NaN in a ragged reference marks an
// undefined position and is skipped, so it must not demand that the implementation be NaN too.
TYPED_TEST(CpuFpReferenceValidationRagged, MaskKeepsSkippingInBlockNaNReference)
{
    const auto maskedValidation = makeMaskedValidation<TypeParam>(validationEpsilon<TypeParam>());
    auto reference = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto implementation = createRaggedSdpaTensor<TypeParam>(0.5f);
    reference.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 1, 1, 2, 1);
    implementation.setHostValue(TypeParam(2.0f), 1, 1, 2, 1);

    EXPECT_TRUE(maskedValidation.allClose(reference, implementation));
}

// Bug caught: a ragged reference that is entirely NaN passing under the mask because NaN matches
// NaN, although no position was graded.
TYPED_TEST(CpuFpReferenceValidationRagged, MaskFailsWhenEveryReferenceValueIsNaN)
{
    const auto maskedValidation = makeMaskedValidation<TypeParam>(validationEpsilon<TypeParam>());
    auto reference = createRaggedSdpaTensor<TypeParam>(std::numeric_limits<float>::quiet_NaN());
    auto implementation
        = createRaggedSdpaTensor<TypeParam>(std::numeric_limits<float>::quiet_NaN());

    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));
}

/* ======== Eight-bit float tests ======== */

// fp8 e5m2 has three NaN encodings (0x7D, 0x7E, 0x7F), so a distinctive payload exists.
TEST(TestCpuFpReferenceValidationNonFiniteMaskFp8E5m2, SentinelIsADistinctNaNEncoding)
{
    const auto sentinelBits = payloadNanSentinelBits<fp8_e5m2>();
    ASSERT_TRUE(sentinelBits.has_value());

    EXPECT_TRUE(hipdnn_data_sdk::types::isnan(payloadNanSentinelValue<fp8_e5m2>()));
    EXPECT_NE(*sentinelBits,
              non_finite_detail::elementBits(std::numeric_limits<fp8_e5m2>::quiet_NaN()));

    Tensor<fp8_e5m2> tensor({2});
    tensor.fillTensorWithValue(1.0f);
    tensor.setHostValue(payloadNanSentinelValue<fp8_e5m2>(), 0);
    EXPECT_EQ(non_finite_detail::elementBits(tensor.getHostValue(0)), *sentinelBits);
}

// Bug caught: the sentinel check or the NaN and Inf matching not working for a one-byte type.
TEST(TestCpuFpReferenceValidationNonFiniteMaskFp8E5m2, MaskMatchesNonFiniteAndFlagsTheSentinel)
{
    const auto detectingValidation
        = makeMaskedValidation<fp8_e5m2>(0.0f, payloadNanSentinelBits<fp8_e5m2>());
    const std::vector<int64_t> dims = {2, 2};

    Tensor<fp8_e5m2> reference(dims);
    Tensor<fp8_e5m2> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    reference.setHostValue(std::numeric_limits<fp8_e5m2>::quiet_NaN(), 0, 0);
    implementation.setHostValue(std::numeric_limits<fp8_e5m2>::quiet_NaN(), 0, 0);
    reference.setHostValue(fp8_e5m2::from_bits(0x7C), 0, 1);
    implementation.setHostValue(fp8_e5m2::from_bits(0x7C), 0, 1);
    reference.setHostValue(fp8_e5m2::from_bits(0xFC), 1, 0);
    implementation.setHostValue(fp8_e5m2::from_bits(0xFC), 1, 0);
    EXPECT_TRUE(detectingValidation.allClose(reference, implementation));

    // +Inf against -Inf.
    implementation.setHostValue(fp8_e5m2::from_bits(0xFC), 0, 1);
    EXPECT_FALSE(detectingValidation.allClose(reference, implementation));
    implementation.setHostValue(fp8_e5m2::from_bits(0x7C), 0, 1);

    // An unwritten element.
    implementation.setHostValue(payloadNanSentinelValue<fp8_e5m2>(), 0, 0);
    EXPECT_FALSE(detectingValidation.allClose(reference, implementation));
}

// fp8 e4m3fn has the single NaN encoding 0x7F/0xFF, so no payload exists and no NaN sentinel
// can be told from a legitimate NaN output.
TEST(TestCpuFpReferenceValidationNonFiniteMaskFp8E4m3, NoPayloadSentinelExists)
{
    EXPECT_FALSE(payloadNanSentinelBits<fp8_e4m3>().has_value());
}

// Documents the consequence for the harness: with the mask on and no sentinel, NaN matches NaN
// and an unwritten NaN-filled element is indistinguishable from a computed NaN, so unwritten
// detection is unavailable for this type. Everything else in the mask still applies.
TEST(TestCpuFpReferenceValidationNonFiniteMaskFp8E4m3, WithoutSentinelNaNMatchesAndFiniteIsGraded)
{
    const auto maskedValidation = makeMaskedValidation<fp8_e4m3>(0.0f);
    const std::vector<int64_t> dims = {2, 2};

    Tensor<fp8_e4m3> reference(dims);
    Tensor<fp8_e4m3> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    reference.setHostValue(std::numeric_limits<fp8_e4m3>::quiet_NaN(), 0, 0);
    implementation.setHostValue(std::numeric_limits<fp8_e4m3>::quiet_NaN(), 0, 0);
    EXPECT_TRUE(maskedValidation.allClose(reference, implementation));

    // NaN against a finite value still fails.
    implementation.setHostValue(fp8_e4m3(1.0f), 0, 0);
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));

    // A finite element that differs still fails.
    implementation.setHostValue(std::numeric_limits<fp8_e4m3>::quiet_NaN(), 0, 0);
    implementation.setHostValue(fp8_e4m3(2.0f), 1, 1);
    EXPECT_FALSE(maskedValidation.allClose(reference, implementation));
}

// The alternative the sentinel parameter supports for e4m3fn: a finite bit pattern the output
// domain excludes. The validator fails an element holding exactly that pattern and no other.
TEST(TestCpuFpReferenceValidationNonFiniteMaskFp8E4m3, FiniteSentinelFlagsOnlyItsOwnBitPattern)
{
    // 0x7E is the largest finite e4m3fn value, 448.
    const auto detectingValidation = makeMaskedValidation<fp8_e4m3>(0.0f, uint64_t{0x7E});
    const std::vector<int64_t> dims = {2, 2};

    Tensor<fp8_e4m3> reference(dims);
    Tensor<fp8_e4m3> implementation(dims);
    reference.fillTensorWithValue(1.0f);
    implementation.fillTensorWithValue(1.0f);
    EXPECT_TRUE(detectingValidation.allClose(reference, implementation));

    reference.setHostValue(fp8_e4m3::from_bits(0x7E), 0, 0);
    implementation.setHostValue(fp8_e4m3::from_bits(0x7E), 0, 0);
    EXPECT_FALSE(detectingValidation.allClose(reference, implementation));
}

/* ================================================= */
