// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Helpers.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <limits>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::helpers;
using namespace hipdnn_data_sdk::types;

TEST(TestCpuReferenceMiopenRmsValidation, NegativeToleranceThrows)
{
    EXPECT_THROW(const CpuReferenceMiopenRmsValidation<float> refValidation(-1e-5f),
                 std::invalid_argument);
}

TEST(TestCpuReferenceMiopenRmsValidation, NaNToleranceThrows)
{
    EXPECT_THROW(const CpuReferenceMiopenRmsValidation<float> refValidation(
                     std::numeric_limits<float>::quiet_NaN()),
                 std::invalid_argument);
}

TEST(TestCpuReferenceMiopenRmsValidation, InfToleranceThrows)
{
    EXPECT_THROW(const CpuReferenceMiopenRmsValidation<float> refValidation(
                     std::numeric_limits<float>::infinity()),
                 std::invalid_argument);
}

// Test MIOpen-specific RMS calculation behavior
TEST(TestCpuReferenceMiopenRmsValidation, MiopenRmsCalculation)
{
    const CpuReferenceMiopenRmsValidation<double> refValidation(0.1);

    Tensor<double> tensor1({4});
    Tensor<double> tensor2({4});

    tensor1.setHostValue(1.0, 0);
    tensor1.setHostValue(2.0, 1);
    tensor1.setHostValue(3.0, 2);
    tensor1.setHostValue(4.0, 3);

    tensor2.setHostValue(1.1, 0);
    tensor2.setHostValue(2.1, 1);
    tensor2.setHostValue(3.1, 2);
    tensor2.setHostValue(4.1, 3);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));

    const CpuReferenceMiopenRmsValidation<double> refValidationTight(0.02);
    EXPECT_FALSE(refValidationTight.allClose(tensor1, tensor2));
}

/* ================= ITensor allClose Tests ================= */

TEST(TestCpuReferenceMiopenRmsValidationITensorBfp16, BasicUsage)
{
    const CpuReferenceMiopenRmsValidation<bfloat16> validator;

    Tensor<bfloat16> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0_bf);

    Tensor<bfloat16> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0_bf);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceMiopenRmsValidationITensorFp16, BasicUsage)
{
    const CpuReferenceMiopenRmsValidation<half> validator;

    Tensor<half> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0_h);

    Tensor<half> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0_h);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceMiopenRmsValidationITensorFp32, BasicUsage)
{
    const CpuReferenceMiopenRmsValidation<float> validator;

    Tensor<float> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0f);

    Tensor<float> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0f);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

TEST(TestCpuReferenceMiopenRmsValidationITensorFp64, BasicUsage)
{
    const CpuReferenceMiopenRmsValidation<double> validator;

    Tensor<double> tensor1({2, 3, 4});
    tensor1.fillWithValue(1.0);

    Tensor<double> tensor2({2, 3, 4});
    tensor2.fillWithValue(1.0);

    EXPECT_TRUE(validator.allClose(tensor1, tensor2));
}

/* ================= Nan / Inf TYPED TESTS ================= */

template <typename T>
class CpuReferenceMiopenRmsValidationNanInf : public ::testing::Test
{
};

using RmsFpValidationTypes = ::testing::Types<float, double, half, bfloat16>;
TYPED_TEST_SUITE(CpuReferenceMiopenRmsValidationNanInf, RmsFpValidationTypes, );

TYPED_TEST(CpuReferenceMiopenRmsValidationNanInf, FailsWhenReferenceHasNaN)
{
    const CpuReferenceMiopenRmsValidation<TypeParam> refValidation(TypeParam(1.0f));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor1.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuReferenceMiopenRmsValidationNanInf, FailsWhenImplementationHasNaN)
{
    const CpuReferenceMiopenRmsValidation<TypeParam> refValidation(TypeParam(1.0f));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    tensor2.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 0, 0);

    EXPECT_FALSE(refValidation.allClose(tensor1, tensor2));
}

TYPED_TEST(CpuReferenceMiopenRmsValidationNanInf, PassesForFiniteValues)
{
    const CpuReferenceMiopenRmsValidation<TypeParam> refValidation(TypeParam(1.0f));
    const std::vector<int64_t> dims = {2, 2};

    Tensor<TypeParam> tensor1(dims);
    Tensor<TypeParam> tensor2(dims);
    tensor1.fillTensorWithValue(1.0f);
    tensor2.fillTensorWithValue(1.0f);

    EXPECT_TRUE(refValidation.allClose(tensor1, tensor2));
}