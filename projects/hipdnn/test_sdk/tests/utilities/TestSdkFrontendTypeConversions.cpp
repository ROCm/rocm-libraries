// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

using hipdnn_test_sdk::utilities::createTensor;
using hipdnn_test_sdk::utilities::frontendToSdkDataType;
using hipdnn_test_sdk::utilities::frontendToSdkPointwiseMode;
using hipdnn_test_sdk::utilities::sdkToFrontendDataType;
using hipdnn_test_sdk::utilities::sdkToFrontendPointwiseMode;

namespace fe = hipdnn_frontend;
namespace sdk = hipdnn_flatbuffers_sdk::data_objects;

// ============================================================================
// DataType round-trip tests
// ============================================================================

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripFloat)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::FLOAT)),
              fe::DataType::FLOAT);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripHalf)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::HALF)), fe::DataType::HALF);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripBfloat16)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::BFLOAT16)),
              fe::DataType::BFLOAT16);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripDouble)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::DOUBLE)),
              fe::DataType::DOUBLE);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripUint8)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::UINT8)),
              fe::DataType::UINT8);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripInt32)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::INT32)),
              fe::DataType::INT32);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripInt8)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::INT8)), fe::DataType::INT8);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripFp8E4M3)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::FP8_E4M3)),
              fe::DataType::FP8_E4M3);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripFp8E5M2)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::FP8_E5M2)),
              fe::DataType::FP8_E5M2);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripFp8E8M0)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::FP8_E8M0)),
              fe::DataType::FP8_E8M0);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripFp4E2M1)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::FP4_E2M1)),
              fe::DataType::FP4_E2M1);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripInt4)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::INT4)), fe::DataType::INT4);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripFp6E2M3)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::FP6_E2M3)),
              fe::DataType::FP6_E2M3);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripFp6E3M2)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::FP6_E3M2)),
              fe::DataType::FP6_E3M2);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripInt64)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::INT64)),
              fe::DataType::INT64);
}

TEST(TestSdkFrontendTypeConversions, DataTypeRoundTripBoolean)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::BOOLEAN)),
              fe::DataType::BOOLEAN);
}

// ============================================================================
// DataType default/fallback tests
// ============================================================================

TEST(TestSdkFrontendTypeConversions, FrontendNotSetMapsToSdkUnset)
{
    EXPECT_EQ(frontendToSdkDataType(fe::DataType::NOT_SET), sdk::DataType::UNSET);
}

TEST(TestSdkFrontendTypeConversions, SdkUnsetMapsToFrontendNotSet)
{
    EXPECT_EQ(sdkToFrontendDataType(sdk::DataType::UNSET), fe::DataType::NOT_SET);
}

TEST(TestSdkFrontendTypeConversions, NotSetRoundTrip)
{
    EXPECT_EQ(sdkToFrontendDataType(frontendToSdkDataType(fe::DataType::NOT_SET)),
              fe::DataType::NOT_SET);
}

// ============================================================================
// createTensor tests
// ============================================================================

TEST(TestSdkFrontendTypeConversions, CreateTensorBoolean)
{
    const std::vector<int64_t> dims = {2, 2};
    const std::vector<int64_t> strides = {2, 1};

    auto tensor = createTensor(fe::DataType::BOOLEAN, dims, strides);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dims(), dims);
    EXPECT_EQ(tensor->strides(), strides);
    EXPECT_EQ(tensor->elementSize(), sizeof(bool));
}

// ============================================================================
// PointwiseMode exhaustive round-trip tests
// ============================================================================

class TestPointwiseModeRoundTrip : public ::testing::TestWithParam<fe::PointwiseMode>
{
};

TEST_P(TestPointwiseModeRoundTrip, RoundTrip)
{
    auto mode = GetParam();
    EXPECT_EQ(sdkToFrontendPointwiseMode(frontendToSdkPointwiseMode(mode)), mode);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestPointwiseModeRoundTrip,
                         ::testing::Values(fe::PointwiseMode::NOT_SET,
                                           fe::PointwiseMode::ABS,
                                           fe::PointwiseMode::ADD,
                                           fe::PointwiseMode::ADD_SQUARE,
                                           fe::PointwiseMode::BINARY_SELECT,
                                           fe::PointwiseMode::CEIL,
                                           fe::PointwiseMode::CMP_EQ,
                                           fe::PointwiseMode::CMP_GE,
                                           fe::PointwiseMode::CMP_GT,
                                           fe::PointwiseMode::CMP_LE,
                                           fe::PointwiseMode::CMP_LT,
                                           fe::PointwiseMode::CMP_NEQ,
                                           fe::PointwiseMode::DIV,
                                           fe::PointwiseMode::ELU_BWD,
                                           fe::PointwiseMode::ELU_FWD,
                                           fe::PointwiseMode::ERF,
                                           fe::PointwiseMode::EXP,
                                           fe::PointwiseMode::FLOOR,
                                           fe::PointwiseMode::GELU_APPROX_TANH_BWD,
                                           fe::PointwiseMode::GELU_APPROX_TANH_FWD,
                                           fe::PointwiseMode::GELU_BWD,
                                           fe::PointwiseMode::GELU_FWD,
                                           fe::PointwiseMode::GEN_INDEX,
                                           fe::PointwiseMode::IDENTITY,
                                           fe::PointwiseMode::LOG,
                                           fe::PointwiseMode::LOGICAL_AND,
                                           fe::PointwiseMode::LOGICAL_NOT,
                                           fe::PointwiseMode::LOGICAL_OR,
                                           fe::PointwiseMode::MAX,
                                           fe::PointwiseMode::MIN,
                                           fe::PointwiseMode::MUL,
                                           fe::PointwiseMode::NEG,
                                           fe::PointwiseMode::RECIPROCAL,
                                           fe::PointwiseMode::RELU_BWD,
                                           fe::PointwiseMode::RELU_FWD,
                                           fe::PointwiseMode::RSQRT,
                                           fe::PointwiseMode::SIGMOID_BWD,
                                           fe::PointwiseMode::SIGMOID_FWD,
                                           fe::PointwiseMode::SIN,
                                           fe::PointwiseMode::SOFTPLUS_BWD,
                                           fe::PointwiseMode::SOFTPLUS_FWD,
                                           fe::PointwiseMode::SQRT,
                                           fe::PointwiseMode::SUB,
                                           fe::PointwiseMode::SWISH_BWD,
                                           fe::PointwiseMode::SWISH_FWD,
                                           fe::PointwiseMode::TAN,
                                           fe::PointwiseMode::TANH_BWD,
                                           fe::PointwiseMode::TANH_FWD));

// ============================================================================
// PointwiseMode direct SDK-to-frontend mapping tests
// ============================================================================

TEST(TestSdkFrontendTypeConversions, PointwiseModeMaxOpMapsToMax)
{
    EXPECT_EQ(sdkToFrontendPointwiseMode(sdk::PointwiseMode::MAX_OP), fe::PointwiseMode::MAX);
}

TEST(TestSdkFrontendTypeConversions, PointwiseModeMinOpMapsToMin)
{
    EXPECT_EQ(sdkToFrontendPointwiseMode(sdk::PointwiseMode::MIN_OP), fe::PointwiseMode::MIN);
}

TEST(TestSdkFrontendTypeConversions, PointwiseModeUnsetMapsToNotSet)
{
    EXPECT_EQ(sdkToFrontendPointwiseMode(sdk::PointwiseMode::UNSET), fe::PointwiseMode::NOT_SET);
}
