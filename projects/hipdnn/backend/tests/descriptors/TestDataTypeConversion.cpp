// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnReduceTensorOp.h"
#include "TestMacros.hpp"
#include "descriptors/DataTypeConversion.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>

#include <string>

namespace hipdnn_backend
{
namespace testing
{

using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;

// =============================================================================
// Parameterized Data Type Conversion Tests
// =============================================================================

struct DataTypeConversionParam
{
    hipdnnDataType_t apiType;
    DataType sdkType;
    int64_t byteSize;
    const char* name;
};

class TestDataTypeConversionRoundTrip : public ::testing::TestWithParam<DataTypeConversionParam>
{
};

TEST_P(TestDataTypeConversionRoundTrip, ToSdkDataType)
{
    auto param = GetParam();
    ASSERT_EQ(toSdkDataType(param.apiType), param.sdkType);
}

TEST_P(TestDataTypeConversionRoundTrip, FromSdkDataType)
{
    auto param = GetParam();
    ASSERT_EQ(fromSdkDataType(param.sdkType), param.apiType);
}

TEST_P(TestDataTypeConversionRoundTrip, GetDataTypeByteSize)
{
    auto param = GetParam();
    ASSERT_EQ(getDataTypeByteSize(param.sdkType), param.byteSize);
}

INSTANTIATE_TEST_SUITE_P(
    DataTypes,
    TestDataTypeConversionRoundTrip,
    ::testing::Values(
        DataTypeConversionParam{HIPDNN_DATA_FLOAT, DataType::FLOAT, 4, "Float"},
        DataTypeConversionParam{HIPDNN_DATA_DOUBLE, DataType::DOUBLE, 8, "Double"},
        DataTypeConversionParam{HIPDNN_DATA_HALF, DataType::HALF, 2, "Half"},
        DataTypeConversionParam{HIPDNN_DATA_INT8, DataType::INT8, 1, "Int8"},
        DataTypeConversionParam{HIPDNN_DATA_INT32, DataType::INT32, 4, "Int32"},
        DataTypeConversionParam{HIPDNN_DATA_UINT8, DataType::UINT8, 1, "Uint8"},
        DataTypeConversionParam{HIPDNN_DATA_BFLOAT16, DataType::BFLOAT16, 2, "Bfloat16"},
        DataTypeConversionParam{HIPDNN_DATA_FP8_E4M3, DataType::FP8_E4M3, 1, "Fp8E4M3"},
        DataTypeConversionParam{HIPDNN_DATA_FP8_E5M2, DataType::FP8_E5M2, 1, "Fp8E5M2"},
        DataTypeConversionParam{HIPDNN_DATA_INT64, DataType::INT64, 8, "Int64"},
        DataTypeConversionParam{HIPDNN_DATA_BOOLEAN, DataType::BOOLEAN, 1, "Boolean"}),
    [](const ::testing::TestParamInfo<DataTypeConversionParam>& info) { return info.param.name; });

// =============================================================================
// Low-precision type conversion tests (no byte size support)
// =============================================================================

TEST(TestDataTypeConversion, ToSdkDataTypeFp8E8M0)
{
    ASSERT_EQ(toSdkDataType(HIPDNN_DATA_FP8_E8M0), DataType::FP8_E8M0);
}

TEST(TestDataTypeConversion, FromSdkDataTypeFp8E8M0)
{
    ASSERT_EQ(fromSdkDataType(DataType::FP8_E8M0), HIPDNN_DATA_FP8_E8M0);
}

TEST(TestDataTypeConversion, ToSdkDataTypeFp4E2M1)
{
    ASSERT_EQ(toSdkDataType(HIPDNN_DATA_FP4_E2M1), DataType::FP4_E2M1);
}

TEST(TestDataTypeConversion, FromSdkDataTypeFp4E2M1)
{
    ASSERT_EQ(fromSdkDataType(DataType::FP4_E2M1), HIPDNN_DATA_FP4_E2M1);
}

TEST(TestDataTypeConversion, ToSdkDataTypeInt4)
{
    ASSERT_EQ(toSdkDataType(HIPDNN_DATA_INT4), DataType::INT4);
}

TEST(TestDataTypeConversion, FromSdkDataTypeInt4)
{
    ASSERT_EQ(fromSdkDataType(DataType::INT4), HIPDNN_DATA_INT4);
}

TEST(TestDataTypeConversion, ToSdkDataTypeFp6E2M3)
{
    ASSERT_EQ(toSdkDataType(HIPDNN_DATA_FP6_E2M3_EXT), DataType::FP6_E2M3);
}

TEST(TestDataTypeConversion, FromSdkDataTypeFp6E2M3)
{
    ASSERT_EQ(fromSdkDataType(DataType::FP6_E2M3), HIPDNN_DATA_FP6_E2M3_EXT);
}

TEST(TestDataTypeConversion, ToSdkDataTypeFp6E3M2)
{
    ASSERT_EQ(toSdkDataType(HIPDNN_DATA_FP6_E3M2_EXT), DataType::FP6_E3M2);
}

TEST(TestDataTypeConversion, FromSdkDataTypeFp6E3M2)
{
    ASSERT_EQ(fromSdkDataType(DataType::FP6_E3M2), HIPDNN_DATA_FP6_E3M2_EXT);
}

TEST(TestDataTypeConversion, GetDataTypeByteSizeThrowsForLowPrecisionTypes)
{
    ASSERT_THROW_HIPDNN_STATUS(getDataTypeByteSize(DataType::FP8_E8M0), HIPDNN_STATUS_BAD_PARAM);
    ASSERT_THROW_HIPDNN_STATUS(getDataTypeByteSize(DataType::FP4_E2M1), HIPDNN_STATUS_BAD_PARAM);
    ASSERT_THROW_HIPDNN_STATUS(getDataTypeByteSize(DataType::INT4), HIPDNN_STATUS_BAD_PARAM);
    ASSERT_THROW_HIPDNN_STATUS(getDataTypeByteSize(DataType::FP6_E2M3), HIPDNN_STATUS_BAD_PARAM);
    ASSERT_THROW_HIPDNN_STATUS(getDataTypeByteSize(DataType::FP6_E3M2), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// toSdkDataType Edge Cases
// =============================================================================

TEST(TestDataTypeConversion, ToSdkDataTypeThrowsOnInvalidEnum)
{
    ASSERT_THROW_HIPDNN_STATUS(toSdkDataType(static_cast<hipdnnDataType_t>(999)),
                               HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// fromSdkDataType Edge Cases
// =============================================================================

TEST(TestDataTypeConversion, FromSdkDataTypeThrowsOnUnset)
{
    ASSERT_THROW_HIPDNN_STATUS(fromSdkDataType(DataType::UNSET), HIPDNN_STATUS_BAD_PARAM);
}

TEST(TestDataTypeConversion, FromSdkDataTypeThrowsOnInvalidEnum)
{
    ASSERT_THROW_HIPDNN_STATUS(fromSdkDataType(static_cast<DataType>(999)),
                               HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// getDataTypeByteSize Edge Cases
// =============================================================================

TEST(TestDataTypeConversion, GetDataTypeByteSizeThrowsOnUnset)
{
    ASSERT_THROW_HIPDNN_STATUS(getDataTypeByteSize(DataType::UNSET), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// toSdkConvMode Tests
// =============================================================================

TEST(TestDataTypeConversion, ToSdkConvModeConvertsConvolution)
{
    ASSERT_EQ(toSdkConvMode(HIPDNN_CONVOLUTION), ConvMode::CONVOLUTION);
}

TEST(TestDataTypeConversion, ToSdkConvModeConvertsCrossCorrelation)
{
    ASSERT_EQ(toSdkConvMode(HIPDNN_CROSS_CORRELATION), ConvMode::CROSS_CORRELATION);
}

TEST(TestDataTypeConversion, ToSdkConvModeThrowsOnInvalidEnum)
{
    ASSERT_THROW_HIPDNN_STATUS(toSdkConvMode(static_cast<hipdnnConvolutionMode_t>(999)),
                               HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// fromSdkConvMode Tests
// =============================================================================

TEST(TestDataTypeConversion, FromSdkConvModeConvertsConvolution)
{
    ASSERT_EQ(fromSdkConvMode(ConvMode::CONVOLUTION), HIPDNN_CONVOLUTION);
}

TEST(TestDataTypeConversion, FromSdkConvModeConvertsCrossCorrelation)
{
    ASSERT_EQ(fromSdkConvMode(ConvMode::CROSS_CORRELATION), HIPDNN_CROSS_CORRELATION);
}

TEST(TestDataTypeConversion, FromSdkConvModeThrowsOnUnset)
{
    ASSERT_THROW_HIPDNN_STATUS(fromSdkConvMode(ConvMode::UNSET), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// Parameterized Pointwise Mode Conversion Tests
// =============================================================================

struct PointwiseModeConversionParam
{
    hipdnnPointwiseMode_t apiMode;
    PointwiseMode sdkMode;
    std::string name;
};

class TestPointwiseModeConversionRoundTrip
    : public ::testing::TestWithParam<PointwiseModeConversionParam>
{
};

TEST_P(TestPointwiseModeConversionRoundTrip, ToSdkPointwiseMode)
{
    const auto& param = GetParam();
    ASSERT_EQ(toSdkPointwiseMode(param.apiMode), param.sdkMode);
}

TEST_P(TestPointwiseModeConversionRoundTrip, FromSdkPointwiseMode)
{
    const auto& param = GetParam();
    ASSERT_EQ(fromSdkPointwiseMode(param.sdkMode), param.apiMode);
}

INSTANTIATE_TEST_SUITE_P(
    PointwiseModes,
    TestPointwiseModeConversionRoundTrip,
    ::testing::Values(
        PointwiseModeConversionParam{HIPDNN_POINTWISE_ABS, PointwiseMode::ABS, "Abs"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_ADD, PointwiseMode::ADD, "Add"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_ADD_SQUARE, PointwiseMode::ADD_SQUARE, "AddSquare"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_BINARY_SELECT, PointwiseMode::BINARY_SELECT, "BinarySelect"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_CEIL, PointwiseMode::CEIL, "Ceil"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_CMP_EQ, PointwiseMode::CMP_EQ, "CmpEq"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_CMP_GE, PointwiseMode::CMP_GE, "CmpGe"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_CMP_GT, PointwiseMode::CMP_GT, "CmpGt"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_CMP_LE, PointwiseMode::CMP_LE, "CmpLe"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_CMP_LT, PointwiseMode::CMP_LT, "CmpLt"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_CMP_NEQ, PointwiseMode::CMP_NEQ, "CmpNeq"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_DIV, PointwiseMode::DIV, "Div"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_ELU_BWD, PointwiseMode::ELU_BWD, "EluBwd"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_ELU_FWD, PointwiseMode::ELU_FWD, "EluFwd"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_ERF, PointwiseMode::ERF, "Erf"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_EXP, PointwiseMode::EXP, "Exp"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_FLOOR, PointwiseMode::FLOOR, "Floor"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_GELU_APPROX_TANH_BWD,
                                     PointwiseMode::GELU_APPROX_TANH_BWD,
                                     "GeluApproxTanhBwd"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_GELU_APPROX_TANH_FWD,
                                     PointwiseMode::GELU_APPROX_TANH_FWD,
                                     "GeluApproxTanhFwd"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_GELU_BWD, PointwiseMode::GELU_BWD, "GeluBwd"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_GELU_FWD, PointwiseMode::GELU_FWD, "GeluFwd"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_GEN_INDEX, PointwiseMode::GEN_INDEX, "GenIndex"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_IDENTITY, PointwiseMode::IDENTITY, "Identity"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_LOG, PointwiseMode::LOG, "Log"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_LOGICAL_AND, PointwiseMode::LOGICAL_AND, "LogicalAnd"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_LOGICAL_NOT, PointwiseMode::LOGICAL_NOT, "LogicalNot"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_LOGICAL_OR, PointwiseMode::LOGICAL_OR, "LogicalOr"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_MAX, PointwiseMode::MAX_OP, "Max"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_MIN, PointwiseMode::MIN_OP, "Min"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_MUL, PointwiseMode::MUL, "Mul"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_NEG, PointwiseMode::NEG, "Neg"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_RECIPROCAL, PointwiseMode::RECIPROCAL, "Reciprocal"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_RELU_BWD, PointwiseMode::RELU_BWD, "ReluBwd"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_RELU_FWD, PointwiseMode::RELU_FWD, "ReluFwd"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_RSQRT, PointwiseMode::RSQRT, "Rsqrt"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_SIGMOID_BWD, PointwiseMode::SIGMOID_BWD, "SigmoidBwd"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_SIGMOID_FWD, PointwiseMode::SIGMOID_FWD, "SigmoidFwd"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_SIN, PointwiseMode::SIN, "Sin"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_SOFTPLUS_BWD, PointwiseMode::SOFTPLUS_BWD, "SoftplusBwd"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_SOFTPLUS_FWD, PointwiseMode::SOFTPLUS_FWD, "SoftplusFwd"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_SQRT, PointwiseMode::SQRT, "Sqrt"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_SUB, PointwiseMode::SUB, "Sub"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_SWISH_BWD, PointwiseMode::SWISH_BWD, "SwishBwd"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_SWISH_FWD, PointwiseMode::SWISH_FWD, "SwishFwd"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_TAN, PointwiseMode::TAN, "Tan"},
        PointwiseModeConversionParam{HIPDNN_POINTWISE_TANH_BWD, PointwiseMode::TANH_BWD, "TanhBwd"},
        PointwiseModeConversionParam{
            HIPDNN_POINTWISE_TANH_FWD, PointwiseMode::TANH_FWD, "TanhFwd"}),
    [](const ::testing::TestParamInfo<PointwiseModeConversionParam>& info) {
        return info.param.name;
    });

// =============================================================================
// toSdkPointwiseMode Edge Cases
// =============================================================================

TEST(TestPointwiseModeConversionRoundTrip, InvalidEnumThrows)
{
    ASSERT_THROW_HIPDNN_STATUS(toSdkPointwiseMode(static_cast<hipdnnPointwiseMode_t>(-1)),
                               HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// fromSdkPointwiseMode Edge Cases
// =============================================================================

TEST(TestPointwiseModeConversionRoundTrip, UnsetSdkModeThrows)
{
    ASSERT_THROW_HIPDNN_STATUS(fromSdkPointwiseMode(PointwiseMode::UNSET), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// Parameterized Reduction Mode Conversion Tests
// =============================================================================

struct ReductionModeConversionParam
{
    hipdnnReduceTensorOp_t apiMode;
    ReductionMode sdkMode;
    std::string name;
};

class TestReductionModeConversionRoundTrip
    : public ::testing::TestWithParam<ReductionModeConversionParam>
{
};

TEST_P(TestReductionModeConversionRoundTrip, ToSdkReductionMode)
{
    const auto& param = GetParam();
    ASSERT_EQ(toSdkReductionMode(param.apiMode), param.sdkMode);
}

TEST_P(TestReductionModeConversionRoundTrip, FromSdkReductionMode)
{
    const auto& param = GetParam();
    ASSERT_EQ(fromSdkReductionMode(param.sdkMode), param.apiMode);
}

INSTANTIATE_TEST_SUITE_P(
    ReductionModes,
    TestReductionModeConversionRoundTrip,
    ::testing::Values(
        ReductionModeConversionParam{HIPDNN_REDUCE_TENSOR_ADD, ReductionMode::ADD, "Add"},
        ReductionModeConversionParam{HIPDNN_REDUCE_TENSOR_MUL, ReductionMode::MUL, "Mul"},
        ReductionModeConversionParam{HIPDNN_REDUCE_TENSOR_MIN, ReductionMode::MIN_OP, "Min"},
        ReductionModeConversionParam{HIPDNN_REDUCE_TENSOR_MAX, ReductionMode::MAX_OP, "Max"},
        ReductionModeConversionParam{HIPDNN_REDUCE_TENSOR_AMAX, ReductionMode::AMAX, "Amax"},
        ReductionModeConversionParam{HIPDNN_REDUCE_TENSOR_AVG, ReductionMode::AVG, "Avg"},
        ReductionModeConversionParam{HIPDNN_REDUCE_TENSOR_NORM1, ReductionMode::NORM1, "Norm1"},
        ReductionModeConversionParam{HIPDNN_REDUCE_TENSOR_NORM2, ReductionMode::NORM2, "Norm2"},
        ReductionModeConversionParam{
            HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS, ReductionMode::MUL_NO_ZEROS, "MulNoZeros"}),
    [](const ::testing::TestParamInfo<ReductionModeConversionParam>& info) {
        return info.param.name;
    });

// =============================================================================
// toSdkReductionMode Edge Cases
// =============================================================================

TEST(TestReductionModeConversionRoundTrip, InvalidEnumThrows)
{
    ASSERT_THROW_HIPDNN_STATUS(toSdkReductionMode(static_cast<hipdnnReduceTensorOp_t>(-1)),
                               HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// fromSdkReductionMode Edge Cases
// =============================================================================

TEST(TestReductionModeConversionRoundTrip, NotSetSdkModeThrows)
{
    ASSERT_THROW_HIPDNN_STATUS(fromSdkReductionMode(ReductionMode::NOT_SET),
                               HIPDNN_STATUS_BAD_PARAM);
}

} // namespace testing
} // namespace hipdnn_backend
