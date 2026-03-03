// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestMacros.hpp"
#include "descriptors/DataTypeConversion.hpp"
#include <gtest/gtest.h>

namespace hipdnn_backend
{
namespace testing
{

using hipdnn_data_sdk::data_objects::ConvMode;
using hipdnn_data_sdk::data_objects::DataType;

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
        DataTypeConversionParam{HIPDNN_DATA_FP8_E5M2, DataType::FP8_E5M2, 1, "Fp8E5M2"}),
    [](const ::testing::TestParamInfo<DataTypeConversionParam>& info) { return info.param.name; });

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
    ASSERT_EQ(toSdkConvMode(HIPDNN_CONVOLUTION_MODE_CONVOLUTION), ConvMode::CONVOLUTION);
}

TEST(TestDataTypeConversion, ToSdkConvModeConvertsCrossCorrelation)
{
    ASSERT_EQ(toSdkConvMode(HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION),
              ConvMode::CROSS_CORRELATION);
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
    ASSERT_EQ(fromSdkConvMode(ConvMode::CONVOLUTION), HIPDNN_CONVOLUTION_MODE_CONVOLUTION);
}

TEST(TestDataTypeConversion, FromSdkConvModeConvertsCrossCorrelation)
{
    ASSERT_EQ(fromSdkConvMode(ConvMode::CROSS_CORRELATION),
              HIPDNN_CONVOLUTION_MODE_CROSS_CORRELATION);
}

TEST(TestDataTypeConversion, FromSdkConvModeThrowsOnUnset)
{
    ASSERT_THROW_HIPDNN_STATUS(fromSdkConvMode(ConvMode::UNSET), HIPDNN_STATUS_BAD_PARAM);
}

} // namespace testing
} // namespace hipdnn_backend
