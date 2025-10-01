// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_frontend/Types.hpp"
#include <gtest/gtest.h>
#include <sstream>

TEST(TestTypes, ToSdkTypeDataTypes)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(toSdkType(DataType::FLOAT), hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(toSdkType(DataType::HALF), hipdnn_sdk::data_objects::DataType::HALF);
    EXPECT_EQ(toSdkType(DataType::BFLOAT16), hipdnn_sdk::data_objects::DataType::BFLOAT16);
    EXPECT_EQ(toSdkType(DataType::DOUBLE), hipdnn_sdk::data_objects::DataType::DOUBLE);
    EXPECT_EQ(toSdkType(DataType::UINT8), hipdnn_sdk::data_objects::DataType::UINT8);
    EXPECT_EQ(toSdkType(DataType::INT32), hipdnn_sdk::data_objects::DataType::INT32);
    EXPECT_EQ(toSdkType(DataType::NOT_SET), hipdnn_sdk::data_objects::DataType::UNSET);
}

TEST(TestTypes, FromSdkTypeDataTypes)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(fromSdkType(hipdnn_sdk::data_objects::DataType::FLOAT), DataType::FLOAT);
    EXPECT_EQ(fromSdkType(hipdnn_sdk::data_objects::DataType::HALF), DataType::HALF);
    EXPECT_EQ(fromSdkType(hipdnn_sdk::data_objects::DataType::BFLOAT16), DataType::BFLOAT16);
    EXPECT_EQ(fromSdkType(hipdnn_sdk::data_objects::DataType::DOUBLE), DataType::DOUBLE);
    EXPECT_EQ(fromSdkType(hipdnn_sdk::data_objects::DataType::UINT8), DataType::UINT8);
    EXPECT_EQ(fromSdkType(hipdnn_sdk::data_objects::DataType::INT32), DataType::INT32);
    EXPECT_EQ(fromSdkType(hipdnn_sdk::data_objects::DataType::UNSET), DataType::NOT_SET);
}

TEST(TestTypes, ConvolutionModeConversion)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(toSdkType(ConvolutionMode::CROSS_CORRELATION),
              hipdnn_sdk::data_objects::ConvMode::CROSS_CORRELATION);
    EXPECT_EQ(toSdkType(ConvolutionMode::CONVOLUTION),
              hipdnn_sdk::data_objects::ConvMode::CONVOLUTION);
    EXPECT_EQ(toSdkType(ConvolutionMode::NOT_SET), hipdnn_sdk::data_objects::ConvMode::UNSET);
}

TEST(TestTypes, PointwiseModeConversion)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(toSdkType(PointwiseMode::RELU_FWD),
              hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD);
    EXPECT_EQ(toSdkType(PointwiseMode::NOT_SET), hipdnn_sdk::data_objects::PointwiseMode::UNSET);
}

TEST(TestTypes, HeuristicModeConversion)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(toBackendType(HeuristicMode::FALLBACK),
              hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK);
}

TEST(TestTypes, GetDataTypeEnumFromType)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(getDataTypeEnumFromType<float>(), DataType::FLOAT);
    EXPECT_EQ(getDataTypeEnumFromType<half>(), DataType::HALF);
    EXPECT_EQ(getDataTypeEnumFromType<hip_bfloat16>(), DataType::BFLOAT16);
    EXPECT_EQ(getDataTypeEnumFromType<double>(), DataType::DOUBLE);
    EXPECT_EQ(getDataTypeEnumFromType<uint8_t>(), DataType::UINT8);
    EXPECT_EQ(getDataTypeEnumFromType<int32_t>(), DataType::INT32);

    EXPECT_EQ(getDataTypeEnumFromType<float*>(), DataType::NOT_SET);
    EXPECT_EQ(getDataTypeEnumFromType<char>(), DataType::NOT_SET);
}

TEST(TestTypes, DataTypeToString)
{
    using namespace hipdnn_frontend;

    EXPECT_STREQ(to_string(DataType::FLOAT), "fp32");
    EXPECT_STREQ(to_string(DataType::HALF), "fp16");
    EXPECT_STREQ(to_string(DataType::BFLOAT16), "bf16");
    EXPECT_STREQ(to_string(DataType::DOUBLE), "fp64");
    EXPECT_STREQ(to_string(DataType::UINT8), "uint8");
    EXPECT_STREQ(to_string(DataType::INT32), "int32");
    EXPECT_STREQ(to_string(DataType::NOT_SET), "unknown");
}

TEST(TestTypes, DataTypeStreamOperator)
{
    using namespace hipdnn_frontend;

    std::ostringstream oss;

    oss << DataType::FLOAT;
    EXPECT_EQ(oss.str(), "fp32");
    oss.str("");

    oss << DataType::HALF;
    EXPECT_EQ(oss.str(), "fp16");
    oss.str("");

    oss << DataType::BFLOAT16;
    EXPECT_EQ(oss.str(), "bf16");
    oss.str("");

    oss << DataType::DOUBLE;
    EXPECT_EQ(oss.str(), "fp64");
    oss.str("");

    oss << DataType::UINT8;
    EXPECT_EQ(oss.str(), "uint8");
    oss.str("");

    oss << DataType::INT32;
    EXPECT_EQ(oss.str(), "int32");
    oss.str("");

    oss << DataType::NOT_SET;
    EXPECT_EQ(oss.str(), "unknown");
}
