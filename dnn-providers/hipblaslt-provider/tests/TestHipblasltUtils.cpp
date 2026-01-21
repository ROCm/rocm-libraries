// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltUtils.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

using namespace hipblaslt_plugin;
using namespace hipblaslt_utils;

TEST(TestHipblasltUtils, TensorDataTypeToHipblasltDataType)
{
    using namespace hipdnn_data_sdk::data_objects;

    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::FLOAT), HIP_R_32F);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::INT32), HIP_R_32I);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::HALF), HIP_R_16F);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::BFLOAT16), HIP_R_16BF);
    EXPECT_EQ(hipblaslt_utils::tensorDataTypeToHipDataType(DataType::INT8), HIP_R_8I);
}

TEST(TestHipblasltUtils, TensorDataTypeToHipblasltDataTypeThrowsOnUnsupported)
{
    // Use a value not in the enum
    EXPECT_THROW(hipblaslt_utils::tensorDataTypeToHipDataType(
                     static_cast<hipdnn_data_sdk::data_objects::DataType>(-1)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
