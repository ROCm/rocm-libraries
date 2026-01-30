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

TEST(TestHipblasltUtils, FindDeviceBufferReturnsCorrectBuffer)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{42, reinterpret_cast<void*>(0x1234)}, {99, reinterpret_cast<void*>(0x5678)}};

    auto result = hipblaslt_utils::findDeviceBuffer(99, buffers.data(), 2);
    EXPECT_EQ(result.uid, 99);
    EXPECT_EQ(result.ptr, reinterpret_cast<void*>(0x5678));
}

TEST(TestHipblasltUtils, FindDeviceBufferThrowsIfNotFound)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{1, reinterpret_cast<void*>(0x1111)}};

    EXPECT_THROW(
        hipblaslt_utils::findDeviceBuffer(2, buffers.data(), static_cast<uint32_t>(buffers.size())),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestHipblasltUtils, FindTensorAttributesReturnsCorrectValue)
{
    flatbuffers::FlatBufferBuilder builder1;
    auto attrOffset1 = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder1, 1);
    builder1.Finish(attrOffset1);

    flatbuffers::FlatBufferBuilder builder2;
    auto attrOffset2 = hipdnn_data_sdk::data_objects::CreateTensorAttributesDirect(builder2, 2);
    builder2.Finish(attrOffset2);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
        builder1.GetBufferPointer());
    auto attrPtr2 = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::TensorAttributes>(
        builder2.GetBufferPointer());

    auto attrMap
        = std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>{
            {1, attrPtr1}, {2, attrPtr2}};

    EXPECT_EQ(hipblaslt_utils::findTensorAttributes(attrMap, 1).uid(), 1);
    EXPECT_EQ(hipblaslt_utils::findTensorAttributes(attrMap, 2).uid(), 2);
}

TEST(TestHipblasltUtils, FindTensorAttributesThrowsIfNotFound)
{
    auto attrMap
        = std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>{};

    EXPECT_THROW(hipblaslt_utils::findTensorAttributes(attrMap, 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
