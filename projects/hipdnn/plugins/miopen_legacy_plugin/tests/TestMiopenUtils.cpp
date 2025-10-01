// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenUtils.hpp"
#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>

using namespace miopen_legacy_plugin;

TEST(TestMiopenUtils, FindDeviceBufferReturnsCorrectBuffer)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{42, reinterpret_cast<void*>(0x1234)}, {99, reinterpret_cast<void*>(0x5678)}};

    auto result = miopen_utils::findDeviceBuffer(99, buffers.data(), 2);
    EXPECT_EQ(result.uid, 99);
    EXPECT_EQ(result.ptr, reinterpret_cast<void*>(0x5678));
}

TEST(TestMiopenUtils, FindDeviceBufferThrowsIfNotFound)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{1, reinterpret_cast<void*>(0x1111)}};

    EXPECT_THROW(
        miopen_utils::findDeviceBuffer(2, buffers.data(), static_cast<uint32_t>(buffers.size())),
        hipdnn_plugin::HipdnnPluginException);
}

TEST(TestMiopenUtils, TensorDataTypeToMiopenDataType)
{
    using namespace hipdnn_sdk::data_objects;

    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType::FLOAT), miopenFloat);
    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType::HALF), miopenHalf);
    EXPECT_EQ(miopen_utils::tensorDataTypeToMiopenDataType(DataType::BFLOAT16), miopenBFloat16);
}

TEST(TestMiopenUtils, TensorDataTypeToMiopenDataTypeThrowsOnUnsupported)
{
    // Use a value not in the enum
    EXPECT_THROW(miopen_utils::tensorDataTypeToMiopenDataType(
                     static_cast<hipdnn_sdk::data_objects::DataType>(-1)),
                 hipdnn_plugin::HipdnnPluginException);
}

TEST(TestMiopenUtils, FindTensorAttributesReturnsCorrectValue)
{
    flatbuffers::FlatBufferBuilder builder1;
    auto attrOffset1 = hipdnn_sdk::data_objects::CreateTensorAttributesDirect(builder1, 1);
    builder1.Finish(attrOffset1);

    flatbuffers::FlatBufferBuilder builder2;
    auto attrOffset2 = hipdnn_sdk::data_objects::CreateTensorAttributesDirect(builder2, 2);
    builder2.Finish(attrOffset2);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_sdk::data_objects::TensorAttributes>(
        builder1.GetBufferPointer());
    auto attrPtr2 = flatbuffers::GetRoot<hipdnn_sdk::data_objects::TensorAttributes>(
        builder2.GetBufferPointer());

    auto attrMap = std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>{
        {1, attrPtr1}, {2, attrPtr2}};

    EXPECT_EQ(miopen_utils::findTensorAttributes(attrMap, 1).uid(), 1);
    EXPECT_EQ(miopen_utils::findTensorAttributes(attrMap, 2).uid(), 2);
}

TEST(TestMiopenUtils, FindTensorAttributesThrowsIfNotFound)
{
    auto attrMap = std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>{};

    EXPECT_THROW(miopen_utils::findTensorAttributes(attrMap, 1),
                 hipdnn_plugin::HipdnnPluginException);
}

TEST(TestMiopenUtils, GetSpatialDimCountReturnsCorrectValue)
{
    std::vector<int64_t> dims = {1, 1, 1, 1, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_sdk::data_objects::DataType::UNSET, nullptr, &dims);
    builder.Finish(attrOffset);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    EXPECT_EQ(miopen_utils::getSpatialDimCount(*attrPtr1), 3);
}

TEST(TestMiopenUtils, GetSpatialDimCountThrowsOnInvalidDims)
{
    std::vector<int64_t> dims = {1, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "", hipdnn_sdk::data_objects::DataType::UNSET, nullptr, &dims);
    builder.Finish(attrOffset);

    auto attrPtr1 = flatbuffers::GetRoot<hipdnn_sdk::data_objects::TensorAttributes>(
        builder.GetBufferPointer());

    EXPECT_THROW(miopen_utils::getSpatialDimCount(*attrPtr1), hipdnn_plugin::HipdnnPluginException);
}
