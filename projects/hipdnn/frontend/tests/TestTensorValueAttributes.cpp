// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <limits>
#include <vector>

// using namespace hipdnn_frontend::graph;
using namespace hipdnn_sdk::data_objects;

constexpr float PI_FLOAT = 3.14159265358979323846f;
constexpr double PI_DOUBLE = 3.14159265358979323846;

TEST(TestTensorValueAttributes, SetGetClearFloat)
{
    hipdnn_frontend::graph::TensorAttributes tensor;
    EXPECT_FALSE(tensor.get_pass_by_value());

    constexpr float TEST_VALUE = PI_FLOAT;
    tensor.set_value(TEST_VALUE);
    EXPECT_TRUE(tensor.get_pass_by_value());

    auto opt = tensor.get_pass_by_value<float>();
    ASSERT_TRUE(opt.has_value());
    EXPECT_FLOAT_EQ(opt.value(), TEST_VALUE);

    tensor.clear_value();
    EXPECT_FALSE(tensor.get_pass_by_value());
    EXPECT_FALSE(tensor.get_pass_by_value<float>().has_value());
}

TEST(TestTensorValueAttributes, ConstructorValues)
{
    constexpr float TEST_VALUE = 42.0f;
    hipdnn_frontend::graph::TensorAttributes tensor(TEST_VALUE);

    auto opt = tensor.get_pass_by_value<float>();
    ASSERT_TRUE(opt.has_value());
    EXPECT_EQ(opt.value(), TEST_VALUE);
}

TEST(TestTensorValueAttributes, PackUnpackFloatValue)
{
    hipdnn_frontend::graph::TensorAttributes tensor;
    tensor.set_uid(7)
        .set_name("value_tensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({1, 2})
        .set_dim({3, 4})
        .set_is_virtual(false)
        .set_value(PI_FLOAT);

    flatbuffers::FlatBufferBuilder builder;
    auto fbOffset = tensor.pack_attributes(builder);
    builder.Finish(fbOffset);

    auto bufferPointer = builder.GetBufferPointer();
    auto fbTensor = flatbuffers::GetRoot<TensorAttributes>(bufferPointer);

    EXPECT_EQ(fbTensor->uid(), 7);
    EXPECT_STREQ(fbTensor->name()->c_str(), "value_tensor");
    EXPECT_EQ(fbTensor->data_type(), DataType::FLOAT);
    EXPECT_EQ(fbTensor->strides()->size(), 1u);
    EXPECT_EQ(fbTensor->dims()->size(), 1u);
    EXPECT_FALSE(fbTensor->virtual_());

    EXPECT_EQ(fbTensor->value_type(), TensorValue::Float32Value);
    auto fval = fbTensor->value_as_Float32Value();
    ASSERT_NE(fval, nullptr);
    EXPECT_FLOAT_EQ(fval->value(), PI_FLOAT);

    auto unpacked = std::unique_ptr<TensorAttributesT>(fbTensor->UnPack());
    EXPECT_EQ(unpacked->uid, 7);
    EXPECT_EQ(unpacked->name, "value_tensor");
    EXPECT_EQ(unpacked->data_type, DataType::FLOAT);

    std::vector<int64_t> expectedStrides = {1};
    std::vector<int64_t> expectedDims = {1};
    EXPECT_EQ(unpacked->strides, expectedStrides);
    EXPECT_EQ(unpacked->dims, expectedDims);

    EXPECT_FALSE(unpacked->virtual_);

    ASSERT_EQ(unpacked->value.type, TensorValue::Float32Value);
    auto* floatVal = unpacked->value.AsFloat32Value();
    ASSERT_NE(floatVal, nullptr);
    EXPECT_FLOAT_EQ(floatVal->value(), PI_FLOAT);
}

TEST(TestTensorValueAttributes, PackUnpackHalfValue)
{
    hipdnn_frontend::graph::TensorAttributes tensor;
    tensor.set_uid(8)
        .set_name("half_tensor")
        .set_data_type(hipdnn_frontend::DataType::HALF)
        .set_is_virtual(false)
        .set_value(1.0_h);

    flatbuffers::FlatBufferBuilder builder;
    auto fbOffset = tensor.pack_attributes(builder);
    builder.Finish(fbOffset);

    auto bufferPointer = builder.GetBufferPointer();
    auto fbTensor = flatbuffers::GetRoot<TensorAttributes>(bufferPointer);

    EXPECT_EQ(fbTensor->value_type(), TensorValue::Float16Value);
    auto hval = fbTensor->value_as_Float16Value();
    ASSERT_NE(hval, nullptr);
    EXPECT_EQ(hval->value(), 1.0_h);

    auto unpacked = std::unique_ptr<TensorAttributesT>(fbTensor->UnPack());
    ASSERT_EQ(unpacked->value.type, TensorValue::Float16Value);
    auto* halfVal = unpacked->value.AsFloat16Value();
    ASSERT_NE(halfVal, nullptr);
    EXPECT_EQ(halfVal->value(), 1.0_h);
}

TEST(TestTensorValueAttributes, PackUnpackBFloat1Value)
{
    hipdnn_frontend::graph::TensorAttributes tensor;
    tensor.set_uid(8)
        .set_name("half_tensor")
        .set_data_type(hipdnn_frontend::DataType::BFLOAT16)
        .set_is_virtual(false)
        .set_value(1.0_bf);

    flatbuffers::FlatBufferBuilder builder;
    auto fbOffset = tensor.pack_attributes(builder);
    builder.Finish(fbOffset);

    auto bufferPointer = builder.GetBufferPointer();
    auto fbTensor = flatbuffers::GetRoot<TensorAttributes>(bufferPointer);

    EXPECT_EQ(fbTensor->value_type(), TensorValue::BFloat16Value);
    auto hval = fbTensor->value_as_BFloat16Value();
    ASSERT_NE(hval, nullptr);
    EXPECT_EQ(hval->value(), 1.0_bf);

    auto unpacked = std::unique_ptr<TensorAttributesT>(fbTensor->UnPack());
    ASSERT_EQ(unpacked->value.type, TensorValue::BFloat16Value);
    auto* halfVal = unpacked->value.AsBFloat16Value();
    ASSERT_NE(halfVal, nullptr);
    EXPECT_EQ(halfVal->value(), 1.0_bf);
}

TEST(TestTensorValueAttributes, PackUnpackDoubleValue)
{
    hipdnn_frontend::graph::TensorAttributes tensor;
    tensor.set_uid(9)
        .set_name("double_tensor")
        .set_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_is_virtual(false)
        .set_value(PI_DOUBLE);

    flatbuffers::FlatBufferBuilder builder;
    auto fbOffset = tensor.pack_attributes(builder);
    builder.Finish(fbOffset);

    auto bufferPointer = builder.GetBufferPointer();
    auto fbTensor = flatbuffers::GetRoot<TensorAttributes>(bufferPointer);

    EXPECT_EQ(fbTensor->value_type(), TensorValue::Float64Value);
    auto dval = fbTensor->value_as_Float64Value();
    ASSERT_NE(dval, nullptr);
    EXPECT_DOUBLE_EQ(dval->value(), PI_DOUBLE);

    auto unpacked = std::unique_ptr<TensorAttributesT>(fbTensor->UnPack());
    ASSERT_EQ(unpacked->value.type, TensorValue::Float64Value);
    auto* doubleVal = unpacked->value.AsFloat64Value();
    ASSERT_NE(doubleVal, nullptr);
    EXPECT_DOUBLE_EQ(doubleVal->value(), PI_DOUBLE);
}

TEST(TestTensorValueAttributes, PackUnpackEmptyValue)
{
    hipdnn_frontend::graph::TensorAttributes tensor;
    tensor.set_uid(10)
        .set_name("empty_tensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_stride({1, 2})
        .set_dim({3, 4})
        .set_is_virtual(true);

    EXPECT_FALSE(tensor.get_pass_by_value());

    flatbuffers::FlatBufferBuilder builder;
    auto fbOffset = tensor.pack_attributes(builder);
    builder.Finish(fbOffset);

    auto bufferPointer = builder.GetBufferPointer();
    auto fbTensor = flatbuffers::GetRoot<TensorAttributes>(bufferPointer);

    EXPECT_EQ(fbTensor->value_type(), TensorValue::NONE);

    auto unpacked = std::unique_ptr<TensorAttributesT>(fbTensor->UnPack());
    EXPECT_EQ(unpacked->value.type, TensorValue::NONE);
}

TEST(TestTensorValueAttributes, TypeSafety)
{
    hipdnn_frontend::graph::TensorAttributes tensor;
    tensor.set_value(42.0f);

    auto floatOpt = tensor.get_pass_by_value<float>();
    ASSERT_TRUE(floatOpt.has_value());
    EXPECT_FLOAT_EQ(floatOpt.value(), 42.0f);

    EXPECT_FALSE(tensor.get_pass_by_value<half>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<hip_bfloat16>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<uint8_t>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<int32_t>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<double>().has_value());

    tensor.set_value(int32_t{123});

    EXPECT_FALSE(tensor.get_pass_by_value<float>().has_value());

    auto intOpt = tensor.get_pass_by_value<int32_t>();
    ASSERT_TRUE(intOpt.has_value());
    EXPECT_EQ(intOpt.value(), 123);
}

TEST(TestTensorValueAttributes, NumericLimits)
{
    hipdnn_frontend::graph::TensorAttributes tensor;

    tensor.set_value(std::numeric_limits<float>::max());
    auto floatOpt = tensor.get_pass_by_value<float>();
    ASSERT_TRUE(floatOpt.has_value());
    EXPECT_FLOAT_EQ(floatOpt.value(), std::numeric_limits<float>::max());

    tensor.set_value(std::numeric_limits<int32_t>::min());
    auto intOpt = tensor.get_pass_by_value<int32_t>();
    ASSERT_TRUE(intOpt.has_value());
    EXPECT_EQ(intOpt.value(), std::numeric_limits<int32_t>::min());

    tensor.set_value(std::numeric_limits<uint8_t>::max());
    auto uint8Opt = tensor.get_pass_by_value<uint8_t>();
    ASSERT_TRUE(uint8Opt.has_value());
    EXPECT_EQ(uint8Opt.value(), std::numeric_limits<uint8_t>::max());

    tensor.set_value(std::numeric_limits<double>::infinity());

    flatbuffers::FlatBufferBuilder builder;
    auto fbOffset = tensor.pack_attributes(builder);
    builder.Finish(fbOffset);

    auto bufferPointer = builder.GetBufferPointer();
    auto fbTensor = flatbuffers::GetRoot<TensorAttributes>(bufferPointer);

    auto dval = fbTensor->value_as_Float64Value();
    ASSERT_NE(dval, nullptr);
    EXPECT_TRUE(std::isinf(dval->value()));
}
