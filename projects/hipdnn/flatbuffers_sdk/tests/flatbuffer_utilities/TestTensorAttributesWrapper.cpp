// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <set>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/TensorAttributesWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;

TEST(TestTensorAttributesWrapper, NullBufferIsInvalid)
{
    EXPECT_THROW(const TensorAttributesWrapper wrapper(nullptr), std::invalid_argument);
}

TEST(TestTensorAttributesWrapper, EnsureTheTensorAttributesIsWrappedCorrectly)
{
    const int64_t uid = 1;
    const std::string name = "x";
    const DataType dataType = DataType::FLOAT;
    const std::vector<int64_t> dims = {10, 2};
    const std::vector<int64_t> strides = {2, 1};
    const bool isVirtual = false;
    const TensorValue valueType = TensorValue::Float32Value;
    const float value = 1.0f;

    flatbuffers::FlatBufferBuilder builder;
    const Float32Value floatValue(value);
    auto valueOffset = builder.CreateStruct(floatValue).Union();
    auto attributeOffset = CreateTensorAttributesDirect(
        builder, uid, name.c_str(), dataType, &strides, &dims, isVirtual, valueType, valueOffset);
    builder.Finish(attributeOffset);

    auto shallowTensorAttributes
        = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const TensorAttributesWrapper wrapper(shallowTensorAttributes);

    EXPECT_TRUE(wrapper.isValid());
    EXPECT_EQ(wrapper.uid(), uid);
    EXPECT_EQ(wrapper.name(), name);
    EXPECT_EQ(wrapper.dataType(), dataType);
    EXPECT_EQ(wrapper.dims(), dims);
    EXPECT_EQ(wrapper.strides(), strides);
    EXPECT_EQ(wrapper.isVirtual(), isVirtual);
    EXPECT_EQ(wrapper.valueType(), valueType);
    EXPECT_EQ(wrapper.valueType(), TensorValue::Float32Value);
    EXPECT_NO_THROW(wrapper.valueAs<Float32Value>());
    EXPECT_EQ(wrapper.valueAs<Float32Value>().value(), value);
    EXPECT_THROW(wrapper.valueAs<Float64Value>(), std::invalid_argument);
}
