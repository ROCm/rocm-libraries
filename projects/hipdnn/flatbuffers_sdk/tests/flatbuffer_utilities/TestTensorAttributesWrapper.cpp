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

namespace
{
// Builds a TensorAttributes flatbuffer across the (runtime flag) x (value present)
// matrix and returns the finished builder so the buffer stays alive for the caller.
flatbuffers::FlatBufferBuilder buildTensorAttributes(bool withValue, bool isRuntimePassByValue)
{
    const int64_t uid = 42;
    const std::string name = "s";
    const DataType dataType = DataType::FLOAT;
    const std::vector<int64_t> dims = {1};
    const std::vector<int64_t> strides = {1};
    const bool isVirtual = false;

    flatbuffers::FlatBufferBuilder builder;
    const TensorValue valueType = withValue ? TensorValue::Float32Value : TensorValue::NONE;
    flatbuffers::Offset<void> valueOffset = 0;
    if(withValue)
    {
        const Float32Value floatValue(1.0f);
        valueOffset = builder.CreateStruct(floatValue).Union();
    }
    const auto attributeOffset = CreateTensorAttributesDirect(builder,
                                                              uid,
                                                              name.c_str(),
                                                              dataType,
                                                              &strides,
                                                              &dims,
                                                              isVirtual,
                                                              valueType,
                                                              valueOffset,
                                                              isRuntimePassByValue);
    builder.Finish(attributeOffset);
    return builder;
}
} // namespace

// Quadrant 1: flag=false, value=NONE -> ordinary tensor (not by value).
TEST(TestTensorAttributesWrapper, OrdinaryTensorIsNotByValue)
{
    const flatbuffers::FlatBufferBuilder builder = buildTensorAttributes(false, false);
    const auto* shallow = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    const TensorAttributesWrapper wrapper(shallow);

    EXPECT_FALSE(wrapper.isByValue());
    EXPECT_FALSE(wrapper.isRuntimePassByValue());
    EXPECT_FALSE(wrapper.hasCompileTimeConstant());
}

// Quadrant 2: flag=false, value present -> compile-time constant.
TEST(TestTensorAttributesWrapper, CompileTimeConstantIsByValueWithConstant)
{
    const flatbuffers::FlatBufferBuilder builder = buildTensorAttributes(true, false);
    const auto* shallow = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    const TensorAttributesWrapper wrapper(shallow);

    EXPECT_TRUE(wrapper.isByValue());
    EXPECT_FALSE(wrapper.isRuntimePassByValue());
    EXPECT_TRUE(wrapper.hasCompileTimeConstant());
}

// Quadrant 3: flag=true, value=NONE -> runtime user-supplied.
TEST(TestTensorAttributesWrapper, RuntimeUserSuppliedIsByValueNoConstant)
{
    const flatbuffers::FlatBufferBuilder builder = buildTensorAttributes(false, true);
    const auto* shallow = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    const TensorAttributesWrapper wrapper(shallow);

    EXPECT_TRUE(wrapper.isByValue());
    EXPECT_TRUE(wrapper.isRuntimePassByValue());
    EXPECT_FALSE(wrapper.hasCompileTimeConstant());
}

// Seam contract (RFC 0016 §4.1): the wrapper's isByValue() umbrella
// (value present || runtime flag) intentionally DIVERGES from the backend C-API
// HIPDNN_ATTR_TENSOR_IS_BY_VALUE (1307), which derives value-presence only
// (value_type() != NONE). For a pure runtime user-supplied tensor the two must
// disagree: wrapper isByValue()==true, C-API basis (value-presence)==false.
// The C-API getter side is covered by
// IntegrationTensorDescriptorApi.FlagTrueWithoutValueIsIndependent; this pins
// the wrapper side against the same tensor shape so an "alignment" refactor that
// collapsed the two would break here.
TEST(TestTensorAttributesWrapper, PureRuntimeIsByValueDivergesFromCApiValuePresence)
{
    const flatbuffers::FlatBufferBuilder builder = buildTensorAttributes(false, true);
    const auto* shallow = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    const TensorAttributesWrapper wrapper(shallow);

    // Wrapper umbrella: this IS a by-value tensor.
    EXPECT_TRUE(wrapper.isByValue());
    // C-API 1307 basis: value-presence only -> false (no baked value).
    EXPECT_EQ(wrapper.valueType(), TensorValue::NONE);
    // The runtime bit (C-API 1308 basis) is what is actually stored.
    EXPECT_TRUE(wrapper.isRuntimePassByValue());
}

// Quadrant 4: flag=true, value present -> runtime with default.
TEST(TestTensorAttributesWrapper, RuntimeWithDefaultIsByValueNoConstant)
{
    const flatbuffers::FlatBufferBuilder builder = buildTensorAttributes(true, true);
    const auto* shallow = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    const TensorAttributesWrapper wrapper(shallow);

    EXPECT_TRUE(wrapper.isByValue());
    EXPECT_TRUE(wrapper.isRuntimePassByValue());
    EXPECT_FALSE(wrapper.hasCompileTimeConstant());
}
