// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_flatbuffers_sdk::utilities::extractValueFromTensorValue;

namespace
{

TensorAttributesT makeBoolValueAttr(bool value)
{
    TensorAttributesT attr;
    attr.uid = 1;
    attr.name = "boolean_value";
    attr.data_type = DataType::BOOLEAN;
    attr.dims = {1};
    attr.strides = {1};
    attr.value.Set(BoolValue(value));
    return attr;
}

} // namespace

TEST(TestFlatbufferUtils, ExtractBoolValueAsBoolTrue)
{
    auto attr = makeBoolValueAttr(true);
    EXPECT_TRUE(extractValueFromTensorValue<bool>(attr, "p"));
}

TEST(TestFlatbufferUtils, ExtractBoolValueAsBoolFalse)
{
    auto attr = makeBoolValueAttr(false);
    EXPECT_FALSE(extractValueFromTensorValue<bool>(attr, "p"));
}

namespace
{

// Builds a scalar TensorAttributes flatbuffer and returns the owning builder
// plus a root pointer, covering the 3 pass-by-value states plus an ordinary
// (non-scalar) data tensor.
flatbuffers::FlatBufferBuilder
    buildTensorAttributes(int64_t uid, bool isRuntimePassByValue, bool withValue)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1};

    flatbuffers::Offset<void> valueOffset = 0;
    TensorValue valueType = TensorValue::NONE;
    if(withValue)
    {
        const Float32Value floatVal(1.0f);
        valueOffset = builder.CreateStruct(floatVal).Union();
        valueType = TensorValue::Float32Value;
    }

    auto attrOffset = CreateTensorAttributesDirect(builder,
                                                   uid,
                                                   "t",
                                                   DataType::FLOAT,
                                                   &dims,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   valueType,
                                                   valueOffset,
                                                   isRuntimePassByValue);
    builder.Finish(attrOffset);
    return builder;
}

} // namespace

using hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor;

TEST(TestFlatbufferUtils, IsPassByValueTensorFalseForOrdinaryDataTensor)
{
    auto builder = buildTensorAttributes(1, /*isRuntimePassByValue=*/false, /*withValue=*/false);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    EXPECT_FALSE(isPassByValueTensor(attr));
}

TEST(TestFlatbufferUtils, IsPassByValueTensorTrueForCompileTimeConstant)
{
    auto builder = buildTensorAttributes(1, /*isRuntimePassByValue=*/false, /*withValue=*/true);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    EXPECT_TRUE(isPassByValueTensor(attr));
}

TEST(TestFlatbufferUtils, IsPassByValueTensorTrueForRuntimeWithDefault)
{
    auto builder = buildTensorAttributes(1, /*isRuntimePassByValue=*/true, /*withValue=*/true);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    EXPECT_TRUE(isPassByValueTensor(attr));
}

TEST(TestFlatbufferUtils, IsPassByValueTensorTrueForRuntimeUserSupplied)
{
    auto builder = buildTensorAttributes(1, /*isRuntimePassByValue=*/true, /*withValue=*/false);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    EXPECT_TRUE(isPassByValueTensor(attr));
}

TEST(TestFlatbufferUtils, IsPassByValueTensorFalseForNullptr)
{
    EXPECT_FALSE(isPassByValueTensor(nullptr));
}
