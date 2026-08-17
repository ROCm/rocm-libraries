// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

#include <stdexcept>
#include <unordered_map>

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

namespace
{

// Builds a scalar FLOAT TensorAttributesT with the given pass-by-value state.
// When withValue is true a baked 2.0f default is set; otherwise the value union
// is left empty (pure runtime user-supplied).
TensorAttributesT makeScalarAttr(int64_t uid, bool isRuntimePassByValue, bool withValue)
{
    TensorAttributesT attr;
    attr.uid = uid;
    attr.name = "scalar";
    attr.data_type = DataType::FLOAT;
    attr.dims = {1};
    attr.strides = {1};
    attr.is_runtime_pass_by_value = isRuntimePassByValue;
    if(withValue)
    {
        attr.value.Set(Float32Value(2.0f));
    }
    return attr;
}

} // namespace

using hipdnn_flatbuffers_sdk::utilities::resolveDoubleScalarFromVariantPack;
using hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack;

TEST(TestFlatbufferUtils, ResolveScalarBakedValueIgnoresPack)
{
    auto attr = makeScalarAttr(7, /*isRuntimePassByValue=*/false, /*withValue=*/true);
    float differing = 99.0f;
    const std::unordered_map<int64_t, void*> pack{{7, &differing}};
    EXPECT_DOUBLE_EQ(resolveDoubleScalarFromVariantPack(attr, pack, "Epsilon"), 2.0);
}

TEST(TestFlatbufferUtils, ResolveScalarRuntimeWithDefaultIgnoresPack)
{
    auto attr = makeScalarAttr(7, /*isRuntimePassByValue=*/true, /*withValue=*/true);
    float differing = 99.0f;
    const std::unordered_map<int64_t, void*> pack{{7, &differing}};
    EXPECT_DOUBLE_EQ(resolveDoubleScalarFromVariantPack(attr, pack, "Epsilon"), 2.0);
}

TEST(TestFlatbufferUtils, ResolvePureRuntimeScalarReadsPack)
{
    auto attr = makeScalarAttr(7, /*isRuntimePassByValue=*/true, /*withValue=*/false);
    float hostValue = 1e-5f;
    const std::unordered_map<int64_t, void*> pack{{7, &hostValue}};
    EXPECT_FLOAT_EQ(resolveScalarFromVariantPack<float>(attr, pack, "Epsilon"), 1e-5f);
}

TEST(TestFlatbufferUtils, ResolvePureRuntimeScalarMissingSlotThrows)
{
    auto attr = makeScalarAttr(7, /*isRuntimePassByValue=*/true, /*withValue=*/false);
    const std::unordered_map<int64_t, void*> pack; // no slot for uid 7
    EXPECT_THROW(resolveScalarFromVariantPack<float>(attr, pack, "Epsilon"), std::runtime_error);
}

TEST(TestFlatbufferUtils, ResolvePureRuntimeScalarNullPointerThrows)
{
    auto attr = makeScalarAttr(7, /*isRuntimePassByValue=*/true, /*withValue=*/false);
    const std::unordered_map<int64_t, void*> pack{{7, nullptr}};
    EXPECT_THROW(resolveScalarFromVariantPack<float>(attr, pack, "Epsilon"), std::runtime_error);
}
