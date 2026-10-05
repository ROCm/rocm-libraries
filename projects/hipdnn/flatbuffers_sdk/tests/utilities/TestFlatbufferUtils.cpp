// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cfenv>
#include <cmath>
#include <cstring>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <limits>

#include <stdexcept>
#include <unordered_map>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_flatbuffers_sdk::utilities::extractFiniteFloatFromTensorValue;
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

namespace
{

template <typename Value>
TensorAttributesT makeStoredScalar(DataType dtype, Value value)
{
    TensorAttributesT attr;
    attr.data_type = dtype;
    attr.dims = {1};
    attr.strides = {1};
    attr.value.Set(value);
    return attr;
}

uint32_t floatBits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void expectFiniteValue(const TensorAttributesT& attr, float expected)
{
    SCOPED_TRACE(EnumNameDataType(attr.data_type));
    EXPECT_EQ(floatBits(extractFiniteFloatFromTensorValue(attr, "scalar")), floatBits(expected));
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateTensorAttributes(builder, &attr));
    const auto* table = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    EXPECT_EQ(floatBits(extractFiniteFloatFromTensorValue(table, "scalar")), floatBits(expected));
}

void expectInvalidFiniteValue(const TensorAttributesT& attr)
{
    SCOPED_TRACE(EnumNameDataType(attr.data_type));
    EXPECT_THROW(extractFiniteFloatFromTensorValue(attr, "scalar"), std::runtime_error);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateTensorAttributes(builder, &attr));
    const auto* table = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    EXPECT_THROW(extractFiniteFloatFromTensorValue(table, "scalar"), std::runtime_error);
}

template <typename Target>
void expectGenericValue(const TensorAttributesT& attr, Target expected)
{
    EXPECT_EQ(extractValueFromTensorValue<Target>(attr, "scalar"), expected);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateTensorAttributes(builder, &attr));
    const auto* table = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    EXPECT_EQ(extractValueFromTensorValue<Target>(table, "scalar"), expected);
}

} // namespace

TEST(TestFlatbufferUtils, ExtractSignedInt8BeforeWidening)
{
    for(const auto bits : {0xff, 0x80, 0x7f})
    {
        const auto attr = makeStoredScalar(DataType::INT8, Float8Value(static_cast<uint8_t>(bits)));
        const auto expected = bits < 128 ? bits : bits - 256;
        expectGenericValue(attr, static_cast<double>(expected));
        expectGenericValue(attr, static_cast<int64_t>(expected));
        expectFiniteValue(attr, static_cast<float>(expected));
    }
    const auto unsignedAttr = makeStoredScalar(DataType::UINT8, Float8Value(0xff));
    expectGenericValue(unsignedAttr, 255.0);
    expectFiniteValue(unsignedAttr, 255.0f);
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatUsesNumericalPayloadForEveryUnionArm)
{
    expectFiniteValue(makeStoredScalar(DataType::FLOAT, Float32Value(1.0001f)), 1.0001f);
    // HALF/BFLOAT16 store float32 values here; do not round them to 16 bits.
    expectFiniteValue(makeStoredScalar(DataType::HALF, Float16Value(1.0001f)), 1.0001f);
    expectFiniteValue(makeStoredScalar(DataType::BFLOAT16, BFloat16Value(1.0001f)), 1.0001f);
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(16777217.0)), 16777216.0f);
    expectFiniteValue(makeStoredScalar(DataType::INT32, Int32Value(16777217)), 16777216.0f);
    expectFiniteValue(makeStoredScalar(DataType::INT64, Int64Value(16777219)), 16777220.0f);
    expectFiniteValue(makeStoredScalar(DataType::BOOLEAN, BoolValue(false)), 0.0f);
    expectFiniteValue(makeStoredScalar(DataType::BOOLEAN, BoolValue(true)), 1.0f);
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatDecodesAllFp8Formats)
{
    struct Case
    {
        DataType dtype;
        uint8_t bits;
        float expected;
    };
    for(const auto& test : {Case{DataType::FP8_E4M3, 0x7e, 448.0f},
                            Case{DataType::FP8_E4M3, 0xfe, -448.0f},
                            Case{DataType::FP8_E4M3, 0x80, -0.0f},
                            Case{DataType::FP8_E5M2, 0x7b, 57344.0f},
                            Case{DataType::FP8_E5M2, 0xfb, -57344.0f},
                            Case{DataType::FP8_E4M3_FNUZ, 0x7f, 240.0f},
                            Case{DataType::FP8_E4M3_FNUZ, 0xff, -240.0f},
                            Case{DataType::FP8_E5M2_FNUZ, 0x7f, 57344.0f},
                            Case{DataType::FP8_E5M2_FNUZ, 0xff, -57344.0f}})
    {
        const auto attr = makeStoredScalar(test.dtype, Float8Value(test.bits));
        expectFiniteValue(attr, test.expected);
        expectGenericValue(attr, static_cast<double>(test.expected));
    }
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatRejectsNonfiniteFp8)
{
    expectInvalidFiniteValue(makeStoredScalar(DataType::FP8_E4M3, Float8Value(0x7f)));
    expectInvalidFiniteValue(makeStoredScalar(DataType::FP8_E4M3, Float8Value(0xff)));
    expectInvalidFiniteValue(makeStoredScalar(DataType::FP8_E5M2, Float8Value(0x7c)));
    expectInvalidFiniteValue(makeStoredScalar(DataType::FP8_E5M2, Float8Value(0xfc)));
    expectInvalidFiniteValue(makeStoredScalar(DataType::FP8_E5M2, Float8Value(0x7f)));
    expectInvalidFiniteValue(makeStoredScalar(DataType::FP8_E4M3_FNUZ, Float8Value(0x80)));
    expectInvalidFiniteValue(makeStoredScalar(DataType::FP8_E5M2_FNUZ, Float8Value(0x80)));
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatRoundsInt64WithoutDoubleStaging)
{
    constexpr int64_t MIDPOINT = (int64_t{1} << 62) + (int64_t{1} << 38);
    expectFiniteValue(makeStoredScalar(DataType::INT64, Int64Value(MIDPOINT - 1)), 0x1p62f);
    expectFiniteValue(makeStoredScalar(DataType::INT64, Int64Value(MIDPOINT)), 0x1p62f);
    expectFiniteValue(makeStoredScalar(DataType::INT64, Int64Value(MIDPOINT + 1)), 0x1.000002p62f);
    expectFiniteValue(makeStoredScalar(DataType::INT64, Int64Value(-MIDPOINT - 1)),
                      -0x1.000002p62f);
    expectFiniteValue(
        makeStoredScalar(DataType::INT64, Int64Value(std::numeric_limits<int64_t>::max())),
        0x1p63f);
    expectFiniteValue(
        makeStoredScalar(DataType::INT64, Int64Value(std::numeric_limits<int64_t>::min())),
        -0x1p63f);
    expectGenericValue(makeStoredScalar(DataType::INT64, Int64Value(MIDPOINT + 1)), MIDPOINT + 1);
    expectGenericValue(makeStoredScalar(DataType::DOUBLE, Float64Value(16777217.0)), 16777217.0);
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatDoubleRoundingBoundaries)
{
    constexpr double MIDPOINT = 1.0 + 0x1p-24;
    expectFiniteValue(
        makeStoredScalar(DataType::DOUBLE, Float64Value(std::nextafter(MIDPOINT, 0.0))), 1.0f);
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(MIDPOINT)), 1.0f);
    expectFiniteValue(
        makeStoredScalar(DataType::DOUBLE, Float64Value(std::nextafter(MIDPOINT, 2.0))),
        0x1.000002p0f);
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(1.0 + 3.0 * 0x1p-24)),
                      0x1.000004p0f);
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(2.0 - 0x1p-24)), 2.0f);

    const auto maxFloat = static_cast<double>(std::numeric_limits<float>::max());
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(maxFloat)),
                      std::numeric_limits<float>::max());
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(-maxFloat)),
                      -std::numeric_limits<float>::max());
    // Reject out-of-range doubles even if rounding would produce FLT_MAX.
    expectInvalidFiniteValue(makeStoredScalar(
        DataType::DOUBLE,
        Float64Value(std::nextafter(maxFloat, std::numeric_limits<double>::infinity()))));
    expectInvalidFiniteValue(makeStoredScalar(
        DataType::DOUBLE,
        Float64Value(std::nextafter(-maxFloat, -std::numeric_limits<double>::infinity()))));
    expectInvalidFiniteValue(
        makeStoredScalar(DataType::DOUBLE, Float64Value(std::numeric_limits<double>::max())));
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatPreservesZerosAndSubnormals)
{
    for(const float value : {0.0f, -0.0f, 0x1p-149f, -0x1p-149f})
    {
        expectFiniteValue(makeStoredScalar(DataType::FLOAT, Float32Value(value)), value);
        expectFiniteValue(makeStoredScalar(DataType::HALF, Float16Value(value)), value);
        expectFiniteValue(makeStoredScalar(DataType::BFLOAT16, BFloat16Value(value)), value);
        expectFiniteValue(
            makeStoredScalar(DataType::DOUBLE, Float64Value(static_cast<double>(value))), value);
    }
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(0x1p-150)), 0.0f);
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(-0x1p-150)), -0.0f);
    expectFiniteValue(
        makeStoredScalar(DataType::DOUBLE, Float64Value(std::nextafter(0x1p-150, 1.0))), 0x1p-149f);
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(3.0 * 0x1p-150)), 0x1p-148f);
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(0x1p-126 - 0x1p-150)),
                      0x1p-126f);
    expectFiniteValue(
        makeStoredScalar(DataType::DOUBLE, Float64Value(std::numeric_limits<double>::denorm_min())),
        0.0f);
    expectFiniteValue(makeStoredScalar(DataType::DOUBLE,
                                       Float64Value(-std::numeric_limits<double>::denorm_min())),
                      -0.0f);
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatRejectsNonfiniteNumericalPayloads)
{
    for(const auto value : {std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()})
    {
        expectInvalidFiniteValue(makeStoredScalar(DataType::FLOAT, Float32Value(value)));
        expectInvalidFiniteValue(makeStoredScalar(DataType::HALF, Float16Value(value)));
        expectInvalidFiniteValue(makeStoredScalar(DataType::BFLOAT16, BFloat16Value(value)));
        expectInvalidFiniteValue(
            makeStoredScalar(DataType::DOUBLE, Float64Value(static_cast<double>(value))));
    }
    // Generic scalar reads allow nonfinite values; publication rejects them.
    expectGenericValue(
        makeStoredScalar(DataType::DOUBLE, Float64Value(std::numeric_limits<double>::infinity())),
        std::numeric_limits<double>::infinity());
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatIgnoresHostRoundingMode)
{
    struct RestoreRounding
    {
        int mode = std::fegetround();
        ~RestoreRounding()
        {
            std::fesetround(mode);
        }
    } const restore;
    ASSERT_NE(restore.mode, -1);
    for(const auto mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
    {
        ASSERT_EQ(std::fesetround(mode), 0);
        expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(16777217.0)),
                          16777216.0f);
        expectFiniteValue(makeStoredScalar(DataType::DOUBLE, Float64Value(0x1p-150)), 0.0f);
        expectFiniteValue(makeStoredScalar(DataType::INT32, Int32Value(-16777217)), -16777216.0f);
        expectFiniteValue(makeStoredScalar(DataType::INT64,
                                           Int64Value((int64_t{1} << 62) + (int64_t{1} << 38) + 1)),
                          0x1.000002p62f);
    }
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatRejectsUnsupportedAndMismatchedPairs)
{
    std::vector<TensorAttributesT> arms;
    arms.push_back(makeStoredScalar(DataType::FLOAT, Float32Value(1.0f)));
    arms.push_back(makeStoredScalar(DataType::HALF, Float16Value(1.0f)));
    arms.push_back(makeStoredScalar(DataType::BFLOAT16, BFloat16Value(1.0f)));
    arms.push_back(makeStoredScalar(DataType::UINT8, Float8Value(1)));
    arms.push_back(makeStoredScalar(DataType::INT32, Int32Value(1)));
    arms.push_back(makeStoredScalar(DataType::DOUBLE, Float64Value(1.0)));
    arms.push_back(makeStoredScalar(DataType::INT64, Int64Value(1)));
    arms.push_back(makeStoredScalar(DataType::BOOLEAN, BoolValue(true)));
    for(auto& attr : arms)
    {
        const auto matchingType = attr.data_type;
        for(const auto dtype : EnumValuesDataType())
        {
            if(dtype == matchingType
               || (attr.value.type == TensorValue::Float8Value
                   && (dtype == DataType::INT8 || dtype == DataType::FP8_E4M3
                       || dtype == DataType::FP8_E5M2 || dtype == DataType::FP8_E4M3_FNUZ
                       || dtype == DataType::FP8_E5M2_FNUZ)))
            {
                continue;
            }
            attr.data_type = dtype;
            expectInvalidFiniteValue(attr);
            EXPECT_THROW(extractValueFromTensorValue<double>(attr, "scalar"), std::runtime_error);
        }
    }
    expectInvalidFiniteValue(makeStoredScalar(static_cast<DataType>(127), Float32Value(1.0f)));
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatRejectsMissingPayloadAndInvalidUnion)
{
    EXPECT_THROW(extractFiniteFloatFromTensorValue(nullptr, "scalar"), std::runtime_error);
    EXPECT_THROW(extractValueFromTensorValue<double>(nullptr, "scalar"), std::runtime_error);
    TensorAttributesT missing;
    missing.data_type = DataType::FLOAT;
    expectInvalidFiniteValue(missing);
    missing.value.type = TensorValue::Float32Value;
    EXPECT_THROW(extractFiniteFloatFromTensorValue(missing, "scalar"), std::runtime_error);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(CreateTensorAttributesDirect(builder,
                                                0,
                                                nullptr,
                                                DataType::FLOAT,
                                                nullptr,
                                                nullptr,
                                                false,
                                                TensorValue::Float32Value,
                                                0));
    EXPECT_THROW(extractFiniteFloatFromTensorValue(
                     flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer()), "scalar"),
                 std::runtime_error);

    auto attr = makeStoredScalar(DataType::FLOAT, Float32Value(1.0f));
    for(const auto invalidType : {TensorValue::NONE, static_cast<TensorValue>(255)})
    {
        attr.value.type = invalidType;
        EXPECT_THROW(extractFiniteFloatFromTensorValue(attr, "scalar"), std::runtime_error);
        builder.Clear();
        const auto payload = builder.CreateStruct(Float32Value(1.0f)).Union();
        builder.Finish(CreateTensorAttributesDirect(
            builder, 0, nullptr, DataType::FLOAT, nullptr, nullptr, false, invalidType, payload));
        EXPECT_THROW(
            extractFiniteFloatFromTensorValue(
                flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer()), "scalar"),
            std::runtime_error);
    }
    attr.value.type = TensorValue::Float32Value; // Restore the tag so the union can free its value.
}

TEST(TestFlatbufferUtils, ExtractFiniteFloatValidatesStoredRuntimeDefault)
{
    auto attr = makeScalarAttr(7, true, true);
    expectFiniteValue(attr, 2.0f);
    attr.value.Set(Float32Value(std::numeric_limits<float>::infinity()));
    expectInvalidFiniteValue(attr);
    expectInvalidFiniteValue(makeScalarAttr(7, true, false));
}
