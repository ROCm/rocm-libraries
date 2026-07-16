// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <limits>
#include <variant>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// Builds a scalar TensorAttributes flatbuffer and returns the owning builder plus the
// finalized root pointer. Mirrors the compile-time-constant / runtime-with-default /
// pure-runtime-user-supplied states.
flatbuffers::FlatBufferBuilder buildScalarTensorAttributes(int64_t uid,
                                                           DataType dataType,
                                                           bool isRuntimePassByValue,
                                                           std::optional<float> floatValue)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1};

    flatbuffers::Offset<void> valueOffset = 0;
    TensorValue valueType = TensorValue::NONE;
    if(floatValue.has_value())
    {
        const Float32Value floatVal(*floatValue);
        valueOffset = builder.CreateStruct(floatVal).Union();
        valueType = TensorValue::Float32Value;
    }

    auto attrOffset = CreateTensorAttributesDirect(builder,
                                                   uid,
                                                   "scalar",
                                                   dataType,
                                                   &dims,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   valueType,
                                                   valueOffset,
                                                   isRuntimePassByValue);
    builder.Finish(attrOffset);
    return builder;
}

// Builds a scalar TensorAttributes flatbuffer with a fully-formed union value of
// the given type, for exercising makeScalarOperand's per-dtype baked switch beyond
// the FLOAT-only coverage buildScalarTensorAttributes provides.
template <typename StructT>
flatbuffers::FlatBufferBuilder buildScalarTensorAttributesWithValue(int64_t uid,
                                                                    DataType dataType,
                                                                    bool isRuntimePassByValue,
                                                                    TensorValue valueType,
                                                                    const StructT& value)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1};

    const auto valueOffset = builder.CreateStruct(value).Union();

    auto attrOffset = CreateTensorAttributesDirect(builder,
                                                   uid,
                                                   "scalar",
                                                   dataType,
                                                   &dims,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   valueType,
                                                   valueOffset,
                                                   isRuntimePassByValue);
    builder.Finish(attrOffset);
    return builder;
}

// Builds a scalar TensorAttributes flatbuffer with UNSET data type or no union
// value at all, for exercising makeScalarOperand's error paths.
flatbuffers::FlatBufferBuilder buildScalarTensorAttributesNoValue(int64_t uid, DataType dataType)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1};

    auto attrOffset = CreateTensorAttributesDirect(builder,
                                                   uid,
                                                   "scalar",
                                                   dataType,
                                                   &dims,
                                                   &dims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   0,
                                                   /*is_runtime_pass_by_value=*/false);
    builder.Finish(attrOffset);
    return builder;
}

// Asserts that invoking fn throws HipdnnPluginException carrying the expected
// status code -- not merely that some HipdnnPluginException was thrown.
template <typename Fn>
void expectPluginThrowWithStatus(Fn&& fn, hipdnnPluginStatus_t expectedStatus)
{
    try
    {
        std::forward<Fn>(fn)();
        ADD_FAILURE() << "Expected HipdnnPluginException, but no exception was thrown";
    }
    catch(const HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), expectedStatus);
    }
}

} // namespace

// --- makeScalarOperand ---

TEST(TestRuntimePassByValue, MakeScalarOperandCompileTimeConstantBakesValue)
{
    auto builder = buildScalarTensorAttributes(1, DataType::FLOAT, false, 1e-5f);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{1, attr}};
    auto op = makeScalarOperand(tensorMap, 1, "Epsilon");

    EXPECT_EQ(op.uid, 1);
    EXPECT_FALSE(op.isRuntimeUserSupplied);
    EXPECT_NEAR(std::get<float>(op.bakedDefault), 1e-5f, 1e-7f);
}

TEST(TestRuntimePassByValue, MakeScalarOperandRuntimeWithDefaultBakesValue)
{
    auto builder = buildScalarTensorAttributes(2, DataType::FLOAT, true, 1e-3f);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{2, attr}};
    auto op = makeScalarOperand(tensorMap, 2, "Epsilon");

    EXPECT_EQ(op.uid, 2);
    EXPECT_FALSE(op.isRuntimeUserSupplied);
    EXPECT_NEAR(std::get<float>(op.bakedDefault), 1e-3f, 1e-6f);
}

TEST(TestRuntimePassByValue, MakeScalarOperandPureRuntimeUserSuppliedDefersRead)
{
    auto builder = buildScalarTensorAttributes(3, DataType::FLOAT, true, std::nullopt);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{3, attr}};
    auto op = makeScalarOperand(tensorMap, 3, "Epsilon");

    EXPECT_EQ(op.uid, 3);
    EXPECT_TRUE(op.isRuntimeUserSupplied);
    EXPECT_EQ(op.dataType, DataType::FLOAT);
}

TEST(TestRuntimePassByValue, MakeScalarOperandBakesDoubleValue)
{
    auto builder = buildScalarTensorAttributesWithValue(
        21, DataType::DOUBLE, false, TensorValue::Float64Value, Float64Value(2.5));
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{21, attr}};
    auto op = makeScalarOperand(tensorMap, 21, "Epsilon");

    EXPECT_FALSE(op.isRuntimeUserSupplied);
    EXPECT_NEAR(std::get<double>(op.bakedDefault), 2.5, 1e-12);
}

TEST(TestRuntimePassByValue, MakeScalarOperandBakesHalfValue)
{
    auto builder = buildScalarTensorAttributesWithValue(
        22, DataType::HALF, false, TensorValue::Float16Value, Float16Value(0.5f));
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{22, attr}};
    auto op = makeScalarOperand(tensorMap, 22, "Epsilon");

    EXPECT_FALSE(op.isRuntimeUserSupplied);
    EXPECT_DOUBLE_EQ(static_cast<double>(std::get<hipdnn_data_sdk::types::half>(op.bakedDefault)),
                     static_cast<double>(hipdnn_data_sdk::types::half(0.5f)));
}

TEST(TestRuntimePassByValue, MakeScalarOperandBakesBfloat16Value)
{
    auto builder = buildScalarTensorAttributesWithValue(
        23, DataType::BFLOAT16, false, TensorValue::BFloat16Value, BFloat16Value(0.25f));
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{23, attr}};
    auto op = makeScalarOperand(tensorMap, 23, "Epsilon");

    EXPECT_FALSE(op.isRuntimeUserSupplied);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(std::get<hipdnn_data_sdk::types::bfloat16>(op.bakedDefault)),
        static_cast<double>(hipdnn_data_sdk::types::bfloat16(0.25f)));
}

TEST(TestRuntimePassByValue, MakeScalarOperandBakesInt32Value)
{
    auto builder = buildScalarTensorAttributesWithValue(
        24, DataType::INT32, false, TensorValue::Int32Value, Int32Value(42));
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{24, attr}};
    auto op = makeScalarOperand(tensorMap, 24, "Epsilon");

    EXPECT_FALSE(op.isRuntimeUserSupplied);
    EXPECT_EQ(std::get<int32_t>(op.bakedDefault), 42);
}

TEST(TestRuntimePassByValue, MakeScalarOperandBakesLargeInt64Value)
{
    // 2^53 + 1: not exactly representable in double. Proves the baked path also
    // carries INT64 without widening (the resolve-side path is covered separately
    // by ResolveScalarOperandPreservesLargeInt64).
    auto builder = buildScalarTensorAttributesWithValue(
        25, DataType::INT64, false, TensorValue::Int64Value, Int64Value(9007199254740993LL));
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{25, attr}};
    auto op = makeScalarOperand(tensorMap, 25, "Epsilon");

    EXPECT_FALSE(op.isRuntimeUserSupplied);
    EXPECT_EQ(std::get<int64_t>(op.bakedDefault), 9007199254740993LL);
}

TEST(TestRuntimePassByValue, MakeScalarOperandBakesBooleanValue)
{
    auto builder = buildScalarTensorAttributesWithValue(
        26, DataType::BOOLEAN, false, TensorValue::BoolValue, BoolValue(true));
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{26, attr}};
    auto op = makeScalarOperand(tensorMap, 26, "Epsilon");

    EXPECT_FALSE(op.isRuntimeUserSupplied);
    EXPECT_EQ(std::get<bool>(op.bakedDefault), true);
}

TEST(TestRuntimePassByValue, MakeScalarOperandThrowsOnUnsetDataType)
{
    auto builder = buildScalarTensorAttributesNoValue(27, DataType::UNSET);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{27, attr}};
    expectPluginThrowWithStatus([&] { makeScalarOperand(tensorMap, 27, "Epsilon"); },
                                HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestRuntimePassByValue, MakeScalarOperandThrowsOnUnsupportedDataType)
{
    // A dtype with no case in the baked switch (e.g. FP8) must hit the default throw,
    // regardless of union-value presence.
    auto builder = buildScalarTensorAttributesNoValue(28, DataType::FP8_E4M3);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{28, attr}};
    expectPluginThrowWithStatus([&] { makeScalarOperand(tensorMap, 28, "Epsilon"); },
                                HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestRuntimePassByValue, MakeScalarOperandThrowsWhenUidNotInTensorMap)
{
    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap;
    expectPluginThrowWithStatus([&] { makeScalarOperand(tensorMap, 999, "Epsilon"); },
                                HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// --- resolveScalarOperand ---

TEST(TestRuntimePassByValue, ResolveScalarOperandReturnsBakedDefaultIgnoringDeviceBuffers)
{
    const ScalarOperand op{1, DataType::FLOAT, false, 1e-5};

    // Even if a (wrong) device buffer exists for this uid, the baked value must win.
    float wrongHostValue = 999.0f;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{1, &wrongHostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_NEAR(std::get<double>(resolved), 1e-5, 1e-10);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandReadsHostFloatForPureRuntimeUserSupplied)
{
    const ScalarOperand op{7, DataType::FLOAT, true, 0.0};

    float hostValue = 5.0f;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{7, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_NEAR(std::get<float>(resolved), 5.0f, 1e-6f);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandReadsHostDoubleForPureRuntimeUserSupplied)
{
    const ScalarOperand op{8, DataType::DOUBLE, true, 0.0};

    double hostValue = 2.5;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{8, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_NEAR(std::get<double>(resolved), 2.5, 1e-12);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandReadsHostInt32ForPureRuntimeUserSupplied)
{
    const ScalarOperand op{9, DataType::INT32, true, 0.0};

    int32_t hostValue = 42;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{9, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_EQ(std::get<int32_t>(resolved), 42);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandThrowsIfPureRuntimeUserSuppliedBufferMissing)
{
    const ScalarOperand op{10, DataType::FLOAT, true, 0.0};
    std::vector<hipdnnPluginDeviceBuffer_t> buffers; // empty: uid 10 absent

    expectPluginThrowWithStatus(
        [&] { resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size())); },
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandThrowsIfPureRuntimeUserSuppliedBufferPtrIsNull)
{
    const ScalarOperand op{10, DataType::FLOAT, true, 0.0};
    // Slot for uid 10 is present but carries a null pointer: findDeviceBuffer
    // succeeds, so the null must be caught before the host-scalar memcpy.
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{10, nullptr}};

    expectPluginThrowWithStatus(
        [&] { resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size())); },
        HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandThrowsOnUnsetDataType)
{
    const ScalarOperand op{11, DataType::UNSET, true, 0.0};

    float hostValue = 1.0f;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{11, &hostValue}};

    expectPluginThrowWithStatus(
        [&] { resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size())); },
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandReadsHostHalfForPureRuntimeUserSupplied)
{
    const ScalarOperand op{12, DataType::HALF, true, 0.0};

    // half is lossy: assert against the value that actually round-trips through fp16,
    // not the source literal, so the test pins the byte-level read, not fp16 precision.
    hipdnn_data_sdk::types::half hostValue(0.5f);
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{12, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_DOUBLE_EQ(static_cast<double>(std::get<hipdnn_data_sdk::types::half>(resolved)),
                     static_cast<double>(hostValue));
}

TEST(TestRuntimePassByValue, ResolveScalarOperandReadsHostBfloat16ForPureRuntimeUserSupplied)
{
    const ScalarOperand op{13, DataType::BFLOAT16, true, 0.0};

    hipdnn_data_sdk::types::bfloat16 hostValue(0.25f);
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{13, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_DOUBLE_EQ(static_cast<double>(std::get<hipdnn_data_sdk::types::bfloat16>(resolved)),
                     static_cast<double>(hostValue));
}

TEST(TestRuntimePassByValue, ResolveScalarOperandReadsHostInt64ForPureRuntimeUserSupplied)
{
    const ScalarOperand op{14, DataType::INT64, true, 0.0};

    int64_t hostValue = 1234567890123LL;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{14, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_EQ(std::get<int64_t>(resolved), hostValue);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandPreservesLargeInt64)
{
    // 2^53 + 1: NOT exactly representable in double. Proves the SDK carries INT64
    // without widening through a double intermediate (the reviewed bug).
    const ScalarOperand op{20, DataType::INT64, true, 0.0};

    int64_t hostValue = 9007199254740993LL;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{20, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_EQ(std::get<int64_t>(resolved), 9007199254740993LL);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandReadsHostBooleanForPureRuntimeUserSupplied)
{
    const ScalarOperand op{15, DataType::BOOLEAN, true, 0.0};

    bool hostValue = true;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{15, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_EQ(std::get<bool>(resolved), true);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandThrowsOnUnsupportedDataType)
{
    // A dtype with no case in the resolve switch (e.g. FP8) must hit the default throw.
    const ScalarOperand op{16, DataType::FP8_E4M3, true, 0.0};

    uint8_t hostValue = 0;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{16, &hostValue}};

    expectPluginThrowWithStatus(
        [&] { resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size())); },
        HIPDNN_PLUGIN_STATUS_BAD_PARAM);
}

// --- toDouble ---

TEST(TestRuntimePassByValue, ToDoubleWidensEachArm)
{
    EXPECT_DOUBLE_EQ(toDouble(ScalarValue{1.5}), 1.5);
    EXPECT_FLOAT_EQ(static_cast<float>(toDouble(ScalarValue{2.5f})), 2.5f);
    EXPECT_EQ(toDouble(ScalarValue{int32_t{42}}), 42.0);
    EXPECT_EQ(toDouble(ScalarValue{true}), 1.0);
    EXPECT_EQ(toDouble(ScalarValue{int64_t{1234567890123LL}}), 1234567890123.0);
}

TEST(TestRuntimePassByValue, ToDoubleThrowsOnUnrepresentableInt64)
{
    // 2^53 + 1: rounds to 2^53 in double, so the round-trip check must catch it
    // rather than silently returning the wrong value.
    expectPluginThrowWithStatus([] { toDouble(ScalarValue{int64_t{9007199254740993LL}}); },
                                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR);
}

TEST(TestRuntimePassByValue, ToDoubleThrowsOnUnrepresentableInt32IsUnreachableButInt64Guarded)
{
    // int32_t is always exactly representable in double (24 vs 53 mantissa bits
    // needed), so this documents that the guard is a no-op for INT32 -- included
    // for completeness of the toDouble arm coverage, not because it can fail.
    EXPECT_EQ(toDouble(ScalarValue{std::numeric_limits<int32_t>::max()}),
              static_cast<double>(std::numeric_limits<int32_t>::max()));
}
