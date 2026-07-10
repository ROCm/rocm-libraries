// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>

using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// Builds a scalar TensorAttributes flatbuffer and returns the owning builder plus the
// finalized root pointer. Mirrors compile-time-constant / runtime-with-default / pure
// runtime-user-supplied states per RFC 0016.
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

} // namespace

// --- findDeviceBuffer ---

TEST(TestRuntimePassByValue, FindDeviceBufferReturnsCorrectBuffer)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers
        = {{42, reinterpret_cast<void*>(0x1234)}, {99, reinterpret_cast<void*>(0x5678)}};

    auto result = findDeviceBuffer(99, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_EQ(result.uid, 99);
    EXPECT_EQ(result.ptr, reinterpret_cast<void*>(0x5678));
}

TEST(TestRuntimePassByValue, FindDeviceBufferThrowsIfNotFound)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{1, reinterpret_cast<void*>(0x1111)}};

    EXPECT_THROW(findDeviceBuffer(2, buffers.data(), static_cast<uint32_t>(buffers.size())),
                 HipdnnPluginException);
}

TEST(TestRuntimePassByValue, FindDeviceBufferThrowsWhenArrayIsEmpty)
{
    EXPECT_THROW(findDeviceBuffer(1, nullptr, 0), HipdnnPluginException);
}

TEST(TestRuntimePassByValue, FindDeviceBufferReturnsFirstMatchWhenDuplicateUidsExist)
{
    int data1 = 0;
    int data2 = 0;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{5, &data1}, {5, &data2}};

    auto result = findDeviceBuffer(5, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_EQ(result.uid, 5);
    EXPECT_EQ(result.ptr, &data1);
}

// --- makeScalarOperand ---

TEST(TestRuntimePassByValue, MakeScalarOperandCompileTimeConstantBakesValue)
{
    auto builder = buildScalarTensorAttributes(1, DataType::FLOAT, false, 1e-5f);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{1, attr}};
    auto op = makeScalarOperand(tensorMap, 1, "Epsilon");

    EXPECT_EQ(op.uid, 1);
    EXPECT_FALSE(op.isRuntimeUserSupplied);
    EXPECT_NEAR(op.bakedDefault, 1e-5, 1e-10);
}

TEST(TestRuntimePassByValue, MakeScalarOperandRuntimeWithDefaultBakesValue)
{
    auto builder = buildScalarTensorAttributes(2, DataType::FLOAT, true, 1e-3f);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    const std::unordered_map<int64_t, const TensorAttributes*> tensorMap{{2, attr}};
    auto op = makeScalarOperand(tensorMap, 2, "Epsilon");

    EXPECT_EQ(op.uid, 2);
    EXPECT_FALSE(op.isRuntimeUserSupplied);
    EXPECT_NEAR(op.bakedDefault, 1e-3, 1e-10);
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

// --- resolveScalarOperand ---

TEST(TestRuntimePassByValue, ResolveScalarOperandReturnsBakedDefaultIgnoringDeviceBuffers)
{
    const ScalarOperand op{1, DataType::FLOAT, false, 1e-5};

    // Even if a (wrong) device buffer exists for this uid, the baked value must win.
    float wrongHostValue = 999.0f;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{1, &wrongHostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_NEAR(resolved, 1e-5, 1e-10);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandReadsHostFloatForPureRuntimeUserSupplied)
{
    const ScalarOperand op{7, DataType::FLOAT, true, 0.0};

    float hostValue = 5.0f;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{7, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_NEAR(resolved, 5.0, 1e-6);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandReadsHostDoubleForPureRuntimeUserSupplied)
{
    const ScalarOperand op{8, DataType::DOUBLE, true, 0.0};

    double hostValue = 2.5;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{8, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_NEAR(resolved, 2.5, 1e-12);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandReadsHostInt32ForPureRuntimeUserSupplied)
{
    const ScalarOperand op{9, DataType::INT32, true, 0.0};

    int32_t hostValue = 42;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{9, &hostValue}};

    auto resolved = resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size()));
    EXPECT_EQ(resolved, 42.0);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandThrowsIfPureRuntimeUserSuppliedBufferMissing)
{
    const ScalarOperand op{10, DataType::FLOAT, true, 0.0};
    std::vector<hipdnnPluginDeviceBuffer_t> buffers; // empty: uid 10 absent

    EXPECT_THROW(resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size())),
                 HipdnnPluginException);
}

TEST(TestRuntimePassByValue, ResolveScalarOperandThrowsOnUnsetDataType)
{
    const ScalarOperand op{11, DataType::UNSET, true, 0.0};

    float hostValue = 1.0f;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers = {{11, &hostValue}};

    EXPECT_THROW(resolveScalarOperand(op, buffers.data(), static_cast<uint32_t>(buffers.size())),
                 HipdnnPluginException);
}
