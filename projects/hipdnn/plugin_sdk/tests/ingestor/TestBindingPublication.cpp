// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/ingestor/BindingPublication.hpp>

namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_flatbuffers_sdk::data_objects;

TensorAttributesT tensor()
{
    TensorAttributesT result;
    result.uid = 0;
    result.data_type = DataType::FLOAT;
    result.dims = {1, 1, 1, 1};
    result.strides = {1, 1, 1, 1};
    return result;
}

template <typename Value>
TensorAttributesT scalar(DataType dtype, Value value)
{
    auto result = tensor();
    result.data_type = dtype;
    result.value.Set(value);
    return result;
}

bool publish(const TensorAttributesT& source, BoundTokens& bound)
{
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(TensorAttributes::Pack(builder, &source));
    return publishTensor(
        bound, "q", flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer()));
}

void expectScalar(const TensorAttributesT& source, float expected)
{
    BoundTokens bound;
    ASSERT_TRUE(publish(source, bound));
    const auto* value = tryGetBoundValue<double>(bound, "q.value_f32");
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(*value, static_cast<double>(expected));
    EXPECT_EQ(std::signbit(*value), std::signbit(expected));
    EXPECT_EQ(tryGetBoundValue<int64_t>(bound, "q.value_f32"), nullptr);
}

void expectInvalid(const TensorAttributesT& source)
{
    BoundTokens bound;
    EXPECT_FALSE(publish(source, bound));
    EXPECT_TRUE(bound.empty());
}

TEST(TestBindingPublication, OwnsCanonicalTensorFieldsAfterSerializedStorageDies)
{
    auto source = tensor();
    source.dims = {2, 3, 4};
    source.strides = {1, 2, 6};
    BoundTokens bound;
    ASSERT_TRUE(publish(source, bound));
    EXPECT_EQ(tryGetBoundInt(bound, "q"), 0);
    EXPECT_EQ(tryGetBoundInt(bound, "q.uid"), 0);
    EXPECT_EQ(tryGetBoundInt(bound, "q.rank"), 3);
    EXPECT_EQ(std::get<std::string>(bound.at("q.dtype")), "FLOAT");
    EXPECT_EQ(std::get<std::vector<int64_t>>(bound.at("q.stride_order")),
              (std::vector<int64_t>{0, 1, 2}));
    EXPECT_TRUE(std::get<bool>(bound.at("q.packed")));
    EXPECT_FALSE(std::get<bool>(bound.at("q.virtual")));
    EXPECT_FALSE(std::get<bool>(bound.at("q.is_runtime_pass_by_value")));
    for(size_t i = 0; i < source.dims.size(); ++i)
    {
        EXPECT_EQ(tryGetBoundInt(bound, tensorElement("q", "dims", i)), source.dims[i]);
        EXPECT_EQ(tryGetBoundInt(bound, tensorElement("q", "strides", i)), source.strides[i]);
    }
    EXPECT_EQ(bound.count("q.value_f32"), 0U);
    EXPECT_EQ(bound.count("q.dims[3]"), 0U);
    EXPECT_EQ(bound.count("q.strides[3]"), 0U);
}

TEST(TestBindingPublication, PublishesEveryScalarArmAndDtypeInterpretation)
{
    expectScalar(scalar(DataType::FLOAT, Float32Value(-2.5f)), -2.5f);
    // HALF/BFLOAT16 store float32 values here; do not round them to 16 bits.
    expectScalar(scalar(DataType::HALF, Float16Value(1.0001f)), 1.0001f);
    expectScalar(scalar(DataType::BFLOAT16, BFloat16Value(1.0001f)), 1.0001f);
    expectScalar(scalar(DataType::UINT8, Float8Value(0xff)), 255.0f);
    expectScalar(scalar(DataType::INT8, Float8Value(0xff)), -1.0f);
    expectScalar(scalar(DataType::INT8, Float8Value(0x80)), -128.0f);
    expectScalar(scalar(DataType::INT8, Float8Value(0x7f)), 127.0f);
    expectScalar(scalar(DataType::FP8_E4M3, Float8Value(0x7e)), 448.0f);
    expectScalar(scalar(DataType::FP8_E5M2, Float8Value(0x7b)), 57344.0f);
    expectScalar(scalar(DataType::FP8_E4M3_FNUZ, Float8Value(0x7f)), 240.0f);
    expectScalar(scalar(DataType::FP8_E4M3_FNUZ, Float8Value(0xff)), -240.0f);
    expectScalar(scalar(DataType::FP8_E5M2_FNUZ, Float8Value(0x7f)), 57344.0f);
    expectScalar(scalar(DataType::FP8_E5M2_FNUZ, Float8Value(0xff)), -57344.0f);
    expectScalar(scalar(DataType::INT32, Int32Value(16777217)), 16777216.0f);
    expectScalar(scalar(DataType::DOUBLE, Float64Value(16777217.0)), 16777216.0f);
    const int64_t halfwayPlusOne = (int64_t{1} << 62) + (int64_t{1} << 38) + 1;
    expectScalar(scalar(DataType::INT64, Int64Value(halfwayPlusOne)), 0x1.000002p62f);
    expectScalar(scalar(DataType::INT64, Int64Value(std::numeric_limits<int64_t>::max())), 0x1p63f);
    expectScalar(scalar(DataType::BOOLEAN, BoolValue(false)), 0.0f);
    expectScalar(scalar(DataType::BOOLEAN, BoolValue(true)), 1.0f);
}

TEST(TestBindingPublication, PreservesFiniteBoundariesSignedZeroAndUnderflow)
{
    for(const float value :
        {-0.0f, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max()})
    {
        expectScalar(scalar(DataType::FLOAT, Float32Value(value)), value);
        expectScalar(scalar(DataType::HALF, Float16Value(value)), value);
        expectScalar(scalar(DataType::BFLOAT16, BFloat16Value(value)), value);
        expectScalar(scalar(DataType::DOUBLE, Float64Value(static_cast<double>(value))), value);
    }
    expectScalar(scalar(DataType::DOUBLE, Float64Value(0x1p-150)), 0.0f);
    expectScalar(scalar(DataType::DOUBLE, Float64Value(-0x1p-150)), -0.0f);
    expectScalar(scalar(DataType::DOUBLE, Float64Value(0x1.8p-149)), 0x1p-148f);
}

TEST(TestBindingPublication, RejectsEveryMismatchedScalarArmDtypePair)
{
    const std::vector<TensorAttributesT> arms{scalar(DataType::FLOAT, Float32Value(1)),
                                              scalar(DataType::HALF, Float16Value(1)),
                                              scalar(DataType::BFLOAT16, BFloat16Value(1)),
                                              scalar(DataType::UINT8, Float8Value(1)),
                                              scalar(DataType::INT32, Int32Value(1)),
                                              scalar(DataType::DOUBLE, Float64Value(1)),
                                              scalar(DataType::INT64, Int64Value(1)),
                                              scalar(DataType::BOOLEAN, BoolValue(true))};
    for(const auto& arm : arms)
    {
        for(const auto dtype : EnumValuesDataType())
        {
            const bool float8Pair
                = arm.value.type == TensorValue::Float8Value
                  && (dtype == DataType::UINT8 || dtype == DataType::INT8
                      || dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2
                      || dtype == DataType::FP8_E4M3_FNUZ || dtype == DataType::FP8_E5M2_FNUZ);
            if(dtype == arm.data_type || float8Pair)
            {
                continue;
            }
            SCOPED_TRACE(EnumNameDataType(dtype));
            auto invalid = arm;
            invalid.data_type = dtype;
            expectInvalid(invalid);
            invalid.is_runtime_pass_by_value = true;
            expectInvalid(invalid);
        }
    }
}

TEST(TestBindingPublication, RejectsNonfiniteAndUnrepresentableStoredDefaultsToo)
{
    std::vector<TensorAttributesT> invalid{
        scalar(DataType::FP8_E4M3, Float8Value(0x7f)),
        scalar(DataType::FP8_E4M3, Float8Value(0xff)),
        scalar(DataType::FP8_E5M2, Float8Value(0x7c)),
        scalar(DataType::FP8_E5M2, Float8Value(0xfc)),
        scalar(DataType::FP8_E4M3_FNUZ, Float8Value(0x80)),
        scalar(DataType::FP8_E5M2_FNUZ, Float8Value(0x80)),
        scalar(DataType::DOUBLE,
               Float64Value(std::nextafter(static_cast<double>(std::numeric_limits<float>::max()),
                                           std::numeric_limits<double>::infinity()))),
        scalar(DataType::DOUBLE, Float64Value(-std::numeric_limits<double>::max()))};
    for(const auto value : {std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()})
    {
        invalid.push_back(scalar(DataType::FLOAT, Float32Value(value)));
        invalid.push_back(scalar(DataType::HALF, Float16Value(value)));
        invalid.push_back(scalar(DataType::BFLOAT16, BFloat16Value(value)));
        invalid.push_back(scalar(DataType::DOUBLE, Float64Value(static_cast<double>(value))));
    }
    for(auto& source : invalid)
    {
        expectInvalid(source);
        source.is_runtime_pass_by_value = true;
        expectInvalid(source);
    }
}

TEST(TestBindingPublication, RuntimeValuesAndDefaultsNeverBecomeCompileTimeConstants)
{
    auto source = tensor();
    for(const bool stored : {false, true})
    {
        if(stored)
        {
            source.value.Set(Float32Value(0));
        }
        source.is_runtime_pass_by_value = true;
        BoundTokens bound;
        ASSERT_TRUE(publish(source, bound));
        EXPECT_TRUE(std::get<bool>(bound.at("q.is_runtime_pass_by_value")));
        EXPECT_EQ(bound.count("q.value_f32"), 0U);
        source.virtual_ = true;
        expectInvalid(source);
        source.virtual_ = false;
        source.dims[0] = 2;
        expectInvalid(source);
        source.dims[0] = 1;
    }
    source.is_runtime_pass_by_value = false;
    source.virtual_ = true;
    expectInvalid(source);
    source.value.Reset();
    BoundTokens bound;
    ASSERT_TRUE(publish(source, bound));
    EXPECT_TRUE(std::get<bool>(bound.at("q.virtual")));
    EXPECT_EQ(bound.count("q.value_f32"), 0U);
    source.virtual_ = false;
    source.value.Set(Float32Value(1));
    source.dims[0] = 2;
    expectInvalid(source);
}

TEST(TestBindingPublication, RejectsMalformedShapeAndScalarPayloads)
{
    for(const auto& dims :
        std::vector<std::vector<int64_t>>{{}, {0}, {-1}, {std::numeric_limits<int64_t>::max(), 2}})
    {
        auto source = tensor();
        source.dims = dims;
        source.strides.assign(dims.size(), 1);
        expectInvalid(source);
    }
    auto source = tensor();
    source.strides.clear();
    expectInvalid(source);
    source.strides = {1, 1};
    expectInvalid(source);
    source = tensor();
    source.data_type = static_cast<DataType>(127);
    expectInvalid(source);
    BoundTokens bound;
    EXPECT_FALSE(publishTensor(bound, "q", nullptr));
    for(const auto type : {TensorValue::Float32Value, static_cast<TensorValue>(127)})
    {
        flatbuffers::FlatBufferBuilder builder;
        const std::vector<int64_t> shape{1};
        builder.Finish(CreateTensorAttributesDirect(
            builder, 0, nullptr, DataType::FLOAT, &shape, &shape, false, type));
        EXPECT_FALSE(publishTensor(
            bound, "q", flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer())));
    }
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> shape{1};
    const auto payload = builder.CreateStruct(Float32Value(1.0f)).Union();
    builder.Finish(CreateTensorAttributesDirect(
        builder, 0, nullptr, DataType::FLOAT, &shape, &shape, false, TensorValue::NONE, payload));
    EXPECT_FALSE(publishTensor(
        bound, "q", flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer())));
}

struct LayoutCase
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    bool packed;
};

TEST(TestBindingPublication, PackednessSeparatesDensePermutationsFromOverlapGapsAndReversal)
{
    const std::vector<LayoutCase> cases{{{2, 3, 4}, {12, 4, 1}, true},
                                        {{2, 3, 4}, {1, 2, 6}, true},
                                        {{2, 1, 3}, {3, 999, 1}, true},
                                        {{2, 1, 3}, {3, 1, 1}, true},
                                        {{2, 1, 3}, {3, 0, 1}, true},
                                        {{2, 1, 3}, {3, -7, 1}, true},
                                        {{2, 2, 2}, {1, 1, 5}, false},
                                        {{2, 3}, {4, 1}, false},
                                        {{2, 3}, {0, 1}, false},
                                        {{2, 3}, {-3, 1}, false},
                                        {{1, 1, 1, 1}, {1, 1, 1, 1}, true},
                                        {{1, 1}, {0, -1}, true}};
    for(const auto& layout : cases)
    {
        auto source = tensor();
        source.dims = layout.dims;
        source.strides = layout.strides;
        BoundTokens bound;
        ASSERT_TRUE(publish(source, bound));
        EXPECT_EQ(std::get<bool>(bound.at("q.packed")), layout.packed);
    }
    auto source = tensor();
    BoundTokens tied;
    ASSERT_TRUE(publish(source, tied));
    EXPECT_EQ(std::get<std::vector<int64_t>>(tied.at("q.stride_order")),
              (std::vector<int64_t>{3, 2, 1, 0}));
    source.ragged_offset_tensor_uid = 9;
    BoundTokens ragged;
    ASSERT_TRUE(publish(source, ragged));
    EXPECT_FALSE(std::get<bool>(ragged.at("q.packed")));
}

TEST(TestBindingPublication, PublishesWholeGraphCountAndActualOverrideFlag)
{
    for(const bool overrideShape : {false, true})
    {
        GraphT source;
        source.is_override_shape_enabled = overrideShape;
        for(size_t i = 0; i < 2; ++i)
        {
            auto node = std::make_unique<NodeT>();
            node->attributes.Set(PointwiseAttributesT{});
            source.nodes.push_back(std::move(node));
        }
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(Graph::Pack(builder, &source));
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
            builder.GetBufferPointer(), builder.GetSize());
        BoundTokens bound;
        ASSERT_TRUE(publishGraph(bound, graph));
        EXPECT_EQ(tryGetBoundInt(bound, "graph.node_count"), 2);
        EXPECT_EQ(std::get<bool>(bound.at("graph.is_override_shape_enabled")), overrideShape);
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
