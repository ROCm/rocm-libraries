// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestReductionAttributes, SetGetMode)
{
    ReductionAttributes attrs;
    EXPECT_FALSE(attrs.get_mode().has_value());

    attrs.set_mode(ReductionMode::ADD);
    EXPECT_TRUE(attrs.get_mode().has_value());
    EXPECT_EQ(attrs.get_mode().value(), ReductionMode::ADD);
}

TEST(TestReductionAttributes, SetGetAllModes)
{
    const std::vector<ReductionMode> modes = {ReductionMode::ADD,
                                              ReductionMode::MUL,
                                              ReductionMode::MIN,
                                              ReductionMode::MAX,
                                              ReductionMode::AMAX,
                                              ReductionMode::AVG,
                                              ReductionMode::NORM1,
                                              ReductionMode::NORM2,
                                              ReductionMode::MUL_NO_ZEROS};

    for(auto mode : modes)
    {
        ReductionAttributes attrs;
        attrs.set_mode(mode);
        EXPECT_EQ(attrs.get_mode().value(), mode);
    }
}

TEST(TestReductionAttributes, SetGetXTensor)
{
    ReductionAttributes attrs;
    EXPECT_EQ(attrs.get_x(), nullptr);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 4})
        .set_stride({4, 1});

    attrs.set_x(x);

    auto retrieved = attrs.get_x();
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "InputTensor");
}

TEST(TestReductionAttributes, SetGetYTensor)
{
    ReductionAttributes attrs;
    EXPECT_EQ(attrs.get_y(), nullptr);

    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 4})
        .set_stride({4, 1});

    attrs.set_y(y);

    auto retrieved = attrs.get_y();
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->get_uid(), 2);
    EXPECT_EQ(retrieved->get_name(), "OutputTensor");
}

TEST(TestReductionAttributes, SetXWithMove)
{
    ReductionAttributes attrs;
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("Input");
    auto rawPtr = x.get();

    attrs.set_x(std::move(x));

    EXPECT_EQ(x, nullptr);
    EXPECT_EQ(attrs.get_x().get(), rawPtr);
}

TEST(TestReductionAttributes, SetYWithMove)
{
    ReductionAttributes attrs;
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_name("Output");
    auto rawPtr = y.get();

    attrs.set_y(std::move(y));

    EXPECT_EQ(y, nullptr);
    EXPECT_EQ(attrs.get_y().get(), rawPtr);
}

TEST(TestReductionAttributes, PackAttributesCorrectMode)
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::AVG);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(10).set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(20).set_dim({1, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    attrs.set_x(x);
    attrs.set_y(y);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = attrs.pack_attributes(builder);
    builder.Finish(offset);

    auto* fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ReductionAttributes>(
        builder.GetBufferPointer());

    EXPECT_EQ(fb->mode(), hipdnn_data_sdk::data_objects::ReductionMode::AVG);
    EXPECT_EQ(fb->in_tensor_uid(), 10);
    EXPECT_EQ(fb->out_tensor_uid(), 20);
}

TEST(TestReductionAttributes, PackAttributesMinMaxMapping)
{
    {
        ReductionAttributes attrs;
        attrs.set_mode(ReductionMode::MIN);
        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(1).set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
        auto y = std::make_shared<TensorAttributes>();
        y->set_uid(2).set_dim({1, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
        attrs.set_x(x);
        attrs.set_y(y);

        flatbuffers::FlatBufferBuilder builder;
        auto offset = attrs.pack_attributes(builder);
        builder.Finish(offset);

        auto* fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ReductionAttributes>(
            builder.GetBufferPointer());
        EXPECT_EQ(fb->mode(), hipdnn_data_sdk::data_objects::ReductionMode::MIN_OP);
    }
    {
        ReductionAttributes attrs;
        attrs.set_mode(ReductionMode::MAX);
        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(1).set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
        auto y = std::make_shared<TensorAttributes>();
        y->set_uid(2).set_dim({1, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
        attrs.set_x(x);
        attrs.set_y(y);

        flatbuffers::FlatBufferBuilder builder;
        auto offset = attrs.pack_attributes(builder);
        builder.Finish(offset);

        auto* fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ReductionAttributes>(
            builder.GetBufferPointer());
        EXPECT_EQ(fb->mode(), hipdnn_data_sdk::data_objects::ReductionMode::MAX_OP);
    }
}

TEST(TestReductionAttributes, FromFlatBufferRoundTrip)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(5).set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(7).set_dim({1, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    ReductionAttributes original;
    original.set_mode(ReductionMode::NORM2);
    original.set_x(x);
    original.set_y(y);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = original.pack_attributes(builder);
    builder.Finish(offset);

    auto* fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ReductionAttributes>(
        builder.GetBufferPointer());

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    tensorMap[5] = x;
    tensorMap[7] = y;

    auto restored = ReductionAttributes::fromFlatBuffer(fb, tensorMap);

    EXPECT_EQ(restored.get_mode().value(), ReductionMode::NORM2);
    EXPECT_EQ(restored.get_x()->get_uid(), 5);
    EXPECT_EQ(restored.get_y()->get_uid(), 7);
}

TEST(TestReductionAttributes, IsDeterministicDefaultsFalse)
{
    const ReductionAttributes attrs;
    EXPECT_FALSE(attrs.get_is_deterministic());
}

TEST(TestReductionAttributes, SetGetIsDeterministic)
{
    ReductionAttributes attrs;
    attrs.set_is_deterministic(true);
    EXPECT_TRUE(attrs.get_is_deterministic());

    attrs.set_is_deterministic(false);
    EXPECT_FALSE(attrs.get_is_deterministic());
}

TEST(TestReductionAttributes, PackAttributesIsDeterministic)
{
    ReductionAttributes attrs;
    attrs.set_mode(ReductionMode::ADD);
    attrs.set_is_deterministic(true);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_dim({1, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
    attrs.set_x(x);
    attrs.set_y(y);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = attrs.pack_attributes(builder);
    builder.Finish(offset);

    auto* fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ReductionAttributes>(
        builder.GetBufferPointer());
    EXPECT_TRUE(fb->is_deterministic());
}

TEST(TestReductionAttributes, FromFlatBufferIsDeterministicRoundTrip)
{
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_dim({4, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(2).set_dim({1, 8}).set_stride({8, 1}).set_data_type(DataType::FLOAT);

    ReductionAttributes original;
    original.set_mode(ReductionMode::ADD);
    original.set_is_deterministic(true);
    original.set_x(x);
    original.set_y(y);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = original.pack_attributes(builder);
    builder.Finish(offset);

    auto* fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::ReductionAttributes>(
        builder.GetBufferPointer());

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    tensorMap[1] = x;
    tensorMap[2] = y;

    auto restored = ReductionAttributes::fromFlatBuffer(fb, tensorMap);
    EXPECT_TRUE(restored.get_is_deterministic());
}

TEST(TestReductionAttributes, ReductionAttributesTypedefExists)
{
    // Verify typedef alias exists and works
    Reduction_attributes attrs;
    attrs.set_mode(ReductionMode::ADD);
    EXPECT_EQ(attrs.get_mode().value(), ReductionMode::ADD);
}
