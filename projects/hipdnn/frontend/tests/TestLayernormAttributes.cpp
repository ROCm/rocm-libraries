// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestLayernormAttributes, DefaultValues)
{
    LayernormAttributes attrs;

    EXPECT_EQ(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_epsilon(), nullptr);
    EXPECT_EQ(attrs.get_y(), nullptr);
    EXPECT_EQ(attrs.get_mean(), nullptr);
    EXPECT_EQ(attrs.get_inv_variance(), nullptr);
    EXPECT_EQ(attrs.get_forward_phase(), NormFwdPhase::NOT_SET);
}

TEST(TestLayernormAttributes, SetRequiredTensors)
{
    LayernormAttributes attrs;

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1);
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(4);
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(5);

    attrs.set_x(x);
    attrs.set_scale(scale);
    attrs.set_bias(bias);
    attrs.set_epsilon(epsilon);
    attrs.set_y(y);

    EXPECT_EQ(attrs.get_x(), x);
    EXPECT_EQ(attrs.get_scale(), scale);
    EXPECT_EQ(attrs.get_bias(), bias);
    EXPECT_EQ(attrs.get_epsilon(), epsilon);
    EXPECT_EQ(attrs.get_y(), y);
    EXPECT_EQ(attrs.get_mean(), nullptr);
    EXPECT_EQ(attrs.get_inv_variance(), nullptr);
}

TEST(TestLayernormAttributes, SetOptionalMean)
{
    LayernormAttributes attrs;

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(30);
    attrs.set_mean(mean);

    EXPECT_EQ(attrs.get_mean(), mean);
    EXPECT_EQ(attrs.get_inv_variance(), nullptr);
}

TEST(TestLayernormAttributes, SetOptionalInvVariance)
{
    LayernormAttributes attrs;

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(40);
    attrs.set_inv_variance(invVariance);

    EXPECT_EQ(attrs.get_inv_variance(), invVariance);
    EXPECT_EQ(attrs.get_mean(), nullptr);
}

TEST(TestLayernormAttributes, SetForwardPhase)
{
    LayernormAttributes attrs;

    attrs.set_forward_phase(NormFwdPhase::TRAINING);
    EXPECT_EQ(attrs.get_forward_phase(), NormFwdPhase::TRAINING);

    attrs.set_forward_phase(NormFwdPhase::INFERENCE);
    EXPECT_EQ(attrs.get_forward_phase(), NormFwdPhase::INFERENCE);
}

TEST(TestLayernormAttributes, PackAttributes)
{
    LayernormAttributes attrs;

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1);
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(4);
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(5);
    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(6);
    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(7);

    attrs.set_x(x);
    attrs.set_scale(scale);
    attrs.set_bias(bias);
    attrs.set_epsilon(epsilon);
    attrs.set_y(y);
    attrs.set_mean(mean);
    attrs.set_inv_variance(invVariance);
    attrs.set_forward_phase(NormFwdPhase::TRAINING);

    flatbuffers::FlatBufferBuilder builder;
    auto packed = attrs.pack_attributes(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::LayernormAttributes>(buf);

    EXPECT_EQ(fb->x_tensor_uid(), 1);
    EXPECT_EQ(fb->scale_tensor_uid(), 2);
    EXPECT_EQ(fb->bias_tensor_uid(), 3);
    EXPECT_EQ(fb->epsilon_tensor_uid(), 4);
    EXPECT_EQ(fb->y_tensor_uid(), 5);
    ASSERT_TRUE(fb->mean_tensor_uid().has_value());
    EXPECT_EQ(*fb->mean_tensor_uid(), 6);
    ASSERT_TRUE(fb->inv_variance_tensor_uid().has_value());
    EXPECT_EQ(*fb->inv_variance_tensor_uid(), 7);
    EXPECT_EQ(fb->forward_phase(), hipdnn_data_sdk::data_objects::NormFwdPhase::TRAINING);
}

TEST(TestLayernormAttributes, PackAttributesNoOptionals)
{
    LayernormAttributes attrs;

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1);
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(2);
    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(3);
    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(4);
    auto y = std::make_shared<TensorAttributes>();
    y->set_uid(5);

    attrs.set_x(x);
    attrs.set_scale(scale);
    attrs.set_bias(bias);
    attrs.set_epsilon(epsilon);
    attrs.set_y(y);

    flatbuffers::FlatBufferBuilder builder;
    auto packed = attrs.pack_attributes(builder);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::LayernormAttributes>(buf);

    EXPECT_EQ(fb->x_tensor_uid(), 1);
    EXPECT_EQ(fb->scale_tensor_uid(), 2);
    EXPECT_EQ(fb->bias_tensor_uid(), 3);
    EXPECT_EQ(fb->epsilon_tensor_uid(), 4);
    EXPECT_EQ(fb->y_tensor_uid(), 5);
    EXPECT_FALSE(fb->mean_tensor_uid().has_value());
    EXPECT_FALSE(fb->inv_variance_tensor_uid().has_value());
}

TEST(TestLayernormAttributes, FromFlatBuffer)
{
    // Build a FlatBuffer with all fields
    flatbuffers::FlatBufferBuilder builder;
    auto packed = hipdnn_data_sdk::data_objects::CreateLayernormAttributes(
        builder,
        10,
        20,
        30,
        40,
        50,
        flatbuffers::Optional<int64_t>(60),
        flatbuffers::Optional<int64_t>(70),
        hipdnn_data_sdk::data_objects::NormFwdPhase::TRAINING);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::LayernormAttributes>(buf);

    // Build a tensorMap with all referenced UIDs
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    for(auto uid : {10, 20, 30, 40, 50, 60, 70})
    {
        auto t = std::make_shared<TensorAttributes>();
        t->set_uid(static_cast<int64_t>(uid));
        tensorMap[static_cast<int64_t>(uid)] = t;
    }

    auto attrs = LayernormAttributes::fromFlatBuffer(fb, tensorMap);

    ASSERT_NE(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_x()->get_uid(), 10);
    ASSERT_NE(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_scale()->get_uid(), 20);
    ASSERT_NE(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_bias()->get_uid(), 30);
    ASSERT_NE(attrs.get_epsilon(), nullptr);
    EXPECT_EQ(attrs.get_epsilon()->get_uid(), 40);
    ASSERT_NE(attrs.get_y(), nullptr);
    EXPECT_EQ(attrs.get_y()->get_uid(), 50);
    ASSERT_NE(attrs.get_mean(), nullptr);
    EXPECT_EQ(attrs.get_mean()->get_uid(), 60);
    ASSERT_NE(attrs.get_inv_variance(), nullptr);
    EXPECT_EQ(attrs.get_inv_variance()->get_uid(), 70);
    EXPECT_EQ(attrs.get_forward_phase(), NormFwdPhase::TRAINING);
}

TEST(TestLayernormAttributes, FromFlatBufferNoOptionals)
{
    // Build a FlatBuffer without optional fields
    flatbuffers::FlatBufferBuilder builder;
    auto packed = hipdnn_data_sdk::data_objects::CreateLayernormAttributes(builder, 1, 2, 3, 4, 5);
    builder.Finish(packed);

    auto buf = builder.GetBufferPointer();
    auto fb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::LayernormAttributes>(buf);

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    for(auto uid : {1, 2, 3, 4, 5})
    {
        auto t = std::make_shared<TensorAttributes>();
        t->set_uid(static_cast<int64_t>(uid));
        tensorMap[static_cast<int64_t>(uid)] = t;
    }

    auto attrs = LayernormAttributes::fromFlatBuffer(fb, tensorMap);

    ASSERT_NE(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_x()->get_uid(), 1);
    ASSERT_NE(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_scale()->get_uid(), 2);
    ASSERT_NE(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_bias()->get_uid(), 3);
    ASSERT_NE(attrs.get_epsilon(), nullptr);
    EXPECT_EQ(attrs.get_epsilon()->get_uid(), 4);
    ASSERT_NE(attrs.get_y(), nullptr);
    EXPECT_EQ(attrs.get_y()->get_uid(), 5);
    EXPECT_EQ(attrs.get_mean(), nullptr);
    EXPECT_EQ(attrs.get_inv_variance(), nullptr);
}
