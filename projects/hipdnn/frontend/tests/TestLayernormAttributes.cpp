// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestLayernormAttributes, DefaultValues)
{
    const LayernormAttributes attrs;

    EXPECT_EQ(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_epsilon(), nullptr);
    EXPECT_EQ(attrs.get_y(), nullptr);
    EXPECT_EQ(attrs.get_normalized_dim_count(), 0);
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

TEST(TestLayernormAttributes, SetNormalizedDimCount)
{
    LayernormAttributes attrs;

    const int64_t normalizedDimCount = 3;
    attrs.set_normalized_dim_count(normalizedDimCount);

    EXPECT_EQ(attrs.get_normalized_dim_count(), normalizedDimCount);
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
