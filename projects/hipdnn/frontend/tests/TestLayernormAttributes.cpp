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

TEST(TestLayernormAttributes, LogicalAndStrictEquality)
{
    hipdnn_frontend::graph::LayernormAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);
    attr1.set_normalized_dim_count(2);

    auto x1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x1->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_x(x1);

    auto scale1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scale1->set_uid(2).set_name("Scale").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_scale(scale1);

    auto bias1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    bias1->set_uid(3).set_name("Bias").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_bias(bias1);

    auto epsilon1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilon1->set_uid(4).set_name("Epsilon").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_epsilon(epsilon1);

    auto y1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    y1->set_uid(5).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_y(y1);

    auto mean1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    mean1->set_uid(6).set_name("Mean").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_mean(mean1);

    auto invVar1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    invVar1->set_uid(7).set_name("InvVariance").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_inv_variance(invVar1);

    hipdnn_frontend::graph::LayernormAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);
    attr2.set_normalized_dim_count(2);

    auto x2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x2->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_x(x2);

    auto scale2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scale2->set_uid(2).set_name("Scale").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_scale(scale2);

    auto bias2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    bias2->set_uid(3).set_name("Bias").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_bias(bias2);

    auto epsilon2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilon2->set_uid(4).set_name("Epsilon").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_epsilon(epsilon2);

    auto y2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    y2->set_uid(5).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_y(y2);

    auto mean2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    mean2->set_uid(6).set_name("Mean").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_mean(mean2);

    auto invVar2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    invVar2->set_uid(7).set_name("InvVariance").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_inv_variance(invVar2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchX = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    structuralMismatchX->set_uid(99).set_name("MismatchedX");
    attr2.set_x(structuralMismatchX);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_x(x2); // Revert

    // Forward-phase mismatch: semantic, must fail both checks
    attr2.set_forward_phase(hipdnn_frontend::NormFwdPhase::INFERENCE);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING); // Revert

    // Normalized-dim-count mismatch: semantic, must fail both checks
    attr2.set_normalized_dim_count(1);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_normalized_dim_count(2); // Revert

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchX = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    logicalMatchX
        ->set_uid(555) // Diverges from attr1's x1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's x1 ("X")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_x(logicalMatchX);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}
