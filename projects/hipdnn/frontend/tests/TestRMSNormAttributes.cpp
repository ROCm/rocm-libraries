// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>

TEST(TestRMSNormAttributes, CreateRMSNormAttributes)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    rmsnormAttributes.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    rmsnormAttributes.set_y(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    rmsnormAttributes.set_scale(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    rmsnormAttributes.set_epsilon(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    auto inputTensor = rmsnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = rmsnormAttributes.get_y();
    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto scaleTensor = rmsnormAttributes.get_scale();
    scaleTensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});

    auto epsilonTensor = rmsnormAttributes.get_epsilon();
    epsilonTensor->set_uid(4).set_name("EpsilonTensor").set_value(1e-5f);

    EXPECT_EQ(inputTensor->get_uid(), 1);
    EXPECT_EQ(inputTensor->get_name(), "InputTensor");
    EXPECT_EQ(inputTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(inputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(inputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(outputTensor->get_uid(), 2);
    EXPECT_EQ(outputTensor->get_name(), "OutputTensor");

    EXPECT_EQ(scaleTensor->get_uid(), 3);
    EXPECT_EQ(scaleTensor->get_name(), "ScaleTensor");

    EXPECT_EQ(epsilonTensor->get_uid(), 4);
    EXPECT_EQ(epsilonTensor->get_name(), "EpsilonTensor");
}

TEST(TestRMSNormAttributes, SetXWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(1).set_name("XTensor");

    auto rawPtr = xTensor.get();

    rmsnormAttributes.set_x(std::move(xTensor));

    auto retrieved = rmsnormAttributes.get_x();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "XTensor");

    EXPECT_EQ(xTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestRMSNormAttributes, SetScaleWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto scaleTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scaleTensor->set_uid(2).set_name("ScaleTensor");

    auto rawPtr = scaleTensor.get();

    rmsnormAttributes.set_scale(std::move(scaleTensor));

    auto retrieved = rmsnormAttributes.get_scale();
    EXPECT_EQ(retrieved->get_uid(), 2);
    EXPECT_EQ(retrieved->get_name(), "ScaleTensor");

    EXPECT_EQ(scaleTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestRMSNormAttributes, SetEpsilonWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilonTensor->set_uid(3).set_name("EpsilonTensor");

    auto rawPtr = epsilonTensor.get();

    rmsnormAttributes.set_epsilon(std::move(epsilonTensor));

    auto retrieved = rmsnormAttributes.get_epsilon();
    EXPECT_EQ(retrieved->get_uid(), 3);
    EXPECT_EQ(retrieved->get_name(), "EpsilonTensor");

    EXPECT_EQ(epsilonTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestRMSNormAttributes, SetBiasWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto biasTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    biasTensor->set_uid(5).set_name("BiasTensor");

    auto rawPtr = biasTensor.get();

    rmsnormAttributes.set_bias(std::move(biasTensor));

    auto retrieved = rmsnormAttributes.get_bias();
    EXPECT_EQ(retrieved->get_uid(), 5);
    EXPECT_EQ(retrieved->get_name(), "BiasTensor");

    EXPECT_EQ(biasTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestRMSNormAttributes, SetYWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto yTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    yTensor->set_uid(4).set_name("YTensor");

    auto rawPtr = yTensor.get();

    rmsnormAttributes.set_y(std::move(yTensor));

    auto retrieved = rmsnormAttributes.get_y();
    EXPECT_EQ(retrieved->get_uid(), 4);
    EXPECT_EQ(retrieved->get_name(), "YTensor");

    EXPECT_EQ(yTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestRMSNormAttributes, SetInvRmsWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto invRmsTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    invRmsTensor->set_uid(5).set_name("InvRmsTensor");

    auto rawPtr = invRmsTensor.get();

    rmsnormAttributes.set_inv_rms(std::move(invRmsTensor));

    auto retrieved = rmsnormAttributes.get_inv_rms();
    EXPECT_EQ(retrieved->get_uid(), 5);
    EXPECT_EQ(retrieved->get_name(), "InvRmsTensor");

    EXPECT_EQ(invRmsTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestRMSNormAttributes, InvRmsIsOptional)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    rmsnormAttributes.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    rmsnormAttributes.set_scale(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    rmsnormAttributes.set_epsilon(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    rmsnormAttributes.set_y(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    // inv_rms should be null when not set
    EXPECT_EQ(rmsnormAttributes.get_inv_rms(), nullptr);
}

TEST(TestRMSNormAttributes, BiasIsOptional)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    rmsnormAttributes.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    rmsnormAttributes.set_scale(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    rmsnormAttributes.set_epsilon(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    rmsnormAttributes.set_y(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    // bias should be null when not set
    EXPECT_EQ(rmsnormAttributes.get_bias(), nullptr);
}

TEST(TestRMSNormAttributes, TypeAliasWorks)
{
    // Verify the compatibility alias compiles and works
    hipdnn_frontend::graph::Rmsnorm_attributes rmsnormAttributes;

    rmsnormAttributes.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    EXPECT_NE(rmsnormAttributes.get_x(), nullptr);
}

// Simplified move tests

TEST(TestRMSNormAttributes, SimplifiedSetXWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    rmsnormAttributes.set_x(std::move(xTensor));

    EXPECT_NE(rmsnormAttributes.get_x(), nullptr);
}

TEST(TestRMSNormAttributes, SimplifiedSetScaleWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto scaleTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    rmsnormAttributes.set_scale(std::move(scaleTensor));

    EXPECT_NE(rmsnormAttributes.get_scale(), nullptr);
}

TEST(TestRMSNormAttributes, SimplifiedSetEpsilonWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    rmsnormAttributes.set_epsilon(std::move(epsilonTensor));

    EXPECT_NE(rmsnormAttributes.get_epsilon(), nullptr);
}

TEST(TestRMSNormAttributes, SimplifiedSetYWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto yTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    rmsnormAttributes.set_y(std::move(yTensor));

    EXPECT_NE(rmsnormAttributes.get_y(), nullptr);
}

TEST(TestRMSNormAttributes, SimplifiedSetInvRmsWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto invRmsTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    rmsnormAttributes.set_inv_rms(std::move(invRmsTensor));

    EXPECT_NE(rmsnormAttributes.get_inv_rms(), nullptr);
}

TEST(TestRMSNormAttributes, SimplifiedSetBiasWithMove)
{
    hipdnn_frontend::graph::RMSNormAttributes rmsnormAttributes;

    auto biasTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    rmsnormAttributes.set_bias(std::move(biasTensor));

    EXPECT_NE(rmsnormAttributes.get_bias(), nullptr);
}

TEST(TestRMSNormAttributes, LogicalAndStrictEquality)
{
    hipdnn_frontend::graph::RMSNormAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);

    auto x1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x1->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_x(x1);

    auto scale1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scale1->set_uid(2).set_name("Scale").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_scale(scale1);

    auto epsilon1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilon1->set_uid(3).set_name("Epsilon").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_epsilon(epsilon1);

    auto bias1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    bias1->set_uid(4).set_name("Bias").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_bias(bias1);

    auto y1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    y1->set_uid(5).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_y(y1);

    auto invRms1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    invRms1->set_uid(6).set_name("InvRms").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_inv_rms(invRms1);

    hipdnn_frontend::graph::RMSNormAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);

    auto x2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x2->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_x(x2);

    auto scale2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scale2->set_uid(2).set_name("Scale").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_scale(scale2);

    auto epsilon2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilon2->set_uid(3).set_name("Epsilon").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_epsilon(epsilon2);

    auto bias2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    bias2->set_uid(4).set_name("Bias").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_bias(bias2);

    auto y2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    y2->set_uid(5).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_y(y2);

    auto invRms2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    invRms2->set_uid(6).set_name("InvRms").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_inv_rms(invRms2);

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

    // forward_phase mismatch: semantic, must fail both checks
    attr2.set_forward_phase(hipdnn_frontend::NormFwdPhase::INFERENCE);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING); // Revert

    // Unset-vs-unset forward_phase: two default-constructed attrs (both
    // NOT_SET) should still compare equal
    const hipdnn_frontend::graph::RMSNormAttributes sparse1;
    hipdnn_frontend::graph::RMSNormAttributes sparse2;
    EXPECT_TRUE(sparse1 == sparse2);
    EXPECT_TRUE(sparse1.logicallyEquals(sparse2));

    // Set-vs-unset forward_phase should differ
    sparse2.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);
    EXPECT_FALSE(sparse1 == sparse2);
    EXPECT_FALSE(sparse1.logicallyEquals(sparse2));

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
