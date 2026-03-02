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
