// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>

TEST(TestBatchnormInferenceAttributes, CreateBatchnormInferenceAttributes)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    batchnormAttributes.set_x(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_inv_variance(
        std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<hipdnn_frontend::graph::TensorAttributes>());

    auto inputTensor = batchnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = batchnormAttributes.get_y();
    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_uid(3)
        .set_name("MeanTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto varianceTensor = batchnormAttributes.get_inv_variance();
    varianceTensor->set_uid(4)
        .set_name("VarianceTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_uid(5)
        .set_name("ScaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_uid(6)
        .set_name("BiasTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    EXPECT_EQ(inputTensor->get_uid(), 1);
    EXPECT_EQ(inputTensor->get_name(), "InputTensor");
    EXPECT_EQ(inputTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(inputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(inputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(outputTensor->get_uid(), 2);
    EXPECT_EQ(outputTensor->get_name(), "OutputTensor");
    EXPECT_EQ(outputTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(meanTensor->get_uid(), 3);
    EXPECT_EQ(meanTensor->get_name(), "MeanTensor");
    EXPECT_EQ(meanTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(meanTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(meanTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(varianceTensor->get_uid(), 4);
    EXPECT_EQ(varianceTensor->get_name(), "VarianceTensor");
    EXPECT_EQ(varianceTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(varianceTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(varianceTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(scaleTensor->get_uid(), 5);
    EXPECT_EQ(scaleTensor->get_name(), "ScaleTensor");
    EXPECT_EQ(scaleTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(scaleTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(biasTensor->get_uid(), 6);
    EXPECT_EQ(biasTensor->get_name(), "BiasTensor");
    EXPECT_EQ(biasTensor->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(biasTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(biasTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBatchnormInferenceAttributes, SetXWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    xTensor->set_uid(1).set_name("XTensor");

    auto rawPtr = xTensor.get();

    batchnormAttributes.set_x(std::move(xTensor));

    auto retrieved = batchnormAttributes.get_x();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "XTensor");

    EXPECT_EQ(xTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormInferenceAttributes, SetMeanWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto meanTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    meanTensor->set_uid(2).set_name("MeanTensor");

    auto rawPtr = meanTensor.get();

    batchnormAttributes.set_mean(std::move(meanTensor));

    auto retrieved = batchnormAttributes.get_mean();
    EXPECT_EQ(retrieved->get_uid(), 2);
    EXPECT_EQ(retrieved->get_name(), "MeanTensor");

    EXPECT_EQ(meanTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormInferenceAttributes, SetInvVarianceWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto invVarianceTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    invVarianceTensor->set_uid(3).set_name("InvVarianceTensor");

    auto rawPtr = invVarianceTensor.get();

    batchnormAttributes.set_inv_variance(std::move(invVarianceTensor));

    auto retrieved = batchnormAttributes.get_inv_variance();
    EXPECT_EQ(retrieved->get_uid(), 3);
    EXPECT_EQ(retrieved->get_name(), "InvVarianceTensor");

    EXPECT_EQ(invVarianceTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormInferenceAttributes, SetScaleWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto scaleTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scaleTensor->set_uid(4).set_name("ScaleTensor");

    auto rawPtr = scaleTensor.get();

    batchnormAttributes.set_scale(std::move(scaleTensor));

    auto retrieved = batchnormAttributes.get_scale();
    EXPECT_EQ(retrieved->get_uid(), 4);
    EXPECT_EQ(retrieved->get_name(), "ScaleTensor");

    EXPECT_EQ(scaleTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormInferenceAttributes, SetBiasWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto biasTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    biasTensor->set_uid(5).set_name("BiasTensor");

    auto rawPtr = biasTensor.get();

    batchnormAttributes.set_bias(std::move(biasTensor));

    auto retrieved = batchnormAttributes.get_bias();
    EXPECT_EQ(retrieved->get_uid(), 5);
    EXPECT_EQ(retrieved->get_name(), "BiasTensor");

    EXPECT_EQ(biasTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestBatchnormInferenceAttributes, SetYWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto yTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    yTensor->set_uid(6).set_name("YTensor");

    auto rawPtr = yTensor.get();

    batchnormAttributes.set_y(std::move(yTensor));

    auto retrieved = batchnormAttributes.get_y();
    EXPECT_EQ(retrieved->get_uid(), 6);
    EXPECT_EQ(retrieved->get_name(), "YTensor");

    EXPECT_EQ(yTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

// Simplified move tests - testing move semantics without setting uid/name

TEST(TestBatchnormInferenceAttributes, SimplifiedSetXWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto xTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    batchnormAttributes.set_x(std::move(xTensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnormAttributes.get_x(), nullptr);
}

TEST(TestBatchnormInferenceAttributes, SimplifiedSetMeanWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto meanTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    batchnormAttributes.set_mean(std::move(meanTensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnormAttributes.get_mean(), nullptr);
}

TEST(TestBatchnormInferenceAttributes, SimplifiedSetInvVarianceWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto invVarianceTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    batchnormAttributes.set_inv_variance(std::move(invVarianceTensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnormAttributes.get_inv_variance(), nullptr);
}

TEST(TestBatchnormInferenceAttributes, SimplifiedSetScaleWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto scaleTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    batchnormAttributes.set_scale(std::move(scaleTensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnormAttributes.get_scale(), nullptr);
}

TEST(TestBatchnormInferenceAttributes, SimplifiedSetBiasWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto biasTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    batchnormAttributes.set_bias(std::move(biasTensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnormAttributes.get_bias(), nullptr);
}

TEST(TestBatchnormInferenceAttributes, SimplifiedSetYWithMove)
{
    hipdnn_frontend::graph::BatchnormInferenceAttributes batchnormAttributes;

    auto yTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    batchnormAttributes.set_y(std::move(yTensor));

    // Just verify the tensor was set
    EXPECT_NE(batchnormAttributes.get_y(), nullptr);
}
