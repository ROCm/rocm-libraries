// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestPointwiseAttributes, CreatePointwiseAttributes)
{
    PointwiseAttributes pointwiseAttributes;

    pointwiseAttributes.set_input_0(std::make_shared<TensorAttributes>());
    pointwiseAttributes.set_output_0(std::make_shared<TensorAttributes>());
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD)
        .set_relu_lower_clip(0.1f)
        .set_relu_upper_clip(6.0f)
        .set_relu_lower_clip_slope(0.01f)
        .set_axis(1)
        .set_swish_beta(1.5f)
        .set_elu_alpha(0.9f)
        .set_softplus_beta(2.0f);

    auto inputTensor = pointwiseAttributes.get_input_0();
    EXPECT_FALSE(inputTensor->has_uid());

    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = pointwiseAttributes.get_output_0();
    EXPECT_FALSE(outputTensor->has_uid());

    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType::HALF)
        .set_dim({5, 6, 7, 8})
        .set_stride({1, 2, 3, 4});

    EXPECT_EQ(inputTensor->get_uid(), 1);
    EXPECT_EQ(inputTensor->get_name(), "InputTensor");
    EXPECT_EQ(inputTensor->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(inputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(inputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(outputTensor->get_uid(), 2);
    EXPECT_EQ(outputTensor->get_name(), "OutputTensor");
    EXPECT_EQ(outputTensor->get_data_type(), DataType::HALF);
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{5, 6, 7, 8}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{1, 2, 3, 4}));

    EXPECT_EQ(pointwiseAttributes.get_mode(), PointwiseMode::RELU_FWD);
    EXPECT_TRUE(pointwiseAttributes.get_relu_lower_clip().has_value());
    EXPECT_EQ(pointwiseAttributes.get_relu_lower_clip().value(), 0.1f);
    EXPECT_TRUE(pointwiseAttributes.get_relu_upper_clip().has_value());
    EXPECT_EQ(pointwiseAttributes.get_relu_upper_clip().value(), 6.0f);
    EXPECT_TRUE(pointwiseAttributes.get_relu_lower_clip_slope().has_value());
    EXPECT_EQ(pointwiseAttributes.get_relu_lower_clip_slope().value(), 0.01f);
    EXPECT_TRUE(pointwiseAttributes.get_axis().has_value());
    EXPECT_EQ(pointwiseAttributes.get_axis().value(), 1);
    EXPECT_TRUE(pointwiseAttributes.get_swish_beta().has_value());
    EXPECT_EQ(pointwiseAttributes.get_swish_beta().value(), 1.5f);
    EXPECT_TRUE(pointwiseAttributes.get_elu_alpha().has_value());
    EXPECT_EQ(pointwiseAttributes.get_elu_alpha().value(), 0.9f);
    EXPECT_TRUE(pointwiseAttributes.get_softplus_beta().has_value());
    EXPECT_EQ(pointwiseAttributes.get_softplus_beta().value(), 2.0f);
}

TEST(TestPointwiseAttributes, CreatePointwiseAttributesWithTwoInputs)
{
    PointwiseAttributes pointwiseAttributes;

    pointwiseAttributes.set_input_0(std::make_shared<TensorAttributes>());
    pointwiseAttributes.set_input_1(std::make_shared<TensorAttributes>());
    pointwiseAttributes.set_output_0(std::make_shared<TensorAttributes>());
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto inputTensor0 = pointwiseAttributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto inputTensor1 = pointwiseAttributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = pointwiseAttributes.get_output_0();
    outputTensor->set_uid(3).set_name("OutputTensor");

    EXPECT_EQ(inputTensor0->get_uid(), 1);
    EXPECT_EQ(inputTensor0->get_name(), "InputTensor0");
    EXPECT_EQ(inputTensor1->get_uid(), 2);
    EXPECT_EQ(inputTensor1->get_name(), "InputTensor1");
    EXPECT_EQ(outputTensor->get_uid(), 3);
    EXPECT_EQ(outputTensor->get_name(), "OutputTensor");
    EXPECT_EQ(pointwiseAttributes.get_mode(), PointwiseMode::RELU_FWD);
}

TEST(TestPointwiseAttributes, SetInput0WithMove)
{
    PointwiseAttributes pointwiseAttributes;

    auto inputTensor = std::make_shared<TensorAttributes>();
    inputTensor->set_uid(1).set_name("InputTensor0");

    auto rawPtr = inputTensor.get();

    pointwiseAttributes.set_input_0(std::move(inputTensor));

    auto retrieved = pointwiseAttributes.get_input_0();
    EXPECT_EQ(retrieved->get_uid(), 1);
    EXPECT_EQ(retrieved->get_name(), "InputTensor0");

    EXPECT_EQ(inputTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestPointwiseAttributes, SetInput1WithMove)
{
    PointwiseAttributes pointwiseAttributes;

    auto inputTensor = std::make_shared<TensorAttributes>();
    inputTensor->set_uid(2).set_name("InputTensor1");

    auto rawPtr = inputTensor.get();

    pointwiseAttributes.set_input_1(std::move(inputTensor));

    auto retrieved = pointwiseAttributes.get_input_1();
    EXPECT_EQ(retrieved->get_uid(), 2);
    EXPECT_EQ(retrieved->get_name(), "InputTensor1");

    EXPECT_EQ(inputTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestPointwiseAttributes, SetInput2WithMove)
{
    PointwiseAttributes pointwiseAttributes;

    auto inputTensor = std::make_shared<TensorAttributes>();
    inputTensor->set_uid(3).set_name("InputTensor2");

    auto rawPtr = inputTensor.get();

    pointwiseAttributes.set_input_2(std::move(inputTensor));

    auto retrieved = pointwiseAttributes.get_input_2();
    EXPECT_EQ(retrieved->get_uid(), 3);
    EXPECT_EQ(retrieved->get_name(), "InputTensor2");

    EXPECT_EQ(inputTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

TEST(TestPointwiseAttributes, SetOutput0WithMove)
{
    PointwiseAttributes pointwiseAttributes;

    auto outputTensor = std::make_shared<TensorAttributes>();
    outputTensor->set_uid(4).set_name("OutputTensor");

    auto rawPtr = outputTensor.get();

    pointwiseAttributes.set_output_0(std::move(outputTensor));

    auto retrieved = pointwiseAttributes.get_output_0();
    EXPECT_EQ(retrieved->get_uid(), 4);
    EXPECT_EQ(retrieved->get_name(), "OutputTensor");

    EXPECT_EQ(outputTensor, nullptr);
    EXPECT_EQ(retrieved.get(), rawPtr);
}

// Simplified move tests - testing move semantics without setting uid/name

TEST(TestPointwiseAttributes, SimplifiedSetInput0WithMove)
{
    PointwiseAttributes pointwiseAttributes;

    auto inputTensor = std::make_shared<TensorAttributes>();
    pointwiseAttributes.set_input_0(std::move(inputTensor));

    // Just verify the tensor was set
    EXPECT_NE(pointwiseAttributes.get_input_0(), nullptr);
}

TEST(TestPointwiseAttributes, SimplifiedSetInput1WithMove)
{
    PointwiseAttributes pointwiseAttributes;

    auto inputTensor = std::make_shared<TensorAttributes>();
    pointwiseAttributes.set_input_1(std::move(inputTensor));

    // Just verify the tensor was set
    EXPECT_NE(pointwiseAttributes.get_input_1(), nullptr);
}

TEST(TestPointwiseAttributes, SimplifiedSetInput2WithMove)
{
    PointwiseAttributes pointwiseAttributes;

    auto inputTensor = std::make_shared<TensorAttributes>();
    pointwiseAttributes.set_input_2(std::move(inputTensor));

    // Just verify the tensor was set
    EXPECT_NE(pointwiseAttributes.get_input_2(), nullptr);
}

TEST(TestPointwiseAttributes, SimplifiedSetOutput0WithMove)
{
    PointwiseAttributes pointwiseAttributes;

    auto outputTensor = std::make_shared<TensorAttributes>();
    pointwiseAttributes.set_output_0(std::move(outputTensor));

    // Just verify the tensor was set
    EXPECT_NE(pointwiseAttributes.get_output_0(), nullptr);
}

TEST(TestPointwiseAttributes, CreatePointwiseAttributesWithThreeInputs)
{
    PointwiseAttributes pointwiseAttributes;

    pointwiseAttributes.set_input_0(std::make_shared<TensorAttributes>());
    pointwiseAttributes.set_input_1(std::make_shared<TensorAttributes>());
    pointwiseAttributes.set_input_2(std::make_shared<TensorAttributes>());
    pointwiseAttributes.set_output_0(std::make_shared<TensorAttributes>());
    pointwiseAttributes.set_mode(PointwiseMode::RELU_FWD);

    auto inputTensor0 = pointwiseAttributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto inputTensor1 = pointwiseAttributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto inputTensor2 = pointwiseAttributes.get_input_2();
    inputTensor2->set_uid(3)
        .set_name("InputTensor2")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = pointwiseAttributes.get_output_0();
    outputTensor->set_uid(4).set_name("OutputTensor");

    EXPECT_EQ(inputTensor0->get_uid(), 1);
    EXPECT_EQ(inputTensor0->get_name(), "InputTensor0");
    EXPECT_EQ(inputTensor1->get_uid(), 2);
    EXPECT_EQ(inputTensor1->get_name(), "InputTensor1");
    EXPECT_EQ(inputTensor2->get_uid(), 3);
    EXPECT_EQ(inputTensor2->get_name(), "InputTensor2");
    EXPECT_EQ(outputTensor->get_uid(), 4);
    EXPECT_EQ(outputTensor->get_name(), "OutputTensor");
    EXPECT_EQ(pointwiseAttributes.get_mode(), PointwiseMode::RELU_FWD);
}

TEST(TestPointwiseAttributes, OptionalParameterDefaults)
{
    PointwiseAttributes pointwiseAttributes;

    EXPECT_FALSE(pointwiseAttributes.get_relu_lower_clip().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_relu_upper_clip().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_relu_lower_clip_slope().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_axis().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_swish_beta().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_elu_alpha().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_softplus_beta().has_value());
}
