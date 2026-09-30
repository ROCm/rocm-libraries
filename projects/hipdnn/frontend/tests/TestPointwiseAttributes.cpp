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
    const PointwiseAttributes pointwiseAttributes;

    EXPECT_FALSE(pointwiseAttributes.get_relu_lower_clip().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_relu_upper_clip().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_relu_lower_clip_slope().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_axis().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_swish_beta().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_elu_alpha().has_value());
    EXPECT_FALSE(pointwiseAttributes.get_softplus_beta().has_value());
}

TEST(TestPointwiseAttributes, LogicalAndStrictEquality)
{
    hipdnn_frontend::graph::PointwiseAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_mode(hipdnn_frontend::PointwiseMode::RELU_FWD);
    attr1.set_relu_lower_clip(0.0f);
    attr1.set_relu_upper_clip(6.0f);
    attr1.set_relu_lower_clip_slope(0.01f);
    attr1.set_axis(1);
    attr1.set_swish_beta(1.0f);
    attr1.set_elu_alpha(1.0f);
    attr1.set_softplus_beta(1.0f);

    auto in01 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    in01->set_uid(1).set_name("IN_0").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_input_0(in01);

    auto out01 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    out01->set_uid(2).set_name("OUT_0").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_output_0(out01);

    hipdnn_frontend::graph::PointwiseAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_mode(hipdnn_frontend::PointwiseMode::RELU_FWD);
    attr2.set_relu_lower_clip(0.0f);
    attr2.set_relu_upper_clip(6.0f);
    attr2.set_relu_lower_clip_slope(0.01f);
    attr2.set_axis(1);
    attr2.set_swish_beta(1.0f);
    attr2.set_elu_alpha(1.0f);
    attr2.set_softplus_beta(1.0f);

    auto in02 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    in02->set_uid(1).set_name("IN_0").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_input_0(in02);

    auto out02 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    out02->set_uid(2).set_name("OUT_0").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_output_0(out02);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchIn0 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    structuralMismatchIn0->set_uid(99).set_name("MismatchedIn0");
    attr2.set_input_0(structuralMismatchIn0);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_input_0(in02); // Revert

    // Mode mismatch: semantic, must fail both checks
    attr2.set_mode(hipdnn_frontend::PointwiseMode::SIGMOID_FWD);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_mode(hipdnn_frontend::PointwiseMode::RELU_FWD); // Revert

    // relu_lower_clip mismatch: semantic, must fail both checks
    attr2.set_relu_lower_clip(0.5f);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_relu_lower_clip(0.0f); // Revert

    // relu_upper_clip mismatch: semantic, must fail both checks
    attr2.set_relu_upper_clip(3.0f);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_relu_upper_clip(6.0f); // Revert

    // relu_lower_clip_slope mismatch: semantic, must fail both checks
    attr2.set_relu_lower_clip_slope(0.02f);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_relu_lower_clip_slope(0.01f); // Revert

    // axis mismatch: semantic, must fail both checks
    attr2.set_axis(2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_axis(1); // Revert

    // swish_beta mismatch: semantic, must fail both checks
    attr2.set_swish_beta(2.0f);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_swish_beta(1.0f); // Revert

    // elu_alpha mismatch: semantic, must fail both checks
    attr2.set_elu_alpha(2.0f);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_elu_alpha(1.0f); // Revert

    // softplus_beta mismatch: semantic, must fail both checks
    attr2.set_softplus_beta(2.0f);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_softplus_beta(1.0f); // Revert

    // Unset-vs-unset optional fields: two attrs with the same field left
    // unset should still compare equal (std::optional::operator== on two
    // disengaged optionals is true)
    hipdnn_frontend::graph::PointwiseAttributes sparse1;
    sparse1.set_mode(hipdnn_frontend::PointwiseMode::ADD);
    hipdnn_frontend::graph::PointwiseAttributes sparse2;
    sparse2.set_mode(hipdnn_frontend::PointwiseMode::ADD);
    EXPECT_TRUE(sparse1 == sparse2);
    EXPECT_TRUE(sparse1.logicallyEquals(sparse2));

    // Set-vs-unset should differ
    sparse2.set_axis(0);
    EXPECT_FALSE(sparse1 == sparse2);
    EXPECT_FALSE(sparse1.logicallyEquals(sparse2));

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchIn0 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    logicalMatchIn0
        ->set_uid(555) // Diverges from attr1's in01 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's in01 ("IN_0")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_input_0(logicalMatchIn0);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}
