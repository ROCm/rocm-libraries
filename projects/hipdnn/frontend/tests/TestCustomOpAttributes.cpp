// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestCustomOpAttributes, DefaultValues)
{
    const CustomOpAttributes attr;

    EXPECT_TRUE(attr.get_name().empty());
    EXPECT_TRUE(attr.get_custom_op_id().empty());
    EXPECT_EQ(attr.get_compute_data_type(), DataType::NOT_SET);
    EXPECT_TRUE(attr.get_inputs().empty());
    EXPECT_TRUE(attr.get_outputs().empty());
    EXPECT_TRUE(attr.get_data().empty());
}

TEST(TestCustomOpAttributes, SettersAndGetters)
{
    CustomOpAttributes attr;
    attr.set_name("test_custom_op")
        .set_custom_op_id("example.my_add")
        .set_compute_data_type(DataType::FLOAT);

    EXPECT_EQ(attr.get_name(), "test_custom_op");
    EXPECT_EQ(attr.get_custom_op_id(), "example.my_add");
    EXPECT_EQ(attr.get_compute_data_type(), DataType::FLOAT);
}

TEST(TestCustomOpAttributes, InputsAndOutputs)
{
    auto inputA = std::make_shared<TensorAttributes>();
    inputA->set_name("input_a").set_uid(1);

    auto inputB = std::make_shared<TensorAttributes>();
    inputB->set_name("input_b").set_uid(2);

    auto output = std::make_shared<TensorAttributes>();
    output->set_name("output").set_uid(3);

    CustomOpAttributes attr;
    attr.set_inputs({inputA, inputB}).set_outputs({output});

    EXPECT_EQ(attr.get_inputs().size(), 2u);
    EXPECT_EQ(attr.get_outputs().size(), 1u);
    EXPECT_EQ(attr.get_inputs()[0]->get_uid(), 1);
    EXPECT_EQ(attr.get_inputs()[1]->get_uid(), 2);
    EXPECT_EQ(attr.get_outputs()[0]->get_uid(), 3);
}

TEST(TestCustomOpAttributes, Data)
{
    const std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0xFF};

    CustomOpAttributes attr;
    attr.set_data(payload);

    EXPECT_EQ(attr.get_data(), payload);
}

TEST(TestCustomOpAttributes, FillFromContext)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);
    graphAttrs.set_io_data_type(DataType::HALF);

    auto inputA = std::make_shared<TensorAttributes>();
    inputA->set_uid(1).set_dim({2, 3}).set_stride({3, 1});

    auto output = std::make_shared<TensorAttributes>();
    output->set_uid(2).set_dim({2, 3}).set_stride({3, 1});

    CustomOpAttributes attr;
    attr.set_inputs({inputA}).set_outputs({output});

    auto err = attr.fill_from_context(graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    EXPECT_EQ(attr.get_compute_data_type(), DataType::FLOAT);

    // Tensors should have their data_type filled from graph context
    EXPECT_EQ(inputA->get_data_type(), DataType::HALF);
    EXPECT_EQ(output->get_data_type(), DataType::HALF);
}

TEST(TestCustomOpAttributes, FillFromContextPreservesExplicitComputeType)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_compute_data_type(DataType::HALF);

    auto err = attr.fill_from_context(graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    EXPECT_EQ(attr.get_compute_data_type(), DataType::HALF);
}
