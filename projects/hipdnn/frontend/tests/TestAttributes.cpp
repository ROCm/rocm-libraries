// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/Attributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

struct Dummy : Attributes<Dummy>
{
    std::unordered_map<int, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<int, std::shared_ptr<TensorAttributes>> outputs;
};

TEST(TestAttributes, FillFromContext)
{
    GraphAttributes graphAttributes;
    graphAttributes.set_compute_data_type(DataType::INT32);
    graphAttributes.set_intermediate_data_type(DataType::BFLOAT16);
    graphAttributes.set_io_data_type(DataType::HALF);

    const int nonVirtualId = 0;
    const int virtualId = 1;
    const int setTypeId = 2;

    Dummy attributes;

    attributes.inputs[nonVirtualId] = std::make_shared<TensorAttributes>();
    attributes.inputs[virtualId] = std::make_shared<TensorAttributes>();
    attributes.inputs[virtualId]->set_is_virtual(true);
    attributes.inputs[setTypeId] = std::make_shared<TensorAttributes>();
    attributes.inputs[setTypeId]->set_data_type(DataType::FLOAT);

    attributes.outputs[nonVirtualId] = std::make_shared<TensorAttributes>();
    attributes.outputs[virtualId] = std::make_shared<TensorAttributes>();
    attributes.outputs[virtualId]->set_is_virtual(true);
    attributes.outputs[setTypeId] = std::make_shared<TensorAttributes>();
    attributes.outputs[setTypeId]->set_data_type(DataType::DOUBLE);

    EXPECT_EQ(attributes.get_compute_data_type(), DataType::NOT_SET);

    attributes.fill_from_context(graphAttributes);

    EXPECT_EQ(attributes.get_compute_data_type(), graphAttributes.get_compute_data_type());

    EXPECT_EQ(attributes.inputs[nonVirtualId]->get_data_type(), graphAttributes.get_io_data_type());
    EXPECT_EQ(attributes.inputs[virtualId]->get_data_type(),
              graphAttributes.get_intermediate_data_type());
    EXPECT_EQ(attributes.inputs[setTypeId]->get_data_type(), DataType::FLOAT);

    EXPECT_EQ(attributes.outputs[nonVirtualId]->get_data_type(),
              graphAttributes.get_io_data_type());
    EXPECT_EQ(attributes.outputs[virtualId]->get_data_type(),
              graphAttributes.get_intermediate_data_type());
    EXPECT_EQ(attributes.outputs[setTypeId]->get_data_type(), DataType::DOUBLE);
}

TEST(TestAttributes, FillFromContextSetNodeComputeTypeNotOverriden)
{
    GraphAttributes graphAttributes;
    graphAttributes.set_compute_data_type(DataType::INT32);

    auto nodeDataType = hipdnn_frontend::DataType::INT32;

    Dummy attributes;
    attributes.set_compute_data_type(nodeDataType);

    attributes.fill_from_context(graphAttributes);

    EXPECT_EQ(attributes.get_compute_data_type(), nodeDataType);
}

TEST(TestAttributes, FillFromContextSetNodeComputeTypeDoesNotUnset)
{
    GraphAttributes graphAttributes;
    graphAttributes.set_compute_data_type(DataType::NOT_SET);

    auto nodeDataType = hipdnn_frontend::DataType::INT32;

    Dummy attributes;
    attributes.set_compute_data_type(nodeDataType);

    attributes.fill_from_context(graphAttributes);

    EXPECT_EQ(attributes.get_compute_data_type(), nodeDataType);
}
