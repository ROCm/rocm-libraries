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
    CustomOpAttributes attr;

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
    std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0xFF};

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

TEST(TestCustomOpAttributes, PackAndFromFlatBuffer)
{
    auto inputA = std::make_shared<TensorAttributes>();
    inputA->set_name("input_a").set_uid(10).set_dim({2, 3}).set_stride({3, 1}).set_data_type(
        DataType::FLOAT);

    auto inputB = std::make_shared<TensorAttributes>();
    inputB->set_name("input_b").set_uid(20).set_dim({2, 3}).set_stride({3, 1}).set_data_type(
        DataType::FLOAT);

    auto output = std::make_shared<TensorAttributes>();
    output->set_name("output").set_uid(30).set_dim({2, 3}).set_stride({3, 1}).set_data_type(
        DataType::FLOAT);

    std::vector<uint8_t> payload = {0xDE, 0xAD, 0xBE, 0xEF};

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({inputA, inputB})
        .set_outputs({output})
        .set_data(payload);

    // Pack
    flatbuffers::FlatBufferBuilder builder;
    auto offset = attr.pack_attributes(builder);
    builder.Finish(offset);

    // Verify and deserialize
    auto buf = builder.GetBufferPointer();
    auto fbAttr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::CustomOpAttributes>(buf);

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    tensorMap[10] = inputA;
    tensorMap[20] = inputB;
    tensorMap[30] = output;

    auto deserialized = CustomOpAttributes::fromFlatBuffer(fbAttr, tensorMap);

    EXPECT_EQ(deserialized.get_custom_op_id(), "example.my_add");
    EXPECT_EQ(deserialized.get_inputs().size(), 2u);
    EXPECT_EQ(deserialized.get_outputs().size(), 1u);
    EXPECT_EQ(deserialized.get_inputs()[0]->get_uid(), 10);
    EXPECT_EQ(deserialized.get_inputs()[1]->get_uid(), 20);
    EXPECT_EQ(deserialized.get_outputs()[0]->get_uid(), 30);
    EXPECT_EQ(deserialized.get_data(), payload);
}

TEST(TestCustomOpAttributes, PackAndFromFlatBufferWithEmptyPayload)
{
    auto input = std::make_shared<TensorAttributes>();
    input->set_name("input").set_uid(1).set_dim({2, 3}).set_stride({3, 1}).set_data_type(
        DataType::FLOAT);

    auto output = std::make_shared<TensorAttributes>();
    output->set_name("output").set_uid(2).set_dim({2, 3}).set_stride({3, 1}).set_data_type(
        DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.passthrough")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({input})
        .set_outputs({output});

    flatbuffers::FlatBufferBuilder builder;
    auto offset = attr.pack_attributes(builder);
    builder.Finish(offset);

    auto buf = builder.GetBufferPointer();
    auto fbAttr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::CustomOpAttributes>(buf);

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    tensorMap[1] = input;
    tensorMap[2] = output;

    auto deserialized = CustomOpAttributes::fromFlatBuffer(fbAttr, tensorMap);

    EXPECT_EQ(deserialized.get_custom_op_id(), "example.passthrough");
    EXPECT_TRUE(deserialized.get_data().empty());
}

TEST(TestCustomOpAttributes, PackAndFromFlatBufferWithMultipleOutputs)
{
    auto input = std::make_shared<TensorAttributes>();
    input->set_name("input").set_uid(1).set_dim({4, 8}).set_stride({8, 1}).set_data_type(
        DataType::FLOAT);

    auto outputA = std::make_shared<TensorAttributes>();
    outputA->set_name("output_a")
        .set_uid(2)
        .set_dim({4, 4})
        .set_stride({4, 1})
        .set_data_type(DataType::FLOAT);

    auto outputB = std::make_shared<TensorAttributes>();
    outputB->set_name("output_b")
        .set_uid(3)
        .set_dim({4, 4})
        .set_stride({4, 1})
        .set_data_type(DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_split")
        .set_custom_op_id("example.split")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({input})
        .set_outputs({outputA, outputB});

    flatbuffers::FlatBufferBuilder builder;
    auto offset = attr.pack_attributes(builder);
    builder.Finish(offset);

    auto buf = builder.GetBufferPointer();
    auto fbAttr = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::CustomOpAttributes>(buf);

    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> tensorMap;
    tensorMap[1] = input;
    tensorMap[2] = outputA;
    tensorMap[3] = outputB;

    auto deserialized = CustomOpAttributes::fromFlatBuffer(fbAttr, tensorMap);

    EXPECT_EQ(deserialized.get_outputs().size(), 2u);
    EXPECT_EQ(deserialized.get_outputs()[0]->get_uid(), 2);
    EXPECT_EQ(deserialized.get_outputs()[1]->get_uid(), 3);
}
