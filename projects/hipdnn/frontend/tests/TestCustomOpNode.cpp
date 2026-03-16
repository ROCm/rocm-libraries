// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/node/CustomOpNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{
std::shared_ptr<TensorAttributes> makeTensor(const std::string& name,
                                             int64_t uid,
                                             const std::vector<int64_t>& dims,
                                             DataType dtype)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_name(name)
        .set_uid(uid)
        .set_dim(dims)
        .set_stride(hipdnn_data_sdk::utilities::generateStrides(dims))
        .set_data_type(dtype);
    return tensor;
}
} // namespace

TEST(TestCustomOpNode, GetNodeName)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("my_custom_op");

    CustomOpNode node(std::move(attr), graphAttrs);
    EXPECT_EQ(node.getNodeName(), "my_custom_op");
}

TEST(TestCustomOpNode, GatherTensors)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op").set_inputs({input}).set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_EQ(allTensors.size(), 2u);
    EXPECT_TRUE(allTensors.count(input));
    EXPECT_TRUE(allTensors.count(output));
}

TEST(TestCustomOpNode, GetInputOutputTensorAttributes)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto inputA = makeTensor("input_a", 1, {2, 3}, DataType::FLOAT);
    auto inputB = makeTensor("input_b", 2, {2, 3}, DataType::FLOAT);
    auto output = makeTensor("output", 3, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op").set_inputs({inputA, inputB}).set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    auto inputs = node.getNodeInputTensorAttributes();
    auto outputs = node.getNodeOutputTensorAttributes();

    EXPECT_EQ(inputs.size(), 2u);
    EXPECT_EQ(outputs.size(), 1u);
}

TEST(TestCustomOpNode, PreValidateSucceeds)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_inputs({input})
        .set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, ErrorCode::OK);
}

TEST(TestCustomOpNode, PreValidateSucceedsWithMultipleInputsAndOutputs)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto inputA = makeTensor("input_a", 1, {2, 3}, DataType::FLOAT);
    auto inputB = makeTensor("input_b", 2, {2, 3}, DataType::FLOAT);
    auto inputC = makeTensor("input_c", 3, {2, 3}, DataType::FLOAT);
    auto outputA = makeTensor("output_a", 4, {2, 3}, DataType::FLOAT);
    auto outputB = makeTensor("output_b", 5, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.multi_io")
        .set_inputs({inputA, inputB, inputC})
        .set_outputs({outputA, outputB});

    CustomOpNode node(std::move(attr), graphAttrs);

    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, ErrorCode::OK);
}

TEST(TestCustomOpNode, PreValidateSucceedsWithNoInputs)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto output = makeTensor("output", 1, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op").set_custom_op_id("example.generator").set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, ErrorCode::OK);
}

TEST(TestCustomOpNode, PreValidateFailsWithNullInput)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_inputs({nullptr})
        .set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
}

TEST(TestCustomOpNode, PreValidateFailsWithNullOutput)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_inputs({input})
        .set_outputs({nullptr});

    CustomOpNode node(std::move(attr), graphAttrs);

    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
}

TEST(TestCustomOpNode, PostValidateSucceeds)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({input})
        .set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    auto err = node.post_validate_node();
    EXPECT_EQ(err.code, ErrorCode::OK);
}

TEST(TestCustomOpNode, PostValidateFailsWithoutCustomOpId)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({input})
        .set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    auto err = node.post_validate_node();
    EXPECT_EQ(err.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestCustomOpNode, PostValidateFailsWithoutComputeDataType)
{
    GraphAttributes graphAttrs;

    auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_inputs({input})
        .set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    auto err = node.post_validate_node();
    EXPECT_EQ(err.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestCustomOpNode, InferPropertiesPropagatesContext)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);
    graphAttrs.set_io_data_type(DataType::HALF);

    auto input = std::make_shared<TensorAttributes>();
    input->set_uid(1).set_dim({2, 3}).set_stride({3, 1});

    auto output = std::make_shared<TensorAttributes>();
    output->set_uid(2).set_dim({2, 3}).set_stride({3, 1});

    CustomOpAttributes attr;
    attr.set_name("test_op").set_inputs({input}).set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, ErrorCode::OK);
    EXPECT_EQ(input->get_data_type(), DataType::HALF);
    EXPECT_EQ(output->get_data_type(), DataType::HALF);
}

TEST(TestCustomOpNode, PackNode)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    std::vector<uint8_t> payload = {0xCA, 0xFE};

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({input})
        .set_outputs({output})
        .set_data(payload);

    CustomOpNode node(std::move(attr), graphAttrs);

    flatbuffers::FlatBufferBuilder builder;
    auto nodeOffset = node.pack_node(builder);
    builder.Finish(nodeOffset);

    auto buf = builder.GetBufferPointer();
    auto fbNode = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(buf);

    EXPECT_EQ(fbNode->attributes_type(),
              hipdnn_data_sdk::data_objects::NodeAttributes::CustomOpAttributes);
    EXPECT_STREQ(fbNode->name()->c_str(), "test_op");

    auto fbCustomOp = fbNode->attributes_as_CustomOpAttributes();
    ASSERT_NE(fbCustomOp, nullptr);
    EXPECT_STREQ(fbCustomOp->custom_op_id()->c_str(), "example.my_add");
    EXPECT_EQ(fbCustomOp->input_tensor_uids()->size(), 1u);
    EXPECT_EQ(fbCustomOp->input_tensor_uids()->Get(0), 1);
    EXPECT_EQ(fbCustomOp->output_tensor_uids()->size(), 1u);
    EXPECT_EQ(fbCustomOp->output_tensor_uids()->Get(0), 2);
    EXPECT_EQ(fbCustomOp->data()->size(), 2u);
    EXPECT_EQ(fbCustomOp->data()->Get(0), 0xCA);
    EXPECT_EQ(fbCustomOp->data()->Get(1), 0xFE);
}

TEST(TestCustomOpNode, PackNodeWithEmptyPayload)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({input})
        .set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    flatbuffers::FlatBufferBuilder builder;
    auto nodeOffset = node.pack_node(builder);
    builder.Finish(nodeOffset);

    auto buf = builder.GetBufferPointer();
    auto fbNode = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(buf);
    auto fbCustomOp = fbNode->attributes_as_CustomOpAttributes();
    ASSERT_NE(fbCustomOp, nullptr);
    EXPECT_EQ(fbCustomOp->data()->size(), 0u);
}

TEST(TestCustomOpNode, PackNodeWithMultipleOutputs)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    auto input = makeTensor("input", 1, {4, 8}, DataType::FLOAT);
    auto outputA = makeTensor("output_a", 2, {4, 4}, DataType::FLOAT);
    auto outputB = makeTensor("output_b", 3, {4, 4}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_split")
        .set_custom_op_id("example.split")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({input})
        .set_outputs({outputA, outputB});

    CustomOpNode node(std::move(attr), graphAttrs);

    flatbuffers::FlatBufferBuilder builder;
    auto nodeOffset = node.pack_node(builder);
    builder.Finish(nodeOffset);

    auto buf = builder.GetBufferPointer();
    auto fbNode = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(buf);
    auto fbCustomOp = fbNode->attributes_as_CustomOpAttributes();
    ASSERT_NE(fbCustomOp, nullptr);
    EXPECT_EQ(fbCustomOp->input_tensor_uids()->size(), 1u);
    EXPECT_EQ(fbCustomOp->output_tensor_uids()->size(), 2u);
    EXPECT_EQ(fbCustomOp->output_tensor_uids()->Get(0), 2);
    EXPECT_EQ(fbCustomOp->output_tensor_uids()->Get(1), 3);
}
