// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/node/CustomOpNode.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>

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

    const CustomOpNode node(std::move(attr), graphAttrs);
    EXPECT_EQ(node.getNodeName(), "my_custom_op");
}

TEST(TestCustomOpNode, GatherTensors)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    const auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op").set_inputs({input}).set_outputs({output});

    const CustomOpNode node(std::move(attr), graphAttrs);

    hipdnn_frontend::detail::OrderedTensorSet allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_EQ(allTensors.size(), 2u);
    EXPECT_TRUE(allTensors.count(input));
    EXPECT_TRUE(allTensors.count(output));
}

TEST(TestCustomOpNode, GatherTensorsIncludesRaggedOffsetAux)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    // The aux tensors carry no UID; gathering them is what lets the graph assign one.
    auto inputAux = std::make_shared<TensorAttributes>();
    inputAux->set_dim({5}).set_stride({1}).set_data_type(DataType::INT32);
    auto outputAux = std::make_shared<TensorAttributes>();
    outputAux->set_dim({5}).set_stride({1}).set_data_type(DataType::INT32);

    auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    input->set_ragged_offset(inputAux);
    auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);
    output->set_ragged_offset(outputAux);

    CustomOpAttributes attr;
    attr.set_name("test_op").set_inputs({input}).set_outputs({output});

    const CustomOpNode node(std::move(attr), graphAttrs);

    hipdnn_frontend::detail::OrderedTensorSet allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_EQ(allTensors.size(), 4u);
    EXPECT_TRUE(allTensors.count(input));
    EXPECT_TRUE(allTensors.count(output));
    EXPECT_TRUE(allTensors.count(inputAux));
    EXPECT_TRUE(allTensors.count(outputAux));
}

TEST(TestCustomOpNode, GetInputOutputTensorAttributes)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const auto inputA = makeTensor("input_a", 1, {2, 3}, DataType::FLOAT);
    const auto inputB = makeTensor("input_b", 2, {2, 3}, DataType::FLOAT);
    const auto output = makeTensor("output", 3, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op").set_inputs({inputA, inputB}).set_outputs({output});

    const CustomOpNode node(std::move(attr), graphAttrs);

    const auto inputs = node.getNodeInputTensorAttributes();
    const auto outputs = node.getNodeOutputTensorAttributes();

    EXPECT_EQ(inputs.size(), 2u);
    EXPECT_EQ(outputs.size(), 1u);
}

TEST(TestCustomOpNode, PreValidateSucceeds)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    const auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_inputs({input})
        .set_outputs({output});

    const CustomOpNode node(std::move(attr), graphAttrs);

    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, ErrorCode::OK);
}

TEST(TestCustomOpNode, PreValidateSucceedsWithMultipleInputsAndOutputs)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const auto inputA = makeTensor("input_a", 1, {2, 3}, DataType::FLOAT);
    const auto inputB = makeTensor("input_b", 2, {2, 3}, DataType::FLOAT);
    const auto inputC = makeTensor("input_c", 3, {2, 3}, DataType::FLOAT);
    const auto outputA = makeTensor("output_a", 4, {2, 3}, DataType::FLOAT);
    const auto outputB = makeTensor("output_b", 5, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.multi_io")
        .set_inputs({inputA, inputB, inputC})
        .set_outputs({outputA, outputB});

    const CustomOpNode node(std::move(attr), graphAttrs);

    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, ErrorCode::OK);
}

TEST(TestCustomOpNode, PreValidateSucceedsWithNoInputs)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const auto output = makeTensor("output", 1, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op").set_custom_op_id("example.generator").set_outputs({output});

    const CustomOpNode node(std::move(attr), graphAttrs);

    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, ErrorCode::OK);
}

TEST(TestCustomOpNode, PreValidateFailsWithNullInput)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_inputs({nullptr})
        .set_outputs({output});

    const CustomOpNode node(std::move(attr), graphAttrs);

    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
}

TEST(TestCustomOpNode, PreValidateFailsWithNullOutput)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_inputs({input})
        .set_outputs({nullptr});

    const CustomOpNode node(std::move(attr), graphAttrs);

    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, ErrorCode::INVALID_VALUE);
}

TEST(TestCustomOpNode, PostValidateSucceeds)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    const auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({input})
        .set_outputs({output});

    const CustomOpNode node(std::move(attr), graphAttrs);

    const auto err = node.post_validate_node();
    EXPECT_EQ(err.code, ErrorCode::OK);
}

TEST(TestCustomOpNode, PostValidateFailsWithoutCustomOpId)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    const auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({input})
        .set_outputs({output});

    const CustomOpNode node(std::move(attr), graphAttrs);

    const auto err = node.post_validate_node();
    EXPECT_EQ(err.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestCustomOpNode, PostValidateFailsWithoutComputeDataType)
{
    const GraphAttributes graphAttrs;

    const auto input = makeTensor("input", 1, {2, 3}, DataType::FLOAT);
    const auto output = makeTensor("output", 2, {2, 3}, DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_name("test_op")
        .set_custom_op_id("example.my_add")
        .set_inputs({input})
        .set_outputs({output});

    const CustomOpNode node(std::move(attr), graphAttrs);

    const auto err = node.post_validate_node();
    EXPECT_EQ(err.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_NE(err.get_message().find("compute_data_type"), std::string::npos) << err.get_message();
    EXPECT_NE(err.get_message().find("Set the graph data types before adding nodes"),
              std::string::npos)
        << err.get_message();
}

TEST(TestCustomOpNode, InferPropertiesPropagatesContext)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);
    graphAttrs.set_io_data_type(DataType::HALF);

    const auto input = std::make_shared<TensorAttributes>();
    input->set_uid(1).set_dim({2, 3}).set_stride({3, 1});

    const auto output = std::make_shared<TensorAttributes>();
    output->set_uid(2).set_dim({2, 3}).set_stride({3, 1});

    CustomOpAttributes attr;
    attr.set_name("test_op").set_inputs({input}).set_outputs({output});

    CustomOpNode node(std::move(attr), graphAttrs);

    const auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, ErrorCode::OK);
    EXPECT_EQ(input->get_data_type(), DataType::HALF);
    EXPECT_EQ(output->get_data_type(), DataType::HALF);
}

TEST(TestCustomOpNode, GetNodeTypeReturnsCustomOp)
{
    const GraphAttributes graphAttrs;
    const CustomOpNode node(CustomOpAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::CUSTOM_OP);
}

TEST(TestCustomOpNode, NodeEqualsItselfAndEquivalentNodes)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const auto makeAttr = [](const std::string& name) {
        CustomOpAttributes attr;
        attr.set_name(name)
            .set_custom_op_id("example.my_add")
            .set_compute_data_type(DataType::FLOAT)
            .set_inputs({makeTensor("input", 1, {2, 3}, DataType::FLOAT)})
            .set_outputs({makeTensor("output", 2, {2, 3}, DataType::FLOAT)})
            .set_data({1, 2, 3});
        return attr;
    };

    const CustomOpNode node(makeAttr("op"), graphAttrs);
    const INode& sameObject = node;
    EXPECT_TRUE(node == sameObject);
    EXPECT_TRUE(node.logicallyEquals(sameObject));

    const CustomOpNode sameNode(makeAttr("op"), graphAttrs);
    EXPECT_TRUE(node == sameNode);
    EXPECT_TRUE(node.logicallyEquals(sameNode));

    const CustomOpNode renamedNode(makeAttr("renamed_op"), graphAttrs);
    EXPECT_FALSE(node == renamedNode);
    EXPECT_TRUE(node.logicallyEquals(renamedNode));

    auto otherIdAttr = makeAttr("op");
    otherIdAttr.set_custom_op_id("example.my_mul");
    const CustomOpNode otherIdNode(std::move(otherIdAttr), graphAttrs);
    EXPECT_FALSE(node == otherIdNode);
    EXPECT_FALSE(node.logicallyEquals(otherIdNode));
}

TEST(TestCustomOpNode, NodeNotEqualToOtherNodeType)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    const CustomOpNode node(CustomOpAttributes{}, graphAttrs);
    const CustomOpNode emptyNode(CustomOpAttributes{}, graphAttrs);
    const PointwiseNode pointwiseNode(PointwiseAttributes{}, graphAttrs);

    EXPECT_TRUE(node == emptyNode);
    EXPECT_FALSE(node == pointwiseNode);
    EXPECT_FALSE(node.logicallyEquals(pointwiseNode));
    EXPECT_FALSE(pointwiseNode == node);
}
