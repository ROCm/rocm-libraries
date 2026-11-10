// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/TopkAttributes.hpp>
#include <hipdnn_frontend/node/TopkNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

namespace
{

std::shared_ptr<TensorAttributes> makeTensor(const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides,
                                             DataType dataType)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim(dims).set_stride(strides).set_data_type(dataType);
    return tensor;
}

TopkAttributes createValidAttributes()
{
    TopkAttributes attrs;
    attrs.set_x(makeTensor({2, 3, 8}, {24, 8, 1}, DataType::FLOAT));
    attrs.set_indices(makeTensor({2, 3, 4}, {12, 4, 1}, DataType::INT64));
    attrs.set_values(makeTensor({2, 3, 4}, {12, 4, 1}, DataType::FLOAT));
    attrs.set_k(4);
    return attrs;
}

Error preValidate(TopkAttributes attrs)
{
    const GraphAttributes graphAttributes;
    const TopkNode node(std::move(attrs), graphAttributes);
    return node.pre_validate_node();
}

} // namespace

TEST(TestTopkNode, GetNodeTypeReturnsTopk)
{
    const GraphAttributes graphAttrs;
    const TopkNode node(TopkAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::TOPK);
}

TEST(TestTopkNode, PreValidateNode)
{
    auto error = preValidate(createValidAttributes());
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
}

TEST(TestTopkNode, PreValidateNodeAcceptsMissingValues)
{
    auto attrs = createValidAttributes();
    attrs.outputs.erase(TopkAttributes::OutputNames::VALUES);
    auto error = preValidate(std::move(attrs));
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
}

TEST(TestTopkNode, PreValidateNodeMissingXTensor)
{
    auto attrs = createValidAttributes();
    attrs.inputs.erase(TopkAttributes::InputNames::X);
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestTopkNode, PreValidateNodeMissingIndicesTensor)
{
    auto attrs = createValidAttributes();
    attrs.outputs.erase(TopkAttributes::OutputNames::INDICES);
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestTopkNode, PreValidateNodeRejectsEmptyXDims)
{
    auto attrs = createValidAttributes();
    attrs.set_x(std::make_shared<TensorAttributes>());
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::INVALID_VALUE);
}

TEST(TestTopkNode, PreValidateNodeRejectsZeroK)
{
    auto attrs = createValidAttributes();
    attrs.set_k(0);
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::INVALID_VALUE);
}

TEST(TestTopkNode, PreValidateNodeRejectsKAboveLastDim)
{
    auto attrs = createValidAttributes();
    attrs.set_k(9);
    attrs.outputs.erase(TopkAttributes::OutputNames::VALUES);
    attrs.get_indices()->set_dim({2, 3, 9});
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::INVALID_VALUE);
}

TEST(TestTopkNode, PreValidateNodeAcceptsKEqualToLastDim)
{
    auto attrs = createValidAttributes();
    attrs.set_k(8);
    attrs.get_indices()->set_dim({2, 3, 8}).set_stride({24, 8, 1});
    attrs.get_values()->set_dim({2, 3, 8}).set_stride({24, 8, 1});
    auto error = preValidate(std::move(attrs));
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
}

TEST(TestTopkNode, PreValidateNodeAcceptsInt32Indices)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_data_type(DataType::INT32);
    auto error = preValidate(std::move(attrs));
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
}

TEST(TestTopkNode, PreValidateNodeAcceptsUnsetIndicesDataType)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_data_type(DataType::NOT_SET);
    auto error = preValidate(std::move(attrs));
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
}

TEST(TestTopkNode, PreValidateNodeRejectsFloatIndices)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_data_type(DataType::FLOAT);
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::INVALID_VALUE);
}

TEST(TestTopkNode, PreValidateNodeRejectsIndicesDimsMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_dim({2, 3, 5});
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::INVALID_VALUE);
}

TEST(TestTopkNode, PreValidateNodeRejectsIndicesRankMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_dim({6, 4});
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::INVALID_VALUE);
}

TEST(TestTopkNode, PreValidateNodeRejectsValuesDimsMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_values()->set_dim({2, 4, 4});
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::INVALID_VALUE);
}

TEST(TestTopkNode, PreValidateNodeRejectsValuesDataTypeMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_values()->set_data_type(DataType::HALF);
    EXPECT_EQ(preValidate(std::move(attrs)).code, ErrorCode::INVALID_VALUE);
}

TEST(TestTopkNode, InferPropertiesNodeFillsUnsetOutputs)
{
    TopkAttributes attrs;
    attrs.set_x(makeTensor({4, 6}, {6, 1}, DataType::BFLOAT16));
    attrs.set_indices(std::make_shared<TensorAttributes>());
    attrs.set_values(std::make_shared<TensorAttributes>());
    attrs.set_k(1);

    GraphAttributes graphAttributes;
    graphAttributes.set_io_data_type(DataType::FLOAT);
    TopkNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    const auto indices = node.attributes.get_indices();
    const auto values = node.attributes.get_values();
    EXPECT_EQ(indices->get_dim(), (std::vector<int64_t>{4, 1}));
    EXPECT_EQ(indices->get_stride(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(indices->get_data_type(), DataType::INT64);
    EXPECT_EQ(values->get_dim(), (std::vector<int64_t>{4, 1}));
    EXPECT_EQ(values->get_stride(), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(values->get_data_type(), DataType::BFLOAT16);
}

TEST(TestTopkNode, InferPropertiesNodeKeepsExplicitOutputs)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_data_type(DataType::INT32).set_stride({1, 8, 2});

    const GraphAttributes graphAttributes;
    TopkNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(node.attributes.get_indices()->get_data_type(), DataType::INT32);
    EXPECT_EQ(node.attributes.get_indices()->get_stride(), (std::vector<int64_t>{1, 8, 2}));
}

TEST(TestTopkNode, InferPropertiesNodeWithoutValues)
{
    auto attrs = createValidAttributes();
    attrs.outputs.erase(TopkAttributes::OutputNames::VALUES);

    const GraphAttributes graphAttributes;
    TopkNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(node.attributes.get_values(), nullptr);
}

TEST(TestTopkNode, InferPropertiesNodeRejectsMismatchedIndicesDims)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_dim({2, 3, 3});

    const GraphAttributes graphAttributes;
    TopkNode node(std::move(attrs), graphAttributes);

    EXPECT_EQ(node.infer_properties_node().code, ErrorCode::INVALID_VALUE);
}

TEST(TestTopkNode, GatherHipdnnTensor)
{
    auto attrs = createValidAttributes();
    const auto xTensor = attrs.get_x();
    const auto indicesTensor = attrs.get_indices();
    const auto valuesTensor = attrs.get_values();

    const GraphAttributes graphAttributes;
    const TopkNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(indicesTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(valuesTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 3u);
}

// NOLINTEND(misc-const-correctness)
