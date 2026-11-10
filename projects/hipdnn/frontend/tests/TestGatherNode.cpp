// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GatherAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/node/GatherNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

GatherAttributes createValidAttributes()
{
    GatherAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 8});
    xTensor->set_stride({8, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto indicesTensor = std::make_shared<TensorAttributes>();
    indicesTensor->set_dim({2, 3});
    indicesTensor->set_stride({3, 1});
    indicesTensor->set_data_type(DataType::INT32);
    attrs.set_indices(indicesTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 3, 8});
    yTensor->set_stride({24, 8, 1});
    yTensor->set_data_type(DataType::FLOAT);
    attrs.set_y(yTensor);

    attrs.set_axis(0);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestGatherNode, GetNodeTypeReturnsGather)
{
    const GraphAttributes graphAttrs;
    const GatherNode node(GatherAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::GATHER);
}

// --- PreValidateNode (success case) ---

TEST(TestGatherNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestGatherNode, PreValidateNodeMissingXTensor)
{
    GatherAttributes attrs;

    // Set all required tensors except x
    auto indicesTensor = std::make_shared<TensorAttributes>();
    indicesTensor->set_dim({2, 3});
    indicesTensor->set_stride({3, 1});
    indicesTensor->set_data_type(DataType::INT32);
    attrs.set_indices(indicesTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 3, 8});
    yTensor->set_stride({24, 8, 1});
    yTensor->set_data_type(DataType::FLOAT);
    attrs.set_y(yTensor);

    attrs.set_axis(0);

    // x tensor is missing
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestGatherNode, PreValidateNodeMissingIndicesTensor)
{
    GatherAttributes attrs;

    // Set all required tensors except indices
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 8});
    xTensor->set_stride({8, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 3, 8});
    yTensor->set_stride({24, 8, 1});
    yTensor->set_data_type(DataType::FLOAT);
    attrs.set_y(yTensor);

    attrs.set_axis(0);

    // indices tensor is missing
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestGatherNode, PreValidateNodeMissingYTensor)
{
    GatherAttributes attrs;

    // Set all required tensors except y
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({16, 8});
    xTensor->set_stride({8, 1});
    xTensor->set_data_type(DataType::FLOAT);
    attrs.set_x(xTensor);
    auto indicesTensor = std::make_shared<TensorAttributes>();
    indicesTensor->set_dim({2, 3});
    indicesTensor->set_stride({3, 1});
    indicesTensor->set_data_type(DataType::INT32);
    attrs.set_indices(indicesTensor);

    attrs.set_axis(0);

    // y tensor is missing
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestGatherNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- InferPropertiesNode ---

TEST(TestGatherNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    GatherNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    EXPECT_EQ(node.attributes.get_y()->get_dim(), (std::vector<int64_t>{2, 3, 8}));
    EXPECT_EQ(node.attributes.get_y()->get_stride(), (std::vector<int64_t>{24, 8, 1}));
}

// --- PreValidateNode: schema rules ---

TEST(TestGatherNode, PreValidateNodeRejectsNegativeAxis)
{
    auto attrs = createValidAttributes();
    attrs.set_axis(-1);
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestGatherNode, PreValidateNodeRejectsAxisEqualToRank)
{
    auto attrs = createValidAttributes();
    attrs.set_axis(2);
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestGatherNode, PreValidateNodeRejectsEmptyXDims)
{
    auto attrs = createValidAttributes();
    attrs.get_x()->set_dim({});
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestGatherNode, PreValidateNodeRejectsEmptyIndicesDims)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_dim({});
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestGatherNode, PreValidateNodeRejectsFloatIndices)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_data_type(DataType::FLOAT);
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestGatherNode, PreValidateNodeRejectsUnsetIndicesDataType)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_data_type(DataType::NOT_SET);
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestGatherNode, PreValidateNodeAcceptsInt64Indices)
{
    auto attrs = createValidAttributes();
    attrs.get_indices()->set_data_type(DataType::INT64);
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::OK);
}

TEST(TestGatherNode, PreValidateNodeRejectsYRankMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_y()->set_dim({6, 8});
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestGatherNode, PreValidateNodeRejectsYDimsMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_y()->set_dim({2, 3, 16});
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestGatherNode, PreValidateNodeRejectsYDataTypeMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_y()->set_data_type(DataType::HALF);
    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.pre_validate_node().code, error_code_t::INVALID_VALUE);
}

// --- InferPropertiesNode: shape and data type inference ---

TEST(TestGatherNode, InferPropertiesNodeMiddleAxis)
{
    GatherAttributes attrs;
    auto x = std::make_shared<TensorAttributes>();
    x->set_dim({2, 5, 4}).set_stride({20, 4, 1}).set_data_type(DataType::HALF);
    auto indices = std::make_shared<TensorAttributes>();
    indices->set_dim({3, 2}).set_stride({2, 1}).set_data_type(DataType::INT64);
    auto y = std::make_shared<TensorAttributes>();
    attrs.set_x(x).set_indices(indices).set_y(y).set_axis(1);

    const GraphAttributes graphAttributes;
    GatherNode node(std::move(attrs), graphAttributes);

    ASSERT_EQ(node.pre_validate_node().code, error_code_t::OK);
    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;
    EXPECT_EQ(y->get_dim(), (std::vector<int64_t>{2, 3, 2, 4}));
    EXPECT_EQ(y->get_stride(), (std::vector<int64_t>{24, 8, 4, 1}));
    EXPECT_EQ(y->get_data_type(), DataType::HALF);
}

TEST(TestGatherNode, InferPropertiesNodeRejectsMismatchedYDims)
{
    auto attrs = createValidAttributes();
    attrs.get_y()->set_dim({3, 2, 8});
    const GraphAttributes graphAttributes;
    GatherNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.infer_properties_node().code, error_code_t::INVALID_VALUE);
}

// --- GatherHipdnnTensors ---

TEST(TestGatherNode, GatherHipdnnTensor)
{
    GatherAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9400).set_name("XTensor");
    attrs.set_x(xTensor);
    auto indicesTensor = std::make_shared<TensorAttributes>();
    indicesTensor->set_uid(9401).set_name("IndicesTensor");
    attrs.set_indices(indicesTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9402).set_name("YTensor");
    attrs.set_y(yTensor);

    attrs.set_axis(0);

    const GraphAttributes graphAttributes;
    const GatherNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(indicesTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 3u);
}

// NOLINTEND(misc-const-correctness)
