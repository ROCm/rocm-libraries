// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeRoutingAttributes.hpp>
#include <hipdnn_frontend/node/MoeRoutingNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

MoeRoutingAttributes createValidAttributes()
{
    MoeRoutingAttributes attrs;

    auto topkIdsTensor = std::make_shared<TensorAttributes>();
    topkIdsTensor->set_dim({4, 2});
    topkIdsTensor->set_stride({2, 1});
    topkIdsTensor->set_data_type(DataType::INT32);
    attrs.set_topk_ids(topkIdsTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({5, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_dim({1, 8, 1});
    tokenIndexTensor->set_stride({8, 1, 1});
    tokenIndexTensor->set_data_type(DataType::INT32);
    attrs.set_token_index(tokenIndexTensor);
    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_dim({1, 8, 1});
    tokenKsTensor->set_stride({8, 1, 1});
    tokenKsTensor->set_data_type(DataType::INT32);
    attrs.set_token_ks(tokenKsTensor);

    attrs.set_num_experts(4);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestMoeRoutingNode, GetNodeTypeReturnsMoeRouting)
{
    const GraphAttributes graphAttrs;
    const MoeRoutingNode node(MoeRoutingAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::MOE_ROUTING);
}

// --- PreValidateNode (success case) ---

TEST(TestMoeRoutingNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const MoeRoutingNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestMoeRoutingNode, PreValidateNodeMissingTopkIdsTensor)
{
    MoeRoutingAttributes attrs;

    // Set all required tensors except topk_ids
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({5, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_dim({1, 8, 1});
    tokenIndexTensor->set_stride({8, 1, 1});
    tokenIndexTensor->set_data_type(DataType::INT32);
    attrs.set_token_index(tokenIndexTensor);
    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_dim({1, 8, 1});
    tokenKsTensor->set_stride({8, 1, 1});
    tokenKsTensor->set_data_type(DataType::INT32);
    attrs.set_token_ks(tokenKsTensor);

    attrs.set_num_experts(4);

    // topk_ids tensor is missing
    const GraphAttributes graphAttributes;
    const MoeRoutingNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeRoutingNode, PreValidateNodeMissingFirstTokenOffsetTensor)
{
    MoeRoutingAttributes attrs;

    // Set all required tensors except first_token_offset
    auto topkIdsTensor = std::make_shared<TensorAttributes>();
    topkIdsTensor->set_dim({4, 2});
    topkIdsTensor->set_stride({2, 1});
    topkIdsTensor->set_data_type(DataType::INT32);
    attrs.set_topk_ids(topkIdsTensor);
    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_dim({1, 8, 1});
    tokenIndexTensor->set_stride({8, 1, 1});
    tokenIndexTensor->set_data_type(DataType::INT32);
    attrs.set_token_index(tokenIndexTensor);
    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_dim({1, 8, 1});
    tokenKsTensor->set_stride({8, 1, 1});
    tokenKsTensor->set_data_type(DataType::INT32);
    attrs.set_token_ks(tokenKsTensor);

    attrs.set_num_experts(4);

    // first_token_offset tensor is missing
    const GraphAttributes graphAttributes;
    const MoeRoutingNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeRoutingNode, PreValidateNodeMissingTokenIndexTensor)
{
    MoeRoutingAttributes attrs;

    // Set all required tensors except token_index
    auto topkIdsTensor = std::make_shared<TensorAttributes>();
    topkIdsTensor->set_dim({4, 2});
    topkIdsTensor->set_stride({2, 1});
    topkIdsTensor->set_data_type(DataType::INT32);
    attrs.set_topk_ids(topkIdsTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({5, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_dim({1, 8, 1});
    tokenKsTensor->set_stride({8, 1, 1});
    tokenKsTensor->set_data_type(DataType::INT32);
    attrs.set_token_ks(tokenKsTensor);

    attrs.set_num_experts(4);

    // token_index tensor is missing
    const GraphAttributes graphAttributes;
    const MoeRoutingNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeRoutingNode, PreValidateNodeMissingTokenKsTensor)
{
    MoeRoutingAttributes attrs;

    // Set all required tensors except token_ks
    auto topkIdsTensor = std::make_shared<TensorAttributes>();
    topkIdsTensor->set_dim({4, 2});
    topkIdsTensor->set_stride({2, 1});
    topkIdsTensor->set_data_type(DataType::INT32);
    attrs.set_topk_ids(topkIdsTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({5, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_dim({1, 8, 1});
    tokenIndexTensor->set_stride({8, 1, 1});
    tokenIndexTensor->set_data_type(DataType::INT32);
    attrs.set_token_index(tokenIndexTensor);

    attrs.set_num_experts(4);

    // token_ks tensor is missing
    const GraphAttributes graphAttributes;
    const MoeRoutingNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeRoutingNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const MoeRoutingNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- InferPropertiesNode ---

TEST(TestMoeRoutingNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    MoeRoutingNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    // Output dims were supplied and already match: inference keeps them.
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- GatherHipdnnTensors ---

TEST(TestMoeRoutingNode, GatherHipdnnTensor)
{
    MoeRoutingAttributes attrs;

    auto topkIdsTensor = std::make_shared<TensorAttributes>();
    topkIdsTensor->set_uid(9700).set_name("TopkIdsTensor");
    attrs.set_topk_ids(topkIdsTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(9701).set_name("FirstTokenOffsetTensor");
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto tokenIndexTensor = std::make_shared<TensorAttributes>();
    tokenIndexTensor->set_uid(9702).set_name("TokenIndexTensor");
    attrs.set_token_index(tokenIndexTensor);
    auto tokenKsTensor = std::make_shared<TensorAttributes>();
    tokenKsTensor->set_uid(9703).set_name("TokenKsTensor");
    attrs.set_token_ks(tokenKsTensor);

    attrs.set_num_experts(4);

    const GraphAttributes graphAttributes;
    const MoeRoutingNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(topkIdsTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(firstTokenOffsetTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(tokenIndexTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(tokenKsTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 4u);
}

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

Error preValidate(MoeRoutingAttributes attrs)
{
    const GraphAttributes graphAttributes;
    const MoeRoutingNode node(std::move(attrs), graphAttributes);
    return node.pre_validate_node();
}

} // namespace

TEST(TestMoeRoutingNode, InferPropertiesNodeFillsUnsetOutputs)
{
    MoeRoutingAttributes attrs;
    attrs.set_topk_ids(makeTensor({3, 2}, {2, 1}, DataType::INT32));
    auto firstTokenOffset = std::make_shared<TensorAttributes>();
    auto tokenIndex = std::make_shared<TensorAttributes>();
    auto tokenKs = std::make_shared<TensorAttributes>();
    attrs.set_first_token_offset(firstTokenOffset);
    attrs.set_token_index(tokenIndex);
    attrs.set_token_ks(tokenKs);
    attrs.set_num_experts(5);

    GraphAttributes graphAttributes;
    graphAttributes.set_io_data_type(DataType::HALF);
    MoeRoutingNode node(std::move(attrs), graphAttributes);

    ASSERT_EQ(node.pre_validate_node().code, error_code_t::OK);
    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;

    EXPECT_EQ(firstTokenOffset->get_dim(), (std::vector<int64_t>{6, 1, 1}));
    EXPECT_EQ(firstTokenOffset->get_stride(), (std::vector<int64_t>{1, 1, 1}));
    EXPECT_EQ(firstTokenOffset->get_data_type(), DataType::INT32);
    EXPECT_EQ(tokenIndex->get_dim(), (std::vector<int64_t>{1, 6, 1}));
    EXPECT_EQ(tokenIndex->get_stride(), (std::vector<int64_t>{6, 1, 1}));
    EXPECT_EQ(tokenIndex->get_data_type(), DataType::INT32);
    EXPECT_EQ(tokenKs->get_dim(), (std::vector<int64_t>{1, 6, 1}));
    EXPECT_EQ(tokenKs->get_data_type(), DataType::INT32);
}

TEST(TestMoeRoutingNode, InferPropertiesNodeKeepsInt64FirstTokenOffset)
{
    auto attrs = createValidAttributes();
    attrs.get_first_token_offset()->set_data_type(DataType::INT64);

    const GraphAttributes graphAttributes;
    MoeRoutingNode node(std::move(attrs), graphAttributes);
    ASSERT_EQ(node.pre_validate_node().code, error_code_t::OK);
    ASSERT_EQ(node.infer_properties_node().code, error_code_t::OK);
    EXPECT_EQ(node.attributes.get_first_token_offset()->get_data_type(), DataType::INT64);
}

TEST(TestMoeRoutingNode, InferPropertiesNodeRejectsMismatchedOutputDims)
{
    auto attrs = createValidAttributes();
    attrs.get_token_ks()->set_dim({1, 7, 1});

    const GraphAttributes graphAttributes;
    MoeRoutingNode node(std::move(attrs), graphAttributes);
    EXPECT_EQ(node.infer_properties_node().code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeRoutingNode, PreValidateNodeRejectsSingleExpert)
{
    auto attrs = createValidAttributes();
    attrs.set_num_experts(1);
    attrs.get_first_token_offset()->set_dim({2, 1, 1});
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeRoutingNode, PreValidateNodeRejectsNonPositiveExperts)
{
    auto attrs = createValidAttributes();
    attrs.set_num_experts(0);
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeRoutingNode, PreValidateNodeRejectsTopkIdsWrongRank)
{
    auto attrs = createValidAttributes();
    attrs.set_topk_ids(makeTensor({1, 4, 2}, {8, 2, 1}, DataType::INT32));
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeRoutingNode, PreValidateNodeRejectsTopkIdsWrongType)
{
    auto attrs = createValidAttributes();
    attrs.get_topk_ids()->set_data_type(DataType::INT64);
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeRoutingNode, PreValidateNodeAcceptsInt64FirstTokenOffset)
{
    auto attrs = createValidAttributes();
    attrs.get_first_token_offset()->set_data_type(DataType::INT64);
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::OK);
}

TEST(TestMoeRoutingNode, PreValidateNodeRejectsFloatFirstTokenOffset)
{
    auto attrs = createValidAttributes();
    attrs.get_first_token_offset()->set_data_type(DataType::FLOAT);
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeRoutingNode, PreValidateNodeRejectsWrongFirstTokenOffsetDims)
{
    auto attrs = createValidAttributes();
    attrs.get_first_token_offset()->set_dim({4, 1, 1});
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeRoutingNode, PreValidateNodeRejectsWrongTokenIndexDims)
{
    auto attrs = createValidAttributes();
    attrs.get_token_index()->set_dim({1, 4, 1});
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeRoutingNode, PreValidateNodeRejectsWrongTokenKsDims)
{
    auto attrs = createValidAttributes();
    attrs.get_token_ks()->set_dim({8, 1, 1});
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeRoutingNode, PreValidateNodeRejectsInt64TokenIndex)
{
    auto attrs = createValidAttributes();
    attrs.get_token_index()->set_data_type(DataType::INT64);
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

TEST(TestMoeRoutingNode, PreValidateNodeRejectsFloatTokenKs)
{
    auto attrs = createValidAttributes();
    attrs.get_token_ks()->set_data_type(DataType::FLOAT);
    EXPECT_EQ(preValidate(std::move(attrs)).code, error_code_t::INVALID_VALUE);
}

// NOLINTEND(misc-const-correctness)
