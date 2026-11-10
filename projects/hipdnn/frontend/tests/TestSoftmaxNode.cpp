// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/SoftmaxAttributes.hpp>
#include <hipdnn_frontend/node/SoftmaxNode.hpp>

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

namespace
{

std::shared_ptr<TensorAttributes> makeTensor(const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
    return tensor;
}

SoftmaxAttributes createValidAttributes()
{
    SoftmaxAttributes attrs;
    attrs.set_p(makeTensor({2, 4, 8, 16}, {512, 128, 16, 1}));
    attrs.set_s(makeTensor({2, 4, 8, 16}, {512, 128, 16, 1}));
    return attrs;
}

} // namespace

TEST(TestSoftmaxNode, GetNodeTypeReturnsSoftmax)
{
    const GraphAttributes graphAttrs;
    const SoftmaxNode node(SoftmaxAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::SOFTMAX);
}

TEST(TestSoftmaxNode, PreValidateNode)
{
    const GraphAttributes graphAttributes;
    const SoftmaxNode node(createValidAttributes(), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestSoftmaxNode, PreValidateNodeWithStats)
{
    auto attrs = createValidAttributes();
    attrs.set_stats(makeTensor({2, 4, 8, 1}, {32, 8, 1, 1}));

    const GraphAttributes graphAttributes;
    const SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestSoftmaxNode, PreValidateNodeRankOne)
{
    SoftmaxAttributes attrs;
    attrs.set_p(makeTensor({5}, {1}));
    attrs.set_s(makeTensor({5}, {1}));
    attrs.set_stats(makeTensor({1}, {1}));

    const GraphAttributes graphAttributes;
    const SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestSoftmaxNode, PreValidateNodeMissingPTensor)
{
    SoftmaxAttributes attrs;
    attrs.set_s(makeTensor({2, 4, 8, 16}, {512, 128, 16, 1}));

    const GraphAttributes graphAttributes;
    const SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSoftmaxNode, PreValidateNodeMissingSTensor)
{
    SoftmaxAttributes attrs;
    attrs.set_p(makeTensor({2, 4, 8, 16}, {512, 128, 16, 1}));

    const GraphAttributes graphAttributes;
    const SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSoftmaxNode, PreValidateNodeRejectsRankZeroP)
{
    SoftmaxAttributes attrs;
    attrs.set_p(std::make_shared<TensorAttributes>());
    attrs.set_s(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestSoftmaxNode, PreValidateNodeRejectsSDimsMismatch)
{
    auto attrs = createValidAttributes();
    attrs.get_s()->set_dim({2, 4, 8, 15});

    const GraphAttributes graphAttributes;
    const SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestSoftmaxNode, PreValidateNodeRejectsStatsLastDimNotOne)
{
    auto attrs = createValidAttributes();
    attrs.set_stats(makeTensor({2, 4, 8, 16}, {512, 128, 16, 1}));

    const GraphAttributes graphAttributes;
    const SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestSoftmaxNode, PreValidateNodeRejectsStatsLeadingDimsMismatch)
{
    auto attrs = createValidAttributes();
    attrs.set_stats(makeTensor({2, 4, 7, 1}, {28, 7, 1, 1}));

    const GraphAttributes graphAttributes;
    const SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestSoftmaxNode, InferPropertiesNodeFillsOutputs)
{
    SoftmaxAttributes attrs;
    // NHWC-style input layout: the inferred outputs keep its stride order.
    attrs.set_p(makeTensor({2, 4, 8, 16}, {512, 1, 64, 4}));
    attrs.set_s(std::make_shared<TensorAttributes>());
    attrs.set_stats(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;

    EXPECT_EQ(node.attributes.get_s()->get_dim(), (std::vector<int64_t>{2, 4, 8, 16}));
    EXPECT_EQ(node.attributes.get_s()->get_stride(), (std::vector<int64_t>{512, 1, 64, 4}));
    EXPECT_EQ(node.attributes.get_stats()->get_dim(), (std::vector<int64_t>{2, 4, 8, 1}));
    EXPECT_EQ(node.attributes.get_stats()->get_stride(), (std::vector<int64_t>{32, 1, 4, 4}));
}

TEST(TestSoftmaxNode, InferPropertiesNodeKeepsUserSetOutputs)
{
    auto attrs = createValidAttributes();
    attrs.get_s()->set_stride({1, 2, 8, 64});

    const GraphAttributes graphAttributes;
    SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, error_code_t::OK) << error.err_msg;
    EXPECT_EQ(node.attributes.get_s()->get_stride(), (std::vector<int64_t>{1, 2, 8, 64}));
}

TEST(TestSoftmaxNode, InferPropertiesNodeMissingInputStrides)
{
    SoftmaxAttributes attrs;
    auto p = std::make_shared<TensorAttributes>();
    p->set_dim({2, 4}).set_data_type(DataType::FLOAT);
    attrs.set_p(p);
    attrs.set_s(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    SoftmaxNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestSoftmaxNode, GatherHipdnnTensor)
{
    SoftmaxAttributes attrs;

    auto pTensor = std::make_shared<TensorAttributes>();
    pTensor->set_uid(9200).set_name("PTensor");
    attrs.set_p(pTensor);
    auto sTensor = std::make_shared<TensorAttributes>();
    sTensor->set_uid(9201).set_name("STensor");
    attrs.set_s(sTensor);
    auto statsTensor = std::make_shared<TensorAttributes>();
    statsTensor->set_uid(9202).set_name("StatsTensor");
    attrs.set_stats(statsTensor);

    const GraphAttributes graphAttributes;
    const SoftmaxNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(pTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(sTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(statsTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 3u);
}

// NOLINTEND(misc-const-correctness)
