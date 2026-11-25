// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/MatmulNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

Error preValidateWithOutputDims(const std::vector<int64_t>& aDims,
                                const std::vector<int64_t>& bDims,
                                const std::vector<int64_t>& cDims)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.get_a()->set_dim(aDims);
    attrs.set_b(std::make_shared<TensorAttributes>());
    attrs.get_b()->set_dim(bDims);
    attrs.set_c(std::make_shared<TensorAttributes>());
    attrs.get_c()->set_dim(cDims);

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    return node.pre_validate_node();
}

} // namespace

TEST(TestMatmulNode, PreValidateNodeSucceeds2D)
{
    MatmulAttributes attrs;
    const auto a = std::make_shared<TensorAttributes>();
    a->set_dim({4, 8}).set_stride({8, 1});
    attrs.set_a(a);

    const auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5}).set_stride({5, 1});
    attrs.set_b(b);

    const auto c = std::make_shared<TensorAttributes>();
    attrs.set_c(c);

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestMatmulNode, PreValidateNodeMissingA)
{
    MatmulAttributes attrs;
    attrs.set_b(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    const MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, PreValidateNodeMissingB)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    const MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, PreValidateNodeMissingC)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_b(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    const MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, PreValidateNodeMismatchedRanks)
{
    MatmulAttributes attrs;
    const auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    const auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, PreValidateNodeMismatchedInnerK)
{
    MatmulAttributes attrs;
    const auto a = std::make_shared<TensorAttributes>();
    a->set_dim({4, 7});
    attrs.set_a(a);

    const auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, PreValidateNodeIncompatibleBatch)
{
    MatmulAttributes attrs;
    const auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    const auto b = std::make_shared<TensorAttributes>();
    b->set_dim({3, 8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, PreValidateNodeBroadcastableBatch)
{
    MatmulAttributes attrs;
    const auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    const auto b = std::make_shared<TensorAttributes>();
    b->set_dim({1, 8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestMatmulNode, PreValidateNodeAcceptsUserSetOutputDims)
{
    auto err = preValidateWithOutputDims({4, 8}, {8, 5}, {4, 5});
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    err = preValidateWithOutputDims({2, 4, 8}, {1, 8, 5}, {2, 4, 5});
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    err = preValidateWithOutputDims({4, 4, 8}, {2, 8, 5}, {4, 4, 5});
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestMatmulNode, PreValidateNodeRejectsOutputRowsNotMatchingA)
{
    const auto err = preValidateWithOutputDims({16, 32}, {32, 24}, {17, 24});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, PreValidateNodeRejectsOutputColsNotMatchingB)
{
    auto err = preValidateWithOutputDims({16, 32}, {32, 23}, {16, 24});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);

    err = preValidateWithOutputDims({1, 4, 5}, {1, 5, 6}, {1, 4, 7});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

// Only the rank check rejects these: {2, 4} matches the batch and row dims of A, and {4, 5, 1}
// on 2-D inputs matches the rows and columns.
TEST(TestMatmulNode, PreValidateNodeRejectsOutputRankMismatch)
{
    const auto lowerRank = preValidateWithOutputDims({2, 4, 8}, {2, 8, 5}, {2, 4});
    EXPECT_EQ(lowerRank.code, error_code_t::INVALID_VALUE);
    EXPECT_NE(lowerRank.err_msg.find("same rank"), std::string::npos) << lowerRank.err_msg;

    const auto higherRank = preValidateWithOutputDims({4, 8}, {8, 5}, {4, 5, 1});
    EXPECT_EQ(higherRank.code, error_code_t::INVALID_VALUE);
    EXPECT_NE(higherRank.err_msg.find("same rank"), std::string::npos) << higherRank.err_msg;
}

TEST(TestMatmulNode, PreValidateNodeRejectsOutputBatchNotBroadcastResult)
{
    auto err = preValidateWithOutputDims({2, 4, 8}, {1, 8, 5}, {1, 4, 5});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);

    err = preValidateWithOutputDims({4, 4, 8}, {2, 8, 5}, {2, 4, 5});
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, InferPropertiesNodeMissingA)
{
    MatmulAttributes attrs;
    attrs.set_b(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, InferPropertiesNodeMissingB)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, InferPropertiesNodeMissingC)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_b(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, InferPropertiesNodeInfer2DOutputDimsAndStrides)
{
    MatmulAttributes attrs;
    const auto a = std::make_shared<TensorAttributes>();
    a->set_dim({4, 8});
    attrs.set_a(a);

    const auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5});
    attrs.set_b(b);

    const auto c = std::make_shared<TensorAttributes>();
    attrs.set_c(c);

    const GraphAttributes graphAttrs;
    MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto dims = c->get_dim();
    ASSERT_EQ(dims.size(), 2u);
    EXPECT_EQ(dims[0], 4);
    EXPECT_EQ(dims[1], 5);

    auto strides = c->get_stride();
    ASSERT_EQ(strides.size(), 2u);
    EXPECT_EQ(strides[0], 5);
    EXPECT_EQ(strides[1], 1);
}

TEST(TestMatmulNode, InferPropertiesNodeInferBatchedDimsAndStrides)
{
    MatmulAttributes attrs;
    const auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    const auto b = std::make_shared<TensorAttributes>();
    b->set_dim({1, 8, 5});
    attrs.set_b(b);

    const auto c = std::make_shared<TensorAttributes>();
    attrs.set_c(c);

    const GraphAttributes graphAttrs;
    MatmulNode node(std::move(attrs), graphAttrs);
    const auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;

    auto dims = c->get_dim();
    ASSERT_EQ(dims.size(), 3u);
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 4);
    EXPECT_EQ(dims[2], 5);

    auto strides = c->get_stride();
    ASSERT_EQ(strides.size(), 3u);
    EXPECT_EQ(strides[2], 1);
    EXPECT_EQ(strides[1], 5);
    EXPECT_EQ(strides[0], 20);
}

TEST(TestMatmulNode, GetNodeTypeReturnsMatmul)
{
    const GraphAttributes graphAttrs;
    const MatmulNode node(MatmulAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::MATMUL);
}
