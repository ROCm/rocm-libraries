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

TEST(TestMatmulNode, PreValidateNodeSucceeds2D)
{
    MatmulAttributes attrs;
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({4, 8}).set_stride({8, 1});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5}).set_stride({5, 1});
    attrs.set_b(b);

    auto c = std::make_shared<TensorAttributes>();
    attrs.set_c(c);

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestMatmulNode, PreValidateNodeMissingA)
{
    MatmulAttributes attrs;
    attrs.set_b(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    const MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, PreValidateNodeMissingB)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    const MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, PreValidateNodeMissingC)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_b(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    const MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, PreValidateNodeMismatchedRanks)
{
    MatmulAttributes attrs;
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, PreValidateNodeMismatchedInnerK)
{
    MatmulAttributes attrs;
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({4, 7});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, PreValidateNodeIncompatibleBatch)
{
    MatmulAttributes attrs;
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({3, 8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, PreValidateNodeBroadcastableBatch)
{
    MatmulAttributes attrs;
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({1, 8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttrs;
    const MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestMatmulNode, InferPropertiesNodeMissingA)
{
    MatmulAttributes attrs;
    attrs.set_b(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, InferPropertiesNodeMissingB)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, InferPropertiesNodeMissingC)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_b(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, InferPropertiesNodeInfer2DOutputDimsAndStrides)
{
    MatmulAttributes attrs;
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({4, 8});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5});
    attrs.set_b(b);

    auto c = std::make_shared<TensorAttributes>();
    attrs.set_c(c);

    const GraphAttributes graphAttrs;
    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
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
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({1, 8, 5});
    attrs.set_b(b);

    auto c = std::make_shared<TensorAttributes>();
    attrs.set_c(c);

    const GraphAttributes graphAttrs;
    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
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
