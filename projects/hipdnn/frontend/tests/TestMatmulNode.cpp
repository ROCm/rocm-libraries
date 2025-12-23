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

TEST(TestMatmulNode, PreValidateNode_Succeeds2D)
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

    GraphAttributes graphAttrs;
    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestMatmulNode, PreValidateNode_MissingA)
{
    MatmulAttributes attrs;
    attrs.set_b(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, PreValidateNode_MissingB)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, PreValidateNode_MissingC)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_b(std::make_shared<TensorAttributes>());
    GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, PreValidateNode_MismatchedRanks)
{
    MatmulAttributes attrs;
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, PreValidateNode_MismatchedInnerK)
{
    MatmulAttributes attrs;
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({4, 7});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, PreValidateNode_IncompatibleBatch)
{
    MatmulAttributes attrs;
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({3, 8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::INVALID_VALUE);
}

TEST(TestMatmulNode, PreValidateNode_BroadcastableBatch)
{
    MatmulAttributes attrs;
    auto a = std::make_shared<TensorAttributes>();
    a->set_dim({2, 4, 8});
    attrs.set_a(a);

    auto b = std::make_shared<TensorAttributes>();
    b->set_dim({1, 8, 5});
    attrs.set_b(b);

    attrs.set_c(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttrs;
    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.pre_validate_node();
    EXPECT_EQ(err.code, error_code_t::OK) << err.err_msg;
}

TEST(TestMatmulNode, InferPropertiesNode_MissingA)
{
    MatmulAttributes attrs;
    attrs.set_b(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, InferPropertiesNode_MissingB)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_c(std::make_shared<TensorAttributes>());
    GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, InferPropertiesNode_MissingC)
{
    MatmulAttributes attrs;
    attrs.set_a(std::make_shared<TensorAttributes>());
    attrs.set_b(std::make_shared<TensorAttributes>());
    GraphAttributes graphAttrs;

    MatmulNode node(std::move(attrs), graphAttrs);
    auto err = node.infer_properties_node();
    EXPECT_EQ(err.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMatmulNode, InferPropertiesNode_Infer2DOutputDimsAndStrides)
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

    GraphAttributes graphAttrs;
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

TEST(TestMatmulNode, InferPropertiesNode_InferBatchedDimsAndStrides)
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

    GraphAttributes graphAttrs;
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
