// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/node/MoeGroupedMatmulNode.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

MoeGroupedMatmulAttributes createValidAttributes()
{
    MoeGroupedMatmulAttributes attrs;

    auto token = std::make_shared<TensorAttributes>();
    token->set_dim({1, 8, 16}).set_stride({128, 16, 1}).set_data_type(DataType::FLOAT);
    attrs.set_token(token);

    auto weight = std::make_shared<TensorAttributes>();
    weight->set_dim({2, 16, 32}).set_stride({512, 32, 1}).set_data_type(DataType::FLOAT);
    attrs.set_weight(weight);

    auto firstTokenOffset = std::make_shared<TensorAttributes>();
    firstTokenOffset->set_dim({2, 1, 1}).set_stride({1, 1, 1}).set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffset);

    auto output = std::make_shared<TensorAttributes>();
    output->set_dim({1, 8, 32}).set_stride({256, 32, 1}).set_data_type(DataType::FLOAT);
    attrs.set_output(output);
    return attrs;
}

std::shared_ptr<TensorAttributes> makeRoutingTensor(int64_t routedTokens = 8,
                                                    DataType dataType = DataType::INT32)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_dim({1, routedTokens, 1}).set_stride({routedTokens, 1, 1}).set_data_type(dataType);
    return tensor;
}

} // namespace

TEST(TestMoeGroupedMatmulNode, GetNodeTypeReturnsMoeGroupedMatmul)
{
    const GraphAttributes graphAttrs;
    const MoeGroupedMatmulNode node(MoeGroupedMatmulAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::MOE_GROUPED_MATMUL);
}

TEST(TestMoeGroupedMatmulNode, NoneModeAcceptsRequiredInputs)
{
    auto attrs = createValidAttributes();
    const GraphAttributes graphAttrs;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
    EXPECT_TRUE(node.pre_validate_node().is_good());
}

TEST(TestMoeGroupedMatmulNode, NotSetModeIsRejected)
{
    auto attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::NOT_SET);

    const GraphAttributes graphAttrs;
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, RequiredTensorsMustBePresent)
{
    const GraphAttributes graphAttrs;

    auto attrs = createValidAttributes();
    attrs.set_token(std::shared_ptr<TensorAttributes>{});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);

    attrs = createValidAttributes();
    attrs.set_weight(std::shared_ptr<TensorAttributes>{});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);

    attrs = createValidAttributes();
    attrs.set_first_token_offset(std::shared_ptr<TensorAttributes>{});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);

    attrs = createValidAttributes();
    attrs.set_output(std::shared_ptr<TensorAttributes>{});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulNode, RequiredTensorShapesAndTypesMustBeConsistent)
{
    const GraphAttributes graphAttrs;

    auto attrs = createValidAttributes();
    attrs.get_token()->set_dim({1, 8, 15});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);

    attrs = createValidAttributes();
    attrs.get_first_token_offset()->set_data_type(DataType::FLOAT);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);

    attrs = createValidAttributes();
    attrs.get_first_token_offset()->set_dim({3, 1, 1});
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, GatherRequiresInt32TokenIndex)
{
    auto attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::GATHER);

    const GraphAttributes graphAttrs;
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);

    attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::GATHER)
        .set_token_index(makeRoutingTensor(8, DataType::FLOAT));
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, ScatterValidatesRoutingAndTopK)
{
    const GraphAttributes graphAttrs;

    auto attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor())
        .set_token_ks(makeRoutingTensor())
        .set_top_k(2);
    EXPECT_TRUE(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().is_good());

    attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor())
        .set_token_ks(makeRoutingTensor())
        .set_top_k(0);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);

    attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor(8))
        .set_token_ks(makeRoutingTensor(7))
        .set_top_k(1);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, ScatterRequiresValidRoutingAndBoundedTopK)
{
    const GraphAttributes graphAttrs;

    auto attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER).set_token_index(makeRoutingTensor()).set_top_k(1);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::ATTRIBUTE_NOT_SET);

    attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor())
        .set_token_ks(makeRoutingTensor(8, DataType::FLOAT))
        .set_top_k(1);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);

    attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_token_index(makeRoutingTensor())
        .set_token_ks(makeRoutingTensor())
        .set_top_k(3);
    EXPECT_EQ(MoeGroupedMatmulNode(std::move(attrs), graphAttrs).pre_validate_node().code,
              ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, GatherInfersOutputDimensionsAndStrides)
{
    auto attrs = createValidAttributes();
    auto output = std::make_shared<TensorAttributes>();
    attrs.set_mode(MoeGroupedMatmulMode::GATHER)
        .set_token_index(makeRoutingTensor(6))
        .set_output(output);

    const GraphAttributes graphAttrs;
    MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
    EXPECT_TRUE(node.pre_validate_node().is_good());
    const auto error = node.infer_properties_node();
    EXPECT_TRUE(error.is_good()) << error.err_msg;
    EXPECT_EQ(output->get_dim(), (std::vector<int64_t>{1, 6, 32}));
    EXPECT_EQ(output->get_stride(), (std::vector<int64_t>{192, 32, 1}));
}

TEST(TestMoeGroupedMatmulNode, NoneInfersOutputDimensionsAndRejectsMismatch)
{
    auto attrs = createValidAttributes();
    auto output = std::make_shared<TensorAttributes>();
    attrs.set_output(output);

    const GraphAttributes graphAttrs;
    MoeGroupedMatmulNode node(std::move(attrs), graphAttrs);
    EXPECT_TRUE(node.pre_validate_node().is_good());
    EXPECT_TRUE(node.infer_properties_node().is_good());
    EXPECT_EQ(output->get_dim(), (std::vector<int64_t>{1, 8, 32}));

    attrs = createValidAttributes();
    attrs.get_output()->set_dim({1, 7, 32});
    MoeGroupedMatmulNode mismatchedNode(std::move(attrs), graphAttrs);
    EXPECT_TRUE(mismatchedNode.pre_validate_node().is_good());
    EXPECT_EQ(mismatchedNode.infer_properties_node().code, ErrorCode::INVALID_VALUE);
}
