// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/node/MoeGroupedMatmulNode.hpp>

#include <memory>
#include <unordered_set>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOLINTBEGIN(misc-const-correctness)

// --- Helper: create fully configured attributes for a valid node ---
namespace
{

MoeGroupedMatmulAttributes createValidAttributes()
{
    MoeGroupedMatmulAttributes attrs;

    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    attrs.set_token(tokenTensor);
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_dim({2, 16, 32});
    weightTensor->set_stride({512, 32, 1});
    attrs.set_weight(weightTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto outputTensor = std::make_shared<TensorAttributes>();
    outputTensor->set_dim({1, 8, 32});
    outputTensor->set_stride({256, 32, 1});
    attrs.set_output(outputTensor);

    attrs.set_top_k(2);

    return attrs;
}

} // namespace

// --- GetNodeType ---

TEST(TestMoeGroupedMatmulNode, GetNodeTypeReturnsMoeGroupedMatmul)
{
    const GraphAttributes graphAttrs;
    const MoeGroupedMatmulNode node(MoeGroupedMatmulAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::MOE_GROUPED_MATMUL);
}

// --- PreValidateNode (success case) ---

TEST(TestMoeGroupedMatmulNode, PreValidateNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- PreValidateNode: missing required tensors ---

TEST(TestMoeGroupedMatmulNode, PreValidateNodeMissingTokenTensor)
{
    MoeGroupedMatmulAttributes attrs;

    // Set all required tensors except token
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_dim({2, 16, 32});
    weightTensor->set_stride({512, 32, 1});
    attrs.set_weight(weightTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto outputTensor = std::make_shared<TensorAttributes>();
    outputTensor->set_dim({1, 8, 32});
    outputTensor->set_stride({256, 32, 1});
    attrs.set_output(outputTensor);

    attrs.set_top_k(2);

    // token tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulNode, PreValidateNodeMissingWeightTensor)
{
    MoeGroupedMatmulAttributes attrs;

    // Set all required tensors except weight
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    attrs.set_token(tokenTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto outputTensor = std::make_shared<TensorAttributes>();
    outputTensor->set_dim({1, 8, 32});
    outputTensor->set_stride({256, 32, 1});
    attrs.set_output(outputTensor);

    attrs.set_top_k(2);

    // weight tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulNode, PreValidateNodeMissingFirstTokenOffsetTensor)
{
    MoeGroupedMatmulAttributes attrs;

    // Set all required tensors except first_token_offset
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    attrs.set_token(tokenTensor);
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_dim({2, 16, 32});
    weightTensor->set_stride({512, 32, 1});
    attrs.set_weight(weightTensor);
    auto outputTensor = std::make_shared<TensorAttributes>();
    outputTensor->set_dim({1, 8, 32});
    outputTensor->set_stride({256, 32, 1});
    attrs.set_output(outputTensor);

    attrs.set_top_k(2);

    // first_token_offset tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulNode, PreValidateNodeMissingOutputTensor)
{
    MoeGroupedMatmulAttributes attrs;

    // Set all required tensors except output
    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_dim({1, 8, 16});
    tokenTensor->set_stride({128, 16, 1});
    attrs.set_token(tokenTensor);
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_dim({2, 16, 32});
    weightTensor->set_stride({512, 32, 1});
    attrs.set_weight(weightTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_dim({2, 1, 1});
    firstTokenOffsetTensor->set_stride({1, 1, 1});
    firstTokenOffsetTensor->set_data_type(DataType::INT32);
    attrs.set_first_token_offset(firstTokenOffsetTensor);

    attrs.set_top_k(2);

    // output tensor is missing
    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulNode, PreValidateNodeAllValuesSet)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestMoeGroupedMatmulNode, GatherRequiresTokenIndex)
{
    auto attrs = createValidAttributes();
    attrs.set_mode(MoeGroupedMatmulMode::GATHER);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);
    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulNode, ScatterRequiresTokenKs)
{
    auto attrs = createValidAttributes();
    auto tokenIndex = std::make_shared<TensorAttributes>();
    tokenIndex->set_dim({1, 8, 1}).set_stride({8, 1, 1}).set_data_type(DataType::INT32);
    attrs.set_token_index(tokenIndex).set_mode(MoeGroupedMatmulMode::SCATTER);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);
    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestMoeGroupedMatmulNode, ScatterRejectsNonpositiveTopK)
{
    auto attrs = createValidAttributes();
    auto tokenIndex = std::make_shared<TensorAttributes>();
    tokenIndex->set_dim({1, 8, 1}).set_stride({8, 1, 1}).set_data_type(DataType::INT32);
    auto tokenKs = std::make_shared<TensorAttributes>();
    tokenKs->set_dim({1, 8, 1}).set_stride({8, 1, 1}).set_data_type(DataType::INT32);
    attrs.set_token_index(tokenIndex)
        .set_token_ks(tokenKs)
        .set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_top_k(0);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);
    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, ScatterRejectsTopKGreaterThanExpertCount)
{
    auto attrs = createValidAttributes();
    auto tokenIndex = std::make_shared<TensorAttributes>();
    tokenIndex->set_dim({1, 8, 1}).set_stride({8, 1, 1}).set_data_type(DataType::INT32);
    auto tokenKs = std::make_shared<TensorAttributes>();
    tokenKs->set_dim({1, 8, 1}).set_stride({8, 1, 1}).set_data_type(DataType::INT32);
    attrs.set_token_index(tokenIndex)
        .set_token_ks(tokenKs)
        .set_mode(MoeGroupedMatmulMode::SCATTER)
        .set_top_k(3);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);
    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, RejectsMismatchedMatmulDimensions)
{
    auto attrs = createValidAttributes();
    attrs.get_token()->set_dim({1, 8, 15});

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);
    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, RejectsNonInt32FirstTokenOffset)
{
    auto attrs = createValidAttributes();
    attrs.get_first_token_offset()->set_data_type(DataType::FLOAT);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);
    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, GatherRejectsNonInt32TokenIndex)
{
    auto attrs = createValidAttributes();
    auto tokenIndex = std::make_shared<TensorAttributes>();
    tokenIndex->set_dim({1, 8, 1}).set_stride({8, 1, 1}).set_data_type(DataType::FLOAT);
    attrs.set_token_index(tokenIndex).set_mode(MoeGroupedMatmulMode::GATHER);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);
    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, ScatterRejectsNonInt32TokenKs)
{
    auto attrs = createValidAttributes();
    auto tokenIndex = std::make_shared<TensorAttributes>();
    tokenIndex->set_dim({1, 8, 1}).set_stride({8, 1, 1}).set_data_type(DataType::INT32);
    auto tokenKs = std::make_shared<TensorAttributes>();
    tokenKs->set_dim({1, 8, 1}).set_stride({8, 1, 1}).set_data_type(DataType::FLOAT);
    attrs.set_token_index(tokenIndex).set_token_ks(tokenKs).set_mode(MoeGroupedMatmulMode::SCATTER);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);
    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestMoeGroupedMatmulNode, GatherInfersOutputDimensionsAndStrides)
{
    auto attrs = createValidAttributes();
    auto tokenIndex = std::make_shared<TensorAttributes>();
    tokenIndex->set_dim({1, 6, 1}).set_stride({6, 1, 1}).set_data_type(DataType::INT32);
    auto output = std::make_shared<TensorAttributes>();
    attrs.set_token_index(tokenIndex).set_output(output).set_mode(MoeGroupedMatmulMode::GATHER);

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);
    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(output->get_dim(), (std::vector<int64_t>{1, 6, 32}));
    EXPECT_EQ(output->get_stride(), (std::vector<int64_t>{192, 32, 1}));
}

// --- InferPropertiesNode ---

TEST(TestMoeGroupedMatmulNode, InferPropertiesNode)
{
    auto attrs = createValidAttributes();

    const GraphAttributes graphAttributes;
    MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    // Stub implementation: verify the method can be called without error
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

// --- GatherHipdnnTensors ---

TEST(TestMoeGroupedMatmulNode, GatherHipdnnTensor)
{
    MoeGroupedMatmulAttributes attrs;

    auto tokenTensor = std::make_shared<TensorAttributes>();
    tokenTensor->set_uid(1900).set_name("TokenTensor");
    attrs.set_token(tokenTensor);
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_uid(1901).set_name("WeightTensor");
    attrs.set_weight(weightTensor);
    auto firstTokenOffsetTensor = std::make_shared<TensorAttributes>();
    firstTokenOffsetTensor->set_uid(1902).set_name("FirstTokenOffsetTensor");
    attrs.set_first_token_offset(firstTokenOffsetTensor);
    auto outputTensor = std::make_shared<TensorAttributes>();
    outputTensor->set_uid(1905).set_name("OutputTensor");
    attrs.set_output(outputTensor);

    attrs.set_top_k(2);

    const GraphAttributes graphAttributes;
    const MoeGroupedMatmulNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(tokenTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(weightTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(firstTokenOffsetTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(outputTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 4u);
}

// NOLINTEND(misc-const-correctness)
