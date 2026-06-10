// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/node/BlockScaleDequantizeNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestBlockScaleDequantizeNode, PreValidateNode)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({2, 2, 32, 32});
    attrs.set_scale(scaleTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_is_virtual(true);
    attrs.set_y(yTensor);
    attrs.set_block_size(std::vector<int32_t>{32});

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeMissingX)
{
    BlockScaleDequantizeAttributes attrs;

    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_block_size(std::vector<int32_t>{32});

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeMissingScale)
{
    BlockScaleDequantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_block_size(std::vector<int32_t>{32});

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeMissingY)
{
    BlockScaleDequantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(std::vector<int32_t>{32});

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeYNotVirtual)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>()); // Not virtual
    attrs.set_block_size(std::vector<int32_t>{32});

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeMissingBlockSize)
{
    BlockScaleDequantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_is_virtual(true);
    attrs.set_y(yTensor);
    // block_size is not set (empty)

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeYShapeMismatch)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({2, 2, 32, 32});
    attrs.set_scale(scaleTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 16, 16}).set_is_virtual(true); // Mismatched dims
    attrs.set_y(yTensor);

    attrs.set_block_size(std::vector<int32_t>{32});

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeYRankMismatch)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    attrs.set_scale(scaleTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 32}).set_is_virtual(true); // Different rank
    attrs.set_y(yTensor);

    attrs.set_block_size(std::vector<int32_t>{32});

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeYDimsNotSetPassesValidation)
{
    // When Y dims are not set, shape match is skipped (will be inferred later)
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    attrs.set_scale(scaleTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_is_virtual(true); // No dims set
    attrs.set_y(yTensor);
    attrs.set_block_size(std::vector<int32_t>{32});

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeBlockSizeZero)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_is_virtual(true);
    attrs.set_y(yTensor);
    attrs.set_block_size(std::vector<int32_t>{0});

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeBlockSizeNegative)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_is_virtual(true);
    attrs.set_y(yTensor);
    attrs.set_block_size(std::vector<int32_t>{32, -1});

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeBlockSizeExceedsRank)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64}); // rank 2
    attrs.set_x(xTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_is_virtual(true);
    attrs.set_y(yTensor);
    attrs.set_block_size(std::vector<int32_t>{32, 16, 8}); // 3 entries > rank 2

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeBlockSizeMatchesRank)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    attrs.set_x(xTensor);

    attrs.set_scale(std::make_shared<TensorAttributes>());
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_is_virtual(true);
    attrs.set_y(yTensor);
    attrs.set_block_size(std::vector<int32_t>{2, 64, 32, 32}); // exactly matches rank

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeXDimsNotSetSkipsDimChecks)
{
    // When X dims are not set, dimension-dependent checks are skipped
    BlockScaleDequantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>()); // No dims
    attrs.set_scale(std::make_shared<TensorAttributes>());
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_is_virtual(true);
    attrs.set_y(yTensor);
    attrs.set_block_size(std::vector<int32_t>{32, 16, 8}); // Would fail if X had rank < 3

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleDequantizeNode, InferPropertiesNode)
{
    BlockScaleDequantizeAttributes attrs;
    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_block_size(std::vector<int32_t>{32});

    auto inputTensor = attrs.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = attrs.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    const GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBlockScaleDequantizeNode, InferPropertiesNodeMissingX)
{
    BlockScaleDequantizeAttributes attrs;
    attrs.set_y(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleDequantizeNode, InferPropertiesNodeMissingY)
{
    BlockScaleDequantizeAttributes attrs;
    attrs.set_x(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleDequantizeNode, GatherHipdnnTensors)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("X");
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(2).set_name("Scale");
    attrs.set_scale(scaleTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(3).set_name("Y");
    attrs.set_y(yTensor);

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());

    EXPECT_EQ(allTensors.size(), 3);
}

TEST(TestBlockScaleDequantizeNode, GetNodeTypeReturnsBlockScaleDequantize)
{
    const GraphAttributes graphAttrs;
    const BlockScaleDequantizeNode node(BlockScaleDequantizeAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BLOCK_SCALE_DEQUANTIZE);
}
