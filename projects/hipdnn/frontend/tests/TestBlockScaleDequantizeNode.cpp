// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/node/BlockScaleDequantizeNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{
ErrorCode preValidateWithScale(const std::vector<int64_t>& xDims,
                               const std::vector<int32_t>& blockSize,
                               const std::vector<int64_t>& scaleDims)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim(xDims);
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim(scaleDims);
    attrs.set_scale(scaleTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_is_virtual(true);
    attrs.set_y(yTensor);
    attrs.set_block_size(blockSize);

    const GraphAttributes graphAttributes;
    const BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);
    return node.pre_validate_node().code;
}
} // namespace

TEST(TestBlockScaleDequantizeNode, PreValidateNode)
{
    BlockScaleDequantizeAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({2, 64, 32, 1});
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
    scaleTensor->set_dim({2, 64, 32, 1});
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

TEST(TestBlockScaleDequantizeNode, PreValidateNodeScaleCoversTrailingBlocks)
{
    // block_size entries apply to the trailing dimensions of x; leading dimensions use block 1
    EXPECT_EQ(preValidateWithScale({4, 64}, {1, 32}, {4, 2}), ErrorCode::OK);
    EXPECT_EQ(preValidateWithScale({4, 64}, {32}, {4, 2}), ErrorCode::OK);
    EXPECT_EQ(preValidateWithScale({2, 64, 32, 32}, {32}, {2, 64, 32, 1}), ErrorCode::OK);

    // A leading-aligned reading of block_size {32} would expect these scale shapes
    EXPECT_EQ(preValidateWithScale({4, 64}, {32}, {1, 64}), ErrorCode::INVALID_VALUE);
    EXPECT_EQ(preValidateWithScale({2, 64, 32, 32}, {32}, {2, 2, 32, 32}),
              ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeScaleDimAtLeastBlockCount)
{
    // Too few scale entries for the blocks of x
    EXPECT_EQ(preValidateWithScale({4, 64}, {1, 32}, {4, 1}), ErrorCode::INVALID_VALUE);
    EXPECT_EQ(preValidateWithScale({4, 64}, {1, 32}, {3, 2}), ErrorCode::INVALID_VALUE);

    // A padded scale layout has more entries than blocks
    EXPECT_EQ(preValidateWithScale({4, 64}, {1, 32}, {4, 3}), ErrorCode::OK);
    EXPECT_EQ(preValidateWithScale({4, 64}, {1, 32}, {8, 4}), ErrorCode::OK);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeScaleDimUsesCeilForPartialBlocks)
{
    // 70 elements in blocks of 32 need ceil(70 / 32) = 3 scale entries
    EXPECT_EQ(preValidateWithScale({4, 70}, {1, 32}, {4, 3}), ErrorCode::OK);
    EXPECT_EQ(preValidateWithScale({4, 70}, {1, 32}, {4, 2}), ErrorCode::INVALID_VALUE);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeScaleRankMismatch)
{
    EXPECT_EQ(preValidateWithScale({4, 64}, {1, 32}, {4, 2, 1}), ErrorCode::INVALID_VALUE);
    EXPECT_EQ(preValidateWithScale({4, 64}, {1, 32}, {4}), ErrorCode::INVALID_VALUE);
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

    hipdnn_frontend::detail::OrderedTensorSet allTensors;
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
