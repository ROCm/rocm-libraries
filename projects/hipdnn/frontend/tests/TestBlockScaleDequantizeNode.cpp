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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeMissingX)
{
    BlockScaleDequantizeAttributes attrs;

    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_block_size(std::vector<int32_t>{32});

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeMissingScale)
{
    BlockScaleDequantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_y(std::make_shared<TensorAttributes>());
    attrs.set_block_size(std::vector<int32_t>{32});

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleDequantizeNode, PreValidateNodeMissingY)
{
    BlockScaleDequantizeAttributes attrs;

    attrs.set_x(std::make_shared<TensorAttributes>());
    attrs.set_scale(std::make_shared<TensorAttributes>());
    attrs.set_block_size(std::vector<int32_t>{32});

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

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

    GraphAttributes graphAttributes;
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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleDequantizeNode, InferPropertiesNodeMissingY)
{
    BlockScaleDequantizeAttributes attrs;
    attrs.set_x(std::make_shared<TensorAttributes>());

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestBlockScaleDequantizeNode, PackNode)
{
    BlockScaleDequantizeAttributes attrs;
    attrs.set_name("BlockScaleDequantize");
    attrs.set_block_size(std::vector<int32_t>{32});

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(2)
        .set_name("ScaleTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    attrs.set_scale(scaleTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(3)
        .set_name("YTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    attrs.set_y(yTensor);

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "BlockScaleDequantize");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_data_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_BlockScaleDequantizeAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    EXPECT_EQ(packedAttributes->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(packedAttributes->scale_tensor_uid(), scaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->y_tensor_uid(), yTensor->get_uid());
    ASSERT_NE(packedAttributes->block_size(), nullptr);
    EXPECT_EQ(packedAttributes->block_size()->size(), 1);
    EXPECT_EQ(packedAttributes->block_size()->Get(0), 32);
    EXPECT_EQ(packedAttributes->is_negative_scale(), false);
}

TEST(TestBlockScaleDequantizeNode, PackNodeWithNegativeScale)
{
    BlockScaleDequantizeAttributes attrs;
    attrs.set_name("BlockScaleDequantize");
    attrs.set_block_size(std::vector<int32_t>{32, 64});
    attrs.set_is_negative_scale(true);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_dim({1, 2, 3, 4}).set_stride({4, 3, 2, 1});
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(2).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    attrs.set_scale(scaleTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(3).set_dim({1, 2, 3, 4}).set_stride({4, 3, 2, 1});
    attrs.set_y(yTensor);

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    builder.Finish(offset);

    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(bufferPointer);
    auto packedAttributes = nodeFlatbuffer->attributes_as_BlockScaleDequantizeAttributes();

    ASSERT_NE(packedAttributes, nullptr);
    EXPECT_EQ(packedAttributes->is_negative_scale(), true);
    ASSERT_NE(packedAttributes->block_size(), nullptr);
    EXPECT_EQ(packedAttributes->block_size()->size(), 2);
    EXPECT_EQ(packedAttributes->block_size()->Get(0), 32);
    EXPECT_EQ(packedAttributes->block_size()->Get(1), 64);
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

    GraphAttributes graphAttributes;
    BlockScaleDequantizeNode node(std::move(attrs), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());

    EXPECT_EQ(allTensors.size(), 3);
}
