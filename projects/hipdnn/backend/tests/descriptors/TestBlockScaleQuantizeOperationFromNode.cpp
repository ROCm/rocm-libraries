// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/BlockScaleQuantizeOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/block_scale_quantize_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BlockScaleQuantizeConstants.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;

// =============================================================================
// BlockScaleQuantizeOperationDescriptor::fromNode() Tests
// =============================================================================

class TestBlockScaleQuantizeOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        // X tensor
        TensorAttributesT xAttrs;
        xAttrs.uid = K_BSQ_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = {K_BSQ_TENSOR_X_DIMS.begin(), K_BSQ_TENSOR_X_DIMS.end()};
        xAttrs.strides = {K_BSQ_TENSOR_X_STRIDES.begin(), K_BSQ_TENSOR_X_STRIDES.end()};
        _tensorMap[K_BSQ_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);

        // Y tensor
        TensorAttributesT yAttrs;
        yAttrs.uid = K_BSQ_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = {K_BSQ_TENSOR_Y_DIMS.begin(), K_BSQ_TENSOR_Y_DIMS.end()};
        yAttrs.strides = {K_BSQ_TENSOR_Y_STRIDES.begin(), K_BSQ_TENSOR_Y_STRIDES.end()};
        _tensorMap[K_BSQ_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);

        // Scale tensor
        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = K_BSQ_TENSOR_SCALE_UID;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims = {K_BSQ_TENSOR_SCALE_DIMS.begin(), K_BSQ_TENSOR_SCALE_DIMS.end()};
        scaleAttrs.strides = {K_BSQ_TENSOR_SCALE_STRIDES.begin(), K_BSQ_TENSOR_SCALE_STRIDES.end()};
        _tensorMap[K_BSQ_TENSOR_SCALE_UID] = TensorDescriptor::fromFlatBuffer(scaleAttrs);
    }

    static BlockScaleQuantizeAttributesT createStandardBsqAttrs()
    {
        BlockScaleQuantizeAttributesT attrs;
        attrs.x_tensor_uid = K_BSQ_TENSOR_X_UID;
        attrs.y_tensor_uid = K_BSQ_TENSOR_Y_UID;
        attrs.scale_tensor_uid = K_BSQ_TENSOR_SCALE_UID;
        attrs.block_size = K_BSQ_BLOCK_SIZE;
        attrs.axis = flatbuffers::nullopt;
        attrs.transpose = false;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardBsqAttrs());
        return node;
    }
};

TEST_F(TestBlockScaleQuantizeOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_BSQ_TENSOR_X_UID);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, NodeFactoryDelegatesCorrectly)
{
    auto node = createStandardNode();

    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);
    ASSERT_NE(graphOp, nullptr);

    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BlockScaleQuantizeAttributes);
    auto desc = std::static_pointer_cast<BlockScaleQuantizeOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    EXPECT_EQ(desc->getData().x_tensor_uid, K_BSQ_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_BSQ_TENSOR_Y_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_BSQ_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getData().block_size, K_BSQ_BLOCK_SIZE);
    EXPECT_FALSE(desc->getData().axis.has_value());
    EXPECT_FALSE(desc->getData().transpose);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, PreservesBlockSize)
{
    auto node = createStandardNode();
    auto attrs = createStandardBsqAttrs();
    attrs.block_size = 64;
    node.attributes.Set(attrs);
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().block_size, 64);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, PreservesAxis)
{
    auto node = createStandardNode();
    auto attrs = createStandardBsqAttrs();
    attrs.axis = 1;
    node.attributes.Set(attrs);
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_TRUE(desc->getData().axis.has_value());
    EXPECT_EQ(desc->getData().axis.value(), 1);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, PreservesTranspose)
{
    auto node = createStandardNode();
    auto attrs = createStandardBsqAttrs();
    attrs.transpose = true;
    node.attributes.Set(attrs);
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_TRUE(desc->getData().transpose);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BSQ_TENSOR_X_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BSQ_TENSOR_Y_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BSQ_TENSOR_SCALE_UID);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_BSQ_TENSOR_X_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_BSQ_TENSOR_Y_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_BSQ_TENSOR_SCALE_UID]);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_BSQ_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_BSQ_TENSOR_Y_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(K_BSQ_TENSOR_SCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, SucceedsWithoutOptionalAxis)
{
    auto attrs = createStandardBsqAttrs();
    attrs.axis = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_FALSE(desc->getData().axis.has_value());
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_BSQ_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_BSQ_TENSOR_Y_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_BSQ_TENSOR_SCALE_UID);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BlockScaleQuantizeAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBlockScaleQuantizeAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_BSQ_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_BSQ_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_BSQ_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->block_size, K_BSQ_BLOCK_SIZE);
    EXPECT_FALSE(rebuiltAttrs->axis.has_value());
    EXPECT_FALSE(rebuiltAttrs->transpose);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(HIPDNN_ATTR_BLOCK_SCALE_QUANTIZE_MATH_PREC_EXT,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &dtCount,
                       &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify block_size
    int32_t blockSize = 0;
    int64_t bsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE_EXT,
                       HIPDNN_TYPE_INT32,
                       1,
                       &bsCount,
                       &blockSize);
    ASSERT_EQ(blockSize, K_BSQ_BLOCK_SIZE);

    // Verify transpose
    bool transpose = true;
    int64_t tCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT,
                       HIPDNN_TYPE_BOOLEAN,
                       1,
                       &tCount,
                       &transpose);
    ASSERT_FALSE(transpose);

    // Verify X tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    int64_t xUid = 0;
    int64_t xUidCount = 0;
    xScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &xUidCount, &xUid);
    EXPECT_EQ(xUid, K_BSQ_TENSOR_X_UID);

    // Verify Y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    int64_t yUid = 0;
    int64_t yUidCount = 0;
    yScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &yUidCount, &yUid);
    EXPECT_EQ(yUid, K_BSQ_TENSOR_Y_UID);

    // Verify operation type
    hipdnnOperationType_t opType = HIPDNN_OPERATION_TYPE_NOT_SET;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BLOCK_SCALE_QUANTIZE);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_bsq_1";

    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_bsq_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_bsq_1");
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
