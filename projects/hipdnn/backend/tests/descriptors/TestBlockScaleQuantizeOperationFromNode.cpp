// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BlockScaleQuantizeOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_quantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BlockScaleQuantizeConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// BlockScaleQuantizeOperationDescriptor::fromNode() Tests
// =============================================================================

class TestBlockScaleQuantizeOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        auto makeTensor = [this](int64_t uid,
                                 const std::vector<int64_t>& dims,
                                 const std::vector<int64_t>& strides) {
            TensorAttributesT attrs;
            attrs.uid = uid;
            attrs.data_type = DataType::FLOAT;
            attrs.dims = dims;
            attrs.strides = strides;
            _tensorMap[uid] = TensorDescriptor::fromFlatBuffer(attrs);
        };

        makeTensor(K_BSQ_TENSOR_X_UID, toVec(K_BSQ_TENSOR_X_DIMS), toVec(K_BSQ_TENSOR_X_STRIDES));
        makeTensor(K_BSQ_TENSOR_Y_UID, toVec(K_BSQ_TENSOR_Y_DIMS), toVec(K_BSQ_TENSOR_Y_STRIDES));
        makeTensor(K_BSQ_TENSOR_SCALE_UID,
                   toVec(K_BSQ_TENSOR_SCALE_DIMS),
                   toVec(K_BSQ_TENSOR_SCALE_STRIDES));
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

// =============================================================================
// Parameterized: missing required tensor
// =============================================================================

class TestBsqMissingRequiredTensor : public TestBlockScaleQuantizeOperationFromNode,
                                     public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestBsqMissingRequiredTensor, FailsWithMissingRequiredTensor)
{
    _tensorMap.erase(GetParam());
    auto node = createStandardNode();
    ASSERT_THROW_HIPDNN_STATUS(BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AllRequiredTensors,
                         TestBsqMissingRequiredTensor,
                         ::testing::Values(K_BSQ_TENSOR_X_UID,
                                           K_BSQ_TENSOR_Y_UID,
                                           K_BSQ_TENSOR_SCALE_UID));

// =============================================================================
// Non-parameterized tests
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR);
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
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, PreservesBlockSize)
{
    auto attrs = createStandardBsqAttrs();
    attrs.block_size = 64;
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().block_size, 64);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, PreservesAxis)
{
    auto attrs = createStandardBsqAttrs();
    attrs.axis = 1;
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_TRUE(desc->getData().axis.has_value());
    EXPECT_EQ(desc->getData().axis.value(), 1);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, PreservesTranspose)
{
    auto attrs = createStandardBsqAttrs();
    attrs.transpose = true;
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
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
    ASSERT_EQ(tensors.size(), 3u);
    EXPECT_EQ(tensors[0]->getData().uid, K_BSQ_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_BSQ_TENSOR_Y_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_BSQ_TENSOR_SCALE_UID);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, BuildNodeRoundTrip)
{
    auto attrs = createStandardBsqAttrs();
    attrs.axis = 1;
    attrs.transpose = true;
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

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
    ASSERT_TRUE(rebuiltAttrs->axis.has_value());
    EXPECT_EQ(rebuiltAttrs->axis.value(), 1);
    EXPECT_TRUE(rebuiltAttrs->transpose);
}

TEST_F(TestBlockScaleQuantizeOperationFromNode, BuildNodeOmitsUnsetOptionalFields)
{
    auto node = createStandardNode();
    auto desc = BlockScaleQuantizeOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBlockScaleQuantizeAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

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
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &dtCount,
                       &computeType);
    ASSERT_EQ(dtCount, 1);
    EXPECT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify block_size
    int32_t blockSize = 0;
    int64_t bsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE,
                       HIPDNN_TYPE_INT32,
                       1,
                       &bsCount,
                       &blockSize);
    ASSERT_EQ(bsCount, 1);
    EXPECT_EQ(blockSize, K_BSQ_BLOCK_SIZE);

    // Verify transpose
    bool transpose = true;
    int64_t tCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT,
                       HIPDNN_TYPE_BOOLEAN,
                       1,
                       &tCount,
                       &transpose);
    ASSERT_EQ(tCount, 1);
    EXPECT_FALSE(transpose);

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BLOCK_SCALE_QUANTIZE_EXT);

    // Verify X tensor — deep check: UID, data type, dims, strides
    ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(xScoped.get(),
                           K_BSQ_TENSOR_X_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_BSQ_TENSOR_X_DIMS),
                           toVec(K_BSQ_TENSOR_X_STRIDES));

    // Verify Y tensor — deep check: UID, data type, dims, strides
    ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    verifyTensorDescriptor(yScoped.get(),
                           K_BSQ_TENSOR_Y_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_BSQ_TENSOR_Y_DIMS),
                           toVec(K_BSQ_TENSOR_Y_STRIDES));

    // Verify Scale tensor — deep check: UID, data type, dims, strides
    ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    verifyTensorDescriptor(scaleScoped.get(),
                           K_BSQ_TENSOR_SCALE_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_BSQ_TENSOR_SCALE_DIMS),
                           toVec(K_BSQ_TENSOR_SCALE_STRIDES));
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
