// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BlockScaleDequantizeOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BlockScaleDequantizeConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// BlockScaleDequantizeOperationDescriptor::fromNode() Tests
// =============================================================================

class TestBlockScaleDequantizeOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_BSD_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_BSD_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_BSD_TENSOR_X_STRIDES);
        _tensorMap[K_BSD_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);

        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = K_BSD_TENSOR_SCALE_UID;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims = toVec(K_BSD_TENSOR_SCALE_DIMS);
        scaleAttrs.strides = toVec(K_BSD_TENSOR_SCALE_STRIDES);
        _tensorMap[K_BSD_TENSOR_SCALE_UID] = TensorDescriptor::fromFlatBuffer(scaleAttrs);

        TensorAttributesT yAttrs;
        yAttrs.uid = K_BSD_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_BSD_TENSOR_Y_DIMS);
        yAttrs.strides = toVec(K_BSD_TENSOR_Y_STRIDES);
        _tensorMap[K_BSD_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
    }

    static BlockScaleDequantizeAttributesT createStandardBlockScaleDequantizeAttrs()
    {
        BlockScaleDequantizeAttributesT attrs;
        attrs.x_tensor_uid = K_BSD_TENSOR_X_UID;
        attrs.scale_tensor_uid = K_BSD_TENSOR_SCALE_UID;
        attrs.y_tensor_uid = K_BSD_TENSOR_Y_UID;
        attrs.block_size = {K_BSD_BLOCK_SIZE};
        attrs.is_negative_scale = false;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardBlockScaleDequantizeAttrs());
        return node;
    }
};

TEST_F(TestBlockScaleDequantizeOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_BSD_TENSOR_X_UID);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, NodeFactoryDelegatesCorrectly)
{
    auto node = createStandardNode();

    // NodeFactory::createOperationFromNode delegates to fromNode internally.
    // Verify the delegation produces a valid, correctly-typed descriptor.
    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);
    ASSERT_NE(graphOp, nullptr);

    // Verify the factory dispatched to the correct operation type, then static_cast.
    // Cannot use dynamic_pointer_cast: backend tests compile with -fno-rtti.
    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BlockScaleDequantizeAttributes);
    auto desc = std::static_pointer_cast<BlockScaleDequantizeOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_BSD_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_BSD_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_BSD_TENSOR_Y_UID);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BSD_TENSOR_X_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BSD_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BSD_TENSOR_Y_UID);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BSD_TENSOR_X_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BSD_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BSD_TENSOR_Y_UID);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_BSD_TENSOR_X_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_BSD_TENSOR_SCALE_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_BSD_TENSOR_Y_UID]);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BSD_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, toVec(K_BSD_TENSOR_X_DIMS));
    EXPECT_EQ(desc->getXDesc()->getData().strides, toVec(K_BSD_TENSOR_X_STRIDES));

    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BSD_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims, toVec(K_BSD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides, toVec(K_BSD_TENSOR_SCALE_STRIDES));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BSD_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, toVec(K_BSD_TENSOR_Y_DIMS));
    EXPECT_EQ(desc->getYDesc()->getData().strides, toVec(K_BSD_TENSOR_Y_STRIDES));
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_BSD_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(K_BSD_TENSOR_SCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_BSD_TENSOR_Y_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_BSD_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_BSD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_BSD_TENSOR_Y_UID);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BlockScaleDequantizeAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBlockScaleDequantizeAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_BSD_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_BSD_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_BSD_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->block_size, (std::vector<int32_t>{K_BSD_BLOCK_SIZE}));
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, FromNodePreservesIsNegativeScale)
{
    auto attrs = createStandardBlockScaleDequantizeAttrs();
    attrs.is_negative_scale = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_EQ(desc->getData().is_negative_scale, true);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBlockScaleDequantizeAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->is_negative_scale, true);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &dtCount,
                       &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(xScoped.get(),
                                                           K_BSD_TENSOR_X_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           toVec(K_BSD_TENSOR_X_DIMS),
                                                           toVec(K_BSD_TENSOR_X_STRIDES));

    // Verify scale tensor
    hipdnn_backend::ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(scaleScoped.get(),
                                                           K_BSD_TENSOR_SCALE_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           toVec(K_BSD_TENSOR_SCALE_DIMS),
                                                           toVec(K_BSD_TENSOR_SCALE_STRIDES));

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(yScoped.get(),
                                                           K_BSD_TENSOR_Y_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           toVec(K_BSD_TENSOR_Y_DIMS),
                                                           toVec(K_BSD_TENSOR_Y_STRIDES));

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BLOCK_SCALE_DEQUANTIZE_EXT);

    // Verify block_size
    int64_t blockSizeCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                       HIPDNN_TYPE_INT32,
                       0,
                       &blockSizeCount,
                       nullptr);
    ASSERT_EQ(blockSizeCount, 1);
    int32_t blockSizeVal = 0;
    int64_t actualBlockSizeCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                       HIPDNN_TYPE_INT32,
                       1,
                       &actualBlockSizeCount,
                       &blockSizeVal);
    EXPECT_EQ(blockSizeVal, K_BSD_BLOCK_SIZE);

    // Verify is_negative_scale
    bool isNegativeScale = true;
    int64_t isNegCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
                       HIPDNN_TYPE_BOOLEAN,
                       1,
                       &isNegCount,
                       &isNegativeScale);
    EXPECT_EQ(isNegativeScale, false);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_blockscaledequantize_1";

    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_blockscaledequantize_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_blockscaledequantize_1");
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestBlockScaleDequantizeOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = BlockScaleDequantizeOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
