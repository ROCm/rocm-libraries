// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GatherOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/gather_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/GatherConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_backend::test_utilities::verifyTensorDescriptor;
using hipdnn_tests::toVec;

// =============================================================================
// GatherOperationDescriptor::fromNode() Tests
// =============================================================================

class TestGatherOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_GATHER_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_GATHER_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_GATHER_TENSOR_X_STRIDES);

        _tensorMap[K_GATHER_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT indicesAttrs;
        indicesAttrs.uid = K_GATHER_TENSOR_INDICES_UID;
        indicesAttrs.data_type = DataType::INT32;
        indicesAttrs.dims = toVec(K_GATHER_TENSOR_INDICES_DIMS);
        indicesAttrs.strides = toVec(K_GATHER_TENSOR_INDICES_STRIDES);

        _tensorMap[K_GATHER_TENSOR_INDICES_UID] = TensorDescriptor::fromFlatBuffer(indicesAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = K_GATHER_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_GATHER_TENSOR_Y_DIMS);
        yAttrs.strides = toVec(K_GATHER_TENSOR_Y_STRIDES);

        _tensorMap[K_GATHER_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::GatherAttributesT createStandardGatherAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::GatherAttributesT attrs;
        attrs.x_tensor_uid = K_GATHER_TENSOR_X_UID;
        attrs.indices_tensor_uid = K_GATHER_TENSOR_INDICES_UID;
        attrs.y_tensor_uid = K_GATHER_TENSOR_Y_UID;
        attrs.axis = 0;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardGatherAttrs());
        return node;
    }
};

TEST_F(TestGatherOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = GatherOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_GATHER_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_GATHER_TENSOR_X_UID);
}

TEST_F(TestGatherOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    const auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::GatherAttributes);
    auto desc = std::static_pointer_cast<GatherOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_GATHER_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().indices_tensor_uid, K_GATHER_TENSOR_INDICES_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_GATHER_TENSOR_Y_UID);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_GATHER_TENSOR_X_UID);
    EXPECT_EQ(desc->getIndicesDesc()->getData().uid, K_GATHER_TENSOR_INDICES_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_GATHER_TENSOR_Y_UID);
}

TEST_F(TestGatherOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = GatherOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_GATHER_TENSOR_X_UID);
    ASSERT_NE(desc->getIndicesDesc(), nullptr);
    EXPECT_EQ(desc->getIndicesDesc()->getData().uid, K_GATHER_TENSOR_INDICES_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_GATHER_TENSOR_Y_UID);
}

TEST_F(TestGatherOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = GatherOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_GATHER_TENSOR_X_UID]);
    EXPECT_EQ(desc->getIndicesDesc(), _tensorMap[K_GATHER_TENSOR_INDICES_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_GATHER_TENSOR_Y_UID]);
}

TEST_F(TestGatherOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = GatherOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_GATHER_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, (std::vector<int64_t>{16, 8}));
    EXPECT_EQ(desc->getXDesc()->getData().strides, (std::vector<int64_t>{8, 1}));

    ASSERT_NE(desc->getIndicesDesc(), nullptr);
    EXPECT_EQ(desc->getIndicesDesc()->getData().uid, K_GATHER_TENSOR_INDICES_UID);
    EXPECT_EQ(desc->getIndicesDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getIndicesDesc()->getData().dims, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(desc->getIndicesDesc()->getData().strides, (std::vector<int64_t>{3, 1}));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_GATHER_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, (std::vector<int64_t>{2, 3, 8}));
    EXPECT_EQ(desc->getYDesc()->getData().strides, (std::vector<int64_t>{24, 8, 1}));
}

// The parameter is the UID of the one required tensor to drop from the tensor map.
class TestGatherOperationFromNodeMissingTensor : public TestGatherOperationFromNode,
                                                 public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestGatherOperationFromNodeMissingTensor, FailsWithMissingTensor)
{
    _tensorMap.erase(GetParam());
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(GatherOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(RequiredTensors,
                         TestGatherOperationFromNodeMissingTensor,
                         ::testing::Values(K_GATHER_TENSOR_X_UID,
                                           K_GATHER_TENSOR_INDICES_UID,
                                           K_GATHER_TENSOR_Y_UID));

TEST_F(TestGatherOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = GatherOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_GATHER_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_GATHER_TENSOR_INDICES_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_GATHER_TENSOR_Y_UID);
}

TEST_F(TestGatherOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = GatherOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::GatherAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsGatherAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_GATHER_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->indices_tensor_uid, K_GATHER_TENSOR_INDICES_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_GATHER_TENSOR_Y_UID);
}

TEST_F(TestGatherOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = GatherOperationDescriptor::fromNode(node, _tensorMap);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GATHER_XDESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(
        xScoped.get(), K_GATHER_TENSOR_X_UID, HIPDNN_DATA_FLOAT, {16, 8}, {8, 1});

    // Verify indices tensor
    hipdnn_backend::ScopedDescriptor indicesScoped;
    int64_t indicesCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GATHER_INDICES_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &indicesCount,
                       static_cast<void*>(indicesScoped.getPtr()));
    ASSERT_EQ(indicesCount, 1);
    ASSERT_NE(indicesScoped.get(), nullptr);
    verifyTensorDescriptor(
        indicesScoped.get(), K_GATHER_TENSOR_INDICES_UID, HIPDNN_DATA_INT32, {2, 3}, {3, 1});

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_GATHER_YDESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    verifyTensorDescriptor(
        yScoped.get(), K_GATHER_TENSOR_Y_UID, HIPDNN_DATA_FLOAT, {2, 3, 8}, {24, 8, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_GATHER_EXT);
}

TEST_F(TestGatherOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_gather_1";

    auto desc = GatherOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_gather_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_gather_1");
}

TEST_F(TestGatherOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = GatherOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestGatherOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = GatherOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
