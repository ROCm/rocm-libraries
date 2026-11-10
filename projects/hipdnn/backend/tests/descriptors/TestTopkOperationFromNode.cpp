// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "descriptors/TopkOperationDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/topk_attributes_generated.h>
#include <hipdnn_test_sdk/constants/TopkConstants.hpp>
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
// TopkOperationDescriptor::fromNode() Tests
// =============================================================================

class TestTopkOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_TOPK_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_TOPK_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_TOPK_TENSOR_X_STRIDES);

        _tensorMap[K_TOPK_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT indicesAttrs;
        indicesAttrs.uid = K_TOPK_TENSOR_INDICES_UID;
        indicesAttrs.data_type = DataType::INT64;
        indicesAttrs.dims = toVec(K_TOPK_TENSOR_INDICES_DIMS);
        indicesAttrs.strides = toVec(K_TOPK_TENSOR_INDICES_STRIDES);

        _tensorMap[K_TOPK_TENSOR_INDICES_UID] = TensorDescriptor::fromFlatBuffer(indicesAttrs);
        TensorAttributesT valuesAttrs;
        valuesAttrs.uid = K_TOPK_TENSOR_VALUES_UID;
        valuesAttrs.data_type = DataType::FLOAT;
        valuesAttrs.dims = toVec(K_TOPK_TENSOR_VALUES_DIMS);
        valuesAttrs.strides = toVec(K_TOPK_TENSOR_VALUES_STRIDES);

        _tensorMap[K_TOPK_TENSOR_VALUES_UID] = TensorDescriptor::fromFlatBuffer(valuesAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::TopkAttributesT createStandardTopkAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::TopkAttributesT attrs;
        attrs.x_tensor_uid = K_TOPK_TENSOR_X_UID;
        attrs.indices_tensor_uid = K_TOPK_TENSOR_INDICES_UID;
        attrs.values_tensor_uid = K_TOPK_TENSOR_VALUES_UID;
        attrs.k = 4;
        attrs.largest = false;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardTopkAttrs());
        return node;
    }
};

TEST_F(TestTopkOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_TOPK_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_TOPK_TENSOR_X_UID);
}

TEST_F(TestTopkOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::TopkAttributes);
    auto desc = std::static_pointer_cast<TopkOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_TOPK_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().indices_tensor_uid, K_TOPK_TENSOR_INDICES_UID);
    EXPECT_EQ(desc->getData().values_tensor_uid, K_TOPK_TENSOR_VALUES_UID);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_TOPK_TENSOR_X_UID);
    EXPECT_EQ(desc->getIndicesDesc()->getData().uid, K_TOPK_TENSOR_INDICES_UID);
    EXPECT_EQ(desc->getValuesDesc()->getData().uid, K_TOPK_TENSOR_VALUES_UID);
}

TEST_F(TestTopkOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_TOPK_TENSOR_X_UID);
    ASSERT_NE(desc->getIndicesDesc(), nullptr);
    EXPECT_EQ(desc->getIndicesDesc()->getData().uid, K_TOPK_TENSOR_INDICES_UID);
    ASSERT_NE(desc->getValuesDesc(), nullptr);
    EXPECT_EQ(desc->getValuesDesc()->getData().uid, K_TOPK_TENSOR_VALUES_UID);
}

TEST_F(TestTopkOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_TOPK_TENSOR_X_UID]);
    EXPECT_EQ(desc->getIndicesDesc(), _tensorMap[K_TOPK_TENSOR_INDICES_UID]);
    EXPECT_EQ(desc->getValuesDesc(), _tensorMap[K_TOPK_TENSOR_VALUES_UID]);
}

TEST_F(TestTopkOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_TOPK_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, (std::vector<int64_t>{2, 3, 8}));
    EXPECT_EQ(desc->getXDesc()->getData().strides, (std::vector<int64_t>{24, 8, 1}));

    ASSERT_NE(desc->getIndicesDesc(), nullptr);
    EXPECT_EQ(desc->getIndicesDesc()->getData().uid, K_TOPK_TENSOR_INDICES_UID);
    EXPECT_EQ(desc->getIndicesDesc()->getData().data_type, DataType::INT64);
    EXPECT_EQ(desc->getIndicesDesc()->getData().dims, (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(desc->getIndicesDesc()->getData().strides, (std::vector<int64_t>{12, 4, 1}));

    ASSERT_NE(desc->getValuesDesc(), nullptr);
    EXPECT_EQ(desc->getValuesDesc()->getData().uid, K_TOPK_TENSOR_VALUES_UID);
    EXPECT_EQ(desc->getValuesDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getValuesDesc()->getData().dims, (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(desc->getValuesDesc()->getData().strides, (std::vector<int64_t>{12, 4, 1}));
}

// The parameter is the UID of the one required tensor to drop from the tensor map.
class TestTopkOperationFromNodeMissingTensor : public TestTopkOperationFromNode,
                                               public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestTopkOperationFromNodeMissingTensor, FailsWithMissingTensor)
{
    _tensorMap.erase(GetParam());
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(TopkOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(RequiredTensors,
                         TestTopkOperationFromNodeMissingTensor,
                         ::testing::Values(K_TOPK_TENSOR_X_UID, K_TOPK_TENSOR_INDICES_UID));

TEST_F(TestTopkOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardTopkAttrs();
    attrs.values_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getXDesc(), nullptr);
    EXPECT_NE(desc->getIndicesDesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getValuesDesc(), nullptr);

    const auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2u);
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getIndicesDesc());
}

TEST_F(TestTopkOperationFromNode, FailsWhenOptionalValuesUidSetButTensorMissing)
{
    _tensorMap.erase(K_TOPK_TENSOR_VALUES_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(TopkOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestTopkOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_TOPK_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_TOPK_TENSOR_INDICES_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_TOPK_TENSOR_VALUES_UID);
}

TEST_F(TestTopkOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::TopkAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsTopkAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_TOPK_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->indices_tensor_uid, K_TOPK_TENSOR_INDICES_UID);
    EXPECT_EQ(rebuiltAttrs->values_tensor_uid, K_TOPK_TENSOR_VALUES_UID);
}

TEST_F(TestTopkOperationFromNode, FromNodePreservesLargest)
{
    auto attrs = createStandardTopkAttrs();
    attrs.largest = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getData().largest, true);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsTopkAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->largest, true);
}

TEST_F(TestTopkOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(
        xScoped.get(), K_TOPK_TENSOR_X_UID, HIPDNN_DATA_FLOAT, {2, 3, 8}, {24, 8, 1});

    // Verify indices tensor
    hipdnn_backend::ScopedDescriptor indicesScoped;
    int64_t indicesCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &indicesCount,
                       static_cast<void*>(indicesScoped.getPtr()));
    ASSERT_EQ(indicesCount, 1);
    ASSERT_NE(indicesScoped.get(), nullptr);
    verifyTensorDescriptor(
        indicesScoped.get(), K_TOPK_TENSOR_INDICES_UID, HIPDNN_DATA_INT64, {2, 3, 4}, {12, 4, 1});

    // Verify values tensor (optional)
    hipdnn_backend::ScopedDescriptor valuesScoped;
    int64_t valuesCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &valuesCount,
                       static_cast<void*>(valuesScoped.getPtr()));
    ASSERT_EQ(valuesCount, 1);
    ASSERT_NE(valuesScoped.get(), nullptr);
    verifyTensorDescriptor(
        valuesScoped.get(), K_TOPK_TENSOR_VALUES_UID, HIPDNN_DATA_FLOAT, {2, 3, 4}, {12, 4, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_TOPK_EXT);
}

TEST_F(TestTopkOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_topk_1";

    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_topk_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_topk_1");
}

TEST_F(TestTopkOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestTopkOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = TopkOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
