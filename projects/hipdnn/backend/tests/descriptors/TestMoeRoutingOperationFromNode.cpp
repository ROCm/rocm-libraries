// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/MoeRoutingOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_routing_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MoeRoutingConstants.hpp>
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
// MoeRoutingOperationDescriptor::fromNode() Tests
// =============================================================================

class TestMoeRoutingOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT topkIdsAttrs;
        topkIdsAttrs.uid = K_MOE_ROUTING_TENSOR_TOPK_IDS_UID;
        topkIdsAttrs.data_type = DataType::INT32;
        topkIdsAttrs.dims = toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS);
        topkIdsAttrs.strides = toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES);

        _tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]
            = TensorDescriptor::fromFlatBuffer(topkIdsAttrs);
        TensorAttributesT firstTokenOffsetAttrs;
        firstTokenOffsetAttrs.uid = K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID;
        firstTokenOffsetAttrs.data_type = DataType::INT32;
        firstTokenOffsetAttrs.dims = toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS);
        firstTokenOffsetAttrs.strides = toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES);

        _tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]
            = TensorDescriptor::fromFlatBuffer(firstTokenOffsetAttrs);
        TensorAttributesT tokenIndexAttrs;
        tokenIndexAttrs.uid = K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID;
        tokenIndexAttrs.data_type = DataType::INT32;
        tokenIndexAttrs.dims = toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS);
        tokenIndexAttrs.strides = toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES);

        _tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]
            = TensorDescriptor::fromFlatBuffer(tokenIndexAttrs);
        TensorAttributesT tokenKsAttrs;
        tokenKsAttrs.uid = K_MOE_ROUTING_TENSOR_TOKEN_KS_UID;
        tokenKsAttrs.data_type = DataType::INT32;
        tokenKsAttrs.dims = toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS);
        tokenKsAttrs.strides = toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES);

        _tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]
            = TensorDescriptor::fromFlatBuffer(tokenKsAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::MoeRoutingAttributesT
        createStandardMoeRoutingAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::MoeRoutingAttributesT attrs;
        attrs.topk_ids_tensor_uid = K_MOE_ROUTING_TENSOR_TOPK_IDS_UID;
        attrs.first_token_offset_tensor_uid = K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID;
        attrs.token_index_tensor_uid = K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID;
        attrs.token_ks_tensor_uid = K_MOE_ROUTING_TENSOR_TOKEN_KS_UID;
        attrs.num_experts = 4;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardMoeRoutingAttrs());
        return node;
    }
};

TEST_F(TestMoeRoutingOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = MoeRoutingOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_MOE_ROUTING_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().topk_ids_tensor_uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
}

TEST_F(TestMoeRoutingOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::MoeRoutingAttributes);
    auto desc = std::static_pointer_cast<MoeRoutingOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().topk_ids_tensor_uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    EXPECT_EQ(desc->getData().first_token_offset_tensor_uid,
              K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(desc->getData().token_index_tensor_uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(desc->getData().token_ks_tensor_uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
    EXPECT_EQ(desc->getTopkIdsDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(desc->getTokenKsDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
}

TEST_F(TestMoeRoutingOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = MoeRoutingOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getTopkIdsDesc(), nullptr);
    EXPECT_EQ(desc->getTopkIdsDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_NE(desc->getTokenIndexDesc(), nullptr);
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    ASSERT_NE(desc->getTokenKsDesc(), nullptr);
    EXPECT_EQ(desc->getTokenKsDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
}

TEST_F(TestMoeRoutingOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = MoeRoutingOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getTopkIdsDesc(), _tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc(),
              _tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]);
    EXPECT_EQ(desc->getTokenIndexDesc(), _tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]);
    EXPECT_EQ(desc->getTokenKsDesc(), _tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]);
}

TEST_F(TestMoeRoutingOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = MoeRoutingOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getTopkIdsDesc(), nullptr);
    EXPECT_EQ(desc->getTopkIdsDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    EXPECT_EQ(desc->getTopkIdsDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getTopkIdsDesc()->getData().dims, (std::vector<int64_t>{4, 2}));
    EXPECT_EQ(desc->getTopkIdsDesc()->getData().strides, (std::vector<int64_t>{2, 1}));

    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().dims, (std::vector<int64_t>{5, 1, 1}));
    EXPECT_EQ(desc->getFirstTokenOffsetDesc()->getData().strides, (std::vector<int64_t>{1, 1, 1}));

    ASSERT_NE(desc->getTokenIndexDesc(), nullptr);
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().dims, (std::vector<int64_t>{1, 8, 1}));
    EXPECT_EQ(desc->getTokenIndexDesc()->getData().strides, (std::vector<int64_t>{8, 1, 1}));

    ASSERT_NE(desc->getTokenKsDesc(), nullptr);
    EXPECT_EQ(desc->getTokenKsDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
    EXPECT_EQ(desc->getTokenKsDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getTokenKsDesc()->getData().dims, (std::vector<int64_t>{1, 8, 1}));
    EXPECT_EQ(desc->getTokenKsDesc()->getData().strides, (std::vector<int64_t>{8, 1, 1}));
}

// The parameter is the UID of the one required tensor to drop from the tensor map.
class TestMoeRoutingOperationFromNodeMissingTensor : public TestMoeRoutingOperationFromNode,
                                                     public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestMoeRoutingOperationFromNodeMissingTensor, FailsWithMissingTensor)
{
    _tensorMap.erase(GetParam());
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(MoeRoutingOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(RequiredTensors,
                         TestMoeRoutingOperationFromNodeMissingTensor,
                         ::testing::Values(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID,
                                           K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                           K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                                           K_MOE_ROUTING_TENSOR_TOKEN_KS_UID));

TEST_F(TestMoeRoutingOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = MoeRoutingOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    EXPECT_EQ(tensors[0]->getData().uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
}

TEST_F(TestMoeRoutingOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = MoeRoutingOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::MoeRoutingAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsMoeRoutingAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->topk_ids_tensor_uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    EXPECT_EQ(rebuiltAttrs->first_token_offset_tensor_uid,
              K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(rebuiltAttrs->token_index_tensor_uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(rebuiltAttrs->token_ks_tensor_uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
}

TEST_F(TestMoeRoutingOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = MoeRoutingOperationDescriptor::fromNode(node, _tensorMap);

    // Verify topk_ids tensor
    hipdnn_backend::ScopedDescriptor topkIdsScoped;
    int64_t topkIdsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &topkIdsCount,
                       static_cast<void*>(topkIdsScoped.getPtr()));
    ASSERT_EQ(topkIdsCount, 1);
    ASSERT_NE(topkIdsScoped.get(), nullptr);
    verifyTensorDescriptor(
        topkIdsScoped.get(), K_MOE_ROUTING_TENSOR_TOPK_IDS_UID, HIPDNN_DATA_INT32, {4, 2}, {2, 1});

    // Verify first_token_offset tensor
    hipdnn_backend::ScopedDescriptor firstTokenOffsetScoped;
    int64_t firstTokenOffsetCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &firstTokenOffsetCount,
                       static_cast<void*>(firstTokenOffsetScoped.getPtr()));
    ASSERT_EQ(firstTokenOffsetCount, 1);
    ASSERT_NE(firstTokenOffsetScoped.get(), nullptr);
    verifyTensorDescriptor(firstTokenOffsetScoped.get(),
                           K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                           HIPDNN_DATA_INT32,
                           {5, 1, 1},
                           {1, 1, 1});

    // Verify token_index tensor
    hipdnn_backend::ScopedDescriptor tokenIndexScoped;
    int64_t tokenIndexCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &tokenIndexCount,
                       static_cast<void*>(tokenIndexScoped.getPtr()));
    ASSERT_EQ(tokenIndexCount, 1);
    ASSERT_NE(tokenIndexScoped.get(), nullptr);
    verifyTensorDescriptor(tokenIndexScoped.get(),
                           K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                           HIPDNN_DATA_INT32,
                           {1, 8, 1},
                           {8, 1, 1});

    // Verify token_ks tensor
    hipdnn_backend::ScopedDescriptor tokenKsScoped;
    int64_t tokenKsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &tokenKsCount,
                       static_cast<void*>(tokenKsScoped.getPtr()));
    ASSERT_EQ(tokenKsCount, 1);
    ASSERT_NE(tokenKsScoped.get(), nullptr);
    verifyTensorDescriptor(tokenKsScoped.get(),
                           K_MOE_ROUTING_TENSOR_TOKEN_KS_UID,
                           HIPDNN_DATA_INT32,
                           {1, 8, 1},
                           {8, 1, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_MOE_ROUTING_EXT);
}

TEST_F(TestMoeRoutingOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_moerouting_1";

    auto desc = MoeRoutingOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_moerouting_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_moerouting_1");
}

TEST_F(TestMoeRoutingOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = MoeRoutingOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestMoeRoutingOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = MoeRoutingOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
