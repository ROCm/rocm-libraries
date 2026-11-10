// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/SoftmaxOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/softmax_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SoftmaxConstants.hpp>
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
// SoftmaxOperationDescriptor::fromNode() Tests
// =============================================================================

class TestSoftmaxOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT pAttrs;
        pAttrs.uid = K_SOFTMAX_TENSOR_P_UID;
        pAttrs.data_type = DataType::FLOAT;
        pAttrs.dims = toVec(K_SOFTMAX_TENSOR_P_DIMS);
        pAttrs.strides = toVec(K_SOFTMAX_TENSOR_P_STRIDES);

        _tensorMap[K_SOFTMAX_TENSOR_P_UID] = TensorDescriptor::fromFlatBuffer(pAttrs);
        TensorAttributesT sAttrs;
        sAttrs.uid = K_SOFTMAX_TENSOR_S_UID;
        sAttrs.data_type = DataType::FLOAT;
        sAttrs.dims = toVec(K_SOFTMAX_TENSOR_S_DIMS);
        sAttrs.strides = toVec(K_SOFTMAX_TENSOR_S_STRIDES);

        _tensorMap[K_SOFTMAX_TENSOR_S_UID] = TensorDescriptor::fromFlatBuffer(sAttrs);
        TensorAttributesT statsAttrs;
        statsAttrs.uid = K_SOFTMAX_TENSOR_STATS_UID;
        statsAttrs.data_type = DataType::FLOAT;
        statsAttrs.dims = toVec(K_SOFTMAX_TENSOR_STATS_DIMS);
        statsAttrs.strides = toVec(K_SOFTMAX_TENSOR_STATS_STRIDES);

        _tensorMap[K_SOFTMAX_TENSOR_STATS_UID] = TensorDescriptor::fromFlatBuffer(statsAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::SoftmaxAttributesT createStandardSoftmaxAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::SoftmaxAttributesT attrs;
        attrs.p_tensor_uid = K_SOFTMAX_TENSOR_P_UID;
        attrs.s_tensor_uid = K_SOFTMAX_TENSOR_S_UID;
        attrs.stats_tensor_uid = K_SOFTMAX_TENSOR_STATS_UID;
        attrs.stats_use_log2 = true;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardSoftmaxAttrs());
        return node;
    }
};

TEST_F(TestSoftmaxOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR);
    EXPECT_EQ(desc->getData().p_tensor_uid, K_SOFTMAX_TENSOR_P_UID);
}

TEST_F(TestSoftmaxOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::SoftmaxAttributes);
    auto desc = std::static_pointer_cast<SoftmaxOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().p_tensor_uid, K_SOFTMAX_TENSOR_P_UID);
    EXPECT_EQ(desc->getData().s_tensor_uid, K_SOFTMAX_TENSOR_S_UID);
    EXPECT_EQ(desc->getData().stats_tensor_uid, K_SOFTMAX_TENSOR_STATS_UID);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getPDesc()->getData().uid, K_SOFTMAX_TENSOR_P_UID);
    EXPECT_EQ(desc->getSDesc()->getData().uid, K_SOFTMAX_TENSOR_S_UID);
    EXPECT_EQ(desc->getStatsDesc()->getData().uid, K_SOFTMAX_TENSOR_STATS_UID);
}

TEST_F(TestSoftmaxOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestSoftmaxOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getPDesc(), nullptr);
    EXPECT_EQ(desc->getPDesc()->getData().uid, K_SOFTMAX_TENSOR_P_UID);
    ASSERT_NE(desc->getSDesc(), nullptr);
    EXPECT_EQ(desc->getSDesc()->getData().uid, K_SOFTMAX_TENSOR_S_UID);
    ASSERT_NE(desc->getStatsDesc(), nullptr);
    EXPECT_EQ(desc->getStatsDesc()->getData().uid, K_SOFTMAX_TENSOR_STATS_UID);
}

TEST_F(TestSoftmaxOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getPDesc(), _tensorMap[K_SOFTMAX_TENSOR_P_UID]);
    EXPECT_EQ(desc->getSDesc(), _tensorMap[K_SOFTMAX_TENSOR_S_UID]);
    EXPECT_EQ(desc->getStatsDesc(), _tensorMap[K_SOFTMAX_TENSOR_STATS_UID]);
}

TEST_F(TestSoftmaxOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getPDesc(), nullptr);
    EXPECT_EQ(desc->getPDesc()->getData().uid, K_SOFTMAX_TENSOR_P_UID);
    EXPECT_EQ(desc->getPDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getPDesc()->getData().dims, (std::vector<int64_t>{2, 4, 8, 16}));
    EXPECT_EQ(desc->getPDesc()->getData().strides, (std::vector<int64_t>{512, 128, 16, 1}));

    ASSERT_NE(desc->getSDesc(), nullptr);
    EXPECT_EQ(desc->getSDesc()->getData().uid, K_SOFTMAX_TENSOR_S_UID);
    EXPECT_EQ(desc->getSDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getSDesc()->getData().dims, (std::vector<int64_t>{2, 4, 8, 16}));
    EXPECT_EQ(desc->getSDesc()->getData().strides, (std::vector<int64_t>{512, 128, 16, 1}));

    ASSERT_NE(desc->getStatsDesc(), nullptr);
    EXPECT_EQ(desc->getStatsDesc()->getData().uid, K_SOFTMAX_TENSOR_STATS_UID);
    EXPECT_EQ(desc->getStatsDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getStatsDesc()->getData().dims, (std::vector<int64_t>{2, 4, 8, 1}));
    EXPECT_EQ(desc->getStatsDesc()->getData().strides, (std::vector<int64_t>{32, 8, 1, 1}));
}

// The parameter is the UID of the one required tensor to drop from the tensor map.
class TestSoftmaxOperationFromNodeMissingTensor : public TestSoftmaxOperationFromNode,
                                                  public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestSoftmaxOperationFromNodeMissingTensor, FailsWithMissingTensor)
{
    _tensorMap.erase(GetParam());
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(SoftmaxOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(RequiredTensors,
                         TestSoftmaxOperationFromNodeMissingTensor,
                         ::testing::Values(K_SOFTMAX_TENSOR_P_UID, K_SOFTMAX_TENSOR_S_UID));

TEST_F(TestSoftmaxOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardSoftmaxAttrs();
    attrs.stats_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getPDesc(), nullptr);
    EXPECT_NE(desc->getSDesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getStatsDesc(), nullptr);

    const auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 2u);
    EXPECT_EQ(tensors[0], desc->getPDesc());
    EXPECT_EQ(tensors[1], desc->getSDesc());
}

TEST_F(TestSoftmaxOperationFromNode, FailsWhenOptionalStatsUidSetButTensorMissing)
{
    _tensorMap.erase(K_SOFTMAX_TENSOR_STATS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(SoftmaxOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestSoftmaxOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0]->getData().uid, K_SOFTMAX_TENSOR_P_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_SOFTMAX_TENSOR_S_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_SOFTMAX_TENSOR_STATS_UID);
}

TEST_F(TestSoftmaxOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::SoftmaxAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSoftmaxAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->p_tensor_uid, K_SOFTMAX_TENSOR_P_UID);
    EXPECT_EQ(rebuiltAttrs->s_tensor_uid, K_SOFTMAX_TENSOR_S_UID);
    EXPECT_EQ(rebuiltAttrs->stats_tensor_uid, K_SOFTMAX_TENSOR_STATS_UID);
}

TEST_F(TestSoftmaxOperationFromNode, FromNodePreservesStatsUseLog2)
{
    auto attrs = createStandardSoftmaxAttrs();
    attrs.stats_use_log2 = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getData().stats_use_log2, true);

    const auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSoftmaxAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->stats_use_log2, true);
}

TEST_F(TestSoftmaxOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify p tensor
    hipdnn_backend::ScopedDescriptor pScoped;
    int64_t pCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &pCount,
                       static_cast<void*>(pScoped.getPtr()));
    ASSERT_EQ(pCount, 1);
    ASSERT_NE(pScoped.get(), nullptr);
    verifyTensorDescriptor(
        pScoped.get(), K_SOFTMAX_TENSOR_P_UID, HIPDNN_DATA_FLOAT, {2, 4, 8, 16}, {512, 128, 16, 1});

    // Verify s tensor
    hipdnn_backend::ScopedDescriptor sScoped;
    int64_t sCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &sCount,
                       static_cast<void*>(sScoped.getPtr()));
    ASSERT_EQ(sCount, 1);
    ASSERT_NE(sScoped.get(), nullptr);
    verifyTensorDescriptor(
        sScoped.get(), K_SOFTMAX_TENSOR_S_UID, HIPDNN_DATA_FLOAT, {2, 4, 8, 16}, {512, 128, 16, 1});

    // Verify stats tensor (optional)
    hipdnn_backend::ScopedDescriptor statsScoped;
    int64_t statsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &statsCount,
                       static_cast<void*>(statsScoped.getPtr()));
    ASSERT_EQ(statsCount, 1);
    ASSERT_NE(statsScoped.get(), nullptr);
    verifyTensorDescriptor(statsScoped.get(),
                           K_SOFTMAX_TENSOR_STATS_UID,
                           HIPDNN_DATA_FLOAT,
                           {2, 4, 8, 1},
                           {32, 8, 1, 1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_SOFTMAX_EXT);
}

TEST_F(TestSoftmaxOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_softmax_1";

    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_softmax_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_softmax_1");
}

TEST_F(TestSoftmaxOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestSoftmaxOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = SoftmaxOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
