// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/BatchnormOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_data_sdk::data_objects;

// =============================================================================
// BatchnormOperationDescriptor::fromNode() Tests
// =============================================================================

class TestBatchnormOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = 50;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = {1, 64, 32, 32};
        xAttrs.strides = {65536, 1024, 32, 1};

        _tensorMap[50] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = 51;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims = {1, 64, 1, 1};
        scaleAttrs.strides = {64, 1, 1, 1};

        _tensorMap[51] = TensorDescriptor::fromFlatBuffer(scaleAttrs);
        TensorAttributesT biasAttrs;
        biasAttrs.uid = 52;
        biasAttrs.data_type = DataType::FLOAT;
        biasAttrs.dims = {1, 64, 1, 1};
        biasAttrs.strides = {64, 1, 1, 1};

        _tensorMap[52] = TensorDescriptor::fromFlatBuffer(biasAttrs);
        TensorAttributesT epsilonAttrs;
        epsilonAttrs.uid = 53;
        epsilonAttrs.data_type = DataType::FLOAT;
        epsilonAttrs.dims = {1, 1, 1, 1};
        epsilonAttrs.strides = {1, 1, 1, 1};

        _tensorMap[53] = TensorDescriptor::fromFlatBuffer(epsilonAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = 54;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = {1, 64, 32, 32};
        yAttrs.strides = {65536, 1024, 32, 1};

        _tensorMap[54] = TensorDescriptor::fromFlatBuffer(yAttrs);
        TensorAttributesT prevRunningMeanAttrs;
        prevRunningMeanAttrs.uid = 6;
        prevRunningMeanAttrs.data_type = DataType::FLOAT;
        prevRunningMeanAttrs.dims = {1};
        prevRunningMeanAttrs.strides = {1};

        _tensorMap[6] = TensorDescriptor::fromFlatBuffer(prevRunningMeanAttrs);
        TensorAttributesT prevRunningVarianceAttrs;
        prevRunningVarianceAttrs.uid = 7;
        prevRunningVarianceAttrs.data_type = DataType::FLOAT;
        prevRunningVarianceAttrs.dims = {1};
        prevRunningVarianceAttrs.strides = {1};

        _tensorMap[7] = TensorDescriptor::fromFlatBuffer(prevRunningVarianceAttrs);
        TensorAttributesT momentumAttrs;
        momentumAttrs.uid = 8;
        momentumAttrs.data_type = DataType::FLOAT;
        momentumAttrs.dims = {1};
        momentumAttrs.strides = {1};

        _tensorMap[8] = TensorDescriptor::fromFlatBuffer(momentumAttrs);
        TensorAttributesT meanAttrs;
        meanAttrs.uid = 9;
        meanAttrs.data_type = DataType::FLOAT;
        meanAttrs.dims = {1};
        meanAttrs.strides = {1};

        _tensorMap[9] = TensorDescriptor::fromFlatBuffer(meanAttrs);
        TensorAttributesT invVarianceAttrs;
        invVarianceAttrs.uid = 10;
        invVarianceAttrs.data_type = DataType::FLOAT;
        invVarianceAttrs.dims = {1};
        invVarianceAttrs.strides = {1};

        _tensorMap[10] = TensorDescriptor::fromFlatBuffer(invVarianceAttrs);
        TensorAttributesT nextRunningMeanAttrs;
        nextRunningMeanAttrs.uid = 11;
        nextRunningMeanAttrs.data_type = DataType::FLOAT;
        nextRunningMeanAttrs.dims = {1};
        nextRunningMeanAttrs.strides = {1};

        _tensorMap[11] = TensorDescriptor::fromFlatBuffer(nextRunningMeanAttrs);
        TensorAttributesT nextRunningVarianceAttrs;
        nextRunningVarianceAttrs.uid = 12;
        nextRunningVarianceAttrs.data_type = DataType::FLOAT;
        nextRunningVarianceAttrs.dims = {1};
        nextRunningVarianceAttrs.strides = {1};

        _tensorMap[12] = TensorDescriptor::fromFlatBuffer(nextRunningVarianceAttrs);
        TensorAttributesT peerStatsAttrs0;
        peerStatsAttrs0.uid = 100;
        peerStatsAttrs0.data_type = DataType::FLOAT;
        peerStatsAttrs0.dims = {1};
        peerStatsAttrs0.strides = {1};

        _tensorMap[100] = TensorDescriptor::fromFlatBuffer(peerStatsAttrs0);
        TensorAttributesT peerStatsAttrs1;
        peerStatsAttrs1.uid = 101;
        peerStatsAttrs1.data_type = DataType::FLOAT;
        peerStatsAttrs1.dims = {1};
        peerStatsAttrs1.strides = {1};

        _tensorMap[101] = TensorDescriptor::fromFlatBuffer(peerStatsAttrs1);
    }

    static hipdnn_data_sdk::data_objects::BatchnormAttributesT createStandardBatchnormAttrs()
    {
        hipdnn_data_sdk::data_objects::BatchnormAttributesT attrs;
        attrs.x_tensor_uid = 50;
        attrs.scale_tensor_uid = 51;
        attrs.bias_tensor_uid = 52;
        attrs.epsilon_tensor_uid = 53;
        attrs.y_tensor_uid = 54;
        attrs.prev_running_mean_tensor_uid = 6;
        attrs.prev_running_variance_tensor_uid = 7;
        attrs.momentum_tensor_uid = 8;
        attrs.mean_tensor_uid = 9;
        attrs.inv_variance_tensor_uid = 10;
        attrs.next_running_mean_tensor_uid = 11;
        attrs.next_running_variance_tensor_uid = 12;
        attrs.peer_stats_tensor_uid = {100, 101};
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardBatchnormAttrs());
        return node;
    }
};

TEST_F(TestBatchnormOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, 50);
}

TEST_F(TestBatchnormOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BatchnormAttributes);
    auto desc = std::static_pointer_cast<BatchnormOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, 50);
    EXPECT_EQ(desc->getData().scale_tensor_uid, 51);
    EXPECT_EQ(desc->getData().bias_tensor_uid, 52);
    EXPECT_EQ(desc->getData().epsilon_tensor_uid, 53);
    EXPECT_EQ(desc->getData().y_tensor_uid, 54);
    EXPECT_EQ(desc->getData().prev_running_mean_tensor_uid, 6);
    EXPECT_EQ(desc->getData().prev_running_variance_tensor_uid, 7);
    EXPECT_EQ(desc->getData().momentum_tensor_uid, 8);
    EXPECT_EQ(desc->getData().mean_tensor_uid, 9);
    EXPECT_EQ(desc->getData().inv_variance_tensor_uid, 10);
    EXPECT_EQ(desc->getData().next_running_mean_tensor_uid, 11);
    EXPECT_EQ(desc->getData().next_running_variance_tensor_uid, 12);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, 50);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, 51);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, 52);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, 53);
    EXPECT_EQ(desc->getYDesc()->getData().uid, 54);
}

TEST_F(TestBatchnormOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestBatchnormOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, 50);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, 51);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, 52);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, 53);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, 54);
}

TEST_F(TestBatchnormOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[50]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[51]);
    EXPECT_EQ(desc->getBiasDesc(), _tensorMap[52]);
    EXPECT_EQ(desc->getEpsilonDesc(), _tensorMap[53]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[54]);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(50);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(51);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMissingBiasTensor)
{
    _tensorMap.erase(52);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMissingEpsilonTensor)
{
    _tensorMap.erase(53);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(54);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardBatchnormAttrs();
    attrs.prev_running_mean_tensor_uid = flatbuffers::nullopt;
    attrs.prev_running_variance_tensor_uid = flatbuffers::nullopt;
    attrs.momentum_tensor_uid = flatbuffers::nullopt;
    attrs.mean_tensor_uid = flatbuffers::nullopt;
    attrs.inv_variance_tensor_uid = flatbuffers::nullopt;
    attrs.next_running_mean_tensor_uid = flatbuffers::nullopt;
    attrs.next_running_variance_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getXDesc(), nullptr);
    EXPECT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_NE(desc->getYDesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getPrevRunningMeanDesc(), nullptr);
    EXPECT_EQ(desc->getPrevRunningVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getMomentumDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getNextRunningMeanDesc(), nullptr);
    EXPECT_EQ(desc->getNextRunningVarianceDesc(), nullptr);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalPrevRunningMeanUidSetButTensorMissing)
{
    _tensorMap.erase(6);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalPrevRunningVarianceUidSetButTensorMissing)
{
    _tensorMap.erase(7);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalMomentumUidSetButTensorMissing)
{
    _tensorMap.erase(8);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalMeanUidSetButTensorMissing)
{
    _tensorMap.erase(9);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalInvVarianceUidSetButTensorMissing)
{
    _tensorMap.erase(10);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalNextRunningMeanUidSetButTensorMissing)
{
    _tensorMap.erase(11);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalNextRunningVarianceUidSetButTensorMissing)
{
    _tensorMap.erase(12);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    // 5 required + 7 optional + 2 peer_stats = 14 total
    ASSERT_EQ(tensors.size(), 14);
    EXPECT_EQ(tensors[0]->getData().uid, 50);
    EXPECT_EQ(tensors[1]->getData().uid, 51);
    EXPECT_EQ(tensors[2]->getData().uid, 52);
    EXPECT_EQ(tensors[3]->getData().uid, 53);
    EXPECT_EQ(tensors[4]->getData().uid, 54);
}

TEST_F(TestBatchnormOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BatchnormAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBatchnormAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, 50);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, 51);
    EXPECT_EQ(rebuiltAttrs->bias_tensor_uid, 52);
    EXPECT_EQ(rebuiltAttrs->epsilon_tensor_uid, 53);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, 54);
    EXPECT_EQ(rebuiltAttrs->prev_running_mean_tensor_uid, 6);
    EXPECT_EQ(rebuiltAttrs->prev_running_variance_tensor_uid, 7);
    EXPECT_EQ(rebuiltAttrs->momentum_tensor_uid, 8);
    EXPECT_EQ(rebuiltAttrs->mean_tensor_uid, 9);
    EXPECT_EQ(rebuiltAttrs->inv_variance_tensor_uid, 10);
    EXPECT_EQ(rebuiltAttrs->next_running_mean_tensor_uid, 11);
    EXPECT_EQ(rebuiltAttrs->next_running_variance_tensor_uid, 12);

    // Verify peer_stats round-trip
    ASSERT_EQ(rebuiltAttrs->peer_stats_tensor_uid.size(), 2);
    EXPECT_EQ(rebuiltAttrs->peer_stats_tensor_uid[0], 100);
    EXPECT_EQ(rebuiltAttrs->peer_stats_tensor_uid[1], 101);
}

TEST_F(TestBatchnormOperationFromNode, BuildNodeRoundTripWithOnlyRequiredTensors)
{
    auto attrs = createStandardBatchnormAttrs();
    attrs.prev_running_mean_tensor_uid = flatbuffers::nullopt;
    attrs.prev_running_variance_tensor_uid = flatbuffers::nullopt;
    attrs.momentum_tensor_uid = flatbuffers::nullopt;
    attrs.mean_tensor_uid = flatbuffers::nullopt;
    attrs.inv_variance_tensor_uid = flatbuffers::nullopt;
    attrs.next_running_mean_tensor_uid = flatbuffers::nullopt;
    attrs.next_running_variance_tensor_uid = flatbuffers::nullopt;
    attrs.peer_stats_tensor_uid.clear();

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBatchnormAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

    // Required tensors preserved
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, 50);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, 51);
    EXPECT_EQ(rebuiltAttrs->bias_tensor_uid, 52);
    EXPECT_EQ(rebuiltAttrs->epsilon_tensor_uid, 53);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, 54);

    // Optional tensors remain unset
    EXPECT_FALSE(rebuiltAttrs->prev_running_mean_tensor_uid.has_value());
    EXPECT_FALSE(rebuiltAttrs->prev_running_variance_tensor_uid.has_value());
    EXPECT_FALSE(rebuiltAttrs->momentum_tensor_uid.has_value());
    EXPECT_FALSE(rebuiltAttrs->mean_tensor_uid.has_value());
    EXPECT_FALSE(rebuiltAttrs->inv_variance_tensor_uid.has_value());
    EXPECT_FALSE(rebuiltAttrs->next_running_mean_tensor_uid.has_value());
    EXPECT_FALSE(rebuiltAttrs->next_running_variance_tensor_uid.has_value());

    // Peer stats empty
    EXPECT_TRUE(rebuiltAttrs->peer_stats_tensor_uid.empty());
}

TEST_F(TestBatchnormOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_batchnorm_1";

    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_batchnorm_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_batchnorm_1");
}

TEST_F(TestBatchnormOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestBatchnormOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestBatchnormOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type via getAttribute
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_BATCHNORM_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(dtCount, 1);
    EXPECT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify operation type
    hipdnnOperationType_t opType = HIPDNN_OPERATION_TYPE_NOT_SET;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BATCHNORM);

    // Verify name (empty default from fixture, count==1 for null terminator)
    int64_t nameCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &nameCount, nullptr);
    EXPECT_EQ(nameCount, 1);

    // --- Required tensor attributes ---

    // X tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT,
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
    EXPECT_EQ(xUid, 50);

    // Scale tensor
    hipdnn_backend::ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    int64_t scaleUid = 0;
    int64_t scaleUidCount = 0;
    scaleScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &scaleUidCount, &scaleUid);
    EXPECT_EQ(scaleUid, 51);

    // Bias tensor
    hipdnn_backend::ScopedDescriptor biasScoped;
    int64_t biasCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &biasCount,
                       static_cast<void*>(biasScoped.getPtr()));
    ASSERT_EQ(biasCount, 1);
    ASSERT_NE(biasScoped.get(), nullptr);
    int64_t biasUid = 0;
    int64_t biasUidCount = 0;
    biasScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &biasUidCount, &biasUid);
    EXPECT_EQ(biasUid, 52);

    // Epsilon tensor
    hipdnn_backend::ScopedDescriptor epsilonScoped;
    int64_t epsilonCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &epsilonCount,
                       static_cast<void*>(epsilonScoped.getPtr()));
    ASSERT_EQ(epsilonCount, 1);
    ASSERT_NE(epsilonScoped.get(), nullptr);
    int64_t epsilonUid = 0;
    int64_t epsilonUidCount = 0;
    epsilonScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &epsilonUidCount, &epsilonUid);
    EXPECT_EQ(epsilonUid, 53);

    // Y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT,
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
    EXPECT_EQ(yUid, 54);

    // --- Optional tensor attributes (all set in standard fixture) ---

    // PrevRunningMean tensor
    hipdnn_backend::ScopedDescriptor prevRunMeanScoped;
    int64_t prevRunMeanCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &prevRunMeanCount,
                       static_cast<void*>(prevRunMeanScoped.getPtr()));
    ASSERT_EQ(prevRunMeanCount, 1);
    ASSERT_NE(prevRunMeanScoped.get(), nullptr);
    int64_t prevRunMeanUid = 0;
    int64_t prevRunMeanUidCount = 0;
    prevRunMeanScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &prevRunMeanUidCount, &prevRunMeanUid);
    EXPECT_EQ(prevRunMeanUid, 6);

    // PrevRunningVariance tensor
    hipdnn_backend::ScopedDescriptor prevRunVarScoped;
    int64_t prevRunVarCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &prevRunVarCount,
                       static_cast<void*>(prevRunVarScoped.getPtr()));
    ASSERT_EQ(prevRunVarCount, 1);
    ASSERT_NE(prevRunVarScoped.get(), nullptr);
    int64_t prevRunVarUid = 0;
    int64_t prevRunVarUidCount = 0;
    prevRunVarScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &prevRunVarUidCount, &prevRunVarUid);
    EXPECT_EQ(prevRunVarUid, 7);

    // Momentum tensor
    hipdnn_backend::ScopedDescriptor momentumScoped;
    int64_t momentumCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &momentumCount,
                       static_cast<void*>(momentumScoped.getPtr()));
    ASSERT_EQ(momentumCount, 1);
    ASSERT_NE(momentumScoped.get(), nullptr);
    int64_t momentumUid = 0;
    int64_t momentumUidCount = 0;
    momentumScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &momentumUidCount, &momentumUid);
    EXPECT_EQ(momentumUid, 8);

    // Mean tensor
    hipdnn_backend::ScopedDescriptor meanScoped;
    int64_t meanCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &meanCount,
                       static_cast<void*>(meanScoped.getPtr()));
    ASSERT_EQ(meanCount, 1);
    ASSERT_NE(meanScoped.get(), nullptr);
    int64_t meanUid = 0;
    int64_t meanUidCount = 0;
    meanScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &meanUidCount, &meanUid);
    EXPECT_EQ(meanUid, 9);

    // InvVariance tensor
    hipdnn_backend::ScopedDescriptor invVarScoped;
    int64_t invVarCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &invVarCount,
                       static_cast<void*>(invVarScoped.getPtr()));
    ASSERT_EQ(invVarCount, 1);
    ASSERT_NE(invVarScoped.get(), nullptr);
    int64_t invVarUid = 0;
    int64_t invVarUidCount = 0;
    invVarScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &invVarUidCount, &invVarUid);
    EXPECT_EQ(invVarUid, 10);

    // NextRunningMean tensor
    hipdnn_backend::ScopedDescriptor nextRunMeanScoped;
    int64_t nextRunMeanCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &nextRunMeanCount,
                       static_cast<void*>(nextRunMeanScoped.getPtr()));
    ASSERT_EQ(nextRunMeanCount, 1);
    ASSERT_NE(nextRunMeanScoped.get(), nullptr);
    int64_t nextRunMeanUid = 0;
    int64_t nextRunMeanUidCount = 0;
    nextRunMeanScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &nextRunMeanUidCount, &nextRunMeanUid);
    EXPECT_EQ(nextRunMeanUid, 11);

    // NextRunningVariance tensor
    hipdnn_backend::ScopedDescriptor nextRunVarScoped;
    int64_t nextRunVarCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &nextRunVarCount,
                       static_cast<void*>(nextRunVarScoped.getPtr()));
    ASSERT_EQ(nextRunVarCount, 1);
    ASSERT_NE(nextRunVarScoped.get(), nullptr);
    int64_t nextRunVarUid = 0;
    int64_t nextRunVarUidCount = 0;
    nextRunVarScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &nextRunVarUidCount, &nextRunVarUid);
    EXPECT_EQ(nextRunVarUid, 12);

    // --- Peer stats tensor array ---

    // Query count first
    int64_t peerStatsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       0,
                       &peerStatsCount,
                       nullptr);
    ASSERT_EQ(peerStatsCount, 2);

    // Retrieve both peer_stats descriptors
    hipdnn_backend::ScopedDescriptor peerStats0Scoped;
    hipdnn_backend::ScopedDescriptor peerStats1Scoped;
    hipdnnBackendDescriptor_t peerStatsArray[2] = {};
    int64_t peerStatsRetrievedCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       2,
                       &peerStatsRetrievedCount,
                       static_cast<void*>(peerStatsArray));
    ASSERT_EQ(peerStatsRetrievedCount, 2);
    // Transfer ownership to ScopedDescriptors
    peerStats0Scoped = hipdnn_backend::ScopedDescriptor(peerStatsArray[0]);
    peerStats1Scoped = hipdnn_backend::ScopedDescriptor(peerStatsArray[1]);
    ASSERT_NE(peerStats0Scoped.get(), nullptr);
    ASSERT_NE(peerStats1Scoped.get(), nullptr);

    int64_t peer0Uid = 0;
    int64_t peer0UidCount = 0;
    peerStats0Scoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &peer0UidCount, &peer0Uid);
    EXPECT_EQ(peer0Uid, 100);

    int64_t peer1Uid = 0;
    int64_t peer1UidCount = 0;
    peerStats1Scoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &peer1UidCount, &peer1Uid);
    EXPECT_EQ(peer1Uid, 101);
}

TEST_F(TestBatchnormOperationFromNode, OperationTypeAttributeReturnsCorrectValue)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    hipdnnOperationType_t opType = HIPDNN_OPERATION_TYPE_NOT_SET;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BATCHNORM);
}

TEST_F(TestBatchnormOperationFromNode, OptionalTensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getPrevRunningMeanDesc(), _tensorMap[6]);
    EXPECT_EQ(desc->getPrevRunningVarianceDesc(), _tensorMap[7]);
    EXPECT_EQ(desc->getMomentumDesc(), _tensorMap[8]);
    EXPECT_EQ(desc->getMeanDesc(), _tensorMap[9]);
    EXPECT_EQ(desc->getInvVarianceDesc(), _tensorMap[10]);
    EXPECT_EQ(desc->getNextRunningMeanDesc(), _tensorMap[11]);
    EXPECT_EQ(desc->getNextRunningVarianceDesc(), _tensorMap[12]);
}

TEST_F(TestBatchnormOperationFromNode, PeerStatsTensorReferencesPopulated)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    // 5 required + 2 (mean, inv_var) + 5 (running stats) + 2 peer_stats = 14 total
    ASSERT_EQ(tensors.size(), 14);

    // Last 2 are peer_stats (UIDs 100, 101)
    EXPECT_EQ(tensors[12]->getData().uid, 100);
    EXPECT_EQ(tensors[13]->getData().uid, 101);

    // Verify they are the same shared_ptr instances from tensorMap
    EXPECT_EQ(tensors[12], _tensorMap[100]);
    EXPECT_EQ(tensors[13], _tensorMap[101]);
}
