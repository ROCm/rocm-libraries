// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormBackwardOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BatchnormBackwardConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// BatchnormBackwardOperationDescriptor::fromNode() Tests
// =============================================================================

class TestBatchnormBackwardOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        auto addTensor =
            [&](int64_t uid, const auto& dims, const auto& strides, DataType dt = DataType::FLOAT) {
                TensorAttributesT attrs;
                attrs.uid = uid;
                attrs.data_type = dt;
                attrs.dims = {dims.begin(), dims.end()};
                attrs.strides = {strides.begin(), strides.end()};
                _tensorMap[uid] = TensorDescriptor::fromFlatBuffer(attrs);
            };

        addTensor(K_BN_BWD_TENSOR_DY_UID, K_BN_BWD_TENSOR_DY_DIMS, K_BN_BWD_TENSOR_DY_STRIDES);
        addTensor(K_BN_BWD_TENSOR_X_UID, K_BN_BWD_TENSOR_X_DIMS, K_BN_BWD_TENSOR_X_STRIDES);
        addTensor(
            K_BN_BWD_TENSOR_SCALE_UID, K_BN_BWD_TENSOR_SCALE_DIMS, K_BN_BWD_TENSOR_SCALE_STRIDES);
        addTensor(K_BN_BWD_TENSOR_DX_UID, K_BN_BWD_TENSOR_DX_DIMS, K_BN_BWD_TENSOR_DX_STRIDES);
        addTensor(K_BN_BWD_TENSOR_DSCALE_UID,
                  K_BN_BWD_TENSOR_DSCALE_DIMS,
                  K_BN_BWD_TENSOR_DSCALE_STRIDES);
        addTensor(
            K_BN_BWD_TENSOR_DBIAS_UID, K_BN_BWD_TENSOR_DBIAS_DIMS, K_BN_BWD_TENSOR_DBIAS_STRIDES);
        addTensor(
            K_BN_BWD_TENSOR_MEAN_UID, K_BN_BWD_TENSOR_MEAN_DIMS, K_BN_BWD_TENSOR_MEAN_STRIDES);
        addTensor(K_BN_BWD_TENSOR_INV_VARIANCE_UID,
                  K_BN_BWD_TENSOR_INV_VARIANCE_DIMS,
                  K_BN_BWD_TENSOR_INV_VARIANCE_STRIDES);
        addTensor(K_BN_BWD_TENSOR_PEER_STAT_0_UID,
                  K_BN_BWD_TENSOR_PEER_STAT_DIMS,
                  K_BN_BWD_TENSOR_PEER_STAT_STRIDES);
        addTensor(K_BN_BWD_TENSOR_PEER_STAT_1_UID,
                  K_BN_BWD_TENSOR_PEER_STAT_DIMS,
                  K_BN_BWD_TENSOR_PEER_STAT_STRIDES);
    }

    static BatchnormBackwardAttributesT createStandardAttrs()
    {
        BatchnormBackwardAttributesT attrs;
        attrs.dy_tensor_uid = K_BN_BWD_TENSOR_DY_UID;
        attrs.x_tensor_uid = K_BN_BWD_TENSOR_X_UID;
        attrs.scale_tensor_uid = K_BN_BWD_TENSOR_SCALE_UID;
        attrs.dx_tensor_uid = K_BN_BWD_TENSOR_DX_UID;
        attrs.dscale_tensor_uid = K_BN_BWD_TENSOR_DSCALE_UID;
        attrs.dbias_tensor_uid = K_BN_BWD_TENSOR_DBIAS_UID;
        attrs.mean_tensor_uid = K_BN_BWD_TENSOR_MEAN_UID;
        attrs.inv_variance_tensor_uid = K_BN_BWD_TENSOR_INV_VARIANCE_UID;
        attrs.peer_stats_tensor_uid
            = {K_BN_BWD_TENSOR_PEER_STAT_0_UID, K_BN_BWD_TENSOR_PEER_STAT_1_UID};
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardAttrs());
        return node;
    }
};

TEST_F(TestBatchnormBackwardOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().dy_tensor_uid, K_BN_BWD_TENSOR_DY_UID);
}

TEST_F(TestBatchnormBackwardOperationFromNode, NodeFactoryDelegatesCorrectly)
{
    auto node = createStandardNode();

    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);
    ASSERT_NE(graphOp, nullptr);

    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BatchnormBackwardAttributes);
    auto desc = std::static_pointer_cast<BatchnormBackwardOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    EXPECT_EQ(desc->getData().dy_tensor_uid, K_BN_BWD_TENSOR_DY_UID);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_BN_BWD_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_BN_BWD_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getData().dx_tensor_uid, K_BN_BWD_TENSOR_DX_UID);
    EXPECT_EQ(desc->getData().dscale_tensor_uid, K_BN_BWD_TENSOR_DSCALE_UID);
    EXPECT_EQ(desc->getData().dbias_tensor_uid, K_BN_BWD_TENSOR_DBIAS_UID);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestBatchnormBackwardOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestBatchnormBackwardOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    // DY tensor
    ASSERT_NE(desc->getDyDesc(), nullptr);
    EXPECT_EQ(desc->getDyDesc()->getData().uid, K_BN_BWD_TENSOR_DY_UID);
    EXPECT_EQ(desc->getDyDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDyDesc()->getData().dims, toVec(K_BN_BWD_TENSOR_DY_DIMS));
    EXPECT_EQ(desc->getDyDesc()->getData().strides, toVec(K_BN_BWD_TENSOR_DY_STRIDES));

    // X tensor
    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BN_BWD_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, toVec(K_BN_BWD_TENSOR_X_DIMS));
    EXPECT_EQ(desc->getXDesc()->getData().strides, toVec(K_BN_BWD_TENSOR_X_STRIDES));

    // Scale tensor
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BN_BWD_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims, toVec(K_BN_BWD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides, toVec(K_BN_BWD_TENSOR_SCALE_STRIDES));

    // DX tensor
    ASSERT_NE(desc->getDxDesc(), nullptr);
    EXPECT_EQ(desc->getDxDesc()->getData().uid, K_BN_BWD_TENSOR_DX_UID);
    EXPECT_EQ(desc->getDxDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDxDesc()->getData().dims, toVec(K_BN_BWD_TENSOR_DX_DIMS));
    EXPECT_EQ(desc->getDxDesc()->getData().strides, toVec(K_BN_BWD_TENSOR_DX_STRIDES));

    // DScale tensor
    ASSERT_NE(desc->getDscaleDesc(), nullptr);
    EXPECT_EQ(desc->getDscaleDesc()->getData().uid, K_BN_BWD_TENSOR_DSCALE_UID);
    EXPECT_EQ(desc->getDscaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDscaleDesc()->getData().dims, toVec(K_BN_BWD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(desc->getDscaleDesc()->getData().strides, toVec(K_BN_BWD_TENSOR_DSCALE_STRIDES));

    // DBias tensor
    ASSERT_NE(desc->getDbiasDesc(), nullptr);
    EXPECT_EQ(desc->getDbiasDesc()->getData().uid, K_BN_BWD_TENSOR_DBIAS_UID);
    EXPECT_EQ(desc->getDbiasDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getDbiasDesc()->getData().dims, toVec(K_BN_BWD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(desc->getDbiasDesc()->getData().strides, toVec(K_BN_BWD_TENSOR_DBIAS_STRIDES));

    // Mean tensor (optional)
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_BN_BWD_TENSOR_MEAN_UID);
    EXPECT_EQ(desc->getMeanDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getMeanDesc()->getData().dims, toVec(K_BN_BWD_TENSOR_MEAN_DIMS));
    EXPECT_EQ(desc->getMeanDesc()->getData().strides, toVec(K_BN_BWD_TENSOR_MEAN_STRIDES));

    // InvVariance tensor (optional)
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid, K_BN_BWD_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().dims, toVec(K_BN_BWD_TENSOR_INV_VARIANCE_DIMS));
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().strides,
              toVec(K_BN_BWD_TENSOR_INV_VARIANCE_STRIDES));
}

TEST_F(TestBatchnormBackwardOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getDyDesc(), _tensorMap[K_BN_BWD_TENSOR_DY_UID]);
    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_BN_BWD_TENSOR_X_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_BN_BWD_TENSOR_SCALE_UID]);
    EXPECT_EQ(desc->getDxDesc(), _tensorMap[K_BN_BWD_TENSOR_DX_UID]);
    EXPECT_EQ(desc->getDscaleDesc(), _tensorMap[K_BN_BWD_TENSOR_DSCALE_UID]);
    EXPECT_EQ(desc->getDbiasDesc(), _tensorMap[K_BN_BWD_TENSOR_DBIAS_UID]);
    EXPECT_EQ(desc->getMeanDesc(), _tensorMap[K_BN_BWD_TENSOR_MEAN_UID]);
    EXPECT_EQ(desc->getInvVarianceDesc(), _tensorMap[K_BN_BWD_TENSOR_INV_VARIANCE_UID]);
}

TEST_F(TestBatchnormBackwardOperationFromNode, FailsWithMissingDyTensor)
{
    _tensorMap.erase(K_BN_BWD_TENSOR_DY_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormBackwardOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_BN_BWD_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormBackwardOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(K_BN_BWD_TENSOR_SCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormBackwardOperationFromNode, FailsWithMissingDxTensor)
{
    _tensorMap.erase(K_BN_BWD_TENSOR_DX_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormBackwardOperationFromNode, FailsWithMissingDscaleTensor)
{
    _tensorMap.erase(K_BN_BWD_TENSOR_DSCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormBackwardOperationFromNode, FailsWithMissingDbiasTensor)
{
    _tensorMap.erase(K_BN_BWD_TENSOR_DBIAS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormBackwardOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardAttrs();
    attrs.mean_tensor_uid = flatbuffers::nullopt;
    attrs.inv_variance_tensor_uid = flatbuffers::nullopt;
    attrs.peer_stats_tensor_uid.clear();

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    EXPECT_NE(desc->getDyDesc(), nullptr);
    EXPECT_NE(desc->getXDesc(), nullptr);
    EXPECT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_NE(desc->getDxDesc(), nullptr);
    EXPECT_NE(desc->getDscaleDesc(), nullptr);
    EXPECT_NE(desc->getDbiasDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc(), nullptr);
}

TEST_F(TestBatchnormBackwardOperationFromNode, FailsWhenOptionalMeanUidSetButTensorMissing)
{
    _tensorMap.erase(K_BN_BWD_TENSOR_MEAN_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormBackwardOperationFromNode, FailsWhenOptionalInvVarianceUidSetButTensorMissing)
{
    _tensorMap.erase(K_BN_BWD_TENSOR_INV_VARIANCE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormBackwardOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    node.name = "bn_bwd_round_trip";
    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BatchnormBackwardAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->dy_tensor_uid, K_BN_BWD_TENSOR_DY_UID);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_BN_BWD_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_BN_BWD_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->dx_tensor_uid, K_BN_BWD_TENSOR_DX_UID);
    EXPECT_EQ(rebuiltAttrs->dscale_tensor_uid, K_BN_BWD_TENSOR_DSCALE_UID);
    EXPECT_EQ(rebuiltAttrs->dbias_tensor_uid, K_BN_BWD_TENSOR_DBIAS_UID);
    EXPECT_EQ(rebuiltAttrs->mean_tensor_uid, K_BN_BWD_TENSOR_MEAN_UID);
    EXPECT_EQ(rebuiltAttrs->inv_variance_tensor_uid, K_BN_BWD_TENSOR_INV_VARIANCE_UID);

    ASSERT_EQ(rebuiltAttrs->peer_stats_tensor_uid.size(), 2);
    EXPECT_EQ(rebuiltAttrs->peer_stats_tensor_uid[0], K_BN_BWD_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(rebuiltAttrs->peer_stats_tensor_uid[1], K_BN_BWD_TENSOR_PEER_STAT_1_UID);

    EXPECT_EQ(rebuiltNode->name, "bn_bwd_round_trip");
}

TEST_F(TestBatchnormBackwardOperationFromNode, BuildNodeRoundTripWithOnlyRequiredTensors)
{
    auto attrs = createStandardAttrs();
    attrs.mean_tensor_uid = flatbuffers::nullopt;
    attrs.inv_variance_tensor_uid = flatbuffers::nullopt;
    attrs.peer_stats_tensor_uid.clear();

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

    EXPECT_EQ(rebuiltAttrs->dy_tensor_uid, K_BN_BWD_TENSOR_DY_UID);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_BN_BWD_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_BN_BWD_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->dx_tensor_uid, K_BN_BWD_TENSOR_DX_UID);
    EXPECT_EQ(rebuiltAttrs->dscale_tensor_uid, K_BN_BWD_TENSOR_DSCALE_UID);
    EXPECT_EQ(rebuiltAttrs->dbias_tensor_uid, K_BN_BWD_TENSOR_DBIAS_UID);

    EXPECT_FALSE(rebuiltAttrs->mean_tensor_uid.has_value());
    EXPECT_FALSE(rebuiltAttrs->inv_variance_tensor_uid.has_value());
    EXPECT_TRUE(rebuiltAttrs->peer_stats_tensor_uid.empty());
}

TEST_F(TestBatchnormBackwardOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_bn_bwd_1";

    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_bn_bwd_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_bn_bwd_1");
}

TEST_F(TestBatchnormBackwardOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestBatchnormBackwardOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestBatchnormBackwardOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    node.name = "bn_bwd_getattr";
    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type via getAttribute
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &dtCount,
                       &computeType);
    ASSERT_EQ(dtCount, 1);
    EXPECT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BATCHNORM_BACKWARD_EXT);

    // Verify name
    int64_t nameCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &nameCount, nullptr);
    ASSERT_EQ(nameCount, static_cast<int64_t>(std::string("bn_bwd_getattr").size() + 1));
    std::vector<char> nameBuffer(static_cast<size_t>(nameCount));
    int64_t actualNameCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       nameCount,
                       &actualNameCount,
                       nameBuffer.data());
    EXPECT_STREQ(nameBuffer.data(), "bn_bwd_getattr");

    // DY tensor - full verification
    hipdnn_backend::ScopedDescriptor dyScoped;
    int64_t dyCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dyCount,
                       static_cast<void*>(dyScoped.getPtr()));
    ASSERT_EQ(dyCount, 1);
    ASSERT_NE(dyScoped.get(), nullptr);
    test_utilities::verifyTensorDescriptor(dyScoped.get(),
                                           K_BN_BWD_TENSOR_DY_UID,
                                           HIPDNN_DATA_FLOAT,
                                           toVec(K_BN_BWD_TENSOR_DY_DIMS),
                                           toVec(K_BN_BWD_TENSOR_DY_STRIDES));

    // X tensor - full verification
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    test_utilities::verifyTensorDescriptor(xScoped.get(),
                                           K_BN_BWD_TENSOR_X_UID,
                                           HIPDNN_DATA_FLOAT,
                                           toVec(K_BN_BWD_TENSOR_X_DIMS),
                                           toVec(K_BN_BWD_TENSOR_X_STRIDES));

    // Scale tensor - full verification
    hipdnn_backend::ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    test_utilities::verifyTensorDescriptor(scaleScoped.get(),
                                           K_BN_BWD_TENSOR_SCALE_UID,
                                           HIPDNN_DATA_FLOAT,
                                           toVec(K_BN_BWD_TENSOR_SCALE_DIMS),
                                           toVec(K_BN_BWD_TENSOR_SCALE_STRIDES));

    // DX tensor - full verification
    hipdnn_backend::ScopedDescriptor dxScoped;
    int64_t dxCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dxCount,
                       static_cast<void*>(dxScoped.getPtr()));
    ASSERT_EQ(dxCount, 1);
    ASSERT_NE(dxScoped.get(), nullptr);
    test_utilities::verifyTensorDescriptor(dxScoped.get(),
                                           K_BN_BWD_TENSOR_DX_UID,
                                           HIPDNN_DATA_FLOAT,
                                           toVec(K_BN_BWD_TENSOR_DX_DIMS),
                                           toVec(K_BN_BWD_TENSOR_DX_STRIDES));

    // DScale tensor - full verification
    hipdnn_backend::ScopedDescriptor dscaleScoped;
    int64_t dscaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dscaleCount,
                       static_cast<void*>(dscaleScoped.getPtr()));
    ASSERT_EQ(dscaleCount, 1);
    ASSERT_NE(dscaleScoped.get(), nullptr);
    test_utilities::verifyTensorDescriptor(dscaleScoped.get(),
                                           K_BN_BWD_TENSOR_DSCALE_UID,
                                           HIPDNN_DATA_FLOAT,
                                           toVec(K_BN_BWD_TENSOR_DSCALE_DIMS),
                                           toVec(K_BN_BWD_TENSOR_DSCALE_STRIDES));

    // DBias tensor - full verification
    hipdnn_backend::ScopedDescriptor dbiasScoped;
    int64_t dbiasCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dbiasCount,
                       static_cast<void*>(dbiasScoped.getPtr()));
    ASSERT_EQ(dbiasCount, 1);
    ASSERT_NE(dbiasScoped.get(), nullptr);
    test_utilities::verifyTensorDescriptor(dbiasScoped.get(),
                                           K_BN_BWD_TENSOR_DBIAS_UID,
                                           HIPDNN_DATA_FLOAT,
                                           toVec(K_BN_BWD_TENSOR_DBIAS_DIMS),
                                           toVec(K_BN_BWD_TENSOR_DBIAS_STRIDES));

    // Mean tensor - full verification (optional, set in standard node)
    hipdnn_backend::ScopedDescriptor meanScoped;
    int64_t meanCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &meanCount,
                       static_cast<void*>(meanScoped.getPtr()));
    ASSERT_EQ(meanCount, 1);
    ASSERT_NE(meanScoped.get(), nullptr);
    test_utilities::verifyTensorDescriptor(meanScoped.get(),
                                           K_BN_BWD_TENSOR_MEAN_UID,
                                           HIPDNN_DATA_FLOAT,
                                           toVec(K_BN_BWD_TENSOR_MEAN_DIMS),
                                           toVec(K_BN_BWD_TENSOR_MEAN_STRIDES));

    // InvVariance tensor - full verification (optional, set in standard node)
    hipdnn_backend::ScopedDescriptor invVarScoped;
    int64_t invVarCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &invVarCount,
                       static_cast<void*>(invVarScoped.getPtr()));
    ASSERT_EQ(invVarCount, 1);
    ASSERT_NE(invVarScoped.get(), nullptr);
    test_utilities::verifyTensorDescriptor(invVarScoped.get(),
                                           K_BN_BWD_TENSOR_INV_VARIANCE_UID,
                                           HIPDNN_DATA_FLOAT,
                                           toVec(K_BN_BWD_TENSOR_INV_VARIANCE_DIMS),
                                           toVec(K_BN_BWD_TENSOR_INV_VARIANCE_STRIDES));

    // --- Peer stats tensor array ---

    // Query count first
    int64_t peerStatsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       0,
                       &peerStatsCount,
                       nullptr);
    ASSERT_EQ(peerStatsCount, 2);

    // Retrieve both peer_stats descriptors
    hipdnn_backend::ScopedDescriptor peerStats0Scoped;
    hipdnn_backend::ScopedDescriptor peerStats1Scoped;
    std::array<hipdnnBackendDescriptor_t, 2> peerStatsArray = {};
    int64_t peerStatsRetrievedCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       2,
                       &peerStatsRetrievedCount,
                       static_cast<void*>(peerStatsArray.data()));
    ASSERT_EQ(peerStatsRetrievedCount, 2);
    // Transfer ownership to ScopedDescriptors
    peerStats0Scoped = hipdnn_backend::ScopedDescriptor(peerStatsArray[0]);
    peerStats1Scoped = hipdnn_backend::ScopedDescriptor(peerStatsArray[1]);
    ASSERT_NE(peerStats0Scoped.get(), nullptr);
    ASSERT_NE(peerStats1Scoped.get(), nullptr);

    test_utilities::verifyTensorDescriptor(
        peerStats0Scoped.get(),
        K_BN_BWD_TENSOR_PEER_STAT_0_UID,
        HIPDNN_DATA_FLOAT,
        {K_BN_BWD_TENSOR_PEER_STAT_DIMS.begin(), K_BN_BWD_TENSOR_PEER_STAT_DIMS.end()},
        {K_BN_BWD_TENSOR_PEER_STAT_STRIDES.begin(), K_BN_BWD_TENSOR_PEER_STAT_STRIDES.end()});
    test_utilities::verifyTensorDescriptor(
        peerStats1Scoped.get(),
        K_BN_BWD_TENSOR_PEER_STAT_1_UID,
        HIPDNN_DATA_FLOAT,
        {K_BN_BWD_TENSOR_PEER_STAT_DIMS.begin(), K_BN_BWD_TENSOR_PEER_STAT_DIMS.end()},
        {K_BN_BWD_TENSOR_PEER_STAT_STRIDES.begin(), K_BN_BWD_TENSOR_PEER_STAT_STRIDES.end()});
}

TEST_F(TestBatchnormBackwardOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    // 6 required + 2 optional (mean, inv_variance) + 2 peer_stats = 10
    ASSERT_EQ(tensors.size(), 10);
    EXPECT_EQ(tensors[0]->getData().uid, K_BN_BWD_TENSOR_DY_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_BN_BWD_TENSOR_X_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_BN_BWD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_BN_BWD_TENSOR_DX_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_BN_BWD_TENSOR_DSCALE_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_BN_BWD_TENSOR_DBIAS_UID);
    EXPECT_EQ(tensors[6]->getData().uid, K_BN_BWD_TENSOR_MEAN_UID);
    EXPECT_EQ(tensors[7]->getData().uid, K_BN_BWD_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(tensors[8]->getData().uid, K_BN_BWD_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(tensors[9]->getData().uid, K_BN_BWD_TENSOR_PEER_STAT_1_UID);
}

TEST_F(TestBatchnormBackwardOperationFromNode, FailsWithWrongAttributesType)
{
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(ConvolutionFwdAttributesT{});

    ASSERT_THROW_HIPDNN_STATUS(BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormBackwardOperationFromNode, FailsWithMissingPeerStatsTensor)
{
    _tensorMap.erase(K_BN_BWD_TENSOR_PEER_STAT_0_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormBackwardOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}
