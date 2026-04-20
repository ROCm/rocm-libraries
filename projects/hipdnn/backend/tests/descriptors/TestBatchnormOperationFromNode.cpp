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
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BatchnormConstants.hpp>

#include <array>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;

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
        xAttrs.uid = K_BATCHNORM_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = {K_BATCHNORM_TENSOR_X_DIMS.begin(), K_BATCHNORM_TENSOR_X_DIMS.end()};
        xAttrs.strides = {K_BATCHNORM_TENSOR_X_STRIDES.begin(), K_BATCHNORM_TENSOR_X_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);

        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = K_BATCHNORM_TENSOR_SCALE_UID;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims
            = {K_BATCHNORM_TENSOR_SCALE_DIMS.begin(), K_BATCHNORM_TENSOR_SCALE_DIMS.end()};
        scaleAttrs.strides
            = {K_BATCHNORM_TENSOR_SCALE_STRIDES.begin(), K_BATCHNORM_TENSOR_SCALE_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_SCALE_UID] = TensorDescriptor::fromFlatBuffer(scaleAttrs);

        TensorAttributesT biasAttrs;
        biasAttrs.uid = K_BATCHNORM_TENSOR_BIAS_UID;
        biasAttrs.data_type = DataType::FLOAT;
        biasAttrs.dims = {K_BATCHNORM_TENSOR_BIAS_DIMS.begin(), K_BATCHNORM_TENSOR_BIAS_DIMS.end()};
        biasAttrs.strides
            = {K_BATCHNORM_TENSOR_BIAS_STRIDES.begin(), K_BATCHNORM_TENSOR_BIAS_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_BIAS_UID] = TensorDescriptor::fromFlatBuffer(biasAttrs);

        TensorAttributesT epsilonAttrs;
        epsilonAttrs.uid = K_BATCHNORM_TENSOR_EPSILON_UID;
        epsilonAttrs.data_type = DataType::FLOAT;
        epsilonAttrs.dims
            = {K_BATCHNORM_TENSOR_EPSILON_DIMS.begin(), K_BATCHNORM_TENSOR_EPSILON_DIMS.end()};
        epsilonAttrs.strides = {K_BATCHNORM_TENSOR_EPSILON_STRIDES.begin(),
                                K_BATCHNORM_TENSOR_EPSILON_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_EPSILON_UID] = TensorDescriptor::fromFlatBuffer(epsilonAttrs);

        TensorAttributesT yAttrs;
        yAttrs.uid = K_BATCHNORM_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = {K_BATCHNORM_TENSOR_Y_DIMS.begin(), K_BATCHNORM_TENSOR_Y_DIMS.end()};
        yAttrs.strides = {K_BATCHNORM_TENSOR_Y_STRIDES.begin(), K_BATCHNORM_TENSOR_Y_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);

        TensorAttributesT prevRunningMeanAttrs;
        prevRunningMeanAttrs.uid = K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID;
        prevRunningMeanAttrs.data_type = DataType::FLOAT;
        prevRunningMeanAttrs.dims = {K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_DIMS.begin(),
                                     K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_DIMS.end()};
        prevRunningMeanAttrs.strides = {K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_STRIDES.begin(),
                                        K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID]
            = TensorDescriptor::fromFlatBuffer(prevRunningMeanAttrs);

        TensorAttributesT prevRunningVarianceAttrs;
        prevRunningVarianceAttrs.uid = K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID;
        prevRunningVarianceAttrs.data_type = DataType::FLOAT;
        prevRunningVarianceAttrs.dims = {K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_DIMS.begin(),
                                         K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_DIMS.end()};
        prevRunningVarianceAttrs.strides
            = {K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_STRIDES.begin(),
               K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID]
            = TensorDescriptor::fromFlatBuffer(prevRunningVarianceAttrs);

        TensorAttributesT momentumAttrs;
        momentumAttrs.uid = K_BATCHNORM_TENSOR_MOMENTUM_UID;
        momentumAttrs.data_type = DataType::FLOAT;
        momentumAttrs.dims
            = {K_BATCHNORM_TENSOR_MOMENTUM_DIMS.begin(), K_BATCHNORM_TENSOR_MOMENTUM_DIMS.end()};
        momentumAttrs.strides = {K_BATCHNORM_TENSOR_MOMENTUM_STRIDES.begin(),
                                 K_BATCHNORM_TENSOR_MOMENTUM_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_MOMENTUM_UID]
            = TensorDescriptor::fromFlatBuffer(momentumAttrs);

        TensorAttributesT meanAttrs;
        meanAttrs.uid = K_BATCHNORM_TENSOR_MEAN_UID;
        meanAttrs.data_type = DataType::FLOAT;
        meanAttrs.dims = {K_BATCHNORM_TENSOR_MEAN_DIMS.begin(), K_BATCHNORM_TENSOR_MEAN_DIMS.end()};
        meanAttrs.strides
            = {K_BATCHNORM_TENSOR_MEAN_STRIDES.begin(), K_BATCHNORM_TENSOR_MEAN_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_MEAN_UID] = TensorDescriptor::fromFlatBuffer(meanAttrs);

        TensorAttributesT invVarianceAttrs;
        invVarianceAttrs.uid = K_BATCHNORM_TENSOR_INV_VARIANCE_UID;
        invVarianceAttrs.data_type = DataType::FLOAT;
        invVarianceAttrs.dims = {K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS.begin(),
                                 K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS.end()};
        invVarianceAttrs.strides = {K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_INV_VARIANCE_UID]
            = TensorDescriptor::fromFlatBuffer(invVarianceAttrs);

        TensorAttributesT nextRunningMeanAttrs;
        nextRunningMeanAttrs.uid = K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID;
        nextRunningMeanAttrs.data_type = DataType::FLOAT;
        nextRunningMeanAttrs.dims = {K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_DIMS.begin(),
                                     K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_DIMS.end()};
        nextRunningMeanAttrs.strides = {K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_STRIDES.begin(),
                                        K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID]
            = TensorDescriptor::fromFlatBuffer(nextRunningMeanAttrs);

        TensorAttributesT nextRunningVarianceAttrs;
        nextRunningVarianceAttrs.uid = K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID;
        nextRunningVarianceAttrs.data_type = DataType::FLOAT;
        nextRunningVarianceAttrs.dims = {K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_DIMS.begin(),
                                         K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_DIMS.end()};
        nextRunningVarianceAttrs.strides
            = {K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_STRIDES.begin(),
               K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID]
            = TensorDescriptor::fromFlatBuffer(nextRunningVarianceAttrs);

        TensorAttributesT peerStatsAttrs0;
        peerStatsAttrs0.uid = K_BATCHNORM_TENSOR_PEER_STAT_0_UID;
        peerStatsAttrs0.data_type = DataType::FLOAT;
        peerStatsAttrs0.dims
            = {K_BATCHNORM_TENSOR_PEER_STAT_DIMS.begin(), K_BATCHNORM_TENSOR_PEER_STAT_DIMS.end()};
        peerStatsAttrs0.strides = {K_BATCHNORM_TENSOR_PEER_STAT_STRIDES.begin(),
                                   K_BATCHNORM_TENSOR_PEER_STAT_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_PEER_STAT_0_UID]
            = TensorDescriptor::fromFlatBuffer(peerStatsAttrs0);

        TensorAttributesT peerStatsAttrs1;
        peerStatsAttrs1.uid = K_BATCHNORM_TENSOR_PEER_STAT_1_UID;
        peerStatsAttrs1.data_type = DataType::FLOAT;
        peerStatsAttrs1.dims
            = {K_BATCHNORM_TENSOR_PEER_STAT_DIMS.begin(), K_BATCHNORM_TENSOR_PEER_STAT_DIMS.end()};
        peerStatsAttrs1.strides = {K_BATCHNORM_TENSOR_PEER_STAT_STRIDES.begin(),
                                   K_BATCHNORM_TENSOR_PEER_STAT_STRIDES.end()};
        _tensorMap[K_BATCHNORM_TENSOR_PEER_STAT_1_UID]
            = TensorDescriptor::fromFlatBuffer(peerStatsAttrs1);
    }

    static hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributesT createStandardBatchnormAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributesT attrs;
        attrs.x_tensor_uid = K_BATCHNORM_TENSOR_X_UID;
        attrs.scale_tensor_uid = K_BATCHNORM_TENSOR_SCALE_UID;
        attrs.bias_tensor_uid = K_BATCHNORM_TENSOR_BIAS_UID;
        attrs.epsilon_tensor_uid = K_BATCHNORM_TENSOR_EPSILON_UID;
        attrs.y_tensor_uid = K_BATCHNORM_TENSOR_Y_UID;
        attrs.prev_running_mean_tensor_uid = K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID;
        attrs.prev_running_variance_tensor_uid = K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID;
        attrs.momentum_tensor_uid = K_BATCHNORM_TENSOR_MOMENTUM_UID;
        attrs.mean_tensor_uid = K_BATCHNORM_TENSOR_MEAN_UID;
        attrs.inv_variance_tensor_uid = K_BATCHNORM_TENSOR_INV_VARIANCE_UID;
        attrs.next_running_mean_tensor_uid = K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID;
        attrs.next_running_variance_tensor_uid = K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID;
        attrs.peer_stats_tensor_uid
            = {K_BATCHNORM_TENSOR_PEER_STAT_0_UID, K_BATCHNORM_TENSOR_PEER_STAT_1_UID};
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardBatchnormAttrs());
        return node;
    }

    // Verifies that a packed tensor descriptor (retrieved via getAttribute) has the
    // expected UID, data_type, dimensions, and strides.
    static void verifyTensorDescriptor(hipdnnBackendDescriptor_t tensorDesc,
                                       int64_t expectedUid,
                                       hipdnnDataType_t expectedDataType,
                                       const std::vector<int64_t>& expectedDims,
                                       const std::vector<int64_t>& expectedStrides)
    {
        // Verify UID
        int64_t uid = 0;
        int64_t uidCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uidCount, &uid);
        EXPECT_EQ(uid, expectedUid);

        // Verify data type
        hipdnnDataType_t dataType = {};
        int64_t dtCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &dataType);
        EXPECT_EQ(dataType, expectedDataType);

        // Verify dimensions
        int64_t dimCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 0, &dimCount, nullptr);
        ASSERT_EQ(dimCount, static_cast<int64_t>(expectedDims.size()));
        std::vector<int64_t> dims(static_cast<size_t>(dimCount));
        int64_t actualDimCount = 0;
        tensorDesc->getAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS,
                                 HIPDNN_TYPE_INT64,
                                 dimCount,
                                 &actualDimCount,
                                 dims.data());
        EXPECT_EQ(dims, expectedDims);

        // Verify strides
        int64_t strideCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 0, &strideCount, nullptr);
        ASSERT_EQ(strideCount, static_cast<int64_t>(expectedStrides.size()));
        std::vector<int64_t> strides(static_cast<size_t>(strideCount));
        int64_t actualStrideCount = 0;
        tensorDesc->getAttribute(HIPDNN_ATTR_TENSOR_STRIDES,
                                 HIPDNN_TYPE_INT64,
                                 strideCount,
                                 &actualStrideCount,
                                 strides.data());
        EXPECT_EQ(strides, expectedStrides);
    }
};

TEST_F(TestBatchnormOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_BATCHNORM_TENSOR_X_UID);
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
    EXPECT_EQ(desc->getData().x_tensor_uid, K_BATCHNORM_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_BATCHNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getData().bias_tensor_uid, K_BATCHNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getData().epsilon_tensor_uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_BATCHNORM_TENSOR_Y_UID);
    EXPECT_EQ(desc->getData().prev_running_mean_tensor_uid,
              K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID);
    EXPECT_EQ(desc->getData().prev_running_variance_tensor_uid,
              K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID);
    EXPECT_EQ(desc->getData().momentum_tensor_uid, K_BATCHNORM_TENSOR_MOMENTUM_UID);
    EXPECT_EQ(desc->getData().mean_tensor_uid, K_BATCHNORM_TENSOR_MEAN_UID);
    EXPECT_EQ(desc->getData().inv_variance_tensor_uid, K_BATCHNORM_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(desc->getData().next_running_mean_tensor_uid,
              K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID);
    EXPECT_EQ(desc->getData().next_running_variance_tensor_uid,
              K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BATCHNORM_TENSOR_X_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BATCHNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_BATCHNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BATCHNORM_TENSOR_Y_UID);
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
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BATCHNORM_TENSOR_X_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BATCHNORM_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_BATCHNORM_TENSOR_BIAS_UID);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BATCHNORM_TENSOR_Y_UID);
}

TEST_F(TestBatchnormOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_BATCHNORM_TENSOR_X_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_BATCHNORM_TENSOR_SCALE_UID]);
    EXPECT_EQ(desc->getBiasDesc(), _tensorMap[K_BATCHNORM_TENSOR_BIAS_UID]);
    EXPECT_EQ(desc->getEpsilonDesc(), _tensorMap[K_BATCHNORM_TENSOR_EPSILON_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_BATCHNORM_TENSOR_Y_UID]);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_SCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMissingBiasTensor)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_BIAS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMissingEpsilonTensor)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_EPSILON_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_Y_UID);
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
    _tensorMap.erase(K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalPrevRunningVarianceUidSetButTensorMissing)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalMomentumUidSetButTensorMissing)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_MOMENTUM_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalMeanUidSetButTensorMissing)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_MEAN_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalInvVarianceUidSetButTensorMissing)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_INV_VARIANCE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalNextRunningMeanUidSetButTensorMissing)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenOptionalNextRunningVarianceUidSetButTensorMissing)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID);
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
    EXPECT_EQ(tensors[0]->getData().uid, K_BATCHNORM_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_BATCHNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_BATCHNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_BATCHNORM_TENSOR_Y_UID);
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
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_BATCHNORM_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_BATCHNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->bias_tensor_uid, K_BATCHNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(rebuiltAttrs->epsilon_tensor_uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_BATCHNORM_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->prev_running_mean_tensor_uid, K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID);
    EXPECT_EQ(rebuiltAttrs->prev_running_variance_tensor_uid,
              K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID);
    EXPECT_EQ(rebuiltAttrs->momentum_tensor_uid, K_BATCHNORM_TENSOR_MOMENTUM_UID);
    EXPECT_EQ(rebuiltAttrs->mean_tensor_uid, K_BATCHNORM_TENSOR_MEAN_UID);
    EXPECT_EQ(rebuiltAttrs->inv_variance_tensor_uid, K_BATCHNORM_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(rebuiltAttrs->next_running_mean_tensor_uid, K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID);
    EXPECT_EQ(rebuiltAttrs->next_running_variance_tensor_uid,
              K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID);

    // Verify peer_stats round-trip
    ASSERT_EQ(rebuiltAttrs->peer_stats_tensor_uid.size(), 2);
    EXPECT_EQ(rebuiltAttrs->peer_stats_tensor_uid[0], K_BATCHNORM_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(rebuiltAttrs->peer_stats_tensor_uid[1], K_BATCHNORM_TENSOR_PEER_STAT_1_UID);
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
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_BATCHNORM_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_BATCHNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->bias_tensor_uid, K_BATCHNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(rebuiltAttrs->epsilon_tensor_uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_BATCHNORM_TENSOR_Y_UID);

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
        HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(dtCount, 1);
    EXPECT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BATCHNORM_EXT);

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
    verifyTensorDescriptor(
        xScoped.get(),
        K_BATCHNORM_TENSOR_X_UID,
        HIPDNN_DATA_FLOAT,
        {K_BATCHNORM_TENSOR_X_DIMS.begin(), K_BATCHNORM_TENSOR_X_DIMS.end()},
        {K_BATCHNORM_TENSOR_X_STRIDES.begin(), K_BATCHNORM_TENSOR_X_STRIDES.end()});

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
    verifyTensorDescriptor(
        scaleScoped.get(),
        K_BATCHNORM_TENSOR_SCALE_UID,
        HIPDNN_DATA_FLOAT,
        {K_BATCHNORM_TENSOR_SCALE_DIMS.begin(), K_BATCHNORM_TENSOR_SCALE_DIMS.end()},
        {K_BATCHNORM_TENSOR_SCALE_STRIDES.begin(), K_BATCHNORM_TENSOR_SCALE_STRIDES.end()});

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
    verifyTensorDescriptor(
        biasScoped.get(),
        K_BATCHNORM_TENSOR_BIAS_UID,
        HIPDNN_DATA_FLOAT,
        {K_BATCHNORM_TENSOR_BIAS_DIMS.begin(), K_BATCHNORM_TENSOR_BIAS_DIMS.end()},
        {K_BATCHNORM_TENSOR_BIAS_STRIDES.begin(), K_BATCHNORM_TENSOR_BIAS_STRIDES.end()});

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
    verifyTensorDescriptor(
        epsilonScoped.get(),
        K_BATCHNORM_TENSOR_EPSILON_UID,
        HIPDNN_DATA_FLOAT,
        {K_BATCHNORM_TENSOR_EPSILON_DIMS.begin(), K_BATCHNORM_TENSOR_EPSILON_DIMS.end()},
        {K_BATCHNORM_TENSOR_EPSILON_STRIDES.begin(), K_BATCHNORM_TENSOR_EPSILON_STRIDES.end()});

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
    verifyTensorDescriptor(
        yScoped.get(),
        K_BATCHNORM_TENSOR_Y_UID,
        HIPDNN_DATA_FLOAT,
        {K_BATCHNORM_TENSOR_Y_DIMS.begin(), K_BATCHNORM_TENSOR_Y_DIMS.end()},
        {K_BATCHNORM_TENSOR_Y_STRIDES.begin(), K_BATCHNORM_TENSOR_Y_STRIDES.end()});

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
    verifyTensorDescriptor(prevRunMeanScoped.get(),
                           K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID,
                           HIPDNN_DATA_FLOAT,
                           {K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_DIMS.begin(),
                            K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_DIMS.end()},
                           {K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_STRIDES.begin(),
                            K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_STRIDES.end()});

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
    verifyTensorDescriptor(prevRunVarScoped.get(),
                           K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID,
                           HIPDNN_DATA_FLOAT,
                           {K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_DIMS.begin(),
                            K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_DIMS.end()},
                           {K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_STRIDES.begin(),
                            K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_STRIDES.end()});

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
    verifyTensorDescriptor(
        momentumScoped.get(),
        K_BATCHNORM_TENSOR_MOMENTUM_UID,
        HIPDNN_DATA_FLOAT,
        {K_BATCHNORM_TENSOR_MOMENTUM_DIMS.begin(), K_BATCHNORM_TENSOR_MOMENTUM_DIMS.end()},
        {K_BATCHNORM_TENSOR_MOMENTUM_STRIDES.begin(), K_BATCHNORM_TENSOR_MOMENTUM_STRIDES.end()});

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
    verifyTensorDescriptor(
        meanScoped.get(),
        K_BATCHNORM_TENSOR_MEAN_UID,
        HIPDNN_DATA_FLOAT,
        {K_BATCHNORM_TENSOR_MEAN_DIMS.begin(), K_BATCHNORM_TENSOR_MEAN_DIMS.end()},
        {K_BATCHNORM_TENSOR_MEAN_STRIDES.begin(), K_BATCHNORM_TENSOR_MEAN_STRIDES.end()});

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
    verifyTensorDescriptor(
        invVarScoped.get(),
        K_BATCHNORM_TENSOR_INV_VARIANCE_UID,
        HIPDNN_DATA_FLOAT,
        {K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS.begin(), K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS.end()},
        {K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES.begin(),
         K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES.end()});

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
    verifyTensorDescriptor(nextRunMeanScoped.get(),
                           K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID,
                           HIPDNN_DATA_FLOAT,
                           {K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_DIMS.begin(),
                            K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_DIMS.end()},
                           {K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_STRIDES.begin(),
                            K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_STRIDES.end()});

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
    verifyTensorDescriptor(nextRunVarScoped.get(),
                           K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID,
                           HIPDNN_DATA_FLOAT,
                           {K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_DIMS.begin(),
                            K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_DIMS.end()},
                           {K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_STRIDES.begin(),
                            K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_STRIDES.end()});

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
    std::array<hipdnnBackendDescriptor_t, 2> peerStatsArray = {};
    int64_t peerStatsRetrievedCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT,
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

    verifyTensorDescriptor(
        peerStats0Scoped.get(),
        K_BATCHNORM_TENSOR_PEER_STAT_0_UID,
        HIPDNN_DATA_FLOAT,
        {K_BATCHNORM_TENSOR_PEER_STAT_DIMS.begin(), K_BATCHNORM_TENSOR_PEER_STAT_DIMS.end()},
        {K_BATCHNORM_TENSOR_PEER_STAT_STRIDES.begin(), K_BATCHNORM_TENSOR_PEER_STAT_STRIDES.end()});
    verifyTensorDescriptor(
        peerStats1Scoped.get(),
        K_BATCHNORM_TENSOR_PEER_STAT_1_UID,
        HIPDNN_DATA_FLOAT,
        {K_BATCHNORM_TENSOR_PEER_STAT_DIMS.begin(), K_BATCHNORM_TENSOR_PEER_STAT_DIMS.end()},
        {K_BATCHNORM_TENSOR_PEER_STAT_STRIDES.begin(), K_BATCHNORM_TENSOR_PEER_STAT_STRIDES.end()});
}

TEST_F(TestBatchnormOperationFromNode, OperationTypeAttributeReturnsCorrectValue)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BATCHNORM_EXT);
}

TEST_F(TestBatchnormOperationFromNode, OptionalTensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getPrevRunningMeanDesc(), _tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID]);
    EXPECT_EQ(desc->getPrevRunningVarianceDesc(),
              _tensorMap[K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID]);
    EXPECT_EQ(desc->getMomentumDesc(), _tensorMap[K_BATCHNORM_TENSOR_MOMENTUM_UID]);
    EXPECT_EQ(desc->getMeanDesc(), _tensorMap[K_BATCHNORM_TENSOR_MEAN_UID]);
    EXPECT_EQ(desc->getInvVarianceDesc(), _tensorMap[K_BATCHNORM_TENSOR_INV_VARIANCE_UID]);
    EXPECT_EQ(desc->getNextRunningMeanDesc(), _tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID]);
    EXPECT_EQ(desc->getNextRunningVarianceDesc(),
              _tensorMap[K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID]);
}

TEST_F(TestBatchnormOperationFromNode, PeerStatsTensorReferencesPopulated)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    // 5 required + 2 (mean, inv_var) + 5 (running stats) + 2 peer_stats = 14 total
    ASSERT_EQ(tensors.size(), 14);

    // Last 2 are peer_stats
    EXPECT_EQ(tensors[12]->getData().uid, K_BATCHNORM_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(tensors[13]->getData().uid, K_BATCHNORM_TENSOR_PEER_STAT_1_UID);

    // Verify they are the same shared_ptr instances from tensorMap
    EXPECT_EQ(tensors[12], _tensorMap[K_BATCHNORM_TENSOR_PEER_STAT_0_UID]);
    EXPECT_EQ(tensors[13], _tensorMap[K_BATCHNORM_TENSOR_PEER_STAT_1_UID]);
}

TEST_F(TestBatchnormOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = BatchnormOperationDescriptor::fromNode(node, _tensorMap);

    // Required: X
    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BATCHNORM_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(
        desc->getXDesc()->getData().dims,
        (std::vector<int64_t>{K_BATCHNORM_TENSOR_X_DIMS.begin(), K_BATCHNORM_TENSOR_X_DIMS.end()}));
    EXPECT_EQ(desc->getXDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_X_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_X_STRIDES.end()}));

    // Required: Scale
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BATCHNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_SCALE_DIMS.begin(),
                                    K_BATCHNORM_TENSOR_SCALE_DIMS.end()}));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_SCALE_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_SCALE_STRIDES.end()}));

    // Required: Bias
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_BATCHNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getBiasDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getBiasDesc()->getData().dims,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_BIAS_DIMS.begin(),
                                    K_BATCHNORM_TENSOR_BIAS_DIMS.end()}));
    EXPECT_EQ(desc->getBiasDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_BIAS_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_BIAS_STRIDES.end()}));

    // Required: Epsilon
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().dims,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_EPSILON_DIMS.begin(),
                                    K_BATCHNORM_TENSOR_EPSILON_DIMS.end()}));
    EXPECT_EQ(desc->getEpsilonDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_EPSILON_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_EPSILON_STRIDES.end()}));

    // Required: Y
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BATCHNORM_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(
        desc->getYDesc()->getData().dims,
        (std::vector<int64_t>{K_BATCHNORM_TENSOR_Y_DIMS.begin(), K_BATCHNORM_TENSOR_Y_DIMS.end()}));
    EXPECT_EQ(desc->getYDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_Y_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_Y_STRIDES.end()}));

    // Optional: PrevRunningMean
    ASSERT_NE(desc->getPrevRunningMeanDesc(), nullptr);
    EXPECT_EQ(desc->getPrevRunningMeanDesc()->getData().uid,
              K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID);
    EXPECT_EQ(desc->getPrevRunningMeanDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getPrevRunningMeanDesc()->getData().dims,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_DIMS.begin(),
                                    K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_DIMS.end()}));
    EXPECT_EQ(desc->getPrevRunningMeanDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_STRIDES.end()}));

    // Optional: PrevRunningVariance
    ASSERT_NE(desc->getPrevRunningVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getPrevRunningVarianceDesc()->getData().uid,
              K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID);
    EXPECT_EQ(desc->getPrevRunningVarianceDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getPrevRunningVarianceDesc()->getData().dims,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_DIMS.begin(),
                                    K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_DIMS.end()}));
    EXPECT_EQ(desc->getPrevRunningVarianceDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_STRIDES.end()}));

    // Optional: Momentum
    ASSERT_NE(desc->getMomentumDesc(), nullptr);
    EXPECT_EQ(desc->getMomentumDesc()->getData().uid, K_BATCHNORM_TENSOR_MOMENTUM_UID);
    EXPECT_EQ(desc->getMomentumDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getMomentumDesc()->getData().dims,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_MOMENTUM_DIMS.begin(),
                                    K_BATCHNORM_TENSOR_MOMENTUM_DIMS.end()}));
    EXPECT_EQ(desc->getMomentumDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_MOMENTUM_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_MOMENTUM_STRIDES.end()}));

    // Optional: Mean
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_BATCHNORM_TENSOR_MEAN_UID);
    EXPECT_EQ(desc->getMeanDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getMeanDesc()->getData().dims,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_MEAN_DIMS.begin(),
                                    K_BATCHNORM_TENSOR_MEAN_DIMS.end()}));
    EXPECT_EQ(desc->getMeanDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_MEAN_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_MEAN_STRIDES.end()}));

    // Optional: InvVariance
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid, K_BATCHNORM_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().dims,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS.begin(),
                                    K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS.end()}));
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES.end()}));

    // Optional: NextRunningMean
    ASSERT_NE(desc->getNextRunningMeanDesc(), nullptr);
    EXPECT_EQ(desc->getNextRunningMeanDesc()->getData().uid,
              K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID);
    EXPECT_EQ(desc->getNextRunningMeanDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getNextRunningMeanDesc()->getData().dims,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_DIMS.begin(),
                                    K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_DIMS.end()}));
    EXPECT_EQ(desc->getNextRunningMeanDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_STRIDES.end()}));

    // Optional: NextRunningVariance
    ASSERT_NE(desc->getNextRunningVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getNextRunningVarianceDesc()->getData().uid,
              K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID);
    EXPECT_EQ(desc->getNextRunningVarianceDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getNextRunningVarianceDesc()->getData().dims,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_DIMS.begin(),
                                    K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_DIMS.end()}));
    EXPECT_EQ(desc->getNextRunningVarianceDesc()->getData().strides,
              (std::vector<int64_t>{K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_STRIDES.begin(),
                                    K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_STRIDES.end()}));
}

TEST_F(TestBatchnormOperationFromNode, FailsWhenPeerStatsUidSetButTensorMissing)
{
    _tensorMap.erase(K_BATCHNORM_TENSOR_PEER_STAT_0_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithMeanWithoutInvVariance)
{
    auto attrs = createStandardBatchnormAttrs();
    attrs.mean_tensor_uid = K_BATCHNORM_TENSOR_MEAN_UID;
    attrs.inv_variance_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithPartialRunningStats)
{
    auto attrs = createStandardBatchnormAttrs();
    attrs.prev_running_mean_tensor_uid = K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID;
    attrs.prev_running_variance_tensor_uid = flatbuffers::nullopt;
    attrs.momentum_tensor_uid = flatbuffers::nullopt;
    attrs.next_running_mean_tensor_uid = flatbuffers::nullopt;
    attrs.next_running_variance_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationFromNode, FailsWithWrongAttributeType)
{
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(PointwiseAttributesT{});

    ASSERT_THROW_HIPDNN_STATUS(BatchnormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}
