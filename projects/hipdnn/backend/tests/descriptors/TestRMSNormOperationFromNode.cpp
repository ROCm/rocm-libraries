// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/RMSNormOperationDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/RMSNormConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// RMSNormOperationDescriptor::fromNode() Tests
// =============================================================================

class TestRMSNormOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_RMSNORM_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_RMSNORM_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_RMSNORM_TENSOR_X_STRIDES);

        _tensorMap[K_RMSNORM_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = K_RMSNORM_TENSOR_SCALE_UID;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims = toVec(K_RMSNORM_TENSOR_SCALE_DIMS);
        scaleAttrs.strides = toVec(K_RMSNORM_TENSOR_SCALE_STRIDES);

        _tensorMap[K_RMSNORM_TENSOR_SCALE_UID] = TensorDescriptor::fromFlatBuffer(scaleAttrs);
        TensorAttributesT epsilonAttrs;
        epsilonAttrs.uid = K_RMSNORM_TENSOR_EPSILON_UID;
        epsilonAttrs.data_type = DataType::FLOAT;
        epsilonAttrs.dims = toVec(K_RMSNORM_TENSOR_EPSILON_DIMS);
        epsilonAttrs.strides = toVec(K_RMSNORM_TENSOR_EPSILON_STRIDES);

        _tensorMap[K_RMSNORM_TENSOR_EPSILON_UID] = TensorDescriptor::fromFlatBuffer(epsilonAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = K_RMSNORM_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_RMSNORM_TENSOR_Y_DIMS);
        yAttrs.strides = toVec(K_RMSNORM_TENSOR_Y_STRIDES);

        _tensorMap[K_RMSNORM_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
        TensorAttributesT biasAttrs;
        biasAttrs.uid = K_RMSNORM_TENSOR_BIAS_UID;
        biasAttrs.data_type = DataType::FLOAT;
        biasAttrs.dims = toVec(K_RMSNORM_TENSOR_BIAS_DIMS);
        biasAttrs.strides = toVec(K_RMSNORM_TENSOR_BIAS_STRIDES);

        _tensorMap[K_RMSNORM_TENSOR_BIAS_UID] = TensorDescriptor::fromFlatBuffer(biasAttrs);
        TensorAttributesT invRmsAttrs;
        invRmsAttrs.uid = K_RMSNORM_TENSOR_INV_RMS_UID;
        invRmsAttrs.data_type = DataType::FLOAT;
        invRmsAttrs.dims = toVec(K_RMSNORM_TENSOR_INV_RMS_DIMS);
        invRmsAttrs.strides = toVec(K_RMSNORM_TENSOR_INV_RMS_STRIDES);

        _tensorMap[K_RMSNORM_TENSOR_INV_RMS_UID] = TensorDescriptor::fromFlatBuffer(invRmsAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributesT createStandardRMSNormAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributesT attrs;
        attrs.x_tensor_uid = K_RMSNORM_TENSOR_X_UID;
        attrs.scale_tensor_uid = K_RMSNORM_TENSOR_SCALE_UID;
        attrs.epsilon_tensor_uid = K_RMSNORM_TENSOR_EPSILON_UID;
        attrs.y_tensor_uid = K_RMSNORM_TENSOR_Y_UID;
        attrs.bias_tensor_uid = K_RMSNORM_TENSOR_BIAS_UID;
        attrs.inv_rms_tensor_uid = K_RMSNORM_TENSOR_INV_RMS_UID;
        attrs.forward_phase = NormFwdPhase::TRAINING;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardRMSNormAttrs());
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
        int64_t uid = 0;
        int64_t uidCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uidCount, &uid);
        EXPECT_EQ(uid, expectedUid);

        hipdnnDataType_t dataType = {};
        int64_t dtCount = 0;
        tensorDesc->getAttribute(
            HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &dataType);
        EXPECT_EQ(dataType, expectedDataType);

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

TEST_F(TestRMSNormOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_RMSNORM_TENSOR_X_UID);
}

TEST_F(TestRMSNormOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::RMSNormAttributes);
    auto desc = std::static_pointer_cast<RMSNormOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getData().epsilon_tensor_uid, K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_RMSNORM_TENSOR_Y_UID);
    EXPECT_EQ(desc->getData().bias_tensor_uid, K_RMSNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getData().inv_rms_tensor_uid, K_RMSNORM_TENSOR_INV_RMS_UID);
    EXPECT_EQ(desc->getData().forward_phase, NormFwdPhase::TRAINING);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_RMSNORM_TENSOR_Y_UID);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_RMSNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getInvRmsDesc()->getData().uid, K_RMSNORM_TENSOR_INV_RMS_UID);
}

TEST_F(TestRMSNormOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestRMSNormOperationFromNode, PreservesNormFwdPhase)
{
    auto node = createStandardNode();
    auto attrs = createStandardRMSNormAttrs();
    attrs.forward_phase = NormFwdPhase::INFERENCE;
    node.attributes.Set(attrs);
    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().forward_phase, NormFwdPhase::INFERENCE);
}

TEST_F(TestRMSNormOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_RMSNORM_TENSOR_X_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_RMSNORM_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_RMSNORM_TENSOR_EPSILON_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_RMSNORM_TENSOR_Y_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_RMSNORM_TENSOR_BIAS_UID);
    ASSERT_NE(desc->getInvRmsDesc(), nullptr);
    EXPECT_EQ(desc->getInvRmsDesc()->getData().uid, K_RMSNORM_TENSOR_INV_RMS_UID);
}

TEST_F(TestRMSNormOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_RMSNORM_TENSOR_X_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_RMSNORM_TENSOR_SCALE_UID]);
    EXPECT_EQ(desc->getEpsilonDesc(), _tensorMap[K_RMSNORM_TENSOR_EPSILON_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_RMSNORM_TENSOR_Y_UID]);
    EXPECT_EQ(desc->getBiasDesc(), _tensorMap[K_RMSNORM_TENSOR_BIAS_UID]);
    EXPECT_EQ(desc->getInvRmsDesc(), _tensorMap[K_RMSNORM_TENSOR_INV_RMS_UID]);
}

TEST_F(TestRMSNormOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, toVec(K_RMSNORM_TENSOR_X_DIMS));
    EXPECT_EQ(desc->getXDesc()->getData().strides, toVec(K_RMSNORM_TENSOR_X_STRIDES));

    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims, toVec(K_RMSNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides, toVec(K_RMSNORM_TENSOR_SCALE_STRIDES));

    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().dims, toVec(K_RMSNORM_TENSOR_EPSILON_DIMS));
    EXPECT_EQ(desc->getEpsilonDesc()->getData().strides, toVec(K_RMSNORM_TENSOR_EPSILON_STRIDES));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_RMSNORM_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, toVec(K_RMSNORM_TENSOR_Y_DIMS));
    EXPECT_EQ(desc->getYDesc()->getData().strides, toVec(K_RMSNORM_TENSOR_Y_STRIDES));

    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_RMSNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getBiasDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getBiasDesc()->getData().dims, toVec(K_RMSNORM_TENSOR_BIAS_DIMS));
    EXPECT_EQ(desc->getBiasDesc()->getData().strides, toVec(K_RMSNORM_TENSOR_BIAS_STRIDES));

    ASSERT_NE(desc->getInvRmsDesc(), nullptr);
    EXPECT_EQ(desc->getInvRmsDesc()->getData().uid, K_RMSNORM_TENSOR_INV_RMS_UID);
    EXPECT_EQ(desc->getInvRmsDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getInvRmsDesc()->getData().dims, toVec(K_RMSNORM_TENSOR_INV_RMS_DIMS));
    EXPECT_EQ(desc->getInvRmsDesc()->getData().strides, toVec(K_RMSNORM_TENSOR_INV_RMS_STRIDES));
}

TEST_F(TestRMSNormOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_RMSNORM_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(RMSNormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestRMSNormOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(K_RMSNORM_TENSOR_SCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(RMSNormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestRMSNormOperationFromNode, FailsWithMissingEpsilonTensor)
{
    _tensorMap.erase(K_RMSNORM_TENSOR_EPSILON_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(RMSNormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestRMSNormOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_RMSNORM_TENSOR_Y_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(RMSNormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestRMSNormOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardRMSNormAttrs();
    attrs.bias_tensor_uid = flatbuffers::nullopt;
    attrs.inv_rms_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getXDesc(), nullptr);
    EXPECT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_NE(desc->getYDesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getInvRmsDesc(), nullptr);
}

TEST_F(TestRMSNormOperationFromNode, FailsWhenOptionalBiasUidSetButTensorMissing)
{
    _tensorMap.erase(K_RMSNORM_TENSOR_BIAS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(RMSNormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestRMSNormOperationFromNode, FailsWhenOptionalInvRmsUidSetButTensorMissing)
{
    _tensorMap.erase(K_RMSNORM_TENSOR_INV_RMS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(RMSNormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestRMSNormOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    EXPECT_EQ(tensors[0]->getData().uid, K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_RMSNORM_TENSOR_Y_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_RMSNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_RMSNORM_TENSOR_INV_RMS_UID);
}

TEST_F(TestRMSNormOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::RMSNormAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsRMSNormAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->epsilon_tensor_uid, K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_RMSNORM_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->bias_tensor_uid, K_RMSNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(rebuiltAttrs->inv_rms_tensor_uid, K_RMSNORM_TENSOR_INV_RMS_UID);
    EXPECT_EQ(rebuiltAttrs->forward_phase, NormFwdPhase::TRAINING);
}

TEST_F(TestRMSNormOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify forward_phase
    hipdnnNormFwdPhase_t forwardPhase = {};
    int64_t forwardPhaseCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT,
                       HIPDNN_TYPE_NORM_FWD_PHASE,
                       1,
                       &forwardPhaseCount,
                       &forwardPhase);
    ASSERT_EQ(forwardPhase, HIPDNN_NORM_FWD_TRAINING);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(xScoped.get(),
                           K_RMSNORM_TENSOR_X_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_RMSNORM_TENSOR_X_DIMS),
                           toVec(K_RMSNORM_TENSOR_X_STRIDES));

    // Verify scale tensor
    hipdnn_backend::ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    verifyTensorDescriptor(scaleScoped.get(),
                           K_RMSNORM_TENSOR_SCALE_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_RMSNORM_TENSOR_SCALE_DIMS),
                           toVec(K_RMSNORM_TENSOR_SCALE_STRIDES));

    // Verify epsilon tensor
    hipdnn_backend::ScopedDescriptor epsilonScoped;
    int64_t epsilonCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &epsilonCount,
                       static_cast<void*>(epsilonScoped.getPtr()));
    ASSERT_EQ(epsilonCount, 1);
    ASSERT_NE(epsilonScoped.get(), nullptr);
    verifyTensorDescriptor(epsilonScoped.get(),
                           K_RMSNORM_TENSOR_EPSILON_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_RMSNORM_TENSOR_EPSILON_DIMS),
                           toVec(K_RMSNORM_TENSOR_EPSILON_STRIDES));

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    verifyTensorDescriptor(yScoped.get(),
                           K_RMSNORM_TENSOR_Y_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_RMSNORM_TENSOR_Y_DIMS),
                           toVec(K_RMSNORM_TENSOR_Y_STRIDES));

    // Verify bias tensor (optional)
    hipdnn_backend::ScopedDescriptor biasScoped;
    int64_t biasCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &biasCount,
                       static_cast<void*>(biasScoped.getPtr()));
    ASSERT_EQ(biasCount, 1);
    ASSERT_NE(biasScoped.get(), nullptr);
    verifyTensorDescriptor(biasScoped.get(),
                           K_RMSNORM_TENSOR_BIAS_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_RMSNORM_TENSOR_BIAS_DIMS),
                           toVec(K_RMSNORM_TENSOR_BIAS_STRIDES));

    // Verify inv_rms tensor (optional)
    hipdnn_backend::ScopedDescriptor invRmsScoped;
    int64_t invRmsCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &invRmsCount,
                       static_cast<void*>(invRmsScoped.getPtr()));
    ASSERT_EQ(invRmsCount, 1);
    ASSERT_NE(invRmsScoped.get(), nullptr);
    verifyTensorDescriptor(invRmsScoped.get(),
                           K_RMSNORM_TENSOR_INV_RMS_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_RMSNORM_TENSOR_INV_RMS_DIMS),
                           toVec(K_RMSNORM_TENSOR_INV_RMS_STRIDES));

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_RMSNORM_EXT);
}

TEST_F(TestRMSNormOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_rmsnorm_1";

    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_rmsnorm_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_rmsnorm_1");
}

TEST_F(TestRMSNormOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestRMSNormOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = RMSNormOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
