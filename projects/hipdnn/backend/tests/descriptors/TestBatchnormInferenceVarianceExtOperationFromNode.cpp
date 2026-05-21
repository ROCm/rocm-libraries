// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormInferenceVarianceExtOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BnInfVarExtConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// BatchnormInferenceVarianceExtOperationDescriptor::fromNode() Tests
// =============================================================================

class TestBatchnormInferenceVarianceExtOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_BN_INF_VAR_EXT_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_BN_INF_VAR_EXT_X_DIMS);
        xAttrs.strides = toVec(K_BN_INF_VAR_EXT_X_STRIDES);

        _tensorMap[K_BN_INF_VAR_EXT_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT meanAttrs;
        meanAttrs.uid = K_BN_INF_VAR_EXT_MEAN_UID;
        meanAttrs.data_type = DataType::FLOAT;
        meanAttrs.dims = toVec(K_BN_INF_VAR_EXT_MEAN_DIMS);
        meanAttrs.strides = toVec(K_BN_INF_VAR_EXT_MEAN_STRIDES);

        _tensorMap[K_BN_INF_VAR_EXT_MEAN_UID] = TensorDescriptor::fromFlatBuffer(meanAttrs);
        TensorAttributesT varianceAttrs;
        varianceAttrs.uid = K_BN_INF_VAR_EXT_VARIANCE_UID;
        varianceAttrs.data_type = DataType::FLOAT;
        varianceAttrs.dims = toVec(K_BN_INF_VAR_EXT_VARIANCE_DIMS);
        varianceAttrs.strides = toVec(K_BN_INF_VAR_EXT_VARIANCE_STRIDES);

        _tensorMap[K_BN_INF_VAR_EXT_VARIANCE_UID] = TensorDescriptor::fromFlatBuffer(varianceAttrs);
        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = K_BN_INF_VAR_EXT_SCALE_UID;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims = toVec(K_BN_INF_VAR_EXT_SCALE_DIMS);
        scaleAttrs.strides = toVec(K_BN_INF_VAR_EXT_SCALE_STRIDES);

        _tensorMap[K_BN_INF_VAR_EXT_SCALE_UID] = TensorDescriptor::fromFlatBuffer(scaleAttrs);
        TensorAttributesT biasAttrs;
        biasAttrs.uid = K_BN_INF_VAR_EXT_BIAS_UID;
        biasAttrs.data_type = DataType::FLOAT;
        biasAttrs.dims = toVec(K_BN_INF_VAR_EXT_BIAS_DIMS);
        biasAttrs.strides = toVec(K_BN_INF_VAR_EXT_BIAS_STRIDES);

        _tensorMap[K_BN_INF_VAR_EXT_BIAS_UID] = TensorDescriptor::fromFlatBuffer(biasAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = K_BN_INF_VAR_EXT_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_BN_INF_VAR_EXT_Y_DIMS);
        yAttrs.strides = toVec(K_BN_INF_VAR_EXT_Y_STRIDES);

        _tensorMap[K_BN_INF_VAR_EXT_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
        TensorAttributesT epsilonAttrs;
        epsilonAttrs.uid = K_BN_INF_VAR_EXT_EPSILON_UID;
        epsilonAttrs.data_type = DataType::FLOAT;
        epsilonAttrs.dims = toVec(K_BN_INF_VAR_EXT_EPSILON_DIMS);
        epsilonAttrs.strides = toVec(K_BN_INF_VAR_EXT_EPSILON_STRIDES);

        _tensorMap[K_BN_INF_VAR_EXT_EPSILON_UID] = TensorDescriptor::fromFlatBuffer(epsilonAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExtT
        createStandardBatchnormInferenceVarianceExtAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExtT attrs;
        attrs.x_tensor_uid = K_BN_INF_VAR_EXT_X_UID;
        attrs.mean_tensor_uid = K_BN_INF_VAR_EXT_MEAN_UID;
        attrs.variance_tensor_uid = K_BN_INF_VAR_EXT_VARIANCE_UID;
        attrs.scale_tensor_uid = K_BN_INF_VAR_EXT_SCALE_UID;
        attrs.bias_tensor_uid = K_BN_INF_VAR_EXT_BIAS_UID;
        attrs.y_tensor_uid = K_BN_INF_VAR_EXT_Y_UID;
        attrs.epsilon_tensor_uid = K_BN_INF_VAR_EXT_EPSILON_UID;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardBatchnormInferenceVarianceExtAttrs());
        return node;
    }
};

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(),
              HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_VARIANCE_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_BN_INF_VAR_EXT_X_UID);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type,
              NodeAttributes::BatchnormInferenceAttributesVarianceExt);
    auto desc = std::static_pointer_cast<BatchnormInferenceVarianceExtOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_BN_INF_VAR_EXT_X_UID);
    EXPECT_EQ(desc->getData().mean_tensor_uid, K_BN_INF_VAR_EXT_MEAN_UID);
    EXPECT_EQ(desc->getData().variance_tensor_uid, K_BN_INF_VAR_EXT_VARIANCE_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_BN_INF_VAR_EXT_SCALE_UID);
    EXPECT_EQ(desc->getData().bias_tensor_uid, K_BN_INF_VAR_EXT_BIAS_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_BN_INF_VAR_EXT_Y_UID);
    EXPECT_EQ(desc->getData().epsilon_tensor_uid, K_BN_INF_VAR_EXT_EPSILON_UID);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BN_INF_VAR_EXT_X_UID);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_BN_INF_VAR_EXT_MEAN_UID);
    EXPECT_EQ(desc->getVarianceDesc()->getData().uid, K_BN_INF_VAR_EXT_VARIANCE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BN_INF_VAR_EXT_SCALE_UID);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_BN_INF_VAR_EXT_BIAS_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BN_INF_VAR_EXT_Y_UID);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_BN_INF_VAR_EXT_EPSILON_UID);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BN_INF_VAR_EXT_X_UID);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_BN_INF_VAR_EXT_MEAN_UID);
    ASSERT_NE(desc->getVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getVarianceDesc()->getData().uid, K_BN_INF_VAR_EXT_VARIANCE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BN_INF_VAR_EXT_SCALE_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_BN_INF_VAR_EXT_BIAS_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BN_INF_VAR_EXT_Y_UID);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_BN_INF_VAR_EXT_EPSILON_UID);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_BN_INF_VAR_EXT_X_UID]);
    EXPECT_EQ(desc->getMeanDesc(), _tensorMap[K_BN_INF_VAR_EXT_MEAN_UID]);
    EXPECT_EQ(desc->getVarianceDesc(), _tensorMap[K_BN_INF_VAR_EXT_VARIANCE_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_BN_INF_VAR_EXT_SCALE_UID]);
    EXPECT_EQ(desc->getBiasDesc(), _tensorMap[K_BN_INF_VAR_EXT_BIAS_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_BN_INF_VAR_EXT_Y_UID]);
    EXPECT_EQ(desc->getEpsilonDesc(), _tensorMap[K_BN_INF_VAR_EXT_EPSILON_UID]);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BN_INF_VAR_EXT_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, toVec(K_BN_INF_VAR_EXT_X_DIMS));
    EXPECT_EQ(desc->getXDesc()->getData().strides, toVec(K_BN_INF_VAR_EXT_X_STRIDES));

    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_BN_INF_VAR_EXT_MEAN_UID);
    EXPECT_EQ(desc->getMeanDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getMeanDesc()->getData().dims, toVec(K_BN_INF_VAR_EXT_MEAN_DIMS));
    EXPECT_EQ(desc->getMeanDesc()->getData().strides, toVec(K_BN_INF_VAR_EXT_MEAN_STRIDES));

    ASSERT_NE(desc->getVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getVarianceDesc()->getData().uid, K_BN_INF_VAR_EXT_VARIANCE_UID);
    EXPECT_EQ(desc->getVarianceDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getVarianceDesc()->getData().dims, toVec(K_BN_INF_VAR_EXT_VARIANCE_DIMS));
    EXPECT_EQ(desc->getVarianceDesc()->getData().strides, toVec(K_BN_INF_VAR_EXT_VARIANCE_STRIDES));

    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BN_INF_VAR_EXT_SCALE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims, toVec(K_BN_INF_VAR_EXT_SCALE_DIMS));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides, toVec(K_BN_INF_VAR_EXT_SCALE_STRIDES));

    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_BN_INF_VAR_EXT_BIAS_UID);
    EXPECT_EQ(desc->getBiasDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getBiasDesc()->getData().dims, toVec(K_BN_INF_VAR_EXT_BIAS_DIMS));
    EXPECT_EQ(desc->getBiasDesc()->getData().strides, toVec(K_BN_INF_VAR_EXT_BIAS_STRIDES));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BN_INF_VAR_EXT_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims, toVec(K_BN_INF_VAR_EXT_Y_DIMS));
    EXPECT_EQ(desc->getYDesc()->getData().strides, toVec(K_BN_INF_VAR_EXT_Y_STRIDES));

    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_BN_INF_VAR_EXT_EPSILON_UID);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().dims, toVec(K_BN_INF_VAR_EXT_EPSILON_DIMS));
    EXPECT_EQ(desc->getEpsilonDesc()->getData().strides, toVec(K_BN_INF_VAR_EXT_EPSILON_STRIDES));
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_BN_INF_VAR_EXT_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(
        BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap),
        HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, FailsWithMissingMeanTensor)
{
    _tensorMap.erase(K_BN_INF_VAR_EXT_MEAN_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(
        BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap),
        HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, FailsWithMissingVarianceTensor)
{
    _tensorMap.erase(K_BN_INF_VAR_EXT_VARIANCE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(
        BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap),
        HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(K_BN_INF_VAR_EXT_SCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(
        BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap),
        HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, FailsWithMissingBiasTensor)
{
    _tensorMap.erase(K_BN_INF_VAR_EXT_BIAS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(
        BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap),
        HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_BN_INF_VAR_EXT_Y_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(
        BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap),
        HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, FailsWithMissingEpsilonTensor)
{
    _tensorMap.erase(K_BN_INF_VAR_EXT_EPSILON_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(
        BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap),
        HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 7);
    EXPECT_EQ(tensors[0]->getData().uid, K_BN_INF_VAR_EXT_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_BN_INF_VAR_EXT_MEAN_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_BN_INF_VAR_EXT_VARIANCE_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_BN_INF_VAR_EXT_SCALE_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_BN_INF_VAR_EXT_BIAS_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_BN_INF_VAR_EXT_Y_UID);
    EXPECT_EQ(tensors[6]->getData().uid, K_BN_INF_VAR_EXT_EPSILON_UID);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type,
              NodeAttributes::BatchnormInferenceAttributesVarianceExt);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBatchnormInferenceAttributesVarianceExt();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_BN_INF_VAR_EXT_X_UID);
    EXPECT_EQ(rebuiltAttrs->mean_tensor_uid, K_BN_INF_VAR_EXT_MEAN_UID);
    EXPECT_EQ(rebuiltAttrs->variance_tensor_uid, K_BN_INF_VAR_EXT_VARIANCE_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_BN_INF_VAR_EXT_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->bias_tensor_uid, K_BN_INF_VAR_EXT_BIAS_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_BN_INF_VAR_EXT_Y_UID);
    EXPECT_EQ(rebuiltAttrs->epsilon_tensor_uid, K_BN_INF_VAR_EXT_EPSILON_UID);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(HIPDNN_ATTR_BATCHNORM_INF_VAR_COMP_TYPE_EXT,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &dtCount,
                       &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(xScoped.get(),
                           K_BN_INF_VAR_EXT_X_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_BN_INF_VAR_EXT_X_DIMS),
                           toVec(K_BN_INF_VAR_EXT_X_STRIDES));

    // Verify mean tensor
    hipdnn_backend::ScopedDescriptor meanScoped;
    int64_t meanCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &meanCount,
                       static_cast<void*>(meanScoped.getPtr()));
    ASSERT_EQ(meanCount, 1);
    ASSERT_NE(meanScoped.get(), nullptr);
    verifyTensorDescriptor(meanScoped.get(),
                           K_BN_INF_VAR_EXT_MEAN_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_BN_INF_VAR_EXT_MEAN_DIMS),
                           toVec(K_BN_INF_VAR_EXT_MEAN_STRIDES));

    // Verify variance tensor
    hipdnn_backend::ScopedDescriptor varianceScoped;
    int64_t varianceCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &varianceCount,
                       static_cast<void*>(varianceScoped.getPtr()));
    ASSERT_EQ(varianceCount, 1);
    ASSERT_NE(varianceScoped.get(), nullptr);
    verifyTensorDescriptor(varianceScoped.get(),
                           K_BN_INF_VAR_EXT_VARIANCE_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_BN_INF_VAR_EXT_VARIANCE_DIMS),
                           toVec(K_BN_INF_VAR_EXT_VARIANCE_STRIDES));

    // Verify scale tensor
    hipdnn_backend::ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    verifyTensorDescriptor(scaleScoped.get(),
                           K_BN_INF_VAR_EXT_SCALE_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_BN_INF_VAR_EXT_SCALE_DIMS),
                           toVec(K_BN_INF_VAR_EXT_SCALE_STRIDES));

    // Verify bias tensor
    hipdnn_backend::ScopedDescriptor biasScoped;
    int64_t biasCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &biasCount,
                       static_cast<void*>(biasScoped.getPtr()));
    ASSERT_EQ(biasCount, 1);
    ASSERT_NE(biasScoped.get(), nullptr);
    verifyTensorDescriptor(biasScoped.get(),
                           K_BN_INF_VAR_EXT_BIAS_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_BN_INF_VAR_EXT_BIAS_DIMS),
                           toVec(K_BN_INF_VAR_EXT_BIAS_STRIDES));

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    verifyTensorDescriptor(yScoped.get(),
                           K_BN_INF_VAR_EXT_Y_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_BN_INF_VAR_EXT_Y_DIMS),
                           toVec(K_BN_INF_VAR_EXT_Y_STRIDES));

    // Verify epsilon tensor
    hipdnn_backend::ScopedDescriptor epsilonScoped;
    int64_t epsilonCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EPSILON_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &epsilonCount,
                       static_cast<void*>(epsilonScoped.getPtr()));
    ASSERT_EQ(epsilonCount, 1);
    ASSERT_NE(epsilonScoped.get(), nullptr);
    verifyTensorDescriptor(epsilonScoped.get(),
                           K_BN_INF_VAR_EXT_EPSILON_UID,
                           HIPDNN_DATA_FLOAT,
                           toVec(K_BN_INF_VAR_EXT_EPSILON_DIMS),
                           toVec(K_BN_INF_VAR_EXT_EPSILON_STRIDES));

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_VARIANCE_EXT);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_batchnorminferencevarianceext_1";

    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count,
              static_cast<int64_t>(std::string("test_batchnorminferencevarianceext_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_batchnorminferencevarianceext_1");
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestBatchnormInferenceVarianceExtOperationFromNode, ToStringIncludesName)
{
    auto node = createStandardNode();
    node.name = "test_to_string";

    auto desc = BatchnormInferenceVarianceExtOperationDescriptor::fromNode(node, _tensorMap);
    auto str = desc->toString();
    EXPECT_NE(str.find("test_to_string"), std::string::npos);
}
