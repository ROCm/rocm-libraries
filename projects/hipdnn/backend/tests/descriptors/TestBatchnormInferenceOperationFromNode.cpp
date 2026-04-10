// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormInferenceOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_test_sdk/constants/BatchnormInferenceConstants.hpp>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;

// =============================================================================
// BatchnormInferenceOperationDescriptor::fromNode() Tests
// =============================================================================

class TestBatchnormInferenceOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_BN_INF_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = {K_BN_INF_SPATIAL_DIMS[0],
                       K_BN_INF_SPATIAL_DIMS[1],
                       K_BN_INF_SPATIAL_DIMS[2],
                       K_BN_INF_SPATIAL_DIMS[3]};
        xAttrs.strides = {K_BN_INF_SPATIAL_STRIDES[0],
                          K_BN_INF_SPATIAL_STRIDES[1],
                          K_BN_INF_SPATIAL_STRIDES[2],
                          K_BN_INF_SPATIAL_STRIDES[3]};

        _tensorMap[K_BN_INF_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT meanAttrs;
        meanAttrs.uid = K_BN_INF_TENSOR_MEAN_UID;
        meanAttrs.data_type = DataType::FLOAT;
        meanAttrs.dims = {K_BN_INF_CHANNEL_DIMS[0],
                          K_BN_INF_CHANNEL_DIMS[1],
                          K_BN_INF_CHANNEL_DIMS[2],
                          K_BN_INF_CHANNEL_DIMS[3]};
        meanAttrs.strides = {K_BN_INF_CHANNEL_STRIDES[0],
                             K_BN_INF_CHANNEL_STRIDES[1],
                             K_BN_INF_CHANNEL_STRIDES[2],
                             K_BN_INF_CHANNEL_STRIDES[3]};

        _tensorMap[K_BN_INF_TENSOR_MEAN_UID] = TensorDescriptor::fromFlatBuffer(meanAttrs);
        TensorAttributesT invVarianceAttrs;
        invVarianceAttrs.uid = K_BN_INF_TENSOR_INV_VARIANCE_UID;
        invVarianceAttrs.data_type = DataType::FLOAT;
        invVarianceAttrs.dims = {K_BN_INF_CHANNEL_DIMS[0],
                                 K_BN_INF_CHANNEL_DIMS[1],
                                 K_BN_INF_CHANNEL_DIMS[2],
                                 K_BN_INF_CHANNEL_DIMS[3]};
        invVarianceAttrs.strides = {K_BN_INF_CHANNEL_STRIDES[0],
                                    K_BN_INF_CHANNEL_STRIDES[1],
                                    K_BN_INF_CHANNEL_STRIDES[2],
                                    K_BN_INF_CHANNEL_STRIDES[3]};

        _tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID]
            = TensorDescriptor::fromFlatBuffer(invVarianceAttrs);
        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = K_BN_INF_TENSOR_SCALE_UID;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims = {K_BN_INF_CHANNEL_DIMS[0],
                           K_BN_INF_CHANNEL_DIMS[1],
                           K_BN_INF_CHANNEL_DIMS[2],
                           K_BN_INF_CHANNEL_DIMS[3]};
        scaleAttrs.strides = {K_BN_INF_CHANNEL_STRIDES[0],
                              K_BN_INF_CHANNEL_STRIDES[1],
                              K_BN_INF_CHANNEL_STRIDES[2],
                              K_BN_INF_CHANNEL_STRIDES[3]};

        _tensorMap[K_BN_INF_TENSOR_SCALE_UID] = TensorDescriptor::fromFlatBuffer(scaleAttrs);
        TensorAttributesT biasAttrs;
        biasAttrs.uid = K_BN_INF_TENSOR_BIAS_UID;
        biasAttrs.data_type = DataType::FLOAT;
        biasAttrs.dims = {K_BN_INF_CHANNEL_DIMS[0],
                          K_BN_INF_CHANNEL_DIMS[1],
                          K_BN_INF_CHANNEL_DIMS[2],
                          K_BN_INF_CHANNEL_DIMS[3]};
        biasAttrs.strides = {K_BN_INF_CHANNEL_STRIDES[0],
                             K_BN_INF_CHANNEL_STRIDES[1],
                             K_BN_INF_CHANNEL_STRIDES[2],
                             K_BN_INF_CHANNEL_STRIDES[3]};

        _tensorMap[K_BN_INF_TENSOR_BIAS_UID] = TensorDescriptor::fromFlatBuffer(biasAttrs);
        TensorAttributesT yAttrs;
        yAttrs.uid = K_BN_INF_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = {K_BN_INF_SPATIAL_DIMS[0],
                       K_BN_INF_SPATIAL_DIMS[1],
                       K_BN_INF_SPATIAL_DIMS[2],
                       K_BN_INF_SPATIAL_DIMS[3]};
        yAttrs.strides = {K_BN_INF_SPATIAL_STRIDES[0],
                          K_BN_INF_SPATIAL_STRIDES[1],
                          K_BN_INF_SPATIAL_STRIDES[2],
                          K_BN_INF_SPATIAL_STRIDES[3]};

        _tensorMap[K_BN_INF_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesT
        createStandardBatchnormInferenceAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesT attrs;
        attrs.x_tensor_uid = K_BN_INF_TENSOR_X_UID;
        attrs.mean_tensor_uid = K_BN_INF_TENSOR_MEAN_UID;
        attrs.inv_variance_tensor_uid = K_BN_INF_TENSOR_INV_VARIANCE_UID;
        attrs.scale_tensor_uid = K_BN_INF_TENSOR_SCALE_UID;
        attrs.bias_tensor_uid = K_BN_INF_TENSOR_BIAS_UID;
        attrs.y_tensor_uid = K_BN_INF_TENSOR_Y_UID;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardBatchnormInferenceAttrs());
        return node;
    }
};

TEST_F(TestBatchnormInferenceOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_BN_INF_TENSOR_X_UID);
}

TEST_F(TestBatchnormInferenceOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BatchnormInferenceAttributes);
    auto desc = std::static_pointer_cast<BatchnormInferenceOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_BN_INF_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().mean_tensor_uid, K_BN_INF_TENSOR_MEAN_UID);
    EXPECT_EQ(desc->getData().inv_variance_tensor_uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_BN_INF_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getData().bias_tensor_uid, K_BN_INF_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_BN_INF_TENSOR_Y_UID);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BN_INF_TENSOR_X_UID);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_BN_INF_TENSOR_MEAN_UID);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BN_INF_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_BN_INF_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BN_INF_TENSOR_Y_UID);
}

TEST_F(TestBatchnormInferenceOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestBatchnormInferenceOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BN_INF_TENSOR_X_UID);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_BN_INF_TENSOR_MEAN_UID);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BN_INF_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_BN_INF_TENSOR_BIAS_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BN_INF_TENSOR_Y_UID);
}

TEST_F(TestBatchnormInferenceOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_BN_INF_TENSOR_X_UID]);
    EXPECT_EQ(desc->getMeanDesc(), _tensorMap[K_BN_INF_TENSOR_MEAN_UID]);
    EXPECT_EQ(desc->getInvVarianceDesc(), _tensorMap[K_BN_INF_TENSOR_INV_VARIANCE_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_BN_INF_TENSOR_SCALE_UID]);
    EXPECT_EQ(desc->getBiasDesc(), _tensorMap[K_BN_INF_TENSOR_BIAS_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_BN_INF_TENSOR_Y_UID]);
}

TEST_F(TestBatchnormInferenceOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_BN_INF_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims,
              (std::vector<int64_t>{K_BN_INF_SPATIAL_DIMS[0],
                                    K_BN_INF_SPATIAL_DIMS[1],
                                    K_BN_INF_SPATIAL_DIMS[2],
                                    K_BN_INF_SPATIAL_DIMS[3]}));
    EXPECT_EQ(desc->getXDesc()->getData().strides,
              (std::vector<int64_t>{K_BN_INF_SPATIAL_STRIDES[0],
                                    K_BN_INF_SPATIAL_STRIDES[1],
                                    K_BN_INF_SPATIAL_STRIDES[2],
                                    K_BN_INF_SPATIAL_STRIDES[3]}));

    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_BN_INF_TENSOR_MEAN_UID);
    EXPECT_EQ(desc->getMeanDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getMeanDesc()->getData().dims,
              (std::vector<int64_t>{K_BN_INF_CHANNEL_DIMS[0],
                                    K_BN_INF_CHANNEL_DIMS[1],
                                    K_BN_INF_CHANNEL_DIMS[2],
                                    K_BN_INF_CHANNEL_DIMS[3]}));
    EXPECT_EQ(desc->getMeanDesc()->getData().strides,
              (std::vector<int64_t>{K_BN_INF_CHANNEL_STRIDES[0],
                                    K_BN_INF_CHANNEL_STRIDES[1],
                                    K_BN_INF_CHANNEL_STRIDES[2],
                                    K_BN_INF_CHANNEL_STRIDES[3]}));

    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().dims,
              (std::vector<int64_t>{K_BN_INF_CHANNEL_DIMS[0],
                                    K_BN_INF_CHANNEL_DIMS[1],
                                    K_BN_INF_CHANNEL_DIMS[2],
                                    K_BN_INF_CHANNEL_DIMS[3]}));
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().strides,
              (std::vector<int64_t>{K_BN_INF_CHANNEL_STRIDES[0],
                                    K_BN_INF_CHANNEL_STRIDES[1],
                                    K_BN_INF_CHANNEL_STRIDES[2],
                                    K_BN_INF_CHANNEL_STRIDES[3]}));

    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_BN_INF_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims,
              (std::vector<int64_t>{K_BN_INF_CHANNEL_DIMS[0],
                                    K_BN_INF_CHANNEL_DIMS[1],
                                    K_BN_INF_CHANNEL_DIMS[2],
                                    K_BN_INF_CHANNEL_DIMS[3]}));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides,
              (std::vector<int64_t>{K_BN_INF_CHANNEL_STRIDES[0],
                                    K_BN_INF_CHANNEL_STRIDES[1],
                                    K_BN_INF_CHANNEL_STRIDES[2],
                                    K_BN_INF_CHANNEL_STRIDES[3]}));

    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_BN_INF_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getBiasDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getBiasDesc()->getData().dims,
              (std::vector<int64_t>{K_BN_INF_CHANNEL_DIMS[0],
                                    K_BN_INF_CHANNEL_DIMS[1],
                                    K_BN_INF_CHANNEL_DIMS[2],
                                    K_BN_INF_CHANNEL_DIMS[3]}));
    EXPECT_EQ(desc->getBiasDesc()->getData().strides,
              (std::vector<int64_t>{K_BN_INF_CHANNEL_STRIDES[0],
                                    K_BN_INF_CHANNEL_STRIDES[1],
                                    K_BN_INF_CHANNEL_STRIDES[2],
                                    K_BN_INF_CHANNEL_STRIDES[3]}));

    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_BN_INF_TENSOR_Y_UID);
    EXPECT_EQ(desc->getYDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getYDesc()->getData().dims,
              (std::vector<int64_t>{K_BN_INF_SPATIAL_DIMS[0],
                                    K_BN_INF_SPATIAL_DIMS[1],
                                    K_BN_INF_SPATIAL_DIMS[2],
                                    K_BN_INF_SPATIAL_DIMS[3]}));
    EXPECT_EQ(desc->getYDesc()->getData().strides,
              (std::vector<int64_t>{K_BN_INF_SPATIAL_STRIDES[0],
                                    K_BN_INF_SPATIAL_STRIDES[1],
                                    K_BN_INF_SPATIAL_STRIDES[2],
                                    K_BN_INF_SPATIAL_STRIDES[3]}));
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_BN_INF_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingMeanTensor)
{
    _tensorMap.erase(K_BN_INF_TENSOR_MEAN_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingInvVarianceTensor)
{
    _tensorMap.erase(K_BN_INF_TENSOR_INV_VARIANCE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(K_BN_INF_TENSOR_SCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingBiasTensor)
{
    _tensorMap.erase(K_BN_INF_TENSOR_BIAS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_BN_INF_TENSOR_Y_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestBatchnormInferenceOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    EXPECT_EQ(tensors[0]->getData().uid, K_BN_INF_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_BN_INF_TENSOR_MEAN_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_BN_INF_TENSOR_SCALE_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_BN_INF_TENSOR_BIAS_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_BN_INF_TENSOR_Y_UID);
}

TEST_F(TestBatchnormInferenceOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::BatchnormInferenceAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsBatchnormInferenceAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_BN_INF_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->mean_tensor_uid, K_BN_INF_TENSOR_MEAN_UID);
    EXPECT_EQ(rebuiltAttrs->inv_variance_tensor_uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_BN_INF_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->bias_tensor_uid, K_BN_INF_TENSOR_BIAS_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_BN_INF_TENSOR_Y_UID);
}

TEST_F(TestBatchnormInferenceOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(xScoped.get(),
                                                           K_BN_INF_TENSOR_X_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           {K_BN_INF_SPATIAL_DIMS[0],
                                                            K_BN_INF_SPATIAL_DIMS[1],
                                                            K_BN_INF_SPATIAL_DIMS[2],
                                                            K_BN_INF_SPATIAL_DIMS[3]},
                                                           {K_BN_INF_SPATIAL_STRIDES[0],
                                                            K_BN_INF_SPATIAL_STRIDES[1],
                                                            K_BN_INF_SPATIAL_STRIDES[2],
                                                            K_BN_INF_SPATIAL_STRIDES[3]});

    // Verify mean tensor
    hipdnn_backend::ScopedDescriptor meanScoped;
    int64_t meanCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &meanCount,
                       static_cast<void*>(meanScoped.getPtr()));
    ASSERT_EQ(meanCount, 1);
    ASSERT_NE(meanScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(meanScoped.get(),
                                                           K_BN_INF_TENSOR_MEAN_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           {K_BN_INF_CHANNEL_DIMS[0],
                                                            K_BN_INF_CHANNEL_DIMS[1],
                                                            K_BN_INF_CHANNEL_DIMS[2],
                                                            K_BN_INF_CHANNEL_DIMS[3]},
                                                           {K_BN_INF_CHANNEL_STRIDES[0],
                                                            K_BN_INF_CHANNEL_STRIDES[1],
                                                            K_BN_INF_CHANNEL_STRIDES[2],
                                                            K_BN_INF_CHANNEL_STRIDES[3]});

    // Verify inv_variance tensor
    hipdnn_backend::ScopedDescriptor invVarianceScoped;
    int64_t invVarianceCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &invVarianceCount,
                       static_cast<void*>(invVarianceScoped.getPtr()));
    ASSERT_EQ(invVarianceCount, 1);
    ASSERT_NE(invVarianceScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(invVarianceScoped.get(),
                                                           K_BN_INF_TENSOR_INV_VARIANCE_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           {K_BN_INF_CHANNEL_DIMS[0],
                                                            K_BN_INF_CHANNEL_DIMS[1],
                                                            K_BN_INF_CHANNEL_DIMS[2],
                                                            K_BN_INF_CHANNEL_DIMS[3]},
                                                           {K_BN_INF_CHANNEL_STRIDES[0],
                                                            K_BN_INF_CHANNEL_STRIDES[1],
                                                            K_BN_INF_CHANNEL_STRIDES[2],
                                                            K_BN_INF_CHANNEL_STRIDES[3]});

    // Verify scale tensor
    hipdnn_backend::ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(scaleScoped.get(),
                                                           K_BN_INF_TENSOR_SCALE_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           {K_BN_INF_CHANNEL_DIMS[0],
                                                            K_BN_INF_CHANNEL_DIMS[1],
                                                            K_BN_INF_CHANNEL_DIMS[2],
                                                            K_BN_INF_CHANNEL_DIMS[3]},
                                                           {K_BN_INF_CHANNEL_STRIDES[0],
                                                            K_BN_INF_CHANNEL_STRIDES[1],
                                                            K_BN_INF_CHANNEL_STRIDES[2],
                                                            K_BN_INF_CHANNEL_STRIDES[3]});

    // Verify bias tensor
    hipdnn_backend::ScopedDescriptor biasScoped;
    int64_t biasCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &biasCount,
                       static_cast<void*>(biasScoped.getPtr()));
    ASSERT_EQ(biasCount, 1);
    ASSERT_NE(biasScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(biasScoped.get(),
                                                           K_BN_INF_TENSOR_BIAS_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           {K_BN_INF_CHANNEL_DIMS[0],
                                                            K_BN_INF_CHANNEL_DIMS[1],
                                                            K_BN_INF_CHANNEL_DIMS[2],
                                                            K_BN_INF_CHANNEL_DIMS[3]},
                                                           {K_BN_INF_CHANNEL_STRIDES[0],
                                                            K_BN_INF_CHANNEL_STRIDES[1],
                                                            K_BN_INF_CHANNEL_STRIDES[2],
                                                            K_BN_INF_CHANNEL_STRIDES[3]});

    // Verify y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &yCount,
                       static_cast<void*>(yScoped.getPtr()));
    ASSERT_EQ(yCount, 1);
    ASSERT_NE(yScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(yScoped.get(),
                                                           K_BN_INF_TENSOR_Y_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           {K_BN_INF_SPATIAL_DIMS[0],
                                                            K_BN_INF_SPATIAL_DIMS[1],
                                                            K_BN_INF_SPATIAL_DIMS[2],
                                                            K_BN_INF_SPATIAL_DIMS[3]},
                                                           {K_BN_INF_SPATIAL_STRIDES[0],
                                                            K_BN_INF_SPATIAL_STRIDES[1],
                                                            K_BN_INF_SPATIAL_STRIDES[2],
                                                            K_BN_INF_SPATIAL_STRIDES[3]});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_BATCHNORM_INFERENCE_EXT);
}

TEST_F(TestBatchnormInferenceOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_batchnorminference_1";

    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_batchnorminference_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_batchnorminference_1");
}

TEST_F(TestBatchnormInferenceOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestBatchnormInferenceOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestBatchnormInferenceOperationFromNode, ToStringIncludesName)
{
    auto node = createStandardNode();
    node.name = "my_batchnorminference_op";

    auto desc = BatchnormInferenceOperationDescriptor::fromNode(node, _tensorMap);
    auto str = desc->toString();

    EXPECT_NE(str.find("name=my_batchnorminference_op"), std::string::npos);
}
