// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/LayernormOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/LayernormConstants.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;

// =============================================================================
// LayernormOperationDescriptor::fromNode() Tests
// =============================================================================

class TestLayernormOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        // X tensor
        TensorAttributesT xAttrs;
        xAttrs.uid = K_LAYERNORM_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = {K_LAYERNORM_TENSOR_X_DIMS.begin(), K_LAYERNORM_TENSOR_X_DIMS.end()};
        xAttrs.strides = {K_LAYERNORM_TENSOR_X_STRIDES.begin(), K_LAYERNORM_TENSOR_X_STRIDES.end()};
        _tensorMap[K_LAYERNORM_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);

        // Scale tensor
        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = K_LAYERNORM_TENSOR_SCALE_UID;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims
            = {K_LAYERNORM_TENSOR_SCALE_DIMS.begin(), K_LAYERNORM_TENSOR_SCALE_DIMS.end()};
        scaleAttrs.strides
            = {K_LAYERNORM_TENSOR_SCALE_STRIDES.begin(), K_LAYERNORM_TENSOR_SCALE_STRIDES.end()};
        _tensorMap[K_LAYERNORM_TENSOR_SCALE_UID] = TensorDescriptor::fromFlatBuffer(scaleAttrs);

        // Bias tensor
        TensorAttributesT biasAttrs;
        biasAttrs.uid = K_LAYERNORM_TENSOR_BIAS_UID;
        biasAttrs.data_type = DataType::FLOAT;
        biasAttrs.dims = {K_LAYERNORM_TENSOR_BIAS_DIMS.begin(), K_LAYERNORM_TENSOR_BIAS_DIMS.end()};
        biasAttrs.strides
            = {K_LAYERNORM_TENSOR_BIAS_STRIDES.begin(), K_LAYERNORM_TENSOR_BIAS_STRIDES.end()};
        _tensorMap[K_LAYERNORM_TENSOR_BIAS_UID] = TensorDescriptor::fromFlatBuffer(biasAttrs);

        // Epsilon tensor
        TensorAttributesT epsilonAttrs;
        epsilonAttrs.uid = K_LAYERNORM_TENSOR_EPSILON_UID;
        epsilonAttrs.data_type = DataType::FLOAT;
        epsilonAttrs.dims
            = {K_LAYERNORM_TENSOR_EPSILON_DIMS.begin(), K_LAYERNORM_TENSOR_EPSILON_DIMS.end()};
        epsilonAttrs.strides = {K_LAYERNORM_TENSOR_EPSILON_STRIDES.begin(),
                                K_LAYERNORM_TENSOR_EPSILON_STRIDES.end()};
        _tensorMap[K_LAYERNORM_TENSOR_EPSILON_UID] = TensorDescriptor::fromFlatBuffer(epsilonAttrs);

        // Y tensor
        TensorAttributesT yAttrs;
        yAttrs.uid = K_LAYERNORM_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = {K_LAYERNORM_TENSOR_Y_DIMS.begin(), K_LAYERNORM_TENSOR_Y_DIMS.end()};
        yAttrs.strides = {K_LAYERNORM_TENSOR_Y_STRIDES.begin(), K_LAYERNORM_TENSOR_Y_STRIDES.end()};
        _tensorMap[K_LAYERNORM_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);

        // Mean tensor (optional)
        TensorAttributesT meanAttrs;
        meanAttrs.uid = K_LAYERNORM_TENSOR_MEAN_UID;
        meanAttrs.data_type = DataType::FLOAT;
        meanAttrs.dims = {K_LAYERNORM_TENSOR_MEAN_DIMS.begin(), K_LAYERNORM_TENSOR_MEAN_DIMS.end()};
        meanAttrs.strides
            = {K_LAYERNORM_TENSOR_MEAN_STRIDES.begin(), K_LAYERNORM_TENSOR_MEAN_STRIDES.end()};
        _tensorMap[K_LAYERNORM_TENSOR_MEAN_UID] = TensorDescriptor::fromFlatBuffer(meanAttrs);

        // InvVariance tensor (optional)
        TensorAttributesT invVarianceAttrs;
        invVarianceAttrs.uid = K_LAYERNORM_TENSOR_INV_VARIANCE_UID;
        invVarianceAttrs.data_type = DataType::FLOAT;
        invVarianceAttrs.dims = {K_LAYERNORM_TENSOR_INV_VARIANCE_DIMS.begin(),
                                 K_LAYERNORM_TENSOR_INV_VARIANCE_DIMS.end()};
        invVarianceAttrs.strides = {K_LAYERNORM_TENSOR_INV_VARIANCE_STRIDES.begin(),
                                    K_LAYERNORM_TENSOR_INV_VARIANCE_STRIDES.end()};
        _tensorMap[K_LAYERNORM_TENSOR_INV_VARIANCE_UID]
            = TensorDescriptor::fromFlatBuffer(invVarianceAttrs);
    }

    static LayernormAttributesT createStandardLayernormAttrs()
    {
        LayernormAttributesT attrs;
        attrs.x_tensor_uid = K_LAYERNORM_TENSOR_X_UID;
        attrs.scale_tensor_uid = K_LAYERNORM_TENSOR_SCALE_UID;
        attrs.bias_tensor_uid = K_LAYERNORM_TENSOR_BIAS_UID;
        attrs.epsilon_tensor_uid = K_LAYERNORM_TENSOR_EPSILON_UID;
        attrs.y_tensor_uid = K_LAYERNORM_TENSOR_Y_UID;
        attrs.normalized_dim_count = 3;
        attrs.mean_tensor_uid = K_LAYERNORM_TENSOR_MEAN_UID;
        attrs.inv_variance_tensor_uid = K_LAYERNORM_TENSOR_INV_VARIANCE_UID;
        attrs.forward_phase = NormFwdPhase::TRAINING;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardLayernormAttrs());
        return node;
    }
};

TEST_F(TestLayernormOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
}

TEST_F(TestLayernormOperationFromNode, NodeFactoryDelegatesCorrectly)
{
    auto node = createStandardNode();

    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);
    ASSERT_NE(graphOp, nullptr);

    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::LayernormAttributes);
    auto desc = std::static_pointer_cast<LayernormOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    EXPECT_EQ(desc->getData().x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_LAYERNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getData().bias_tensor_uid, K_LAYERNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(desc->getData().epsilon_tensor_uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_LAYERNORM_TENSOR_Y_UID);
    EXPECT_EQ(desc->getData().normalized_dim_count, 3);
    ASSERT_TRUE(desc->getData().mean_tensor_uid.has_value());
    EXPECT_EQ(desc->getData().mean_tensor_uid.value(), K_LAYERNORM_TENSOR_MEAN_UID);
    ASSERT_TRUE(desc->getData().inv_variance_tensor_uid.has_value());
    EXPECT_EQ(desc->getData().inv_variance_tensor_uid.value(), K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(desc->getData().forward_phase, NormFwdPhase::TRAINING);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestLayernormOperationFromNode, PreservesComputeDataType)
{
    auto node = createStandardNode(DataType::HALF);
    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestLayernormOperationFromNode, PreservesForwardPhase)
{
    auto node = createStandardNode();
    auto attrs = createStandardLayernormAttrs();
    attrs.forward_phase = NormFwdPhase::INFERENCE;
    node.attributes.Set(attrs);
    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().forward_phase, NormFwdPhase::INFERENCE);
}

TEST_F(TestLayernormOperationFromNode, PreservesNormalizedDimCount)
{
    auto node = createStandardNode();
    auto attrs = createStandardLayernormAttrs();
    attrs.normalized_dim_count = 2;
    node.attributes.Set(attrs);
    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().normalized_dim_count, 2);
}

TEST_F(TestLayernormOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_LAYERNORM_TENSOR_X_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_LAYERNORM_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    EXPECT_EQ(desc->getBiasDesc()->getData().uid, K_LAYERNORM_TENSOR_BIAS_UID);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    EXPECT_EQ(desc->getEpsilonDesc()->getData().uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
    EXPECT_EQ(desc->getYDesc()->getData().uid, K_LAYERNORM_TENSOR_Y_UID);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getMeanDesc()->getData().uid, K_LAYERNORM_TENSOR_MEAN_UID);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc()->getData().uid, K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
}

TEST_F(TestLayernormOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_LAYERNORM_TENSOR_X_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_LAYERNORM_TENSOR_SCALE_UID]);
    EXPECT_EQ(desc->getBiasDesc(), _tensorMap[K_LAYERNORM_TENSOR_BIAS_UID]);
    EXPECT_EQ(desc->getEpsilonDesc(), _tensorMap[K_LAYERNORM_TENSOR_EPSILON_UID]);
    EXPECT_EQ(desc->getYDesc(), _tensorMap[K_LAYERNORM_TENSOR_Y_UID]);
    EXPECT_EQ(desc->getMeanDesc(), _tensorMap[K_LAYERNORM_TENSOR_MEAN_UID]);
    EXPECT_EQ(desc->getInvVarianceDesc(), _tensorMap[K_LAYERNORM_TENSOR_INV_VARIANCE_UID]);
}

TEST_F(TestLayernormOperationFromNode, FailsWithMissingXTensor)
{
    _tensorMap.erase(K_LAYERNORM_TENSOR_X_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormOperationFromNode, FailsWithMissingScaleTensor)
{
    _tensorMap.erase(K_LAYERNORM_TENSOR_SCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormOperationFromNode, FailsWithMissingBiasTensor)
{
    _tensorMap.erase(K_LAYERNORM_TENSOR_BIAS_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormOperationFromNode, FailsWithMissingEpsilonTensor)
{
    _tensorMap.erase(K_LAYERNORM_TENSOR_EPSILON_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormOperationFromNode, FailsWithMissingYTensor)
{
    _tensorMap.erase(K_LAYERNORM_TENSOR_Y_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormOperationFromNode, SucceedsWithoutOptionalTensors)
{
    auto attrs = createStandardLayernormAttrs();
    attrs.mean_tensor_uid = flatbuffers::nullopt;
    attrs.inv_variance_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getMeanDesc(), nullptr);
    EXPECT_EQ(desc->getInvVarianceDesc(), nullptr);
}

TEST_F(TestLayernormOperationFromNode, FailsWhenOptionalMeanUidSetButTensorMissing)
{
    _tensorMap.erase(K_LAYERNORM_TENSOR_MEAN_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormOperationFromNode, FailsWhenOptionalInvVarianceUidSetButTensorMissing)
{
    _tensorMap.erase(K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(LayernormOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestLayernormOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    // 5 required + 2 optional = 7
    ASSERT_EQ(tensors.size(), 7);
    EXPECT_EQ(tensors[0]->getData().uid, K_LAYERNORM_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_LAYERNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_LAYERNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_LAYERNORM_TENSOR_Y_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_LAYERNORM_TENSOR_MEAN_UID);
    EXPECT_EQ(tensors[6]->getData().uid, K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
}

TEST_F(TestLayernormOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::LayernormAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsLayernormAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_LAYERNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->bias_tensor_uid, K_LAYERNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(rebuiltAttrs->epsilon_tensor_uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(rebuiltAttrs->y_tensor_uid, K_LAYERNORM_TENSOR_Y_UID);
    EXPECT_EQ(rebuiltAttrs->normalized_dim_count, 3);
    ASSERT_TRUE(rebuiltAttrs->mean_tensor_uid.has_value());
    EXPECT_EQ(rebuiltAttrs->mean_tensor_uid.value(), K_LAYERNORM_TENSOR_MEAN_UID);
    ASSERT_TRUE(rebuiltAttrs->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(rebuiltAttrs->inv_variance_tensor_uid.value(), K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(rebuiltAttrs->forward_phase, NormFwdPhase::TRAINING);
}

TEST_F(TestLayernormOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify forward phase
    hipdnnNormFwdPhase_t fwdPhase = {};
    int64_t fwdCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                       HIPDNN_TYPE_NORM_FWD_PHASE,
                       1,
                       &fwdCount,
                       &fwdPhase);
    ASSERT_EQ(fwdPhase, HIPDNN_NORM_FWD_TRAINING);

    // Verify normalized dim count
    int64_t normalizedDimCount = 0;
    int64_t ndcCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT,
                       HIPDNN_TYPE_INT64,
                       1,
                       &ndcCount,
                       &normalizedDimCount);
    ASSERT_EQ(normalizedDimCount, 3);

    // Verify X tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT,
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
    EXPECT_EQ(xUid, K_LAYERNORM_TENSOR_X_UID);

    // Verify Y tensor
    hipdnn_backend::ScopedDescriptor yScoped;
    int64_t yCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT,
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
    EXPECT_EQ(yUid, K_LAYERNORM_TENSOR_Y_UID);

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_LAYERNORM_EXT);
}

TEST_F(TestLayernormOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_layernorm_1";

    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_layernorm_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_layernorm_1");
}

TEST_F(TestLayernormOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestLayernormOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = LayernormOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
