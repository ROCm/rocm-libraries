// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestMacros.hpp"
#include "descriptors/ConvolutionFwdOperationDescriptor.hpp"
#include "descriptors/LayernormOperationDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/SdpaBwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/constants/LayernormConstants.hpp>
#include <hipdnn_test_sdk/constants/SdpaBwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// NodeFactory::createOperationFromNode() Tests
// =============================================================================

class TestNodeFactory : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_FPROP_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_FPROP_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_FPROP_TENSOR_X_STRIDES);

        TensorAttributesT wAttrs;
        wAttrs.uid = K_FPROP_TENSOR_W_UID;
        wAttrs.data_type = DataType::FLOAT;
        wAttrs.dims = toVec(K_FPROP_TENSOR_W_DIMS);
        wAttrs.strides = toVec(K_FPROP_TENSOR_W_STRIDES);

        TensorAttributesT yAttrs;
        yAttrs.uid = K_FPROP_TENSOR_Y_UID;
        yAttrs.data_type = DataType::FLOAT;
        yAttrs.dims = toVec(K_FPROP_TENSOR_Y_DIMS);
        yAttrs.strides = toVec(K_FPROP_TENSOR_Y_STRIDES);

        _tensorMap[K_FPROP_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        _tensorMap[K_FPROP_TENSOR_W_UID] = TensorDescriptor::fromFlatBuffer(wAttrs);
        _tensorMap[K_FPROP_TENSOR_Y_UID] = TensorDescriptor::fromFlatBuffer(yAttrs);
    }

    static ConvolutionFwdAttributesT createStandardConvAttrs()
    {
        ConvolutionFwdAttributesT attrs;
        attrs.x_tensor_uid = K_FPROP_TENSOR_X_UID;
        attrs.w_tensor_uid = K_FPROP_TENSOR_W_UID;
        attrs.y_tensor_uid = K_FPROP_TENSOR_Y_UID;
        attrs.pre_padding = toVec(K_FPROP_CONV_PADDING);
        attrs.post_padding = toVec(K_FPROP_CONV_PADDING);
        attrs.stride = toVec(K_FPROP_CONV_STRIDE);
        attrs.dilation = toVec(K_FPROP_CONV_DILATION);
        attrs.conv_mode = ConvMode::CROSS_CORRELATION;
        return attrs;
    }

    static NodeT createStandardNode()
    {
        NodeT node;
        node.compute_data_type = DataType::FLOAT;
        node.attributes.Set(createStandardConvAttrs());
        return node;
    }
};

TEST_F(TestNodeFactory, CreateOperationFromNodeConvFwd)
{
    auto node = createStandardNode();
    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);

    ASSERT_NE(graphOp, nullptr);

    // Verify the factory dispatched to the correct operation type, then static_cast.
    // Cannot use dynamic_pointer_cast: backend tests compile with -fno-rtti.
    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::ConvolutionFwdAttributes);
    auto desc = std::static_pointer_cast<ConvolutionFwdOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getData().x_tensor_uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().w_tensor_uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(desc->getData().y_tensor_uid, K_FPROP_TENSOR_Y_UID);
}

TEST_F(TestNodeFactory, CreateOperationFromNodeLayernorm)
{
    // Build a minimal layernorm tensor map
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> lnTensorMap;
    auto makeTensor = [](int64_t uid) {
        TensorAttributesT attrs;
        attrs.uid = uid;
        attrs.data_type = DataType::FLOAT;
        attrs.dims = {2, 64, 32, 32};
        attrs.strides = {65536, 1024, 32, 1};
        return TensorDescriptor::fromFlatBuffer(attrs);
    };
    lnTensorMap[K_LAYERNORM_TENSOR_X_UID] = makeTensor(K_LAYERNORM_TENSOR_X_UID);
    lnTensorMap[K_LAYERNORM_TENSOR_SCALE_UID] = makeTensor(K_LAYERNORM_TENSOR_SCALE_UID);
    lnTensorMap[K_LAYERNORM_TENSOR_BIAS_UID] = makeTensor(K_LAYERNORM_TENSOR_BIAS_UID);
    lnTensorMap[K_LAYERNORM_TENSOR_EPSILON_UID] = makeTensor(K_LAYERNORM_TENSOR_EPSILON_UID);
    lnTensorMap[K_LAYERNORM_TENSOR_Y_UID] = makeTensor(K_LAYERNORM_TENSOR_Y_UID);

    LayernormAttributesT lnAttrs;
    lnAttrs.x_tensor_uid = K_LAYERNORM_TENSOR_X_UID;
    lnAttrs.scale_tensor_uid = K_LAYERNORM_TENSOR_SCALE_UID;
    lnAttrs.bias_tensor_uid = K_LAYERNORM_TENSOR_BIAS_UID;
    lnAttrs.epsilon_tensor_uid = K_LAYERNORM_TENSOR_EPSILON_UID;
    lnAttrs.y_tensor_uid = K_LAYERNORM_TENSOR_Y_UID;
    lnAttrs.normalized_dim_count = 3;
    lnAttrs.forward_phase = NormFwdPhase::TRAINING;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(lnAttrs);

    auto graphOp = NodeFactory::createOperationFromNode(node, lnTensorMap);
    ASSERT_NE(graphOp, nullptr);

    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::LayernormAttributes);

    auto desc = std::static_pointer_cast<LayernormOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getData().x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().normalized_dim_count, 3);
    EXPECT_EQ(desc->getData().forward_phase, NormFwdPhase::TRAINING);
}

TEST_F(TestNodeFactory, CreateOperationFromNodeSdpaBwd)
{
    // Set up tensor map with 9 required SDPA bwd tensors
    auto makeTensor =
        [](int64_t uid, const std::array<int64_t, 4>& dims, const std::array<int64_t, 4>& strides) {
            TensorAttributesT t;
            t.uid = uid;
            t.data_type = DataType::FLOAT;
            t.dims = toVec(dims);
            t.strides = toVec(strides);
            return TensorDescriptor::fromFlatBuffer(t);
        };

    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> sdpaTensorMap;
    sdpaTensorMap[K_SDPA_BWD_TENSOR_Q_UID] = makeTensor(
        K_SDPA_BWD_TENSOR_Q_UID, K_SDPA_BWD_TENSOR_Q_DIMS, K_SDPA_BWD_TENSOR_Q_STRIDES);
    sdpaTensorMap[K_SDPA_BWD_TENSOR_K_UID] = makeTensor(
        K_SDPA_BWD_TENSOR_K_UID, K_SDPA_BWD_TENSOR_K_DIMS, K_SDPA_BWD_TENSOR_K_STRIDES);
    sdpaTensorMap[K_SDPA_BWD_TENSOR_V_UID] = makeTensor(
        K_SDPA_BWD_TENSOR_V_UID, K_SDPA_BWD_TENSOR_V_DIMS, K_SDPA_BWD_TENSOR_V_STRIDES);
    sdpaTensorMap[K_SDPA_BWD_TENSOR_O_UID] = makeTensor(
        K_SDPA_BWD_TENSOR_O_UID, K_SDPA_BWD_TENSOR_O_DIMS, K_SDPA_BWD_TENSOR_O_STRIDES);
    sdpaTensorMap[K_SDPA_BWD_TENSOR_DO_UID] = makeTensor(
        K_SDPA_BWD_TENSOR_DO_UID, K_SDPA_BWD_TENSOR_DO_DIMS, K_SDPA_BWD_TENSOR_DO_STRIDES);
    sdpaTensorMap[K_SDPA_BWD_TENSOR_STATS_UID] = makeTensor(
        K_SDPA_BWD_TENSOR_STATS_UID, K_SDPA_BWD_TENSOR_STATS_DIMS, K_SDPA_BWD_TENSOR_STATS_STRIDES);
    sdpaTensorMap[K_SDPA_BWD_TENSOR_DQ_UID] = makeTensor(
        K_SDPA_BWD_TENSOR_DQ_UID, K_SDPA_BWD_TENSOR_DQ_DIMS, K_SDPA_BWD_TENSOR_DQ_STRIDES);
    sdpaTensorMap[K_SDPA_BWD_TENSOR_DK_UID] = makeTensor(
        K_SDPA_BWD_TENSOR_DK_UID, K_SDPA_BWD_TENSOR_DK_DIMS, K_SDPA_BWD_TENSOR_DK_STRIDES);
    sdpaTensorMap[K_SDPA_BWD_TENSOR_DV_UID] = makeTensor(
        K_SDPA_BWD_TENSOR_DV_UID, K_SDPA_BWD_TENSOR_DV_DIMS, K_SDPA_BWD_TENSOR_DV_STRIDES);

    // Create SDPA backward attributes
    SdpaBackwardAttributesT sdpaAttrs;
    sdpaAttrs.q_tensor_uid = K_SDPA_BWD_TENSOR_Q_UID;
    sdpaAttrs.k_tensor_uid = K_SDPA_BWD_TENSOR_K_UID;
    sdpaAttrs.v_tensor_uid = K_SDPA_BWD_TENSOR_V_UID;
    sdpaAttrs.o_tensor_uid = K_SDPA_BWD_TENSOR_O_UID;
    sdpaAttrs.do_tensor_uid = K_SDPA_BWD_TENSOR_DO_UID;
    sdpaAttrs.stats_tensor_uid = K_SDPA_BWD_TENSOR_STATS_UID;
    sdpaAttrs.dq_tensor_uid = K_SDPA_BWD_TENSOR_DQ_UID;
    sdpaAttrs.dk_tensor_uid = K_SDPA_BWD_TENSOR_DK_UID;
    sdpaAttrs.dv_tensor_uid = K_SDPA_BWD_TENSOR_DV_UID;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(sdpaAttrs);

    auto graphOp = NodeFactory::createOperationFromNode(node, sdpaTensorMap);
    ASSERT_NE(graphOp, nullptr);

    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::SdpaBackwardAttributes);

    auto desc = std::static_pointer_cast<SdpaBwdOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getData().q_tensor_uid, K_SDPA_BWD_TENSOR_Q_UID);
    EXPECT_EQ(desc->getData().k_tensor_uid, K_SDPA_BWD_TENSOR_K_UID);
    EXPECT_EQ(desc->getData().v_tensor_uid, K_SDPA_BWD_TENSOR_V_UID);
    EXPECT_EQ(desc->getData().o_tensor_uid, K_SDPA_BWD_TENSOR_O_UID);
    EXPECT_EQ(desc->getData().do_tensor_uid, K_SDPA_BWD_TENSOR_DO_UID);
    EXPECT_EQ(desc->getData().stats_tensor_uid, K_SDPA_BWD_TENSOR_STATS_UID);
    EXPECT_EQ(desc->getData().dq_tensor_uid, K_SDPA_BWD_TENSOR_DQ_UID);
    EXPECT_EQ(desc->getData().dk_tensor_uid, K_SDPA_BWD_TENSOR_DK_UID);
    EXPECT_EQ(desc->getData().dv_tensor_uid, K_SDPA_BWD_TENSOR_DV_UID);
}

TEST_F(TestNodeFactory, CreateOperationFromNodeUnsupportedType)
{
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    // attributes.type defaults to NodeAttributes::NONE (unsupported)

    ASSERT_THROW_HIPDNN_STATUS(NodeFactory::createOperationFromNode(node, _tensorMap),
                               HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// NodeFactory::buildTensorMap() Tests
// =============================================================================

TEST_F(TestNodeFactory, BuildTensorMapSuccess)
{
    std::vector<std::unique_ptr<TensorAttributesT>> tensors;

    auto t1 = std::make_unique<TensorAttributesT>();
    t1->uid = 10;
    t1->data_type = DataType::FLOAT;
    t1->dims = {1, 3, 32, 32};
    t1->strides = {3072, 1024, 32, 1};
    tensors.push_back(std::move(t1));

    auto t2 = std::make_unique<TensorAttributesT>();
    t2->uid = 20;
    t2->data_type = DataType::FLOAT;
    t2->dims = {1, 3, 32, 32};
    t2->strides = {3072, 1024, 32, 1};
    tensors.push_back(std::move(t2));

    auto t3 = std::make_unique<TensorAttributesT>();
    t3->uid = 30;
    t3->data_type = DataType::FLOAT;
    t3->dims = {1, 3, 32, 32};
    t3->strides = {3072, 1024, 32, 1};
    tensors.push_back(std::move(t3));

    auto result = NodeFactory::buildTensorMap(tensors);

    ASSERT_EQ(result.size(), 3u);
    ASSERT_NE(result.find(10), result.end());
    ASSERT_NE(result.find(20), result.end());
    ASSERT_NE(result.find(30), result.end());
    EXPECT_EQ(result[10]->getData().uid, 10);
    EXPECT_EQ(result[20]->getData().uid, 20);
    EXPECT_EQ(result[30]->getData().uid, 30);
}

TEST_F(TestNodeFactory, BuildTensorMapEmptyVector)
{
    const std::vector<std::unique_ptr<TensorAttributesT>> tensors;

    auto result = NodeFactory::buildTensorMap(tensors);

    EXPECT_TRUE(result.empty());
}

TEST_F(TestNodeFactory, BuildTensorMapNullTensor)
{
    std::vector<std::unique_ptr<TensorAttributesT>> tensors;

    auto t1 = std::make_unique<TensorAttributesT>();
    t1->uid = 10;
    t1->data_type = DataType::FLOAT;
    t1->dims = {1, 3, 32, 32};
    t1->strides = {3072, 1024, 32, 1};
    tensors.push_back(std::move(t1));

    tensors.push_back(nullptr);

    ASSERT_THROW_HIPDNN_STATUS(NodeFactory::buildTensorMap(tensors), HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestNodeFactory, BuildTensorMapDuplicateUID)
{
    std::vector<std::unique_ptr<TensorAttributesT>> tensors;

    auto t1 = std::make_unique<TensorAttributesT>();
    t1->uid = 10;
    t1->data_type = DataType::FLOAT;
    t1->dims = {1, 3, 32, 32};
    t1->strides = {3072, 1024, 32, 1};
    tensors.push_back(std::move(t1));

    auto t2 = std::make_unique<TensorAttributesT>();
    t2->uid = 10;
    t2->data_type = DataType::FLOAT;
    t2->dims = {1, 3, 32, 32};
    t2->strides = {3072, 1024, 32, 1};
    tensors.push_back(std::move(t2));

    ASSERT_THROW_HIPDNN_STATUS(NodeFactory::buildTensorMap(tensors), HIPDNN_STATUS_INTERNAL_ERROR);
}
