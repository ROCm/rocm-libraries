// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/SdpaBwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SdpaBwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// SdpaBwdOperationDescriptor::fromNode() Tests
// =============================================================================

class TestSdpaBwdOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        auto makeTensor = [this](int64_t uid,
                                 const std::vector<int64_t>& dims,
                                 const std::vector<int64_t>& strides) {
            TensorAttributesT attrs;
            attrs.uid = uid;
            attrs.data_type = DataType::FLOAT;
            attrs.dims = dims;
            attrs.strides = strides;
            _tensorMap[uid] = TensorDescriptor::fromFlatBuffer(attrs);
        };

        auto makeScalarTensor = [this](int64_t uid) {
            TensorAttributesT attrs;
            attrs.uid = uid;
            attrs.data_type = DataType::FLOAT;
            attrs.dims = toVec(K_SDPA_BWD_TENSOR_SCALAR_DIMS);
            attrs.strides = toVec(K_SDPA_BWD_TENSOR_SCALAR_STRIDES);
            _tensorMap[uid] = TensorDescriptor::fromFlatBuffer(attrs);
        };

        // Required tensors
        makeTensor(K_SDPA_BWD_TENSOR_Q_UID,
                   toVec(K_SDPA_BWD_TENSOR_Q_DIMS),
                   toVec(K_SDPA_BWD_TENSOR_Q_STRIDES));
        makeTensor(K_SDPA_BWD_TENSOR_K_UID,
                   toVec(K_SDPA_BWD_TENSOR_K_DIMS),
                   toVec(K_SDPA_BWD_TENSOR_K_STRIDES));
        makeTensor(K_SDPA_BWD_TENSOR_V_UID,
                   toVec(K_SDPA_BWD_TENSOR_V_DIMS),
                   toVec(K_SDPA_BWD_TENSOR_V_STRIDES));
        makeTensor(K_SDPA_BWD_TENSOR_O_UID,
                   toVec(K_SDPA_BWD_TENSOR_O_DIMS),
                   toVec(K_SDPA_BWD_TENSOR_O_STRIDES));
        makeTensor(K_SDPA_BWD_TENSOR_DO_UID,
                   toVec(K_SDPA_BWD_TENSOR_DO_DIMS),
                   toVec(K_SDPA_BWD_TENSOR_DO_STRIDES));
        makeTensor(K_SDPA_BWD_TENSOR_STATS_UID,
                   toVec(K_SDPA_BWD_TENSOR_STATS_DIMS),
                   toVec(K_SDPA_BWD_TENSOR_STATS_STRIDES));
        makeTensor(K_SDPA_BWD_TENSOR_DQ_UID,
                   toVec(K_SDPA_BWD_TENSOR_DQ_DIMS),
                   toVec(K_SDPA_BWD_TENSOR_DQ_STRIDES));
        makeTensor(K_SDPA_BWD_TENSOR_DK_UID,
                   toVec(K_SDPA_BWD_TENSOR_DK_DIMS),
                   toVec(K_SDPA_BWD_TENSOR_DK_STRIDES));
        makeTensor(K_SDPA_BWD_TENSOR_DV_UID,
                   toVec(K_SDPA_BWD_TENSOR_DV_DIMS),
                   toVec(K_SDPA_BWD_TENSOR_DV_STRIDES));

        // Optional tensors
        makeScalarTensor(K_SDPA_BWD_TENSOR_SCALE_UID);
        makeScalarTensor(K_SDPA_BWD_TENSOR_ATTN_MASK_UID);
        makeScalarTensor(K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID);
        makeScalarTensor(K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID);
        makeScalarTensor(K_SDPA_BWD_TENSOR_SEED_UID);
        makeScalarTensor(K_SDPA_BWD_TENSOR_OFFSET_UID);
        makeScalarTensor(K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID);
        makeScalarTensor(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID);
        makeScalarTensor(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID);
        makeScalarTensor(K_SDPA_BWD_TENSOR_DBIAS_UID);
    }

    static SdpaBackwardAttributesT createRequiredOnlyAttrs()
    {
        SdpaBackwardAttributesT attrs;
        attrs.q_tensor_uid = K_SDPA_BWD_TENSOR_Q_UID;
        attrs.k_tensor_uid = K_SDPA_BWD_TENSOR_K_UID;
        attrs.v_tensor_uid = K_SDPA_BWD_TENSOR_V_UID;
        attrs.o_tensor_uid = K_SDPA_BWD_TENSOR_O_UID;
        attrs.do_tensor_uid = K_SDPA_BWD_TENSOR_DO_UID;
        attrs.stats_tensor_uid = K_SDPA_BWD_TENSOR_STATS_UID;
        attrs.dq_tensor_uid = K_SDPA_BWD_TENSOR_DQ_UID;
        attrs.dk_tensor_uid = K_SDPA_BWD_TENSOR_DK_UID;
        attrs.dv_tensor_uid = K_SDPA_BWD_TENSOR_DV_UID;
        return attrs;
    }

    static SdpaBackwardAttributesT createAllAttrs()
    {
        auto attrs = createRequiredOnlyAttrs();
        attrs.scale_tensor_uid = K_SDPA_BWD_TENSOR_SCALE_UID;
        attrs.attn_mask_tensor_uid = K_SDPA_BWD_TENSOR_ATTN_MASK_UID;
        attrs.seq_len_q_tensor_uid = K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID;
        attrs.seq_len_kv_tensor_uid = K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID;
        attrs.seed_tensor_uid = K_SDPA_BWD_TENSOR_SEED_UID;
        attrs.offset_tensor_uid = K_SDPA_BWD_TENSOR_OFFSET_UID;
        attrs.dropout_mask_tensor_uid = K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID;
        attrs.dropout_scale_tensor_uid = K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID;
        attrs.dropout_scale_inv_tensor_uid = K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID;
        attrs.dbias_tensor_uid = K_SDPA_BWD_TENSOR_DBIAS_UID;
        attrs.alibi_mask = true;
        attrs.causal_mask = true;
        attrs.padding_mask = true;
        attrs.causal_mask_bottom_right = true;
        attrs.dropout_probability = 0.5f;
        attrs.attn_scale_value = 0.125f;
        attrs.left_bound = 5;
        attrs.right_bound = 15;
        attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;
        return attrs;
    }

    static NodeT createRequiredOnlyNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createRequiredOnlyAttrs());
        return node;
    }

    static NodeT createAllNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createAllAttrs());
        return node;
    }
};

// =============================================================================
// Parameterized: missing required tensor
// =============================================================================

class TestSdpaBwdMissingRequiredTensor : public TestSdpaBwdOperationFromNode,
                                         public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestSdpaBwdMissingRequiredTensor, FailsWithMissingRequiredTensor)
{
    _tensorMap.erase(GetParam());
    auto node = createRequiredOnlyNode();
    ASSERT_THROW_HIPDNN_STATUS(SdpaBwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(AllRequiredTensors,
                         TestSdpaBwdMissingRequiredTensor,
                         ::testing::Values(K_SDPA_BWD_TENSOR_Q_UID,
                                           K_SDPA_BWD_TENSOR_K_UID,
                                           K_SDPA_BWD_TENSOR_V_UID,
                                           K_SDPA_BWD_TENSOR_O_UID,
                                           K_SDPA_BWD_TENSOR_DO_UID,
                                           K_SDPA_BWD_TENSOR_STATS_UID,
                                           K_SDPA_BWD_TENSOR_DQ_UID,
                                           K_SDPA_BWD_TENSOR_DK_UID,
                                           K_SDPA_BWD_TENSOR_DV_UID));

// =============================================================================
// Non-parameterized tests
// =============================================================================

TEST_F(TestSdpaBwdOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createRequiredOnlyNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().q_tensor_uid, K_SDPA_BWD_TENSOR_Q_UID);
}

TEST_F(TestSdpaBwdOperationFromNode, NodeFactoryDelegatesCorrectly)
{
    auto node = createRequiredOnlyNode();

    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);
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
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestSdpaBwdOperationFromNode, PreservesComputeDataType)
{
    auto node = createRequiredOnlyNode(DataType::HALF);
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), DataType::HALF);
}

TEST_F(TestSdpaBwdOperationFromNode, SetsTensorReferences)
{
    auto node = createRequiredOnlyNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getQDesc(), nullptr);
    EXPECT_EQ(desc->getQDesc()->getData().uid, K_SDPA_BWD_TENSOR_Q_UID);
    ASSERT_NE(desc->getKDesc(), nullptr);
    EXPECT_EQ(desc->getKDesc()->getData().uid, K_SDPA_BWD_TENSOR_K_UID);
    ASSERT_NE(desc->getVDesc(), nullptr);
    EXPECT_EQ(desc->getVDesc()->getData().uid, K_SDPA_BWD_TENSOR_V_UID);
    ASSERT_NE(desc->getODesc(), nullptr);
    EXPECT_EQ(desc->getODesc()->getData().uid, K_SDPA_BWD_TENSOR_O_UID);
    ASSERT_NE(desc->getDoDesc(), nullptr);
    EXPECT_EQ(desc->getDoDesc()->getData().uid, K_SDPA_BWD_TENSOR_DO_UID);
    ASSERT_NE(desc->getStatsDesc(), nullptr);
    EXPECT_EQ(desc->getStatsDesc()->getData().uid, K_SDPA_BWD_TENSOR_STATS_UID);
    ASSERT_NE(desc->getDqDesc(), nullptr);
    EXPECT_EQ(desc->getDqDesc()->getData().uid, K_SDPA_BWD_TENSOR_DQ_UID);
    ASSERT_NE(desc->getDkDesc(), nullptr);
    EXPECT_EQ(desc->getDkDesc()->getData().uid, K_SDPA_BWD_TENSOR_DK_UID);
    ASSERT_NE(desc->getDvDesc(), nullptr);
    EXPECT_EQ(desc->getDvDesc()->getData().uid, K_SDPA_BWD_TENSOR_DV_UID);
}

TEST_F(TestSdpaBwdOperationFromNode, OptionalTensorReferencesSetWhenAllFieldsPresent)
{
    auto node = createAllNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_SDPA_BWD_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getAttnMaskDesc(), nullptr);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().uid, K_SDPA_BWD_TENSOR_ATTN_MASK_UID);
    ASSERT_NE(desc->getSeqLenQDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().uid, K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID);
    ASSERT_NE(desc->getSeqLenKvDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().uid, K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID);
    ASSERT_NE(desc->getSeedDesc(), nullptr);
    EXPECT_EQ(desc->getSeedDesc()->getData().uid, K_SDPA_BWD_TENSOR_SEED_UID);
    ASSERT_NE(desc->getOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getOffsetDesc()->getData().uid, K_SDPA_BWD_TENSOR_OFFSET_UID);
    ASSERT_NE(desc->getDropoutMaskDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutMaskDesc()->getData().uid, K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID);
    ASSERT_NE(desc->getDropoutScaleDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleDesc()->getData().uid, K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID);
    ASSERT_NE(desc->getDropoutScaleInvDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleInvDesc()->getData().uid,
              K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID);
    ASSERT_NE(desc->getDBiasDesc(), nullptr);
    EXPECT_EQ(desc->getDBiasDesc()->getData().uid, K_SDPA_BWD_TENSOR_DBIAS_UID);
}

TEST_F(TestSdpaBwdOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createRequiredOnlyNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getQDesc(), _tensorMap[K_SDPA_BWD_TENSOR_Q_UID]);
    EXPECT_EQ(desc->getKDesc(), _tensorMap[K_SDPA_BWD_TENSOR_K_UID]);
    EXPECT_EQ(desc->getVDesc(), _tensorMap[K_SDPA_BWD_TENSOR_V_UID]);
    EXPECT_EQ(desc->getODesc(), _tensorMap[K_SDPA_BWD_TENSOR_O_UID]);
    EXPECT_EQ(desc->getDoDesc(), _tensorMap[K_SDPA_BWD_TENSOR_DO_UID]);
    EXPECT_EQ(desc->getStatsDesc(), _tensorMap[K_SDPA_BWD_TENSOR_STATS_UID]);
    EXPECT_EQ(desc->getDqDesc(), _tensorMap[K_SDPA_BWD_TENSOR_DQ_UID]);
    EXPECT_EQ(desc->getDkDesc(), _tensorMap[K_SDPA_BWD_TENSOR_DK_UID]);
    EXPECT_EQ(desc->getDvDesc(), _tensorMap[K_SDPA_BWD_TENSOR_DV_UID]);
}

TEST_F(TestSdpaBwdOperationFromNode, RequiredOnlyHasNullOptionalTensors)
{
    auto node = createRequiredOnlyNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    // Optional tensor descs are null
    EXPECT_EQ(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getAttnMaskDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenQDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenKvDesc(), nullptr);
    EXPECT_EQ(desc->getSeedDesc(), nullptr);
    EXPECT_EQ(desc->getOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutMaskDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleInvDesc(), nullptr);
    EXPECT_EQ(desc->getDBiasDesc(), nullptr);

    // Optional scalars unset
    EXPECT_FALSE(desc->getData().dropout_probability.has_value());
    EXPECT_FALSE(desc->getData().attn_scale_value.has_value());
    EXPECT_FALSE(desc->getData().left_bound.has_value());
    EXPECT_FALSE(desc->getData().right_bound.has_value());

    // Bool defaults false
    EXPECT_FALSE(desc->getData().alibi_mask);
    EXPECT_FALSE(desc->getData().causal_mask);
    EXPECT_FALSE(desc->getData().padding_mask);
    EXPECT_FALSE(desc->getData().causal_mask_bottom_right);
}

TEST_F(TestSdpaBwdOperationFromNode, AllFieldsScalarsAndBoolsPreserved)
{
    auto node = createAllNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_TRUE(desc->getData().alibi_mask);
    EXPECT_TRUE(desc->getData().causal_mask);
    EXPECT_TRUE(desc->getData().padding_mask);
    EXPECT_TRUE(desc->getData().causal_mask_bottom_right);

    ASSERT_TRUE(desc->getData().dropout_probability.has_value());
    EXPECT_FLOAT_EQ(desc->getData().dropout_probability.value(), 0.5f);
    ASSERT_TRUE(desc->getData().attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(desc->getData().attn_scale_value.value(), 0.125f);
    ASSERT_TRUE(desc->getData().left_bound.has_value());
    EXPECT_EQ(desc->getData().left_bound.value(), 5);
    ASSERT_TRUE(desc->getData().right_bound.has_value());
    EXPECT_EQ(desc->getData().right_bound.value(), 15);
    EXPECT_EQ(desc->getData().diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
}

TEST_F(TestSdpaBwdOperationFromNode, SucceedsWithOnlyRequiredTensorsInAttrs)
{
    // Explicitly nullopt all optional UIDs
    auto attrs = createRequiredOnlyAttrs();
    attrs.scale_tensor_uid = flatbuffers::nullopt;
    attrs.attn_mask_tensor_uid = flatbuffers::nullopt;
    attrs.seq_len_q_tensor_uid = flatbuffers::nullopt;
    attrs.seq_len_kv_tensor_uid = flatbuffers::nullopt;
    attrs.seed_tensor_uid = flatbuffers::nullopt;
    attrs.offset_tensor_uid = flatbuffers::nullopt;
    attrs.dropout_mask_tensor_uid = flatbuffers::nullopt;
    attrs.dropout_scale_tensor_uid = flatbuffers::nullopt;
    attrs.dropout_scale_inv_tensor_uid = flatbuffers::nullopt;
    attrs.dbias_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getAttnMaskDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenQDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenKvDesc(), nullptr);
    EXPECT_EQ(desc->getSeedDesc(), nullptr);
    EXPECT_EQ(desc->getOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutMaskDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleInvDesc(), nullptr);
    EXPECT_EQ(desc->getDBiasDesc(), nullptr);
}

TEST_F(TestSdpaBwdOperationFromNode, OptionalTensorUidSetButMissingInMap)
{
    // Set scale_tensor_uid in attrs but erase it from the map
    _tensorMap.erase(K_SDPA_BWD_TENSOR_SCALE_UID);

    auto allAttrs = createAllAttrs();
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(allAttrs);

    // findOptional returns nullptr gracefully when UID is set but missing from map
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getScaleDesc(), nullptr);
}

TEST_F(TestSdpaBwdOperationFromNode, GetTensorDescriptorsRequiredOnly)
{
    auto node = createRequiredOnlyNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 9u);
    EXPECT_EQ(tensors[0]->getData().uid, K_SDPA_BWD_TENSOR_Q_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_SDPA_BWD_TENSOR_K_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_SDPA_BWD_TENSOR_V_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_SDPA_BWD_TENSOR_O_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_SDPA_BWD_TENSOR_DO_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_SDPA_BWD_TENSOR_STATS_UID);
    EXPECT_EQ(tensors[6]->getData().uid, K_SDPA_BWD_TENSOR_DQ_UID);
    EXPECT_EQ(tensors[7]->getData().uid, K_SDPA_BWD_TENSOR_DK_UID);
    EXPECT_EQ(tensors[8]->getData().uid, K_SDPA_BWD_TENSOR_DV_UID);
}

TEST_F(TestSdpaBwdOperationFromNode, GetTensorDescriptorsAllOptionals)
{
    auto node = createAllNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    // 9 required + 10 optional = 19
    ASSERT_EQ(tensors.size(), 19u);
}

TEST_F(TestSdpaBwdOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createAllNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::SdpaBackwardAttributes);

    const auto* attrs = rebuiltNode->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->q_tensor_uid, K_SDPA_BWD_TENSOR_Q_UID);
    EXPECT_EQ(attrs->k_tensor_uid, K_SDPA_BWD_TENSOR_K_UID);
    EXPECT_EQ(attrs->v_tensor_uid, K_SDPA_BWD_TENSOR_V_UID);
    EXPECT_EQ(attrs->o_tensor_uid, K_SDPA_BWD_TENSOR_O_UID);
    EXPECT_EQ(attrs->do_tensor_uid, K_SDPA_BWD_TENSOR_DO_UID);
    EXPECT_EQ(attrs->stats_tensor_uid, K_SDPA_BWD_TENSOR_STATS_UID);
    EXPECT_EQ(attrs->dq_tensor_uid, K_SDPA_BWD_TENSOR_DQ_UID);
    EXPECT_EQ(attrs->dk_tensor_uid, K_SDPA_BWD_TENSOR_DK_UID);
    EXPECT_EQ(attrs->dv_tensor_uid, K_SDPA_BWD_TENSOR_DV_UID);

    ASSERT_TRUE(attrs->scale_tensor_uid.has_value());
    EXPECT_EQ(attrs->scale_tensor_uid.value(), K_SDPA_BWD_TENSOR_SCALE_UID);
    ASSERT_TRUE(attrs->attn_mask_tensor_uid.has_value());
    EXPECT_EQ(attrs->attn_mask_tensor_uid.value(), K_SDPA_BWD_TENSOR_ATTN_MASK_UID);

    EXPECT_TRUE(attrs->alibi_mask);
    EXPECT_TRUE(attrs->causal_mask);
    EXPECT_TRUE(attrs->padding_mask);
    EXPECT_TRUE(attrs->causal_mask_bottom_right);

    ASSERT_TRUE(attrs->dropout_probability.has_value());
    EXPECT_FLOAT_EQ(attrs->dropout_probability.value(), 0.5f);
    ASSERT_TRUE(attrs->attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(attrs->attn_scale_value.value(), 0.125f);
    ASSERT_TRUE(attrs->left_bound.has_value());
    EXPECT_EQ(attrs->left_bound.value(), 5);
    ASSERT_TRUE(attrs->right_bound.has_value());
    EXPECT_EQ(attrs->right_bound.value(), 15);
    EXPECT_EQ(attrs->diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
}

TEST_F(TestSdpaBwdOperationFromNode, BuildNodeOmitsUnsetOptionalScalars)
{
    auto node = createRequiredOnlyNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    const auto* attrs = rebuiltNode->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_FALSE(attrs->scale_tensor_uid.has_value());
    EXPECT_FALSE(attrs->dropout_probability.has_value());
    EXPECT_FALSE(attrs->attn_scale_value.has_value());
    EXPECT_FALSE(attrs->left_bound.has_value());
    EXPECT_FALSE(attrs->right_bound.has_value());
}

TEST_F(TestSdpaBwdOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createRequiredOnlyNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BWD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(dtCount, 1);
    EXPECT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify Q tensor
    hipdnn_backend::ScopedDescriptor qScoped;
    int64_t qCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BWD_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &qCount,
                       static_cast<void*>(qScoped.getPtr()));
    ASSERT_EQ(qCount, 1);
    ASSERT_NE(qScoped.get(), nullptr);
    int64_t qUid = 0;
    int64_t qUidCount = 0;
    qScoped.get()->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &qUidCount, &qUid);
    EXPECT_EQ(qUid, K_SDPA_BWD_TENSOR_Q_UID);

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_SDPA_BACKWARD_EXT);
}

TEST_F(TestSdpaBwdOperationFromNode, NamePreservedFromNode)
{
    auto node = createRequiredOnlyNode();
    node.name = "test_sdpa_bwd_1";

    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_sdpa_bwd_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_sdpa_bwd_1");
}

TEST_F(TestSdpaBwdOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createRequiredOnlyNode();
    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestSdpaBwdOperationFromNode, BuildNodePreservesName)
{
    auto node = createRequiredOnlyNode();
    node.name = "test_build_name";

    auto desc = SdpaBwdOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
