// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/SdpaFwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SdpaFwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// SdpaFwdOperationDescriptor::fromNode() Tests
// =============================================================================

class TestSdpaFwdOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        // Required tensors: Q, K, V, O
        auto addTensor = [this](int64_t uid,
                                const std::vector<int64_t>& dims,
                                const std::vector<int64_t>& strides) {
            TensorAttributesT attrs;
            attrs.uid = uid;
            attrs.data_type = DataType::FLOAT;
            attrs.dims = dims;
            attrs.strides = strides;
            _tensorMap[uid] = TensorDescriptor::fromFlatBuffer(attrs);
        };

        addTensor(K_SDPA_TENSOR_Q_UID, toVec(K_SDPA_TENSOR_Q_DIMS), toVec(K_SDPA_TENSOR_Q_STRIDES));
        addTensor(K_SDPA_TENSOR_K_UID, toVec(K_SDPA_TENSOR_K_DIMS), toVec(K_SDPA_TENSOR_K_STRIDES));
        addTensor(K_SDPA_TENSOR_V_UID, toVec(K_SDPA_TENSOR_V_DIMS), toVec(K_SDPA_TENSOR_V_STRIDES));
        addTensor(K_SDPA_TENSOR_O_UID, toVec(K_SDPA_TENSOR_O_DIMS), toVec(K_SDPA_TENSOR_O_STRIDES));

        // Optional tensors
        addTensor(K_SDPA_TENSOR_ATTN_MASK_UID,
                  toVec(K_SDPA_TENSOR_ATTN_MASK_DIMS),
                  toVec(K_SDPA_TENSOR_ATTN_MASK_STRIDES));
        addTensor(K_SDPA_TENSOR_SCALE_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_SEQ_LEN_Q_UID, {1}, {1});
        addTensor(K_SDPA_TENSOR_SEQ_LEN_KV_UID, {1}, {1});
        addTensor(K_SDPA_TENSOR_SEED_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_OFFSET_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_DROPOUT_MASK_UID,
                  toVec(K_SDPA_TENSOR_Q_DIMS),
                  toVec(K_SDPA_TENSOR_Q_STRIDES));
        addTensor(K_SDPA_TENSOR_DROPOUT_SCALE_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_PAGE_TABLE_K_UID, {1}, {1});
        addTensor(K_SDPA_TENSOR_PAGE_TABLE_V_UID, {1}, {1});
        addTensor(K_SDPA_TENSOR_BLOCK_MASK_UID, {1}, {1});
        addTensor(K_SDPA_TENSOR_SINK_TOKEN_UID, {1}, {1});
        addTensor(K_SDPA_TENSOR_DESCALE_Q_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_DESCALE_K_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_DESCALE_V_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_DESCALE_S_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_SCALE_S_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_SCALE_O_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_STATS_UID,
                  toVec(K_SDPA_TENSOR_STATS_DIMS),
                  toVec(K_SDPA_TENSOR_STATS_STRIDES));
        addTensor(K_SDPA_TENSOR_MAX_UID,
                  toVec(K_SDPA_TENSOR_STATS_DIMS),
                  toVec(K_SDPA_TENSOR_STATS_STRIDES));
        addTensor(K_SDPA_TENSOR_SUM_EXP_UID,
                  toVec(K_SDPA_TENSOR_STATS_DIMS),
                  toVec(K_SDPA_TENSOR_STATS_STRIDES));
        addTensor(K_SDPA_TENSOR_RNG_DUMP_UID,
                  toVec(K_SDPA_TENSOR_Q_DIMS),
                  toVec(K_SDPA_TENSOR_Q_STRIDES));
        addTensor(K_SDPA_TENSOR_AMAX_S_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
        addTensor(K_SDPA_TENSOR_AMAX_O_UID,
                  toVec(K_SDPA_TENSOR_SCALAR_DIMS),
                  toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
    }

    static hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT createStandardSdpaFwdAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT attrs;
        attrs.q_tensor_uid = K_SDPA_TENSOR_Q_UID;
        attrs.k_tensor_uid = K_SDPA_TENSOR_K_UID;
        attrs.v_tensor_uid = K_SDPA_TENSOR_V_UID;
        attrs.o_tensor_uid = K_SDPA_TENSOR_O_UID;
        attrs.attn_mask_tensor_uid = K_SDPA_TENSOR_ATTN_MASK_UID;
        attrs.scale_tensor_uid = K_SDPA_TENSOR_SCALE_UID;
        attrs.seq_len_q_tensor_uid = K_SDPA_TENSOR_SEQ_LEN_Q_UID;
        attrs.seq_len_kv_tensor_uid = K_SDPA_TENSOR_SEQ_LEN_KV_UID;
        attrs.seed_tensor_uid = K_SDPA_TENSOR_SEED_UID;
        attrs.offset_tensor_uid = K_SDPA_TENSOR_OFFSET_UID;
        attrs.dropout_mask_tensor_uid = K_SDPA_TENSOR_DROPOUT_MASK_UID;
        attrs.dropout_scale_tensor_uid = K_SDPA_TENSOR_DROPOUT_SCALE_UID;
        attrs.page_table_k_tensor_uid = K_SDPA_TENSOR_PAGE_TABLE_K_UID;
        attrs.page_table_v_tensor_uid = K_SDPA_TENSOR_PAGE_TABLE_V_UID;
        attrs.block_mask_tensor_uid = K_SDPA_TENSOR_BLOCK_MASK_UID;
        attrs.sink_token_tensor_uid = K_SDPA_TENSOR_SINK_TOKEN_UID;
        attrs.descale_q_tensor_uid = K_SDPA_TENSOR_DESCALE_Q_UID;
        attrs.descale_k_tensor_uid = K_SDPA_TENSOR_DESCALE_K_UID;
        attrs.descale_v_tensor_uid = K_SDPA_TENSOR_DESCALE_V_UID;
        attrs.descale_s_tensor_uid = K_SDPA_TENSOR_DESCALE_S_UID;
        attrs.scale_s_tensor_uid = K_SDPA_TENSOR_SCALE_S_UID;
        attrs.scale_o_tensor_uid = K_SDPA_TENSOR_SCALE_O_UID;
        attrs.stats_tensor_uid = K_SDPA_TENSOR_STATS_UID;
        attrs.max_tensor_uid = K_SDPA_TENSOR_MAX_UID;
        attrs.sum_exp_tensor_uid = K_SDPA_TENSOR_SUM_EXP_UID;
        attrs.rng_dump_tensor_uid = K_SDPA_TENSOR_RNG_DUMP_UID;
        attrs.amax_s_tensor_uid = K_SDPA_TENSOR_AMAX_S_UID;
        attrs.amax_o_tensor_uid = K_SDPA_TENSOR_AMAX_O_UID;
        attrs.diagonal_alignment = DiagonalAlignment::TOP_LEFT;
        attrs.implementation = AttentionImplementation::AUTO;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardSdpaFwdAttrs());
        return node;
    }
};

TEST_F(TestSdpaFwdOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR);
    EXPECT_EQ(desc->getData().q_tensor_uid, K_SDPA_TENSOR_Q_UID);
}

TEST_F(TestSdpaFwdOperationFromNode, NodeFactoryDelegatesCorrectly)
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
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::SdpaAttributes);
    auto desc = std::static_pointer_cast<SdpaFwdOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().q_tensor_uid, K_SDPA_TENSOR_Q_UID);
    EXPECT_EQ(desc->getData().k_tensor_uid, K_SDPA_TENSOR_K_UID);
    EXPECT_EQ(desc->getData().v_tensor_uid, K_SDPA_TENSOR_V_UID);
    EXPECT_EQ(desc->getData().o_tensor_uid, K_SDPA_TENSOR_O_UID);
    EXPECT_EQ(desc->getData().attn_mask_tensor_uid, K_SDPA_TENSOR_ATTN_MASK_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_SDPA_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getData().seq_len_q_tensor_uid, K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    EXPECT_EQ(desc->getData().seq_len_kv_tensor_uid, K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    EXPECT_EQ(desc->getData().seed_tensor_uid, K_SDPA_TENSOR_SEED_UID);
    EXPECT_EQ(desc->getData().offset_tensor_uid, K_SDPA_TENSOR_OFFSET_UID);
    EXPECT_EQ(desc->getData().dropout_mask_tensor_uid, K_SDPA_TENSOR_DROPOUT_MASK_UID);
    EXPECT_EQ(desc->getData().dropout_scale_tensor_uid, K_SDPA_TENSOR_DROPOUT_SCALE_UID);
    EXPECT_EQ(desc->getData().page_table_k_tensor_uid, K_SDPA_TENSOR_PAGE_TABLE_K_UID);
    EXPECT_EQ(desc->getData().page_table_v_tensor_uid, K_SDPA_TENSOR_PAGE_TABLE_V_UID);
    EXPECT_EQ(desc->getData().block_mask_tensor_uid, K_SDPA_TENSOR_BLOCK_MASK_UID);
    EXPECT_EQ(desc->getData().sink_token_tensor_uid, K_SDPA_TENSOR_SINK_TOKEN_UID);
    EXPECT_EQ(desc->getData().descale_q_tensor_uid, K_SDPA_TENSOR_DESCALE_Q_UID);
    EXPECT_EQ(desc->getData().descale_k_tensor_uid, K_SDPA_TENSOR_DESCALE_K_UID);
    EXPECT_EQ(desc->getData().descale_v_tensor_uid, K_SDPA_TENSOR_DESCALE_V_UID);
    EXPECT_EQ(desc->getData().descale_s_tensor_uid, K_SDPA_TENSOR_DESCALE_S_UID);
    EXPECT_EQ(desc->getData().scale_s_tensor_uid, K_SDPA_TENSOR_SCALE_S_UID);
    EXPECT_EQ(desc->getData().scale_o_tensor_uid, K_SDPA_TENSOR_SCALE_O_UID);
    EXPECT_EQ(desc->getData().stats_tensor_uid, K_SDPA_TENSOR_STATS_UID);
    EXPECT_EQ(desc->getData().max_tensor_uid, K_SDPA_TENSOR_MAX_UID);
    EXPECT_EQ(desc->getData().sum_exp_tensor_uid, K_SDPA_TENSOR_SUM_EXP_UID);
    EXPECT_EQ(desc->getData().rng_dump_tensor_uid, K_SDPA_TENSOR_RNG_DUMP_UID);
    EXPECT_EQ(desc->getData().amax_s_tensor_uid, K_SDPA_TENSOR_AMAX_S_UID);
    EXPECT_EQ(desc->getData().amax_o_tensor_uid, K_SDPA_TENSOR_AMAX_O_UID);
    EXPECT_EQ(desc->getData().diagonal_alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(desc->getData().implementation, AttentionImplementation::AUTO);
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);
    EXPECT_EQ(desc->getQDesc()->getData().uid, K_SDPA_TENSOR_Q_UID);
    EXPECT_EQ(desc->getKDesc()->getData().uid, K_SDPA_TENSOR_K_UID);
    EXPECT_EQ(desc->getVDesc()->getData().uid, K_SDPA_TENSOR_V_UID);
    EXPECT_EQ(desc->getODesc()->getData().uid, K_SDPA_TENSOR_O_UID);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().uid, K_SDPA_TENSOR_ATTN_MASK_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_SDPA_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().uid, K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().uid, K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    EXPECT_EQ(desc->getSeedDesc()->getData().uid, K_SDPA_TENSOR_SEED_UID);
    EXPECT_EQ(desc->getOffsetDesc()->getData().uid, K_SDPA_TENSOR_OFFSET_UID);
    EXPECT_EQ(desc->getDropoutMaskDesc()->getData().uid, K_SDPA_TENSOR_DROPOUT_MASK_UID);
    EXPECT_EQ(desc->getDropoutScaleDesc()->getData().uid, K_SDPA_TENSOR_DROPOUT_SCALE_UID);
    EXPECT_EQ(desc->getPageTableKDesc()->getData().uid, K_SDPA_TENSOR_PAGE_TABLE_K_UID);
    EXPECT_EQ(desc->getPageTableVDesc()->getData().uid, K_SDPA_TENSOR_PAGE_TABLE_V_UID);
    EXPECT_EQ(desc->getBlockMaskDesc()->getData().uid, K_SDPA_TENSOR_BLOCK_MASK_UID);
    EXPECT_EQ(desc->getSinkTokenDesc()->getData().uid, K_SDPA_TENSOR_SINK_TOKEN_UID);
    EXPECT_EQ(desc->getDescaleQDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_Q_UID);
    EXPECT_EQ(desc->getDescaleKDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_K_UID);
    EXPECT_EQ(desc->getDescaleVDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_V_UID);
    EXPECT_EQ(desc->getDescaleSDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_S_UID);
    EXPECT_EQ(desc->getScaleSDesc()->getData().uid, K_SDPA_TENSOR_SCALE_S_UID);
    EXPECT_EQ(desc->getScaleODesc()->getData().uid, K_SDPA_TENSOR_SCALE_O_UID);
    EXPECT_EQ(desc->getStatsDesc()->getData().uid, K_SDPA_TENSOR_STATS_UID);
    EXPECT_EQ(desc->getMaxDesc()->getData().uid, K_SDPA_TENSOR_MAX_UID);
    EXPECT_EQ(desc->getSumExpDesc()->getData().uid, K_SDPA_TENSOR_SUM_EXP_UID);
    EXPECT_EQ(desc->getRngDumpDesc()->getData().uid, K_SDPA_TENSOR_RNG_DUMP_UID);
    EXPECT_EQ(desc->getAmaxSDesc()->getData().uid, K_SDPA_TENSOR_AMAX_S_UID);
    EXPECT_EQ(desc->getAmaxODesc()->getData().uid, K_SDPA_TENSOR_AMAX_O_UID);
}

// -- 3c: Parameterized compute data type tests --

class TestSdpaFwdComputeDataType : public TestSdpaFwdOperationFromNode,
                                   public ::testing::WithParamInterface<DataType>
{
};

TEST_P(TestSdpaFwdComputeDataType, PreservesComputeDataType)
{
    auto node = createStandardNode(GetParam());
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getComputeDataType(), GetParam());
}

INSTANTIATE_TEST_SUITE_P(ComputeDataTypes,
                         TestSdpaFwdComputeDataType,
                         ::testing::Values(DataType::HALF,
                                           DataType::BFLOAT16,
                                           DataType::FLOAT,
                                           DataType::FP8_E4M3,
                                           DataType::FP8_E5M2));

// -- 3c: Parameterized diagonal alignment tests --

class TestSdpaFwdDiagonalAlignment : public TestSdpaFwdOperationFromNode,
                                     public ::testing::WithParamInterface<DiagonalAlignment>
{
};

TEST_P(TestSdpaFwdDiagonalAlignment, PreservesDiagonalAlignment)
{
    auto node = createStandardNode();
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.diagonal_alignment = GetParam();
    node.attributes.Set(attrs);
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().diagonal_alignment, GetParam());
}

INSTANTIATE_TEST_SUITE_P(DiagonalAlignments,
                         TestSdpaFwdDiagonalAlignment,
                         ::testing::Values(DiagonalAlignment::TOP_LEFT,
                                           DiagonalAlignment::BOTTOM_RIGHT));

// -- 3c: Parameterized attention implementation tests --

class TestSdpaFwdAttentionImplementation
    : public TestSdpaFwdOperationFromNode,
      public ::testing::WithParamInterface<AttentionImplementation>
{
};

TEST_P(TestSdpaFwdAttentionImplementation, PreservesAttentionImplementation)
{
    auto node = createStandardNode();
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.implementation = GetParam();
    node.attributes.Set(attrs);
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_EQ(desc->getData().implementation, GetParam());
}

INSTANTIATE_TEST_SUITE_P(AttentionImplementations,
                         TestSdpaFwdAttentionImplementation,
                         ::testing::Values(AttentionImplementation::AUTO,
                                           AttentionImplementation::COMPOSITE,
                                           AttentionImplementation::UNIFIED));

TEST_F(TestSdpaFwdOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getQDesc(), nullptr);
    EXPECT_EQ(desc->getQDesc()->getData().uid, K_SDPA_TENSOR_Q_UID);
    ASSERT_NE(desc->getKDesc(), nullptr);
    EXPECT_EQ(desc->getKDesc()->getData().uid, K_SDPA_TENSOR_K_UID);
    ASSERT_NE(desc->getVDesc(), nullptr);
    EXPECT_EQ(desc->getVDesc()->getData().uid, K_SDPA_TENSOR_V_UID);
    ASSERT_NE(desc->getODesc(), nullptr);
    EXPECT_EQ(desc->getODesc()->getData().uid, K_SDPA_TENSOR_O_UID);
    ASSERT_NE(desc->getAttnMaskDesc(), nullptr);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().uid, K_SDPA_TENSOR_ATTN_MASK_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_SDPA_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getSeqLenQDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().uid, K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    ASSERT_NE(desc->getSeqLenKvDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().uid, K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    ASSERT_NE(desc->getSeedDesc(), nullptr);
    EXPECT_EQ(desc->getSeedDesc()->getData().uid, K_SDPA_TENSOR_SEED_UID);
    ASSERT_NE(desc->getOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getOffsetDesc()->getData().uid, K_SDPA_TENSOR_OFFSET_UID);
    ASSERT_NE(desc->getDropoutMaskDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutMaskDesc()->getData().uid, K_SDPA_TENSOR_DROPOUT_MASK_UID);
    ASSERT_NE(desc->getDropoutScaleDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleDesc()->getData().uid, K_SDPA_TENSOR_DROPOUT_SCALE_UID);
    ASSERT_NE(desc->getPageTableKDesc(), nullptr);
    EXPECT_EQ(desc->getPageTableKDesc()->getData().uid, K_SDPA_TENSOR_PAGE_TABLE_K_UID);
    ASSERT_NE(desc->getPageTableVDesc(), nullptr);
    EXPECT_EQ(desc->getPageTableVDesc()->getData().uid, K_SDPA_TENSOR_PAGE_TABLE_V_UID);
    ASSERT_NE(desc->getBlockMaskDesc(), nullptr);
    EXPECT_EQ(desc->getBlockMaskDesc()->getData().uid, K_SDPA_TENSOR_BLOCK_MASK_UID);
    ASSERT_NE(desc->getSinkTokenDesc(), nullptr);
    EXPECT_EQ(desc->getSinkTokenDesc()->getData().uid, K_SDPA_TENSOR_SINK_TOKEN_UID);
    ASSERT_NE(desc->getDescaleQDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleQDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_Q_UID);
    ASSERT_NE(desc->getDescaleKDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleKDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_K_UID);
    ASSERT_NE(desc->getDescaleVDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleVDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_V_UID);
    ASSERT_NE(desc->getDescaleSDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleSDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_S_UID);
    ASSERT_NE(desc->getScaleSDesc(), nullptr);
    EXPECT_EQ(desc->getScaleSDesc()->getData().uid, K_SDPA_TENSOR_SCALE_S_UID);
    ASSERT_NE(desc->getScaleODesc(), nullptr);
    EXPECT_EQ(desc->getScaleODesc()->getData().uid, K_SDPA_TENSOR_SCALE_O_UID);
    ASSERT_NE(desc->getStatsDesc(), nullptr);
    EXPECT_EQ(desc->getStatsDesc()->getData().uid, K_SDPA_TENSOR_STATS_UID);
    ASSERT_NE(desc->getMaxDesc(), nullptr);
    EXPECT_EQ(desc->getMaxDesc()->getData().uid, K_SDPA_TENSOR_MAX_UID);
    ASSERT_NE(desc->getSumExpDesc(), nullptr);
    EXPECT_EQ(desc->getSumExpDesc()->getData().uid, K_SDPA_TENSOR_SUM_EXP_UID);
    ASSERT_NE(desc->getRngDumpDesc(), nullptr);
    EXPECT_EQ(desc->getRngDumpDesc()->getData().uid, K_SDPA_TENSOR_RNG_DUMP_UID);
    ASSERT_NE(desc->getAmaxSDesc(), nullptr);
    EXPECT_EQ(desc->getAmaxSDesc()->getData().uid, K_SDPA_TENSOR_AMAX_S_UID);
    ASSERT_NE(desc->getAmaxODesc(), nullptr);
    EXPECT_EQ(desc->getAmaxODesc()->getData().uid, K_SDPA_TENSOR_AMAX_O_UID);
}

TEST_F(TestSdpaFwdOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getQDesc(), _tensorMap[K_SDPA_TENSOR_Q_UID]);
    EXPECT_EQ(desc->getKDesc(), _tensorMap[K_SDPA_TENSOR_K_UID]);
    EXPECT_EQ(desc->getVDesc(), _tensorMap[K_SDPA_TENSOR_V_UID]);
    EXPECT_EQ(desc->getODesc(), _tensorMap[K_SDPA_TENSOR_O_UID]);
    EXPECT_EQ(desc->getAttnMaskDesc(), _tensorMap[K_SDPA_TENSOR_ATTN_MASK_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_SDPA_TENSOR_SCALE_UID]);
    EXPECT_EQ(desc->getSeqLenQDesc(), _tensorMap[K_SDPA_TENSOR_SEQ_LEN_Q_UID]);
    EXPECT_EQ(desc->getSeqLenKvDesc(), _tensorMap[K_SDPA_TENSOR_SEQ_LEN_KV_UID]);
    EXPECT_EQ(desc->getSeedDesc(), _tensorMap[K_SDPA_TENSOR_SEED_UID]);
    EXPECT_EQ(desc->getOffsetDesc(), _tensorMap[K_SDPA_TENSOR_OFFSET_UID]);
    EXPECT_EQ(desc->getDropoutMaskDesc(), _tensorMap[K_SDPA_TENSOR_DROPOUT_MASK_UID]);
    EXPECT_EQ(desc->getDropoutScaleDesc(), _tensorMap[K_SDPA_TENSOR_DROPOUT_SCALE_UID]);
    EXPECT_EQ(desc->getPageTableKDesc(), _tensorMap[K_SDPA_TENSOR_PAGE_TABLE_K_UID]);
    EXPECT_EQ(desc->getPageTableVDesc(), _tensorMap[K_SDPA_TENSOR_PAGE_TABLE_V_UID]);
    EXPECT_EQ(desc->getBlockMaskDesc(), _tensorMap[K_SDPA_TENSOR_BLOCK_MASK_UID]);
    EXPECT_EQ(desc->getSinkTokenDesc(), _tensorMap[K_SDPA_TENSOR_SINK_TOKEN_UID]);
    EXPECT_EQ(desc->getDescaleQDesc(), _tensorMap[K_SDPA_TENSOR_DESCALE_Q_UID]);
    EXPECT_EQ(desc->getDescaleKDesc(), _tensorMap[K_SDPA_TENSOR_DESCALE_K_UID]);
    EXPECT_EQ(desc->getDescaleVDesc(), _tensorMap[K_SDPA_TENSOR_DESCALE_V_UID]);
    EXPECT_EQ(desc->getDescaleSDesc(), _tensorMap[K_SDPA_TENSOR_DESCALE_S_UID]);
    EXPECT_EQ(desc->getScaleSDesc(), _tensorMap[K_SDPA_TENSOR_SCALE_S_UID]);
    EXPECT_EQ(desc->getScaleODesc(), _tensorMap[K_SDPA_TENSOR_SCALE_O_UID]);
    EXPECT_EQ(desc->getStatsDesc(), _tensorMap[K_SDPA_TENSOR_STATS_UID]);
    EXPECT_EQ(desc->getMaxDesc(), _tensorMap[K_SDPA_TENSOR_MAX_UID]);
    EXPECT_EQ(desc->getSumExpDesc(), _tensorMap[K_SDPA_TENSOR_SUM_EXP_UID]);
    EXPECT_EQ(desc->getRngDumpDesc(), _tensorMap[K_SDPA_TENSOR_RNG_DUMP_UID]);
    EXPECT_EQ(desc->getAmaxSDesc(), _tensorMap[K_SDPA_TENSOR_AMAX_S_UID]);
    EXPECT_EQ(desc->getAmaxODesc(), _tensorMap[K_SDPA_TENSOR_AMAX_O_UID]);
}

TEST_F(TestSdpaFwdOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getQDesc(), nullptr);
    EXPECT_EQ(desc->getQDesc()->getData().uid, K_SDPA_TENSOR_Q_UID);
    EXPECT_EQ(desc->getQDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getQDesc()->getData().dims, toVec(K_SDPA_TENSOR_Q_DIMS));
    EXPECT_EQ(desc->getQDesc()->getData().strides, toVec(K_SDPA_TENSOR_Q_STRIDES));

    ASSERT_NE(desc->getKDesc(), nullptr);
    EXPECT_EQ(desc->getKDesc()->getData().uid, K_SDPA_TENSOR_K_UID);
    EXPECT_EQ(desc->getKDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getKDesc()->getData().dims, toVec(K_SDPA_TENSOR_K_DIMS));
    EXPECT_EQ(desc->getKDesc()->getData().strides, toVec(K_SDPA_TENSOR_K_STRIDES));

    ASSERT_NE(desc->getVDesc(), nullptr);
    EXPECT_EQ(desc->getVDesc()->getData().uid, K_SDPA_TENSOR_V_UID);
    EXPECT_EQ(desc->getVDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getVDesc()->getData().dims, toVec(K_SDPA_TENSOR_V_DIMS));
    EXPECT_EQ(desc->getVDesc()->getData().strides, toVec(K_SDPA_TENSOR_V_STRIDES));

    ASSERT_NE(desc->getODesc(), nullptr);
    EXPECT_EQ(desc->getODesc()->getData().uid, K_SDPA_TENSOR_O_UID);
    EXPECT_EQ(desc->getODesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getODesc()->getData().dims, toVec(K_SDPA_TENSOR_O_DIMS));
    EXPECT_EQ(desc->getODesc()->getData().strides, toVec(K_SDPA_TENSOR_O_STRIDES));

    ASSERT_NE(desc->getAttnMaskDesc(), nullptr);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().uid, K_SDPA_TENSOR_ATTN_MASK_UID);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().dims, toVec(K_SDPA_TENSOR_ATTN_MASK_DIMS));
    EXPECT_EQ(desc->getAttnMaskDesc()->getData().strides, toVec(K_SDPA_TENSOR_ATTN_MASK_STRIDES));

    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_SDPA_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims, toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides, toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    ASSERT_NE(desc->getSeqLenQDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().uid, K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().dims, (std::vector<int64_t>{1}));
    EXPECT_EQ(desc->getSeqLenQDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getSeqLenKvDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().uid, K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().dims, (std::vector<int64_t>{1}));
    EXPECT_EQ(desc->getSeqLenKvDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getSeedDesc(), nullptr);
    EXPECT_EQ(desc->getSeedDesc()->getData().uid, K_SDPA_TENSOR_SEED_UID);

    ASSERT_NE(desc->getOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getOffsetDesc()->getData().uid, K_SDPA_TENSOR_OFFSET_UID);

    ASSERT_NE(desc->getDropoutMaskDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutMaskDesc()->getData().uid, K_SDPA_TENSOR_DROPOUT_MASK_UID);

    ASSERT_NE(desc->getDropoutScaleDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleDesc()->getData().uid, K_SDPA_TENSOR_DROPOUT_SCALE_UID);

    ASSERT_NE(desc->getStatsDesc(), nullptr);
    EXPECT_EQ(desc->getStatsDesc()->getData().uid, K_SDPA_TENSOR_STATS_UID);
    EXPECT_EQ(desc->getStatsDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getStatsDesc()->getData().dims, toVec(K_SDPA_TENSOR_STATS_DIMS));
    EXPECT_EQ(desc->getStatsDesc()->getData().strides, toVec(K_SDPA_TENSOR_STATS_STRIDES));

    ASSERT_NE(desc->getAmaxSDesc(), nullptr);
    EXPECT_EQ(desc->getAmaxSDesc()->getData().uid, K_SDPA_TENSOR_AMAX_S_UID);
    EXPECT_EQ(desc->getAmaxSDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getAmaxSDesc()->getData().dims, toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    EXPECT_EQ(desc->getAmaxSDesc()->getData().strides, toVec(K_SDPA_TENSOR_SCALAR_STRIDES));

    ASSERT_NE(desc->getAmaxODesc(), nullptr);
    EXPECT_EQ(desc->getAmaxODesc()->getData().uid, K_SDPA_TENSOR_AMAX_O_UID);
    EXPECT_EQ(desc->getAmaxODesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getAmaxODesc()->getData().dims, toVec(K_SDPA_TENSOR_SCALAR_DIMS));
    EXPECT_EQ(desc->getAmaxODesc()->getData().strides, toVec(K_SDPA_TENSOR_SCALAR_STRIDES));
}

TEST_F(TestSdpaFwdOperationFromNode, FailsWithMissingQTensor)
{
    _tensorMap.erase(K_SDPA_TENSOR_Q_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(SdpaFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestSdpaFwdOperationFromNode, FailsWithMissingKTensor)
{
    _tensorMap.erase(K_SDPA_TENSOR_K_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(SdpaFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestSdpaFwdOperationFromNode, FailsWithMissingVTensor)
{
    _tensorMap.erase(K_SDPA_TENSOR_V_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(SdpaFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestSdpaFwdOperationFromNode, FailsWithMissingOTensor)
{
    _tensorMap.erase(K_SDPA_TENSOR_O_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(SdpaFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestSdpaFwdOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.attn_mask_tensor_uid = flatbuffers::nullopt;
    attrs.scale_tensor_uid = flatbuffers::nullopt;
    attrs.seq_len_q_tensor_uid = flatbuffers::nullopt;
    attrs.seq_len_kv_tensor_uid = flatbuffers::nullopt;
    attrs.seed_tensor_uid = flatbuffers::nullopt;
    attrs.offset_tensor_uid = flatbuffers::nullopt;
    attrs.dropout_mask_tensor_uid = flatbuffers::nullopt;
    attrs.dropout_scale_tensor_uid = flatbuffers::nullopt;
    attrs.page_table_k_tensor_uid = flatbuffers::nullopt;
    attrs.page_table_v_tensor_uid = flatbuffers::nullopt;
    attrs.block_mask_tensor_uid = flatbuffers::nullopt;
    attrs.sink_token_tensor_uid = flatbuffers::nullopt;
    attrs.descale_q_tensor_uid = flatbuffers::nullopt;
    attrs.descale_k_tensor_uid = flatbuffers::nullopt;
    attrs.descale_v_tensor_uid = flatbuffers::nullopt;
    attrs.descale_s_tensor_uid = flatbuffers::nullopt;
    attrs.scale_s_tensor_uid = flatbuffers::nullopt;
    attrs.scale_o_tensor_uid = flatbuffers::nullopt;
    attrs.stats_tensor_uid = flatbuffers::nullopt;
    attrs.max_tensor_uid = flatbuffers::nullopt;
    attrs.sum_exp_tensor_uid = flatbuffers::nullopt;
    attrs.rng_dump_tensor_uid = flatbuffers::nullopt;
    attrs.amax_s_tensor_uid = flatbuffers::nullopt;
    attrs.amax_o_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getQDesc(), nullptr);
    EXPECT_NE(desc->getKDesc(), nullptr);
    EXPECT_NE(desc->getVDesc(), nullptr);
    EXPECT_NE(desc->getODesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getAttnMaskDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenQDesc(), nullptr);
    EXPECT_EQ(desc->getSeqLenKvDesc(), nullptr);
    EXPECT_EQ(desc->getSeedDesc(), nullptr);
    EXPECT_EQ(desc->getOffsetDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutMaskDesc(), nullptr);
    EXPECT_EQ(desc->getDropoutScaleDesc(), nullptr);
    EXPECT_EQ(desc->getPageTableKDesc(), nullptr);
    EXPECT_EQ(desc->getPageTableVDesc(), nullptr);
    EXPECT_EQ(desc->getBlockMaskDesc(), nullptr);
    EXPECT_EQ(desc->getSinkTokenDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleQDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleKDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleVDesc(), nullptr);
    EXPECT_EQ(desc->getDescaleSDesc(), nullptr);
    EXPECT_EQ(desc->getScaleSDesc(), nullptr);
    EXPECT_EQ(desc->getScaleODesc(), nullptr);
    EXPECT_EQ(desc->getStatsDesc(), nullptr);
    EXPECT_EQ(desc->getMaxDesc(), nullptr);
    EXPECT_EQ(desc->getSumExpDesc(), nullptr);
    EXPECT_EQ(desc->getRngDumpDesc(), nullptr);
    EXPECT_EQ(desc->getAmaxSDesc(), nullptr);
    EXPECT_EQ(desc->getAmaxODesc(), nullptr);
}

// -- 3d: Parameterized optional tensor missing tests --

class TestSdpaFwdOptionalTensorMissing : public TestSdpaFwdOperationFromNode,
                                         public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestSdpaFwdOptionalTensorMissing, SucceedsWhenOptionalTensorMissing)
{
    const int64_t erasedUid = GetParam();
    _tensorMap.erase(erasedUid);
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Verify the erased tensor's getter returns nullptr
    const std::unordered_map<int64_t, std::function<std::shared_ptr<TensorDescriptor>()>> getterMap
        = {
            {K_SDPA_TENSOR_ATTN_MASK_UID, [&] { return desc->getAttnMaskDesc(); }},
            {K_SDPA_TENSOR_SCALE_UID, [&] { return desc->getScaleDesc(); }},
            {K_SDPA_TENSOR_SEQ_LEN_Q_UID, [&] { return desc->getSeqLenQDesc(); }},
            {K_SDPA_TENSOR_SEQ_LEN_KV_UID, [&] { return desc->getSeqLenKvDesc(); }},
            {K_SDPA_TENSOR_SEED_UID, [&] { return desc->getSeedDesc(); }},
            {K_SDPA_TENSOR_OFFSET_UID, [&] { return desc->getOffsetDesc(); }},
            {K_SDPA_TENSOR_DROPOUT_MASK_UID, [&] { return desc->getDropoutMaskDesc(); }},
            {K_SDPA_TENSOR_DROPOUT_SCALE_UID, [&] { return desc->getDropoutScaleDesc(); }},
            {K_SDPA_TENSOR_PAGE_TABLE_K_UID, [&] { return desc->getPageTableKDesc(); }},
            {K_SDPA_TENSOR_PAGE_TABLE_V_UID, [&] { return desc->getPageTableVDesc(); }},
            {K_SDPA_TENSOR_BLOCK_MASK_UID, [&] { return desc->getBlockMaskDesc(); }},
            {K_SDPA_TENSOR_SINK_TOKEN_UID, [&] { return desc->getSinkTokenDesc(); }},
            {K_SDPA_TENSOR_DESCALE_Q_UID, [&] { return desc->getDescaleQDesc(); }},
            {K_SDPA_TENSOR_DESCALE_K_UID, [&] { return desc->getDescaleKDesc(); }},
            {K_SDPA_TENSOR_DESCALE_V_UID, [&] { return desc->getDescaleVDesc(); }},
            {K_SDPA_TENSOR_DESCALE_S_UID, [&] { return desc->getDescaleSDesc(); }},
            {K_SDPA_TENSOR_SCALE_S_UID, [&] { return desc->getScaleSDesc(); }},
            {K_SDPA_TENSOR_SCALE_O_UID, [&] { return desc->getScaleODesc(); }},
            {K_SDPA_TENSOR_STATS_UID, [&] { return desc->getStatsDesc(); }},
            {K_SDPA_TENSOR_MAX_UID, [&] { return desc->getMaxDesc(); }},
            {K_SDPA_TENSOR_SUM_EXP_UID, [&] { return desc->getSumExpDesc(); }},
            {K_SDPA_TENSOR_RNG_DUMP_UID, [&] { return desc->getRngDumpDesc(); }},
            {K_SDPA_TENSOR_AMAX_S_UID, [&] { return desc->getAmaxSDesc(); }},
            {K_SDPA_TENSOR_AMAX_O_UID, [&] { return desc->getAmaxODesc(); }},
        };
    auto it = getterMap.find(erasedUid);
    ASSERT_NE(it, getterMap.end());
    EXPECT_EQ(it->second(), nullptr);
}

INSTANTIATE_TEST_SUITE_P(OptionalTensors,
                         TestSdpaFwdOptionalTensorMissing,
                         ::testing::Values(K_SDPA_TENSOR_ATTN_MASK_UID,
                                           K_SDPA_TENSOR_SCALE_UID,
                                           K_SDPA_TENSOR_SEQ_LEN_Q_UID,
                                           K_SDPA_TENSOR_SEQ_LEN_KV_UID,
                                           K_SDPA_TENSOR_SEED_UID,
                                           K_SDPA_TENSOR_OFFSET_UID,
                                           K_SDPA_TENSOR_DROPOUT_MASK_UID,
                                           K_SDPA_TENSOR_DROPOUT_SCALE_UID,
                                           K_SDPA_TENSOR_PAGE_TABLE_K_UID,
                                           K_SDPA_TENSOR_PAGE_TABLE_V_UID,
                                           K_SDPA_TENSOR_BLOCK_MASK_UID,
                                           K_SDPA_TENSOR_SINK_TOKEN_UID,
                                           K_SDPA_TENSOR_DESCALE_Q_UID,
                                           K_SDPA_TENSOR_DESCALE_K_UID,
                                           K_SDPA_TENSOR_DESCALE_V_UID,
                                           K_SDPA_TENSOR_DESCALE_S_UID,
                                           K_SDPA_TENSOR_SCALE_S_UID,
                                           K_SDPA_TENSOR_SCALE_O_UID,
                                           K_SDPA_TENSOR_STATS_UID,
                                           K_SDPA_TENSOR_MAX_UID,
                                           K_SDPA_TENSOR_SUM_EXP_UID,
                                           K_SDPA_TENSOR_RNG_DUMP_UID,
                                           K_SDPA_TENSOR_AMAX_S_UID,
                                           K_SDPA_TENSOR_AMAX_O_UID));

TEST_F(TestSdpaFwdOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 28);
    EXPECT_EQ(tensors[0]->getData().uid, K_SDPA_TENSOR_Q_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_SDPA_TENSOR_K_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_SDPA_TENSOR_V_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_SDPA_TENSOR_O_UID);
    EXPECT_EQ(tensors[4]->getData().uid, K_SDPA_TENSOR_ATTN_MASK_UID);
    EXPECT_EQ(tensors[5]->getData().uid, K_SDPA_TENSOR_SCALE_UID);
    EXPECT_EQ(tensors[6]->getData().uid, K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    EXPECT_EQ(tensors[7]->getData().uid, K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    EXPECT_EQ(tensors[8]->getData().uid, K_SDPA_TENSOR_SEED_UID);
    EXPECT_EQ(tensors[9]->getData().uid, K_SDPA_TENSOR_OFFSET_UID);
    EXPECT_EQ(tensors[10]->getData().uid, K_SDPA_TENSOR_DROPOUT_MASK_UID);
    EXPECT_EQ(tensors[11]->getData().uid, K_SDPA_TENSOR_DROPOUT_SCALE_UID);
    EXPECT_EQ(tensors[12]->getData().uid, K_SDPA_TENSOR_PAGE_TABLE_K_UID);
    EXPECT_EQ(tensors[13]->getData().uid, K_SDPA_TENSOR_PAGE_TABLE_V_UID);
    EXPECT_EQ(tensors[14]->getData().uid, K_SDPA_TENSOR_BLOCK_MASK_UID);
    EXPECT_EQ(tensors[15]->getData().uid, K_SDPA_TENSOR_SINK_TOKEN_UID);
    EXPECT_EQ(tensors[16]->getData().uid, K_SDPA_TENSOR_DESCALE_Q_UID);
    EXPECT_EQ(tensors[17]->getData().uid, K_SDPA_TENSOR_DESCALE_K_UID);
    EXPECT_EQ(tensors[18]->getData().uid, K_SDPA_TENSOR_DESCALE_V_UID);
    EXPECT_EQ(tensors[19]->getData().uid, K_SDPA_TENSOR_DESCALE_S_UID);
    EXPECT_EQ(tensors[20]->getData().uid, K_SDPA_TENSOR_SCALE_S_UID);
    EXPECT_EQ(tensors[21]->getData().uid, K_SDPA_TENSOR_SCALE_O_UID);
    EXPECT_EQ(tensors[22]->getData().uid, K_SDPA_TENSOR_STATS_UID);
    EXPECT_EQ(tensors[23]->getData().uid, K_SDPA_TENSOR_MAX_UID);
    EXPECT_EQ(tensors[24]->getData().uid, K_SDPA_TENSOR_SUM_EXP_UID);
    EXPECT_EQ(tensors[25]->getData().uid, K_SDPA_TENSOR_RNG_DUMP_UID);
    EXPECT_EQ(tensors[26]->getData().uid, K_SDPA_TENSOR_AMAX_S_UID);
    EXPECT_EQ(tensors[27]->getData().uid, K_SDPA_TENSOR_AMAX_O_UID);
}

TEST_F(TestSdpaFwdOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::SdpaAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->q_tensor_uid, K_SDPA_TENSOR_Q_UID);
    EXPECT_EQ(rebuiltAttrs->k_tensor_uid, K_SDPA_TENSOR_K_UID);
    EXPECT_EQ(rebuiltAttrs->v_tensor_uid, K_SDPA_TENSOR_V_UID);
    EXPECT_EQ(rebuiltAttrs->o_tensor_uid, K_SDPA_TENSOR_O_UID);
    EXPECT_EQ(rebuiltAttrs->attn_mask_tensor_uid, K_SDPA_TENSOR_ATTN_MASK_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_SDPA_TENSOR_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->seq_len_q_tensor_uid, K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    EXPECT_EQ(rebuiltAttrs->seq_len_kv_tensor_uid, K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    EXPECT_EQ(rebuiltAttrs->seed_tensor_uid, K_SDPA_TENSOR_SEED_UID);
    EXPECT_EQ(rebuiltAttrs->offset_tensor_uid, K_SDPA_TENSOR_OFFSET_UID);
    EXPECT_EQ(rebuiltAttrs->dropout_mask_tensor_uid, K_SDPA_TENSOR_DROPOUT_MASK_UID);
    EXPECT_EQ(rebuiltAttrs->dropout_scale_tensor_uid, K_SDPA_TENSOR_DROPOUT_SCALE_UID);
    EXPECT_EQ(rebuiltAttrs->page_table_k_tensor_uid, K_SDPA_TENSOR_PAGE_TABLE_K_UID);
    EXPECT_EQ(rebuiltAttrs->page_table_v_tensor_uid, K_SDPA_TENSOR_PAGE_TABLE_V_UID);
    EXPECT_EQ(rebuiltAttrs->block_mask_tensor_uid, K_SDPA_TENSOR_BLOCK_MASK_UID);
    EXPECT_EQ(rebuiltAttrs->sink_token_tensor_uid, K_SDPA_TENSOR_SINK_TOKEN_UID);
    EXPECT_EQ(rebuiltAttrs->descale_q_tensor_uid, K_SDPA_TENSOR_DESCALE_Q_UID);
    EXPECT_EQ(rebuiltAttrs->descale_k_tensor_uid, K_SDPA_TENSOR_DESCALE_K_UID);
    EXPECT_EQ(rebuiltAttrs->descale_v_tensor_uid, K_SDPA_TENSOR_DESCALE_V_UID);
    EXPECT_EQ(rebuiltAttrs->descale_s_tensor_uid, K_SDPA_TENSOR_DESCALE_S_UID);
    EXPECT_EQ(rebuiltAttrs->scale_s_tensor_uid, K_SDPA_TENSOR_SCALE_S_UID);
    EXPECT_EQ(rebuiltAttrs->scale_o_tensor_uid, K_SDPA_TENSOR_SCALE_O_UID);
    EXPECT_EQ(rebuiltAttrs->stats_tensor_uid, K_SDPA_TENSOR_STATS_UID);
    EXPECT_EQ(rebuiltAttrs->max_tensor_uid, K_SDPA_TENSOR_MAX_UID);
    EXPECT_EQ(rebuiltAttrs->sum_exp_tensor_uid, K_SDPA_TENSOR_SUM_EXP_UID);
    EXPECT_EQ(rebuiltAttrs->rng_dump_tensor_uid, K_SDPA_TENSOR_RNG_DUMP_UID);
    EXPECT_EQ(rebuiltAttrs->amax_s_tensor_uid, K_SDPA_TENSOR_AMAX_S_UID);
    EXPECT_EQ(rebuiltAttrs->amax_o_tensor_uid, K_SDPA_TENSOR_AMAX_O_UID);
    EXPECT_EQ(rebuiltAttrs->diagonal_alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(rebuiltAttrs->implementation, AttentionImplementation::AUTO);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesMmaCoreMode)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.mma_core_mode = DataType::HALF;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_EQ(desc->getData().mma_core_mode, DataType::HALF);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->mma_core_mode, DataType::HALF);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodeMmaCoreModeDefaultsToNotSet)
{
    // When mma_core_mode is NOT_SET, the packer omits the attribute.
    // The unpacker should leave it at the default (NOT_SET).
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_EQ(desc->getData().mma_core_mode, DataType::UNSET);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesGenerateStats)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.generate_stats = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().generate_stats.has_value());
    EXPECT_EQ(desc->getData().generate_stats.value(), true);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->generate_stats.has_value());
    EXPECT_EQ(rebuiltAttrs->generate_stats.value(), true);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesAlibiMask)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.alibi_mask = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().alibi_mask);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_TRUE(rebuiltAttrs->alibi_mask);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesPaddingMask)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.padding_mask = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().padding_mask);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_TRUE(rebuiltAttrs->padding_mask);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesCausalMask)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.causal_mask = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().causal_mask);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_TRUE(rebuiltAttrs->causal_mask);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesCausalMaskBottomRight)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.causal_mask_bottom_right = true;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().causal_mask_bottom_right);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_TRUE(rebuiltAttrs->causal_mask_bottom_right);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesDropoutProbability)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.dropout_probability = 0.5F;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().dropout_probability.has_value());
    EXPECT_FLOAT_EQ(desc->getData().dropout_probability.value(), 0.5F);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->dropout_probability.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->dropout_probability.value(), 0.5F);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesAttnScaleValue)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.attn_scale_value = 0.5F;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(desc->getData().attn_scale_value.value(), 0.5F);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(rebuiltAttrs->attn_scale_value.value(), 0.5F);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesLeftBound)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.left_bound = 2;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().left_bound.has_value());
    EXPECT_EQ(desc->getData().left_bound.value(), 2);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->left_bound.has_value());
    EXPECT_EQ(rebuiltAttrs->left_bound.value(), 2);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesRightBound)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.right_bound = 2;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().right_bound.has_value());
    EXPECT_EQ(desc->getData().right_bound.value(), 2);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->right_bound.has_value());
    EXPECT_EQ(rebuiltAttrs->right_bound.value(), 2);
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodePreservesMaxSeqLenKv)
{
    auto attrs = createStandardSdpaFwdAttrs();
    attrs.max_seq_len_kv = 2;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);

    EXPECT_TRUE(desc->getData().max_seq_len_kv.has_value());
    EXPECT_EQ(desc->getData().max_seq_len_kv.value(), 2);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    ASSERT_TRUE(rebuiltAttrs->max_seq_len_kv.has_value());
    EXPECT_EQ(rebuiltAttrs->max_seq_len_kv.value(), 2);
}

TEST_F(TestSdpaFwdOperationFromNode, BuildNodeOmitsUnsetOptionalScalars)
{
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    const auto* rebuiltAttrs = rebuiltNode->attributes.AsSdpaAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);

    EXPECT_FALSE(rebuiltAttrs->generate_stats.has_value());
    EXPECT_FALSE(rebuiltAttrs->alibi_mask);
    EXPECT_FALSE(rebuiltAttrs->padding_mask);
    EXPECT_FALSE(rebuiltAttrs->causal_mask);
    EXPECT_FALSE(rebuiltAttrs->causal_mask_bottom_right);
    EXPECT_FALSE(rebuiltAttrs->dropout_probability.has_value());
    EXPECT_FALSE(rebuiltAttrs->attn_scale_value.has_value());
    EXPECT_FALSE(rebuiltAttrs->left_bound.has_value());
    EXPECT_FALSE(rebuiltAttrs->right_bound.has_value());
    EXPECT_FALSE(rebuiltAttrs->max_seq_len_kv.has_value());
}

TEST_F(TestSdpaFwdOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    // Verify compute type
    hipdnnDataType_t computeType = {};
    int64_t dtCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_FWD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &computeType);
    ASSERT_EQ(computeType, HIPDNN_DATA_FLOAT);

    // Verify diagonal_alignment
    hipdnnDiagonalAlignment_t diagonalAlignment = {};
    int64_t diagonalAlignmentCount = 0;
    desc->getAttribute(HIPDNN_ATTR_SDPA_FWD_DIAGONAL_ALIGNMENT_EXT,
                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT,
                       1,
                       &diagonalAlignmentCount,
                       &diagonalAlignment);
    ASSERT_EQ(diagonalAlignment, HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT);

    // Verify implementation
    hipdnnAttentionImplementation_t implementation = {};
    int64_t implementationCount = 0;
    desc->getAttribute(HIPDNN_ATTR_SDPA_FWD_IMPLEMENTATION_EXT,
                       HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT,
                       1,
                       &implementationCount,
                       &implementation);
    ASSERT_EQ(implementation, HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT);

    // Verify q tensor
    hipdnn_backend::ScopedDescriptor qScoped;
    int64_t qCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_QDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &qCount,
                       static_cast<void*>(qScoped.getPtr()));
    ASSERT_EQ(qCount, 1);
    ASSERT_NE(qScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(qScoped.get(),
                                                           K_SDPA_TENSOR_Q_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           toVec(K_SDPA_TENSOR_Q_DIMS),
                                                           toVec(K_SDPA_TENSOR_Q_STRIDES));

    // Verify k tensor
    hipdnn_backend::ScopedDescriptor kScoped;
    int64_t kCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_KDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &kCount,
                       static_cast<void*>(kScoped.getPtr()));
    ASSERT_EQ(kCount, 1);
    ASSERT_NE(kScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(kScoped.get(),
                                                           K_SDPA_TENSOR_K_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           toVec(K_SDPA_TENSOR_K_DIMS),
                                                           toVec(K_SDPA_TENSOR_K_STRIDES));

    // Verify v tensor
    hipdnn_backend::ScopedDescriptor vScoped;
    int64_t vCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_VDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &vCount,
                       static_cast<void*>(vScoped.getPtr()));
    ASSERT_EQ(vCount, 1);
    ASSERT_NE(vScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(vScoped.get(),
                                                           K_SDPA_TENSOR_V_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           toVec(K_SDPA_TENSOR_V_DIMS),
                                                           toVec(K_SDPA_TENSOR_V_STRIDES));

    // Verify o tensor
    hipdnn_backend::ScopedDescriptor oScoped;
    int64_t oCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_ODESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &oCount,
                       static_cast<void*>(oScoped.getPtr()));
    ASSERT_EQ(oCount, 1);
    ASSERT_NE(oScoped.get(), nullptr);
    hipdnn_backend::test_utilities::verifyTensorDescriptor(oScoped.get(),
                                                           K_SDPA_TENSOR_O_UID,
                                                           HIPDNN_DATA_FLOAT,
                                                           toVec(K_SDPA_TENSOR_O_DIMS),
                                                           toVec(K_SDPA_TENSOR_O_STRIDES));
}

TEST_F(TestSdpaFwdOperationFromNode, OperationTypeIsCorrectAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_SDPA_FORWARD_EXT);
}

TEST_F(TestSdpaFwdOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_sdpa_fwd_1";

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_sdpa_fwd_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_sdpa_fwd_1");
}

TEST_F(TestSdpaFwdOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestSdpaFwdOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = SdpaFwdOperationDescriptor::fromNode(node, _tensorMap);
    auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}

TEST_F(TestSdpaFwdOperationFromNode, FromNodeFailsWithWrongAttributeType)
{
    // A node with ConvolutionFwdAttributes should fail SDPA fromNode — wrong union type.
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributesT{});

    ASSERT_THROW_HIPDNN_STATUS(SdpaFwdOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}
