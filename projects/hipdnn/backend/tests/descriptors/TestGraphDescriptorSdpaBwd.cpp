// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/SdpaBwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SdpaBwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Helper: set a single tensor attribute on a descriptor
inline void setTensorAttr(SdpaBwdOperationDescriptor& desc,
                          hipdnnBackendAttributeName_t attr,
                          HipdnnBackendDescriptor* tensor)
{
    desc.setAttribute(attr, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, static_cast<const void*>(&tensor));
}

// Helper: create and finalize a SdpaBwdOperationDescriptor with required tensors.
// Pass nullptr for any optional tensor to omit it.
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedSdpaBwdOp(HipdnnBackendDescriptor* qDesc,
                             HipdnnBackendDescriptor* kDesc,
                             HipdnnBackendDescriptor* vDesc,
                             HipdnnBackendDescriptor* oDesc,
                             HipdnnBackendDescriptor* doDesc,
                             HipdnnBackendDescriptor* statsDesc,
                             HipdnnBackendDescriptor* dqDesc,
                             HipdnnBackendDescriptor* dkDesc,
                             HipdnnBackendDescriptor* dvDesc,
                             HipdnnBackendDescriptor* scaleDesc = nullptr,
                             HipdnnBackendDescriptor* attnMaskDesc = nullptr,
                             HipdnnBackendDescriptor* seqLenQDesc = nullptr,
                             HipdnnBackendDescriptor* seqLenKvDesc = nullptr,
                             HipdnnBackendDescriptor* seedDesc = nullptr,
                             HipdnnBackendDescriptor* offsetDesc = nullptr,
                             HipdnnBackendDescriptor* dropoutMaskDesc = nullptr,
                             HipdnnBackendDescriptor* dropoutScaleDesc = nullptr,
                             HipdnnBackendDescriptor* dropoutScaleInvDesc = nullptr,
                             HipdnnBackendDescriptor* dbiasDesc = nullptr,
                             hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<SdpaBwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SdpaBwdOperationDescriptor>();

    // Required input tensors
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_Q_EXT, qDesc);
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_K_EXT, kDesc);
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_V_EXT, vDesc);
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_O_EXT, oDesc);
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DO_EXT, doDesc);
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_STATS_EXT, statsDesc);

    // Required output tensors
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DQ_EXT, dqDesc);
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DK_EXT, dkDesc);
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DV_EXT, dvDesc);

    // Optional tensors — only set when provided
    if(scaleDesc != nullptr)
    {
        setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_SCALE_EXT, scaleDesc);
    }
    if(attnMaskDesc != nullptr)
    {
        setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_ATTN_MASK_EXT, attnMaskDesc);
    }
    if(seqLenQDesc != nullptr)
    {
        setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_Q_EXT, seqLenQDesc);
    }
    if(seqLenKvDesc != nullptr)
    {
        setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KV_EXT, seqLenKvDesc);
    }
    if(seedDesc != nullptr)
    {
        setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_SEED_EXT, seedDesc);
    }
    if(offsetDesc != nullptr)
    {
        setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_OFFSET_EXT, offsetDesc);
    }
    if(dropoutMaskDesc != nullptr)
    {
        setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_MASK_EXT, dropoutMaskDesc);
    }
    if(dropoutScaleDesc != nullptr)
    {
        setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_EXT, dropoutScaleDesc);
    }
    if(dropoutScaleInvDesc != nullptr)
    {
        setTensorAttr(
            *desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_INV_EXT, dropoutScaleInvDesc);
    }
    if(dbiasDesc != nullptr)
    {
        setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DBIAS_EXT, dbiasDesc);
    }

    desc->setAttribute(HIPDNN_ATTR_SDPA_BWD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorSdpaBwd : public ::testing::Test
{
public:
    static const TensorAttributesT* findTensorByUid(const GraphT& graphT, int64_t uid)
    {
        for(const auto& tensor : graphT.tensors)
        {
            if(tensor->uid == uid)
            {
                return tensor.get();
            }
        }
        return nullptr;
    }

    static void verifyTensor(const TensorAttributesT* tensor,
                             int64_t expectedUid,
                             const std::vector<int64_t>& expectedDims,
                             const std::vector<int64_t>& expectedStrides,
                             DataType expectedDataType,
                             bool expectedVirtual = false)
    {
        ASSERT_NE(tensor, nullptr) << "Tensor with UID " << expectedUid << " not found";
        EXPECT_EQ(tensor->uid, expectedUid);
        EXPECT_EQ(tensor->dims, expectedDims);
        EXPECT_EQ(tensor->strides, expectedStrides);
        EXPECT_EQ(tensor->data_type, expectedDataType);
        EXPECT_EQ(tensor->virtual_, expectedVirtual);
    }

    static void verifyRequiredTensorUids(const SdpaBackwardAttributesT* attrs)
    {
        EXPECT_EQ(attrs->q_tensor_uid, K_SDPA_BWD_TENSOR_Q_UID);
        EXPECT_EQ(attrs->k_tensor_uid, K_SDPA_BWD_TENSOR_K_UID);
        EXPECT_EQ(attrs->v_tensor_uid, K_SDPA_BWD_TENSOR_V_UID);
        EXPECT_EQ(attrs->o_tensor_uid, K_SDPA_BWD_TENSOR_O_UID);
        EXPECT_EQ(attrs->do_tensor_uid, K_SDPA_BWD_TENSOR_DO_UID);
        EXPECT_EQ(attrs->stats_tensor_uid, K_SDPA_BWD_TENSOR_STATS_UID);
        EXPECT_EQ(attrs->dq_tensor_uid, K_SDPA_BWD_TENSOR_DQ_UID);
        EXPECT_EQ(attrs->dk_tensor_uid, K_SDPA_BWD_TENSOR_DK_UID);
        EXPECT_EQ(attrs->dv_tensor_uid, K_SDPA_BWD_TENSOR_DV_UID);
    }

    static void verifyDefaultScalarAttrs(const SdpaBackwardAttributesT* attrs)
    {
        EXPECT_EQ(attrs->diagonal_alignment, DiagonalAlignment::TOP_LEFT);
        EXPECT_EQ(attrs->alibi_mask, false);
        EXPECT_EQ(attrs->padding_mask, false);
        EXPECT_EQ(attrs->causal_mask, false);
        EXPECT_EQ(attrs->causal_mask_bottom_right, false);
        EXPECT_FALSE(attrs->dropout_probability.has_value());
        EXPECT_FALSE(attrs->attn_scale_value.has_value());
        EXPECT_FALSE(attrs->left_bound.has_value());
        EXPECT_FALSE(attrs->right_bound.has_value());
    }

    std::shared_ptr<GraphDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<GraphDescriptor>();
    }

    void setHandle() const
    {
        auto desc = getDescriptor();
        hipdnnHandle_t handle = &_mockHandle;
        desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                           HIPDNN_TYPE_HANDLE,
                           1,
                           static_cast<const void*>(&handle));
    }

    // Wires the op into the graph, finalizes, serializes, and unpacks. Returns the unpacked graph.
    std::unique_ptr<GraphT>
        finalizeGraphAndUnpack(std::unique_ptr<HipdnnBackendDescriptor>& opWrapper) const
    {
        auto graphDesc = getDescriptor();
        setHandle();

        auto* opDescPtr = opWrapper.get();
        graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                1,
                                static_cast<const void*>(&opDescPtr));
        graphDesc->finalize();

        auto serialized = graphDesc->getSerializedGraph();
        return UnPackGraph(serialized.ptr);
    }

    // Creates the 9 required tensor descriptors used across all tests.
    struct RequiredTensors
    {
        std::unique_ptr<HipdnnBackendDescriptor> q;
        std::unique_ptr<HipdnnBackendDescriptor> k;
        std::unique_ptr<HipdnnBackendDescriptor> v;
        std::unique_ptr<HipdnnBackendDescriptor> o;
        std::unique_ptr<HipdnnBackendDescriptor> dO;
        std::unique_ptr<HipdnnBackendDescriptor> stats;
        std::unique_ptr<HipdnnBackendDescriptor> dq;
        std::unique_ptr<HipdnnBackendDescriptor> dk;
        std::unique_ptr<HipdnnBackendDescriptor> dv;
    };

    static RequiredTensors createRequiredTensors()
    {
        return {
            createFinalizedTensor(K_SDPA_BWD_TENSOR_Q_UID,
                                  toVec(K_SDPA_BWD_TENSOR_Q_DIMS),
                                  toVec(K_SDPA_BWD_TENSOR_Q_STRIDES)),
            createFinalizedTensor(K_SDPA_BWD_TENSOR_K_UID,
                                  toVec(K_SDPA_BWD_TENSOR_K_DIMS),
                                  toVec(K_SDPA_BWD_TENSOR_K_STRIDES)),
            createFinalizedTensor(K_SDPA_BWD_TENSOR_V_UID,
                                  toVec(K_SDPA_BWD_TENSOR_V_DIMS),
                                  toVec(K_SDPA_BWD_TENSOR_V_STRIDES)),
            createFinalizedTensor(K_SDPA_BWD_TENSOR_O_UID,
                                  toVec(K_SDPA_BWD_TENSOR_O_DIMS),
                                  toVec(K_SDPA_BWD_TENSOR_O_STRIDES)),
            createFinalizedTensor(K_SDPA_BWD_TENSOR_DO_UID,
                                  toVec(K_SDPA_BWD_TENSOR_DO_DIMS),
                                  toVec(K_SDPA_BWD_TENSOR_DO_STRIDES)),
            createFinalizedTensor(K_SDPA_BWD_TENSOR_STATS_UID,
                                  toVec(K_SDPA_BWD_TENSOR_STATS_DIMS),
                                  toVec(K_SDPA_BWD_TENSOR_STATS_STRIDES)),
            createFinalizedTensor(K_SDPA_BWD_TENSOR_DQ_UID,
                                  toVec(K_SDPA_BWD_TENSOR_DQ_DIMS),
                                  toVec(K_SDPA_BWD_TENSOR_DQ_STRIDES)),
            createFinalizedTensor(K_SDPA_BWD_TENSOR_DK_UID,
                                  toVec(K_SDPA_BWD_TENSOR_DK_DIMS),
                                  toVec(K_SDPA_BWD_TENSOR_DK_STRIDES)),
            createFinalizedTensor(K_SDPA_BWD_TENSOR_DV_UID,
                                  toVec(K_SDPA_BWD_TENSOR_DV_DIMS),
                                  toVec(K_SDPA_BWD_TENSOR_DV_STRIDES)),
        };
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    mutable MockHandle _mockHandle;

    void SetUp() override
    {
        _wrapper = createDescriptor<GraphDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
    }
};

// =============================================================================
// Full operation with all optional tensors
// =============================================================================

TEST_F(TestGraphDescriptorSdpaBwd, BuildFromSingleOperation)
{
    auto t = createRequiredTensors();
    auto scaleDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_SCALE_UID);
    auto attnMaskDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_ATTN_MASK_UID);
    auto seqLenQDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID);
    auto seqLenKvDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID);
    auto seedDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_SEED_UID);
    auto offsetDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_OFFSET_UID);
    auto dropoutMaskDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID);
    auto dropoutScaleDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID);
    auto dropoutScaleInvDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID);
    auto dbiasDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_DBIAS_UID);

    auto opWrapper = createFinalizedSdpaBwdOp(t.q.get(),
                                              t.k.get(),
                                              t.v.get(),
                                              t.o.get(),
                                              t.dO.get(),
                                              t.stats.get(),
                                              t.dq.get(),
                                              t.dk.get(),
                                              t.dv.get(),
                                              scaleDesc.get(),
                                              attnMaskDesc.get(),
                                              seqLenQDesc.get(),
                                              seqLenKvDesc.get(),
                                              seedDesc.get(),
                                              offsetDesc.get(),
                                              dropoutMaskDesc.get(),
                                              dropoutScaleDesc.get(),
                                              dropoutScaleInvDesc.get(),
                                              dbiasDesc.get());

    auto graphDesc = getDescriptor();
    setHandle();

    auto* opDescPtr = opWrapper.get();
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(&opDescPtr));

    ASSERT_NO_THROW(graphDesc->finalize());
    ASSERT_TRUE(graphDesc->isFinalized());

    auto serialized = graphDesc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_NE(graphT, nullptr);
    ASSERT_EQ(graphT->nodes.size(), 1u);

    // Exact tensor count: 9 required + 10 optional = 19
    ASSERT_EQ(graphT->tensors.size(), 19u);

    auto& node = graphT->nodes[0];
    ASSERT_EQ(node->attributes.type, NodeAttributes::SdpaBackwardAttributes);

    auto* attrs = node->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    verifyRequiredTensorUids(attrs);

    // Verify all 10 optional tensor UIDs
    ASSERT_TRUE(attrs->scale_tensor_uid.has_value());
    EXPECT_EQ(attrs->scale_tensor_uid.value(), K_SDPA_BWD_TENSOR_SCALE_UID);

    ASSERT_TRUE(attrs->attn_mask_tensor_uid.has_value());
    EXPECT_EQ(attrs->attn_mask_tensor_uid.value(), K_SDPA_BWD_TENSOR_ATTN_MASK_UID);

    ASSERT_TRUE(attrs->seq_len_q_tensor_uid.has_value());
    EXPECT_EQ(attrs->seq_len_q_tensor_uid.value(), K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID);

    ASSERT_TRUE(attrs->seq_len_kv_tensor_uid.has_value());
    EXPECT_EQ(attrs->seq_len_kv_tensor_uid.value(), K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID);

    ASSERT_TRUE(attrs->seed_tensor_uid.has_value());
    EXPECT_EQ(attrs->seed_tensor_uid.value(), K_SDPA_BWD_TENSOR_SEED_UID);

    ASSERT_TRUE(attrs->offset_tensor_uid.has_value());
    EXPECT_EQ(attrs->offset_tensor_uid.value(), K_SDPA_BWD_TENSOR_OFFSET_UID);

    ASSERT_TRUE(attrs->dropout_mask_tensor_uid.has_value());
    EXPECT_EQ(attrs->dropout_mask_tensor_uid.value(), K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID);

    ASSERT_TRUE(attrs->dropout_scale_tensor_uid.has_value());
    EXPECT_EQ(attrs->dropout_scale_tensor_uid.value(), K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID);

    ASSERT_TRUE(attrs->dropout_scale_inv_tensor_uid.has_value());
    EXPECT_EQ(attrs->dropout_scale_inv_tensor_uid.value(), K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID);

    ASSERT_TRUE(attrs->dbias_tensor_uid.has_value());
    EXPECT_EQ(attrs->dbias_tensor_uid.value(), K_SDPA_BWD_TENSOR_DBIAS_UID);

    // Verify tensor attributes survive serialization for required tensors
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_Q_UID),
                 K_SDPA_BWD_TENSOR_Q_UID,
                 toVec(K_SDPA_BWD_TENSOR_Q_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_Q_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_K_UID),
                 K_SDPA_BWD_TENSOR_K_UID,
                 toVec(K_SDPA_BWD_TENSOR_K_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_K_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_V_UID),
                 K_SDPA_BWD_TENSOR_V_UID,
                 toVec(K_SDPA_BWD_TENSOR_V_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_V_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_O_UID),
                 K_SDPA_BWD_TENSOR_O_UID,
                 toVec(K_SDPA_BWD_TENSOR_O_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_O_STRIDES),
                 DataType::FLOAT);

    verifyDefaultScalarAttrs(attrs);
}

// =============================================================================
// Required-only serialization test
// =============================================================================

TEST_F(TestGraphDescriptorSdpaBwd, BuildFromRequiredOnlyOperation)
{
    auto t = createRequiredTensors();

    auto opWrapper = createFinalizedSdpaBwdOp(t.q.get(),
                                              t.k.get(),
                                              t.v.get(),
                                              t.o.get(),
                                              t.dO.get(),
                                              t.stats.get(),
                                              t.dq.get(),
                                              t.dk.get(),
                                              t.dv.get());

    auto graphDesc = getDescriptor();
    setHandle();

    auto* opDescPtr = opWrapper.get();
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(&opDescPtr));

    ASSERT_NO_THROW(graphDesc->finalize());
    ASSERT_TRUE(graphDesc->isFinalized());

    auto serialized = graphDesc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_NE(graphT, nullptr);
    ASSERT_EQ(graphT->nodes.size(), 1u);

    // Exact tensor count: exactly 9 required tensors, no optional
    ASSERT_EQ(graphT->tensors.size(), 9u);

    // Verify required tensor attributes survive serialization
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_Q_UID),
                 K_SDPA_BWD_TENSOR_Q_UID,
                 toVec(K_SDPA_BWD_TENSOR_Q_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_Q_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_K_UID),
                 K_SDPA_BWD_TENSOR_K_UID,
                 toVec(K_SDPA_BWD_TENSOR_K_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_K_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_V_UID),
                 K_SDPA_BWD_TENSOR_V_UID,
                 toVec(K_SDPA_BWD_TENSOR_V_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_V_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_O_UID),
                 K_SDPA_BWD_TENSOR_O_UID,
                 toVec(K_SDPA_BWD_TENSOR_O_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_O_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_DO_UID),
                 K_SDPA_BWD_TENSOR_DO_UID,
                 toVec(K_SDPA_BWD_TENSOR_DO_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_DO_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_STATS_UID),
                 K_SDPA_BWD_TENSOR_STATS_UID,
                 toVec(K_SDPA_BWD_TENSOR_STATS_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_STATS_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_DQ_UID),
                 K_SDPA_BWD_TENSOR_DQ_UID,
                 toVec(K_SDPA_BWD_TENSOR_DQ_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_DQ_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_DK_UID),
                 K_SDPA_BWD_TENSOR_DK_UID,
                 toVec(K_SDPA_BWD_TENSOR_DK_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_DK_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BWD_TENSOR_DV_UID),
                 K_SDPA_BWD_TENSOR_DV_UID,
                 toVec(K_SDPA_BWD_TENSOR_DV_DIMS),
                 toVec(K_SDPA_BWD_TENSOR_DV_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::SdpaBackwardAttributes);
    auto* attrs = graphT->nodes[0]->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    verifyRequiredTensorUids(attrs);

    // No optional tensor UIDs should be set
    EXPECT_FALSE(attrs->scale_tensor_uid.has_value());
    EXPECT_FALSE(attrs->attn_mask_tensor_uid.has_value());
    EXPECT_FALSE(attrs->seq_len_q_tensor_uid.has_value());
    EXPECT_FALSE(attrs->seq_len_kv_tensor_uid.has_value());
    EXPECT_FALSE(attrs->seed_tensor_uid.has_value());
    EXPECT_FALSE(attrs->offset_tensor_uid.has_value());
    EXPECT_FALSE(attrs->dropout_mask_tensor_uid.has_value());
    EXPECT_FALSE(attrs->dropout_scale_tensor_uid.has_value());
    EXPECT_FALSE(attrs->dropout_scale_inv_tensor_uid.has_value());
    EXPECT_FALSE(attrs->dbias_tensor_uid.has_value());

    verifyDefaultScalarAttrs(attrs);

    // Compute data type should be FLOAT
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::FLOAT);
}

// =============================================================================
// Compute data type preservation
// =============================================================================

TEST_F(TestGraphDescriptorSdpaBwd, ComputeDataTypePreserved)
{
    auto t = createRequiredTensors();

    auto opWrapper = createFinalizedSdpaBwdOp(t.q.get(),
                                              t.k.get(),
                                              t.v.get(),
                                              t.o.get(),
                                              t.dO.get(),
                                              t.stats.get(),
                                              t.dq.get(),
                                              t.dk.get(),
                                              t.dv.get(),
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              HIPDNN_DATA_HALF);

    auto graphT = finalizeGraphAndUnpack(opWrapper);

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

// =============================================================================
// Non-default scalar/enum fields survive serialization
// =============================================================================

TEST_F(TestGraphDescriptorSdpaBwd, NonDefaultScalarsPreservedInSerialization)
{
    auto t = createRequiredTensors();

    auto wrapper = createDescriptor<SdpaBwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SdpaBwdOperationDescriptor>();

    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_Q_EXT, t.q.get());
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_K_EXT, t.k.get());
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_V_EXT, t.v.get());
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_O_EXT, t.o.get());
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DO_EXT, t.dO.get());
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_STATS_EXT, t.stats.get());
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DQ_EXT, t.dq.get());
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DK_EXT, t.dk.get());
    setTensorAttr(*desc, HIPDNN_ATTR_OPERATION_SDPA_BWD_DV_EXT, t.dv.get());

    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BWD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Set non-default boolean flags
    bool trueVal = true;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BWD_ALIBI_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(HIPDNN_ATTR_SDPA_BWD_PADDING_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_BOTTOM_RIGHT_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);

    // Set non-default optional float scalars
    float dropoutProb = 0.3f;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BWD_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 1, &dropoutProb);
    float attnScale = 0.125f;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BWD_ATTN_SCALE_VALUE_EXT, HIPDNN_TYPE_FLOAT, 1, &attnScale);

    // Set non-default optional int64 scalars
    int64_t leftBound = 5;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BWD_LEFT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &leftBound);
    int64_t rightBound = 15;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BWD_RIGHT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &rightBound);

    // Set non-default diagonal alignment
    auto diagAlign = HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BWD_DIAGONAL_ALIGNMENT_EXT,
                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT,
                       1,
                       &diagAlign);

    desc->finalize();

    auto graphT = finalizeGraphAndUnpack(wrapper);
    ASSERT_NE(graphT, nullptr);
    ASSERT_EQ(graphT->nodes.size(), 1u);
    ASSERT_EQ(graphT->tensors.size(), 9u); // required only

    auto* attrs = graphT->nodes[0]->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify all non-default boolean flags
    EXPECT_TRUE(attrs->alibi_mask);
    EXPECT_TRUE(attrs->padding_mask);
    EXPECT_TRUE(attrs->causal_mask);
    EXPECT_TRUE(attrs->causal_mask_bottom_right);

    // Verify non-default optional float scalars
    ASSERT_TRUE(attrs->dropout_probability.has_value());
    EXPECT_FLOAT_EQ(attrs->dropout_probability.value(), 0.3f);
    ASSERT_TRUE(attrs->attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(attrs->attn_scale_value.value(), 0.125f);

    // Verify non-default optional int64 scalars
    ASSERT_TRUE(attrs->left_bound.has_value());
    EXPECT_EQ(attrs->left_bound.value(), 5);
    ASSERT_TRUE(attrs->right_bound.has_value());
    EXPECT_EQ(attrs->right_bound.value(), 15);

    // Verify non-default diagonal alignment
    EXPECT_EQ(attrs->diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
}

// =============================================================================
// Verify all optional tensor UIDs present in serialized tensor list
// =============================================================================

TEST_F(TestGraphDescriptorSdpaBwd, AllOptionalTensorsPresentInSerializedTensorList)
{
    auto t = createRequiredTensors();
    auto scaleDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_SCALE_UID);
    auto attnMaskDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_ATTN_MASK_UID);
    auto seqLenQDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID);
    auto seqLenKvDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID);
    auto seedDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_SEED_UID);
    auto offsetDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_OFFSET_UID);
    auto dropoutMaskDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID);
    auto dropoutScaleDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID);
    auto dropoutScaleInvDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID);
    auto dbiasDesc = createFinalizedTensor(K_SDPA_BWD_TENSOR_DBIAS_UID);

    auto opWrapper = createFinalizedSdpaBwdOp(t.q.get(),
                                              t.k.get(),
                                              t.v.get(),
                                              t.o.get(),
                                              t.dO.get(),
                                              t.stats.get(),
                                              t.dq.get(),
                                              t.dk.get(),
                                              t.dv.get(),
                                              scaleDesc.get(),
                                              attnMaskDesc.get(),
                                              seqLenQDesc.get(),
                                              seqLenKvDesc.get(),
                                              seedDesc.get(),
                                              offsetDesc.get(),
                                              dropoutMaskDesc.get(),
                                              dropoutScaleDesc.get(),
                                              dropoutScaleInvDesc.get(),
                                              dbiasDesc.get());

    auto graphT = finalizeGraphAndUnpack(opWrapper);
    ASSERT_NE(graphT, nullptr);

    // Collect all tensor UIDs from the serialized tensor list
    std::set<int64_t> tensorUids;
    for(const auto& tensor : graphT->tensors)
    {
        tensorUids.insert(tensor->uid);
    }

    // Verify all 9 required tensor UIDs are present in the tensor list
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_Q_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_K_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_V_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_O_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_DO_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_STATS_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_DQ_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_DK_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_DV_UID) > 0);

    // Verify all 10 optional tensor UIDs are present in the tensor list
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_SCALE_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_ATTN_MASK_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_SEED_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_OFFSET_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BWD_TENSOR_DBIAS_UID) > 0);
}

// =============================================================================
// Required-only: no optional tensor UIDs in tensor list
// =============================================================================

TEST_F(TestGraphDescriptorSdpaBwd, RequiredOnlyNoOptionalTensorsInList)
{
    auto t = createRequiredTensors();

    auto opWrapper = createFinalizedSdpaBwdOp(t.q.get(),
                                              t.k.get(),
                                              t.v.get(),
                                              t.o.get(),
                                              t.dO.get(),
                                              t.stats.get(),
                                              t.dq.get(),
                                              t.dk.get(),
                                              t.dv.get());

    auto graphT = finalizeGraphAndUnpack(opWrapper);
    ASSERT_NE(graphT, nullptr);

    // Collect all tensor UIDs
    std::set<int64_t> tensorUids;
    for(const auto& tensor : graphT->tensors)
    {
        tensorUids.insert(tensor->uid);
    }

    // No optional tensor UIDs should be present
    EXPECT_EQ(tensorUids.count(K_SDPA_BWD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BWD_TENSOR_ATTN_MASK_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BWD_TENSOR_SEQ_LEN_Q_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BWD_TENSOR_SEQ_LEN_KV_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BWD_TENSOR_SEED_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BWD_TENSOR_OFFSET_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BWD_TENSOR_DROPOUT_MASK_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BWD_TENSOR_DROPOUT_SCALE_INV_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BWD_TENSOR_DBIAS_UID), 0u);
}

} // namespace
