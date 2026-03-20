// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/SdpaBpropOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SdpaBpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized SdpaBpropOperationDescriptor with all tensors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedSdpaBpropOp(HipdnnBackendDescriptor* qDesc,
                               HipdnnBackendDescriptor* kDesc,
                               HipdnnBackendDescriptor* vDesc,
                               HipdnnBackendDescriptor* oDesc,
                               HipdnnBackendDescriptor* doDesc,
                               HipdnnBackendDescriptor* statsDesc,
                               HipdnnBackendDescriptor* dqDesc,
                               HipdnnBackendDescriptor* dkDesc,
                               HipdnnBackendDescriptor* dvDesc,
                               HipdnnBackendDescriptor* scaleDesc,
                               HipdnnBackendDescriptor* attnMaskDesc,
                               HipdnnBackendDescriptor* seqLenQDesc,
                               HipdnnBackendDescriptor* seqLenKvDesc,
                               HipdnnBackendDescriptor* seedDesc,
                               HipdnnBackendDescriptor* offsetDesc,
                               HipdnnBackendDescriptor* dropoutMaskDesc,
                               HipdnnBackendDescriptor* dropoutScaleDesc,
                               HipdnnBackendDescriptor* dropoutScaleInvDesc,
                               HipdnnBackendDescriptor* dbiasDesc,
                               hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<SdpaBpropOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SdpaBpropOperationDescriptor>();

    // Required input tensors
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&qDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&kDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&vDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&oDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&doDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&statsDesc));

    // Required output tensors
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dqDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dkDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dvDesc));

    // Optional tensors
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_ATTN_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&attnMaskDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEQ_LEN_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&seqLenQDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEQ_LEN_KV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&seqLenKvDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEED_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&seedDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_OFFSET_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&offsetDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dropoutMaskDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dropoutScaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_SCALE_INV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dropoutScaleInvDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DBIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dbiasDesc));
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedSdpaBpropOpRequiredOnly(HipdnnBackendDescriptor* qDesc,
                                           HipdnnBackendDescriptor* kDesc,
                                           HipdnnBackendDescriptor* vDesc,
                                           HipdnnBackendDescriptor* oDesc,
                                           HipdnnBackendDescriptor* doDesc,
                                           HipdnnBackendDescriptor* statsDesc,
                                           HipdnnBackendDescriptor* dqDesc,
                                           HipdnnBackendDescriptor* dkDesc,
                                           HipdnnBackendDescriptor* dvDesc,
                                           hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<SdpaBpropOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SdpaBpropOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&qDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&kDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&vDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&oDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&doDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&statsDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dqDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dkDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dvDesc));
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorSdpaBprop : public ::testing::Test
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

TEST_F(TestGraphDescriptorSdpaBprop, BuildFromSingleOperation)
{
    auto qDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_Q_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_K_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_V_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_O_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
    auto doDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
    auto statsDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
    auto dqDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
    auto dkDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
    auto dvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SCALE_UID);
    auto attnMaskDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);
    auto seqLenQDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID);
    auto seqLenKvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID);
    auto seedDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEED_UID);
    auto offsetDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_OFFSET_UID);
    auto dropoutMaskDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID);
    auto dropoutScaleDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID);
    auto dropoutScaleInvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID);
    auto dbiasDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DBIAS_UID);

    auto opWrapper = createFinalizedSdpaBpropOp(qDesc.get(),
                                                kDesc.get(),
                                                vDesc.get(),
                                                oDesc.get(),
                                                doDesc.get(),
                                                statsDesc.get(),
                                                dqDesc.get(),
                                                dkDesc.get(),
                                                dvDesc.get(),
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

    // Verify all 9 required tensor UIDs
    EXPECT_EQ(attrs->q_tensor_uid, K_SDPA_BPROP_TENSOR_Q_UID);
    EXPECT_EQ(attrs->k_tensor_uid, K_SDPA_BPROP_TENSOR_K_UID);
    EXPECT_EQ(attrs->v_tensor_uid, K_SDPA_BPROP_TENSOR_V_UID);
    EXPECT_EQ(attrs->o_tensor_uid, K_SDPA_BPROP_TENSOR_O_UID);
    EXPECT_EQ(attrs->do_tensor_uid, K_SDPA_BPROP_TENSOR_DO_UID);
    EXPECT_EQ(attrs->stats_tensor_uid, K_SDPA_BPROP_TENSOR_STATS_UID);
    EXPECT_EQ(attrs->dq_tensor_uid, K_SDPA_BPROP_TENSOR_DQ_UID);
    EXPECT_EQ(attrs->dk_tensor_uid, K_SDPA_BPROP_TENSOR_DK_UID);
    EXPECT_EQ(attrs->dv_tensor_uid, K_SDPA_BPROP_TENSOR_DV_UID);

    // Verify all 10 optional tensor UIDs
    ASSERT_TRUE(attrs->scale_tensor_uid.has_value());
    EXPECT_EQ(attrs->scale_tensor_uid.value(), K_SDPA_BPROP_TENSOR_SCALE_UID);

    ASSERT_TRUE(attrs->attn_mask_tensor_uid.has_value());
    EXPECT_EQ(attrs->attn_mask_tensor_uid.value(), K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);

    ASSERT_TRUE(attrs->seq_len_q_tensor_uid.has_value());
    EXPECT_EQ(attrs->seq_len_q_tensor_uid.value(), K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID);

    ASSERT_TRUE(attrs->seq_len_kv_tensor_uid.has_value());
    EXPECT_EQ(attrs->seq_len_kv_tensor_uid.value(), K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID);

    ASSERT_TRUE(attrs->seed_tensor_uid.has_value());
    EXPECT_EQ(attrs->seed_tensor_uid.value(), K_SDPA_BPROP_TENSOR_SEED_UID);

    ASSERT_TRUE(attrs->offset_tensor_uid.has_value());
    EXPECT_EQ(attrs->offset_tensor_uid.value(), K_SDPA_BPROP_TENSOR_OFFSET_UID);

    ASSERT_TRUE(attrs->dropout_mask_tensor_uid.has_value());
    EXPECT_EQ(attrs->dropout_mask_tensor_uid.value(), K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID);

    ASSERT_TRUE(attrs->dropout_scale_tensor_uid.has_value());
    EXPECT_EQ(attrs->dropout_scale_tensor_uid.value(), K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID);

    ASSERT_TRUE(attrs->dropout_scale_inv_tensor_uid.has_value());
    EXPECT_EQ(attrs->dropout_scale_inv_tensor_uid.value(),
              K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID);

    ASSERT_TRUE(attrs->dbias_tensor_uid.has_value());
    EXPECT_EQ(attrs->dbias_tensor_uid.value(), K_SDPA_BPROP_TENSOR_DBIAS_UID);

    // Verify tensor attributes survive serialization for required tensors
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_Q_UID),
                 K_SDPA_BPROP_TENSOR_Q_UID,
                 toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_K_UID),
                 K_SDPA_BPROP_TENSOR_K_UID,
                 toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_K_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_V_UID),
                 K_SDPA_BPROP_TENSOR_V_UID,
                 toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_V_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_O_UID),
                 K_SDPA_BPROP_TENSOR_O_UID,
                 toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_O_STRIDES),
                 DataType::FLOAT);

    // Verify default scalar/enum attributes survive serialization
    EXPECT_EQ(attrs->diagonal_alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(attrs->alibi_mask, false);
    EXPECT_EQ(attrs->padding_mask, false);
    EXPECT_EQ(attrs->causal_mask, false);
    EXPECT_EQ(attrs->causal_mask_bottom_right, false);

    // Optional scalars should not be set (defaults)
    EXPECT_FALSE(attrs->dropout_probability.has_value());
    EXPECT_FALSE(attrs->attn_scale_value.has_value());
    EXPECT_FALSE(attrs->left_bound.has_value());
    EXPECT_FALSE(attrs->right_bound.has_value());
}

// =============================================================================
// Required-only serialization test
// =============================================================================

TEST_F(TestGraphDescriptorSdpaBprop, BuildFromRequiredOnlyOperation)
{
    auto qDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_Q_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_K_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_V_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_O_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
    auto doDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
    auto statsDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
    auto dqDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
    auto dkDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
    auto dvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));

    auto opWrapper = createFinalizedSdpaBpropOpRequiredOnly(qDesc.get(),
                                                            kDesc.get(),
                                                            vDesc.get(),
                                                            oDesc.get(),
                                                            doDesc.get(),
                                                            statsDesc.get(),
                                                            dqDesc.get(),
                                                            dkDesc.get(),
                                                            dvDesc.get());

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
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_Q_UID),
                 K_SDPA_BPROP_TENSOR_Q_UID,
                 toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_K_UID),
                 K_SDPA_BPROP_TENSOR_K_UID,
                 toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_K_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_V_UID),
                 K_SDPA_BPROP_TENSOR_V_UID,
                 toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_V_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_O_UID),
                 K_SDPA_BPROP_TENSOR_O_UID,
                 toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_O_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_DO_UID),
                 K_SDPA_BPROP_TENSOR_DO_UID,
                 toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_STATS_UID),
                 K_SDPA_BPROP_TENSOR_STATS_UID,
                 toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_DQ_UID),
                 K_SDPA_BPROP_TENSOR_DQ_UID,
                 toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_DK_UID),
                 K_SDPA_BPROP_TENSOR_DK_UID,
                 toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_BPROP_TENSOR_DV_UID),
                 K_SDPA_BPROP_TENSOR_DV_UID,
                 toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                 toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::SdpaBackwardAttributes);
    auto* attrs = graphT->nodes[0]->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Required tensor UIDs
    EXPECT_EQ(attrs->q_tensor_uid, K_SDPA_BPROP_TENSOR_Q_UID);
    EXPECT_EQ(attrs->k_tensor_uid, K_SDPA_BPROP_TENSOR_K_UID);
    EXPECT_EQ(attrs->v_tensor_uid, K_SDPA_BPROP_TENSOR_V_UID);
    EXPECT_EQ(attrs->o_tensor_uid, K_SDPA_BPROP_TENSOR_O_UID);
    EXPECT_EQ(attrs->do_tensor_uid, K_SDPA_BPROP_TENSOR_DO_UID);
    EXPECT_EQ(attrs->stats_tensor_uid, K_SDPA_BPROP_TENSOR_STATS_UID);
    EXPECT_EQ(attrs->dq_tensor_uid, K_SDPA_BPROP_TENSOR_DQ_UID);
    EXPECT_EQ(attrs->dk_tensor_uid, K_SDPA_BPROP_TENSOR_DK_UID);
    EXPECT_EQ(attrs->dv_tensor_uid, K_SDPA_BPROP_TENSOR_DV_UID);

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

    // Verify default scalar/enum values
    EXPECT_EQ(attrs->diagonal_alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(attrs->alibi_mask, false);
    EXPECT_EQ(attrs->padding_mask, false);
    EXPECT_EQ(attrs->causal_mask, false);
    EXPECT_EQ(attrs->causal_mask_bottom_right, false);

    // Optional scalar fields should have default (unset) values
    EXPECT_FALSE(attrs->dropout_probability.has_value());
    EXPECT_FALSE(attrs->attn_scale_value.has_value());
    EXPECT_FALSE(attrs->left_bound.has_value());
    EXPECT_FALSE(attrs->right_bound.has_value());

    // Compute data type should be FLOAT
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::FLOAT);
}

// =============================================================================
// Compute data type preservation
// =============================================================================

TEST_F(TestGraphDescriptorSdpaBprop, ComputeDataTypePreserved)
{
    auto qDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_Q_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_K_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_V_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_O_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
    auto doDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
    auto statsDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
    auto dqDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
    auto dkDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
    auto dvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));

    auto opWrapper = createFinalizedSdpaBpropOpRequiredOnly(qDesc.get(),
                                                            kDesc.get(),
                                                            vDesc.get(),
                                                            oDesc.get(),
                                                            doDesc.get(),
                                                            statsDesc.get(),
                                                            dqDesc.get(),
                                                            dkDesc.get(),
                                                            dvDesc.get(),
                                                            HIPDNN_DATA_HALF);

    auto graphDesc = getDescriptor();
    setHandle();

    auto* opDescPtr = opWrapper.get();
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(&opDescPtr));
    graphDesc->finalize();

    auto serialized = graphDesc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

// =============================================================================
// Non-default scalar/enum fields survive serialization
// =============================================================================

TEST_F(TestGraphDescriptorSdpaBprop, NonDefaultScalarsPreservedInSerialization)
{
    auto qDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_Q_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_K_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_V_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_O_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
    auto doDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
    auto statsDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
    auto dqDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
    auto dkDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
    auto dvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));

    // Build op with non-default scalar/enum values
    auto wrapper = createDescriptor<SdpaBpropOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SdpaBpropOperationDescriptor>();

    // Required tensors
    auto* q = qDesc.get();
    auto* k = kDesc.get();
    auto* v = vDesc.get();
    auto* o = oDesc.get();
    auto* dOp = doDesc.get();
    auto* stats = statsDesc.get();
    auto* dq = dqDesc.get();
    auto* dk = dkDesc.get();
    auto* dv = dvDesc.get();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&q));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&k));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&v));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&o));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dOp));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&stats));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dq));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dk));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dv));

    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Set non-default boolean flags
    bool trueVal = true;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_ALIBI_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_PADDING_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_BOTTOM_RIGHT_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);

    // Set non-default optional float scalars
    float dropoutProb = 0.3f;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 1, &dropoutProb);
    float attnScale = 0.125f;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_ATTN_SCALE_VALUE_EXT, HIPDNN_TYPE_FLOAT, 1, &attnScale);

    // Set non-default optional int64 scalars
    int64_t leftBound = 5;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_LEFT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &leftBound);
    int64_t rightBound = 15;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_RIGHT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &rightBound);

    // Set non-default diagonal alignment
    auto diagAlign = HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_DIAGONAL_ALIGNMENT_EXT,
                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                       1,
                       &diagAlign);

    desc->finalize();

    auto graphDesc = getDescriptor();
    setHandle();

    auto* opDescPtr = wrapper.get();
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(&opDescPtr));
    ASSERT_NO_THROW(graphDesc->finalize());

    auto serialized = graphDesc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);

    auto graphT = UnPackGraph(serialized.ptr);
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

TEST_F(TestGraphDescriptorSdpaBprop, AllOptionalTensorsPresentInSerializedTensorList)
{
    auto qDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_Q_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_K_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_V_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_O_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
    auto doDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
    auto statsDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
    auto dqDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
    auto dkDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
    auto dvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SCALE_UID);
    auto attnMaskDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);
    auto seqLenQDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID);
    auto seqLenKvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID);
    auto seedDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEED_UID);
    auto offsetDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_OFFSET_UID);
    auto dropoutMaskDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID);
    auto dropoutScaleDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID);
    auto dropoutScaleInvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID);
    auto dbiasDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DBIAS_UID);

    auto opWrapper = createFinalizedSdpaBpropOp(qDesc.get(),
                                                kDesc.get(),
                                                vDesc.get(),
                                                oDesc.get(),
                                                doDesc.get(),
                                                statsDesc.get(),
                                                dqDesc.get(),
                                                dkDesc.get(),
                                                dvDesc.get(),
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
    graphDesc->finalize();

    auto serialized = graphDesc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_NE(graphT, nullptr);

    // Collect all tensor UIDs from the serialized tensor list
    std::set<int64_t> tensorUids;
    for(const auto& tensor : graphT->tensors)
    {
        tensorUids.insert(tensor->uid);
    }

    // Verify all 9 required tensor UIDs are present in the tensor list
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_Q_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_K_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_V_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_O_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_DO_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_STATS_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_DQ_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_DK_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_DV_UID) > 0);

    // Verify all 10 optional tensor UIDs are present in the tensor list
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_SCALE_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_SEED_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_OFFSET_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID) > 0);
    EXPECT_TRUE(tensorUids.count(K_SDPA_BPROP_TENSOR_DBIAS_UID) > 0);
}

// =============================================================================
// Required-only: no optional tensor UIDs in tensor list
// =============================================================================

TEST_F(TestGraphDescriptorSdpaBprop, RequiredOnlyNoOptionalTensorsInList)
{
    auto qDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_Q_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_K_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_V_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_O_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
    auto doDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
    auto statsDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
    auto dqDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
    auto dkDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
    auto dvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));

    auto opWrapper = createFinalizedSdpaBpropOpRequiredOnly(qDesc.get(),
                                                            kDesc.get(),
                                                            vDesc.get(),
                                                            oDesc.get(),
                                                            doDesc.get(),
                                                            statsDesc.get(),
                                                            dqDesc.get(),
                                                            dkDesc.get(),
                                                            dvDesc.get());

    auto graphDesc = getDescriptor();
    setHandle();

    auto* opDescPtr = opWrapper.get();
    graphDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                            HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                            1,
                            static_cast<const void*>(&opDescPtr));
    graphDesc->finalize();

    auto serialized = graphDesc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);
    ASSERT_NE(graphT, nullptr);

    // Collect all tensor UIDs
    std::set<int64_t> tensorUids;
    for(const auto& tensor : graphT->tensors)
    {
        tensorUids.insert(tensor->uid);
    }

    // No optional tensor UIDs should be present
    EXPECT_EQ(tensorUids.count(K_SDPA_BPROP_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BPROP_TENSOR_SEED_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BPROP_TENSOR_OFFSET_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID), 0u);
    EXPECT_EQ(tensorUids.count(K_SDPA_BPROP_TENSOR_DBIAS_UID), 0u);
}

} // namespace
