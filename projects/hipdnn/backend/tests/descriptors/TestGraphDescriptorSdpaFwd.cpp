// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/SdpaFwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SdpaFwdConstants.hpp>
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

// Helper: create a finalized SdpaFwdOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedSdpaFwdOp(HipdnnBackendDescriptor* qDesc,
                             HipdnnBackendDescriptor* kDesc,
                             HipdnnBackendDescriptor* vDesc,
                             HipdnnBackendDescriptor* oDesc,
                             HipdnnBackendDescriptor* attnMaskDesc,
                             HipdnnBackendDescriptor* scaleDesc,
                             HipdnnBackendDescriptor* seqLenQDesc,
                             HipdnnBackendDescriptor* seqLenKvDesc,
                             HipdnnBackendDescriptor* seedDesc,
                             HipdnnBackendDescriptor* offsetDesc,
                             HipdnnBackendDescriptor* dropoutMaskDesc,
                             HipdnnBackendDescriptor* dropoutScaleDesc,
                             HipdnnBackendDescriptor* pageTableKDesc,
                             HipdnnBackendDescriptor* pageTableVDesc,
                             HipdnnBackendDescriptor* blockMaskDesc,
                             HipdnnBackendDescriptor* sinkTokenDesc,
                             HipdnnBackendDescriptor* descaleQDesc,
                             HipdnnBackendDescriptor* descaleKDesc,
                             HipdnnBackendDescriptor* descaleVDesc,
                             HipdnnBackendDescriptor* descaleSDesc,
                             HipdnnBackendDescriptor* scaleSDesc,
                             HipdnnBackendDescriptor* scaleODesc,
                             HipdnnBackendDescriptor* statsDesc,
                             HipdnnBackendDescriptor* maxDesc,
                             HipdnnBackendDescriptor* sumExpDesc,
                             HipdnnBackendDescriptor* rngDumpDesc,
                             HipdnnBackendDescriptor* amaxSDesc,
                             HipdnnBackendDescriptor* amaxODesc,
                             hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                             const std::string& name = "")
{
    auto wrapper = createDescriptor<SdpaFwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SdpaFwdOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_QDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&qDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_KDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&kDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_VDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&vDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_ODESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&oDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_ATTN_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&attnMaskDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&seqLenQDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&seqLenKvDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_SEED_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&seedDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_OFFSET_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&offsetDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dropoutMaskDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dropoutScaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&pageTableKDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&pageTableVDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&blockMaskDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_SINK_TOKEN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&sinkTokenDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&descaleQDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&descaleKDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&descaleVDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&descaleSDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleSDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALE_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleODesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&statsDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_MAX_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&maxDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_SUM_EXP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&sumExpDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_RNG_DUMP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&rngDumpDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_AMAX_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&amaxSDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_AMAX_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&amaxODesc));
    desc->setAttribute(HIPDNN_ATTR_SDPA_FWD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    if(!name.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                           HIPDNN_TYPE_CHAR,
                           static_cast<int64_t>(name.size()),
                           name.c_str());
    }

    desc->finalize();
    return wrapper;
}

inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedSdpaFwdOpRequiredOnly(HipdnnBackendDescriptor* qDesc,
                                         HipdnnBackendDescriptor* kDesc,
                                         HipdnnBackendDescriptor* vDesc,
                                         HipdnnBackendDescriptor* oDesc,
                                         hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                         const std::string& name = "")
{
    auto wrapper = createDescriptor<SdpaFwdOperationDescriptor>();
    auto desc = wrapper->asDescriptor<SdpaFwdOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_QDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&qDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_KDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&kDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_VDESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&vDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FWD_ODESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&oDesc));
    desc->setAttribute(HIPDNN_ATTR_SDPA_FWD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    if(!name.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                           HIPDNN_TYPE_CHAR,
                           static_cast<int64_t>(name.size()),
                           name.c_str());
    }

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorSdpaFwd : public ::testing::Test
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

TEST_F(TestGraphDescriptorSdpaFwd, BuildFromSingleOperation)
{
    auto qDesc = createFinalizedTensor(
        K_SDPA_TENSOR_Q_UID, toVec(K_SDPA_TENSOR_Q_DIMS), toVec(K_SDPA_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(
        K_SDPA_TENSOR_K_UID, toVec(K_SDPA_TENSOR_K_DIMS), toVec(K_SDPA_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(
        K_SDPA_TENSOR_V_UID, toVec(K_SDPA_TENSOR_V_DIMS), toVec(K_SDPA_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(
        K_SDPA_TENSOR_O_UID, toVec(K_SDPA_TENSOR_O_DIMS), toVec(K_SDPA_TENSOR_O_STRIDES));
    auto attnMaskDesc = createFinalizedTensor(K_SDPA_TENSOR_ATTN_MASK_UID);
    auto scaleDesc = createFinalizedTensor(K_SDPA_TENSOR_SCALE_UID);
    auto seqLenQDesc = createFinalizedTensor(K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    auto seqLenKvDesc = createFinalizedTensor(K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    auto seedDesc = createFinalizedTensor(K_SDPA_TENSOR_SEED_UID);
    auto offsetDesc = createFinalizedTensor(K_SDPA_TENSOR_OFFSET_UID);
    auto dropoutMaskDesc = createFinalizedTensor(K_SDPA_TENSOR_DROPOUT_MASK_UID);
    auto dropoutScaleDesc = createFinalizedTensor(K_SDPA_TENSOR_DROPOUT_SCALE_UID);
    auto pageTableKDesc = createFinalizedTensor(K_SDPA_TENSOR_PAGE_TABLE_K_UID);
    auto pageTableVDesc = createFinalizedTensor(K_SDPA_TENSOR_PAGE_TABLE_V_UID);
    auto blockMaskDesc = createFinalizedTensor(K_SDPA_TENSOR_BLOCK_MASK_UID);
    auto sinkTokenDesc = createFinalizedTensor(K_SDPA_TENSOR_SINK_TOKEN_UID);
    auto descaleQDesc = createFinalizedTensor(K_SDPA_TENSOR_DESCALE_Q_UID);
    auto descaleKDesc = createFinalizedTensor(K_SDPA_TENSOR_DESCALE_K_UID);
    auto descaleVDesc = createFinalizedTensor(K_SDPA_TENSOR_DESCALE_V_UID);
    auto descaleSDesc = createFinalizedTensor(K_SDPA_TENSOR_DESCALE_S_UID);
    auto scaleSDesc = createFinalizedTensor(K_SDPA_TENSOR_SCALE_S_UID);
    auto scaleODesc = createFinalizedTensor(K_SDPA_TENSOR_SCALE_O_UID);
    auto statsDesc = createFinalizedTensor(K_SDPA_TENSOR_STATS_UID);
    auto maxDesc = createFinalizedTensor(K_SDPA_TENSOR_MAX_UID);
    auto sumExpDesc = createFinalizedTensor(K_SDPA_TENSOR_SUM_EXP_UID);
    auto rngDumpDesc = createFinalizedTensor(K_SDPA_TENSOR_RNG_DUMP_UID);
    auto amaxSDesc = createFinalizedTensor(K_SDPA_TENSOR_AMAX_S_UID);
    auto amaxODesc = createFinalizedTensor(K_SDPA_TENSOR_AMAX_O_UID);
    auto opDesc = createFinalizedSdpaFwdOp(qDesc.get(),
                                           kDesc.get(),
                                           vDesc.get(),
                                           oDesc.get(),
                                           attnMaskDesc.get(),
                                           scaleDesc.get(),
                                           seqLenQDesc.get(),
                                           seqLenKvDesc.get(),
                                           seedDesc.get(),
                                           offsetDesc.get(),
                                           dropoutMaskDesc.get(),
                                           dropoutScaleDesc.get(),
                                           pageTableKDesc.get(),
                                           pageTableVDesc.get(),
                                           blockMaskDesc.get(),
                                           sinkTokenDesc.get(),
                                           descaleQDesc.get(),
                                           descaleKDesc.get(),
                                           descaleVDesc.get(),
                                           descaleSDesc.get(),
                                           scaleSDesc.get(),
                                           scaleODesc.get(),
                                           statsDesc.get(),
                                           maxDesc.get(),
                                           sumExpDesc.get(),
                                           rngDumpDesc.get(),
                                           amaxSDesc.get(),
                                           amaxODesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    // Verify the built graph
    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 28);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::SdpaAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsSdpaAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->q_tensor_uid, K_SDPA_TENSOR_Q_UID);
    EXPECT_EQ(attrs->k_tensor_uid, K_SDPA_TENSOR_K_UID);
    EXPECT_EQ(attrs->v_tensor_uid, K_SDPA_TENSOR_V_UID);
    EXPECT_EQ(attrs->o_tensor_uid, K_SDPA_TENSOR_O_UID);
    EXPECT_EQ(attrs->attn_mask_tensor_uid, K_SDPA_TENSOR_ATTN_MASK_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_SDPA_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->seq_len_q_tensor_uid, K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    EXPECT_EQ(attrs->seq_len_kv_tensor_uid, K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    EXPECT_EQ(attrs->seed_tensor_uid, K_SDPA_TENSOR_SEED_UID);
    EXPECT_EQ(attrs->offset_tensor_uid, K_SDPA_TENSOR_OFFSET_UID);
    EXPECT_EQ(attrs->dropout_mask_tensor_uid, K_SDPA_TENSOR_DROPOUT_MASK_UID);
    EXPECT_EQ(attrs->dropout_scale_tensor_uid, K_SDPA_TENSOR_DROPOUT_SCALE_UID);
    EXPECT_EQ(attrs->page_table_k_tensor_uid, K_SDPA_TENSOR_PAGE_TABLE_K_UID);
    EXPECT_EQ(attrs->page_table_v_tensor_uid, K_SDPA_TENSOR_PAGE_TABLE_V_UID);
    EXPECT_EQ(attrs->block_mask_tensor_uid, K_SDPA_TENSOR_BLOCK_MASK_UID);
    EXPECT_EQ(attrs->sink_token_tensor_uid, K_SDPA_TENSOR_SINK_TOKEN_UID);
    EXPECT_EQ(attrs->descale_q_tensor_uid, K_SDPA_TENSOR_DESCALE_Q_UID);
    EXPECT_EQ(attrs->descale_k_tensor_uid, K_SDPA_TENSOR_DESCALE_K_UID);
    EXPECT_EQ(attrs->descale_v_tensor_uid, K_SDPA_TENSOR_DESCALE_V_UID);
    EXPECT_EQ(attrs->descale_s_tensor_uid, K_SDPA_TENSOR_DESCALE_S_UID);
    EXPECT_EQ(attrs->scale_s_tensor_uid, K_SDPA_TENSOR_SCALE_S_UID);
    EXPECT_EQ(attrs->scale_o_tensor_uid, K_SDPA_TENSOR_SCALE_O_UID);
    EXPECT_EQ(attrs->stats_tensor_uid, K_SDPA_TENSOR_STATS_UID);
    EXPECT_EQ(attrs->max_tensor_uid, K_SDPA_TENSOR_MAX_UID);
    EXPECT_EQ(attrs->sum_exp_tensor_uid, K_SDPA_TENSOR_SUM_EXP_UID);
    EXPECT_EQ(attrs->rng_dump_tensor_uid, K_SDPA_TENSOR_RNG_DUMP_UID);
    EXPECT_EQ(attrs->amax_s_tensor_uid, K_SDPA_TENSOR_AMAX_S_UID);
    EXPECT_EQ(attrs->amax_o_tensor_uid, K_SDPA_TENSOR_AMAX_O_UID);

    // Verify tensor attributes survive serialization (dims, strides, data_type, virtual)
    verifyTensor(findTensorByUid(*graphT, K_SDPA_TENSOR_Q_UID),
                 K_SDPA_TENSOR_Q_UID,
                 toVec(K_SDPA_TENSOR_Q_DIMS),
                 toVec(K_SDPA_TENSOR_Q_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_TENSOR_K_UID),
                 K_SDPA_TENSOR_K_UID,
                 toVec(K_SDPA_TENSOR_K_DIMS),
                 toVec(K_SDPA_TENSOR_K_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_TENSOR_V_UID),
                 K_SDPA_TENSOR_V_UID,
                 toVec(K_SDPA_TENSOR_V_DIMS),
                 toVec(K_SDPA_TENSOR_V_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_TENSOR_O_UID),
                 K_SDPA_TENSOR_O_UID,
                 toVec(K_SDPA_TENSOR_O_DIMS),
                 toVec(K_SDPA_TENSOR_O_STRIDES),
                 DataType::FLOAT);

    // Verify scalar/enum SDPA attributes survive serialization
    EXPECT_EQ(attrs->diagonal_alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(attrs->mma_core_mode, DataType::UNSET);
    EXPECT_EQ(attrs->implementation, AttentionImplementation::AUTO);
    EXPECT_EQ(attrs->alibi_mask, false);
    EXPECT_EQ(attrs->padding_mask, false);
    EXPECT_EQ(attrs->causal_mask, false);
    EXPECT_EQ(attrs->causal_mask_bottom_right, false);
}

TEST_F(TestGraphDescriptorSdpaFwd, ComputeDataTypePreserved)
{
    auto qDesc = createFinalizedTensor(
        K_SDPA_TENSOR_Q_UID, toVec(K_SDPA_TENSOR_Q_DIMS), toVec(K_SDPA_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(
        K_SDPA_TENSOR_K_UID, toVec(K_SDPA_TENSOR_K_DIMS), toVec(K_SDPA_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(
        K_SDPA_TENSOR_V_UID, toVec(K_SDPA_TENSOR_V_DIMS), toVec(K_SDPA_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(
        K_SDPA_TENSOR_O_UID, toVec(K_SDPA_TENSOR_O_DIMS), toVec(K_SDPA_TENSOR_O_STRIDES));
    auto opDesc = createFinalizedSdpaFwdOpRequiredOnly(
        qDesc.get(), kDesc.get(), vDesc.get(), oDesc.get(), HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

TEST_F(TestGraphDescriptorSdpaFwd, BuildFromRequiredTensorsOnly)
{
    auto qDesc = createFinalizedTensor(
        K_SDPA_TENSOR_Q_UID, toVec(K_SDPA_TENSOR_Q_DIMS), toVec(K_SDPA_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(
        K_SDPA_TENSOR_K_UID, toVec(K_SDPA_TENSOR_K_DIMS), toVec(K_SDPA_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(
        K_SDPA_TENSOR_V_UID, toVec(K_SDPA_TENSOR_V_DIMS), toVec(K_SDPA_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(
        K_SDPA_TENSOR_O_UID, toVec(K_SDPA_TENSOR_O_DIMS), toVec(K_SDPA_TENSOR_O_STRIDES));

    auto opDesc
        = createFinalizedSdpaFwdOpRequiredOnly(qDesc.get(), kDesc.get(), vDesc.get(), oDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graphT = UnPackGraph(serialized.ptr);

    // Only 4 required tensors should be present
    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 4);

    // Verify tensor attributes
    verifyTensor(findTensorByUid(*graphT, K_SDPA_TENSOR_Q_UID),
                 K_SDPA_TENSOR_Q_UID,
                 toVec(K_SDPA_TENSOR_Q_DIMS),
                 toVec(K_SDPA_TENSOR_Q_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_TENSOR_K_UID),
                 K_SDPA_TENSOR_K_UID,
                 toVec(K_SDPA_TENSOR_K_DIMS),
                 toVec(K_SDPA_TENSOR_K_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_TENSOR_V_UID),
                 K_SDPA_TENSOR_V_UID,
                 toVec(K_SDPA_TENSOR_V_DIMS),
                 toVec(K_SDPA_TENSOR_V_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_SDPA_TENSOR_O_UID),
                 K_SDPA_TENSOR_O_UID,
                 toVec(K_SDPA_TENSOR_O_DIMS),
                 toVec(K_SDPA_TENSOR_O_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::SdpaAttributes);
    auto* attrs = graphT->nodes[0]->attributes.AsSdpaAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_EQ(attrs->q_tensor_uid, K_SDPA_TENSOR_Q_UID);
    EXPECT_EQ(attrs->k_tensor_uid, K_SDPA_TENSOR_K_UID);
    EXPECT_EQ(attrs->v_tensor_uid, K_SDPA_TENSOR_V_UID);
    EXPECT_EQ(attrs->o_tensor_uid, K_SDPA_TENSOR_O_UID);

    // Optional tensor UIDs should not be set
    EXPECT_FALSE(attrs->attn_mask_tensor_uid.has_value());
    EXPECT_FALSE(attrs->scale_tensor_uid.has_value());
    EXPECT_FALSE(attrs->seq_len_q_tensor_uid.has_value());
    EXPECT_FALSE(attrs->seq_len_kv_tensor_uid.has_value());
    EXPECT_FALSE(attrs->seed_tensor_uid.has_value());
    EXPECT_FALSE(attrs->offset_tensor_uid.has_value());
    EXPECT_FALSE(attrs->dropout_mask_tensor_uid.has_value());
    EXPECT_FALSE(attrs->dropout_scale_tensor_uid.has_value());

    // Verify default scalar/enum values
    EXPECT_EQ(attrs->diagonal_alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_EQ(attrs->mma_core_mode, DataType::UNSET);
    EXPECT_EQ(attrs->implementation, AttentionImplementation::AUTO);
    EXPECT_EQ(attrs->alibi_mask, false);
    EXPECT_EQ(attrs->padding_mask, false);
    EXPECT_EQ(attrs->causal_mask, false);
    EXPECT_EQ(attrs->causal_mask_bottom_right, false);

    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::FLOAT);
}

TEST_F(TestGraphDescriptorSdpaFwd, OperationNamePreservedInSerialization)
{
    auto qDesc = createFinalizedTensor(
        K_SDPA_TENSOR_Q_UID, toVec(K_SDPA_TENSOR_Q_DIMS), toVec(K_SDPA_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(
        K_SDPA_TENSOR_K_UID, toVec(K_SDPA_TENSOR_K_DIMS), toVec(K_SDPA_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(
        K_SDPA_TENSOR_V_UID, toVec(K_SDPA_TENSOR_V_DIMS), toVec(K_SDPA_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(
        K_SDPA_TENSOR_O_UID, toVec(K_SDPA_TENSOR_O_DIMS), toVec(K_SDPA_TENSOR_O_STRIDES));

    auto opDesc = createFinalizedSdpaFwdOpRequiredOnly(
        qDesc.get(), kDesc.get(), vDesc.get(), oDesc.get(), HIPDNN_DATA_FLOAT, "my_sdpa_op");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->name, "my_sdpa_op");
}

TEST_F(TestGraphDescriptorSdpaFwd, OperationNameRoundTripThroughLifting)
{
    auto qDesc = createFinalizedTensor(
        K_SDPA_TENSOR_Q_UID, toVec(K_SDPA_TENSOR_Q_DIMS), toVec(K_SDPA_TENSOR_Q_STRIDES));
    auto kDesc = createFinalizedTensor(
        K_SDPA_TENSOR_K_UID, toVec(K_SDPA_TENSOR_K_DIMS), toVec(K_SDPA_TENSOR_K_STRIDES));
    auto vDesc = createFinalizedTensor(
        K_SDPA_TENSOR_V_UID, toVec(K_SDPA_TENSOR_V_DIMS), toVec(K_SDPA_TENSOR_V_STRIDES));
    auto oDesc = createFinalizedTensor(
        K_SDPA_TENSOR_O_UID, toVec(K_SDPA_TENSOR_O_DIMS), toVec(K_SDPA_TENSOR_O_STRIDES));

    auto opDesc = createFinalizedSdpaFwdOpRequiredOnly(
        qDesc.get(), kDesc.get(), vDesc.get(), oDesc.get(), HIPDNN_DATA_FLOAT, "round_trip_name");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Serialize and deserialize via FlatBuffer
    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    // Rebuild from the deserialized graph using NodeFactory
    auto tensorMap = NodeFactory::buildTensorMap(graphT->tensors);
    ASSERT_EQ(graphT->nodes.size(), 1);

    auto rebuilt = NodeFactory::createOperationFromNode(*graphT->nodes[0], tensorMap);
    ASSERT_NE(rebuilt, nullptr);

    auto* graphOp = rebuilt->asGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the name survived the round-trip by building a node from the rebuilt operation
    auto rebuiltNode = graphOp->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "round_trip_name");
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::SdpaAttributes);
}

} // namespace
