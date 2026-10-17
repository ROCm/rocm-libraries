// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/KdaOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/kda_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/KdaConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized KdaOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedKdaOp(HipdnnBackendDescriptor* qDesc,
                         HipdnnBackendDescriptor* kDesc,
                         HipdnnBackendDescriptor* vDesc,
                         HipdnnBackendDescriptor* gDesc,
                         HipdnnBackendDescriptor* betaDesc,
                         HipdnnBackendDescriptor* cuSeqlensDesc,
                         HipdnnBackendDescriptor* initialStateDesc,
                         HipdnnBackendDescriptor* aLogDesc,
                         HipdnnBackendDescriptor* dtBiasDesc,
                         HipdnnBackendDescriptor* stateIndicesDesc,
                         HipdnnBackendDescriptor* oDesc,
                         HipdnnBackendDescriptor* finalStateDesc,
                         HipdnnBackendDescriptor* stateCheckpointsDesc,
                         hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                         const std::string& name = "")
{
    auto wrapper = createDescriptor<KdaOperationDescriptor>();
    auto desc = wrapper->asDescriptor<KdaOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&qDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_K_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&kDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_V_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&vDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_G_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&gDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_BETA_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&betaDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_CU_SEQLENS_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&cuSeqlensDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_INITIAL_STATE_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&initialStateDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_A_LOG_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&aLogDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_DT_BIAS_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dtBiasDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_STATE_INDICES_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&stateIndicesDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_O_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&oDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_FINAL_STATE_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&finalStateDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_STATE_CHECKPOINTS_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&stateCheckpointsDesc));

    auto checkpointEveryNTokens = static_cast<int64_t>(4);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_CHECKPOINT_EVERY_N_TOKENS_EXT,
                       HIPDNN_TYPE_INT64,
                       1,
                       &checkpointEveryNTokens);

    // Hand-written: also set the optional and bool scalars that the generated
    // read-back tests expect (test values from configs/kda.yaml).
    auto scale = 0.35355339F;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_SCALE_EXT, HIPDNN_TYPE_FLOAT, 1, &scale);
    auto gateLowerBound = -5.0F;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_GATE_LOWER_BOUND_EXT, HIPDNN_TYPE_FLOAT, 1, &gateLowerBound);
    const bool enabled = true;
    for(const auto boolAttr : {HIPDNN_ATTR_OPERATION_KDA_OUTPUT_FINAL_STATE_EXT,
                               HIPDNN_ATTR_OPERATION_KDA_USE_QK_L2NORM_EXT,
                               HIPDNN_ATTR_OPERATION_KDA_USE_BETA_SIGMOID_EXT,
                               HIPDNN_ATTR_OPERATION_KDA_ALLOW_NEG_EIGVAL_EXT,
                               HIPDNN_ATTR_OPERATION_KDA_SAFE_GATE_EXT,
                               HIPDNN_ATTR_OPERATION_KDA_BATCH_INVARIANT_EXT})
    {
        desc->setAttribute(boolAttr, HIPDNN_TYPE_BOOLEAN, 1, &enabled);
    }

    auto gateDomain = HIPDNN_GATE_DOMAIN_LOG_EXT;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT, HIPDNN_TYPE_GATE_DOMAIN_EXT, 1, &gateDomain);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    if(!name.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                           HIPDNN_TYPE_CHAR,
                           static_cast<int64_t>(name.size()),
                           name.data());
    }

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorKda : public ::testing::Test
{
public:
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
        ASSERT_NE(tensor, nullptr) << "Tensor with UID " << expectedUid
                                   << " not found"; // NOLINT(readability-implicit-bool-conversion)
        EXPECT_EQ(tensor->uid, expectedUid);
        EXPECT_EQ(tensor->dims, expectedDims);
        EXPECT_EQ(tensor->strides, expectedStrides);
        EXPECT_EQ(tensor->data_type, expectedDataType);
        EXPECT_EQ(tensor->virtual_, expectedVirtual);
    }

    static void verifyKdaNode(const NodeT& node,
                              DataType expectedComputeType,
                              int64_t expectedQUid,
                              int64_t expectedKUid,
                              int64_t expectedVUid,
                              int64_t expectedGUid,
                              int64_t expectedBetaUid,
                              int64_t expectedCuSeqlensUid,
                              int64_t expectedInitialStateUid,
                              int64_t expectedALogUid,
                              int64_t expectedDtBiasUid,
                              int64_t expectedStateIndicesUid,
                              int64_t expectedOUid,
                              int64_t expectedFinalStateUid,
                              int64_t expectedStateCheckpointsUid,
                              float expectedScale,
                              bool expectedOutputFinalState,
                              bool expectedUseQkL2norm,
                              int64_t expectedCheckpointEveryNTokens,
                              bool expectedUseBetaSigmoid,
                              bool expectedAllowNegEigval,
                              bool expectedSafeGate,
                              GateDomain expectedGateDomain,
                              float expectedGateLowerBound,
                              bool expectedBatchInvariant,
                              bool expectedOverwriteInitialState)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::KdaAttributes);

        auto* attrs = node.attributes.AsKdaAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->q_tensor_uid, expectedQUid);
        EXPECT_EQ(attrs->k_tensor_uid, expectedKUid);
        EXPECT_EQ(attrs->v_tensor_uid, expectedVUid);
        EXPECT_EQ(attrs->g_tensor_uid, expectedGUid);
        EXPECT_EQ(attrs->beta_tensor_uid, expectedBetaUid);
        EXPECT_EQ(attrs->cu_seqlens_tensor_uid, expectedCuSeqlensUid);
        EXPECT_EQ(attrs->initial_state_tensor_uid, expectedInitialStateUid);
        EXPECT_EQ(attrs->a_log_tensor_uid, expectedALogUid);
        EXPECT_EQ(attrs->dt_bias_tensor_uid, expectedDtBiasUid);
        EXPECT_EQ(attrs->state_indices_tensor_uid, expectedStateIndicesUid);
        EXPECT_EQ(attrs->o_tensor_uid, expectedOUid);
        EXPECT_EQ(attrs->final_state_tensor_uid, expectedFinalStateUid);
        EXPECT_EQ(attrs->state_checkpoints_tensor_uid, expectedStateCheckpointsUid);
        EXPECT_EQ(attrs->scale, expectedScale);
        EXPECT_EQ(attrs->output_final_state, expectedOutputFinalState);
        EXPECT_EQ(attrs->use_qk_l2norm, expectedUseQkL2norm);
        EXPECT_EQ(attrs->checkpoint_every_n_tokens, expectedCheckpointEveryNTokens);
        EXPECT_EQ(attrs->use_beta_sigmoid, expectedUseBetaSigmoid);
        EXPECT_EQ(attrs->allow_neg_eigval, expectedAllowNegEigval);
        EXPECT_EQ(attrs->safe_gate, expectedSafeGate);
        EXPECT_EQ(attrs->gate_domain, expectedGateDomain);
        EXPECT_EQ(attrs->gate_lower_bound, expectedGateLowerBound);
        EXPECT_EQ(attrs->batch_invariant, expectedBatchInvariant);
        EXPECT_EQ(attrs->overwrite_initial_state, expectedOverwriteInitialState);
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

TEST_F(TestGraphDescriptorKda, BuildFromSingleOperation)
{
    auto qDesc = createFinalizedTensor(K_KDA_TENSOR_Q_UID,
                                       toVec(K_KDA_TENSOR_Q_DIMS),
                                       toVec(K_KDA_TENSOR_Q_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto kDesc = createFinalizedTensor(K_KDA_TENSOR_K_UID,
                                       toVec(K_KDA_TENSOR_K_DIMS),
                                       toVec(K_KDA_TENSOR_K_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto vDesc = createFinalizedTensor(K_KDA_TENSOR_V_UID,
                                       toVec(K_KDA_TENSOR_V_DIMS),
                                       toVec(K_KDA_TENSOR_V_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto gDesc = createFinalizedTensor(K_KDA_TENSOR_G_UID,
                                       toVec(K_KDA_TENSOR_G_DIMS),
                                       toVec(K_KDA_TENSOR_G_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto betaDesc = createFinalizedTensor(K_KDA_TENSOR_BETA_UID,
                                          toVec(K_KDA_TENSOR_BETA_DIMS),
                                          toVec(K_KDA_TENSOR_BETA_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto cuSeqlensDesc = createFinalizedTensor(K_KDA_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
    auto initialStateDesc = createFinalizedTensor(K_KDA_TENSOR_INITIAL_STATE_UID,
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_DIMS),
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_STRIDES),
                                                  HIPDNN_DATA_FLOAT);
    auto aLogDesc = createFinalizedTensor(K_KDA_TENSOR_A_LOG_UID,
                                          toVec(K_KDA_TENSOR_A_LOG_DIMS),
                                          toVec(K_KDA_TENSOR_A_LOG_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto dtBiasDesc = createFinalizedTensor(K_KDA_TENSOR_DT_BIAS_UID,
                                            toVec(K_KDA_TENSOR_DT_BIAS_DIMS),
                                            toVec(K_KDA_TENSOR_DT_BIAS_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto stateIndicesDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_INDICES_UID,
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_DIMS),
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_STRIDES),
                                                  HIPDNN_DATA_INT32);
    auto oDesc = createFinalizedTensor(K_KDA_TENSOR_O_UID,
                                       toVec(K_KDA_TENSOR_O_DIMS),
                                       toVec(K_KDA_TENSOR_O_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto finalStateDesc = createFinalizedTensor(K_KDA_TENSOR_FINAL_STATE_UID,
                                                toVec(K_KDA_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_KDA_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
    auto stateCheckpointsDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_CHECKPOINTS_UID,
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_DIMS),
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_STRIDES),
                                                      HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedKdaOp(qDesc.get(),
                                       kDesc.get(),
                                       vDesc.get(),
                                       gDesc.get(),
                                       betaDesc.get(),
                                       cuSeqlensDesc.get(),
                                       initialStateDesc.get(),
                                       aLogDesc.get(),
                                       dtBiasDesc.get(),
                                       stateIndicesDesc.get(),
                                       oDesc.get(),
                                       finalStateDesc.get(),
                                       stateCheckpointsDesc.get());

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

    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 13);

    // Verify tensor attributes
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_Q_UID),
                 K_KDA_TENSOR_Q_UID,
                 toVec(K_KDA_TENSOR_Q_DIMS),
                 toVec(K_KDA_TENSOR_Q_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_K_UID),
                 K_KDA_TENSOR_K_UID,
                 toVec(K_KDA_TENSOR_K_DIMS),
                 toVec(K_KDA_TENSOR_K_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_V_UID),
                 K_KDA_TENSOR_V_UID,
                 toVec(K_KDA_TENSOR_V_DIMS),
                 toVec(K_KDA_TENSOR_V_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_G_UID),
                 K_KDA_TENSOR_G_UID,
                 toVec(K_KDA_TENSOR_G_DIMS),
                 toVec(K_KDA_TENSOR_G_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_BETA_UID),
                 K_KDA_TENSOR_BETA_UID,
                 toVec(K_KDA_TENSOR_BETA_DIMS),
                 toVec(K_KDA_TENSOR_BETA_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_CU_SEQLENS_UID),
                 K_KDA_TENSOR_CU_SEQLENS_UID,
                 toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS),
                 toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_INITIAL_STATE_UID),
                 K_KDA_TENSOR_INITIAL_STATE_UID,
                 toVec(K_KDA_TENSOR_INITIAL_STATE_DIMS),
                 toVec(K_KDA_TENSOR_INITIAL_STATE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_A_LOG_UID),
                 K_KDA_TENSOR_A_LOG_UID,
                 toVec(K_KDA_TENSOR_A_LOG_DIMS),
                 toVec(K_KDA_TENSOR_A_LOG_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_DT_BIAS_UID),
                 K_KDA_TENSOR_DT_BIAS_UID,
                 toVec(K_KDA_TENSOR_DT_BIAS_DIMS),
                 toVec(K_KDA_TENSOR_DT_BIAS_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_STATE_INDICES_UID),
                 K_KDA_TENSOR_STATE_INDICES_UID,
                 toVec(K_KDA_TENSOR_STATE_INDICES_DIMS),
                 toVec(K_KDA_TENSOR_STATE_INDICES_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_O_UID),
                 K_KDA_TENSOR_O_UID,
                 toVec(K_KDA_TENSOR_O_DIMS),
                 toVec(K_KDA_TENSOR_O_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_FINAL_STATE_UID),
                 K_KDA_TENSOR_FINAL_STATE_UID,
                 toVec(K_KDA_TENSOR_FINAL_STATE_DIMS),
                 toVec(K_KDA_TENSOR_FINAL_STATE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_STATE_CHECKPOINTS_UID),
                 K_KDA_TENSOR_STATE_CHECKPOINTS_UID,
                 toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_DIMS),
                 toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    verifyKdaNode(*graphT->nodes[0],
                  DataType::FLOAT,
                  K_KDA_TENSOR_Q_UID,
                  K_KDA_TENSOR_K_UID,
                  K_KDA_TENSOR_V_UID,
                  K_KDA_TENSOR_G_UID,
                  K_KDA_TENSOR_BETA_UID,
                  K_KDA_TENSOR_CU_SEQLENS_UID,
                  K_KDA_TENSOR_INITIAL_STATE_UID,
                  K_KDA_TENSOR_A_LOG_UID,
                  K_KDA_TENSOR_DT_BIAS_UID,
                  K_KDA_TENSOR_STATE_INDICES_UID,
                  K_KDA_TENSOR_O_UID,
                  K_KDA_TENSOR_FINAL_STATE_UID,
                  K_KDA_TENSOR_STATE_CHECKPOINTS_UID,
                  0.35355339F,
                  true,
                  true,
                  static_cast<int64_t>(4),
                  true,
                  true,
                  true,
                  GateDomain::LOG,
                  -5.0F,
                  true,
                  false);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorKda, ComputeDataTypePreserved)
{
    auto qDesc = createFinalizedTensor(K_KDA_TENSOR_Q_UID,
                                       toVec(K_KDA_TENSOR_Q_DIMS),
                                       toVec(K_KDA_TENSOR_Q_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto kDesc = createFinalizedTensor(K_KDA_TENSOR_K_UID,
                                       toVec(K_KDA_TENSOR_K_DIMS),
                                       toVec(K_KDA_TENSOR_K_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto vDesc = createFinalizedTensor(K_KDA_TENSOR_V_UID,
                                       toVec(K_KDA_TENSOR_V_DIMS),
                                       toVec(K_KDA_TENSOR_V_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto gDesc = createFinalizedTensor(K_KDA_TENSOR_G_UID,
                                       toVec(K_KDA_TENSOR_G_DIMS),
                                       toVec(K_KDA_TENSOR_G_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto betaDesc = createFinalizedTensor(K_KDA_TENSOR_BETA_UID,
                                          toVec(K_KDA_TENSOR_BETA_DIMS),
                                          toVec(K_KDA_TENSOR_BETA_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto cuSeqlensDesc = createFinalizedTensor(K_KDA_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
    auto initialStateDesc = createFinalizedTensor(K_KDA_TENSOR_INITIAL_STATE_UID,
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_DIMS),
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_STRIDES),
                                                  HIPDNN_DATA_FLOAT);
    auto aLogDesc = createFinalizedTensor(K_KDA_TENSOR_A_LOG_UID,
                                          toVec(K_KDA_TENSOR_A_LOG_DIMS),
                                          toVec(K_KDA_TENSOR_A_LOG_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto dtBiasDesc = createFinalizedTensor(K_KDA_TENSOR_DT_BIAS_UID,
                                            toVec(K_KDA_TENSOR_DT_BIAS_DIMS),
                                            toVec(K_KDA_TENSOR_DT_BIAS_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto stateIndicesDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_INDICES_UID,
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_DIMS),
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_STRIDES),
                                                  HIPDNN_DATA_INT32);
    auto oDesc = createFinalizedTensor(K_KDA_TENSOR_O_UID,
                                       toVec(K_KDA_TENSOR_O_DIMS),
                                       toVec(K_KDA_TENSOR_O_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto finalStateDesc = createFinalizedTensor(K_KDA_TENSOR_FINAL_STATE_UID,
                                                toVec(K_KDA_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_KDA_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
    auto stateCheckpointsDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_CHECKPOINTS_UID,
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_DIMS),
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_STRIDES),
                                                      HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedKdaOp(qDesc.get(),
                                       kDesc.get(),
                                       vDesc.get(),
                                       gDesc.get(),
                                       betaDesc.get(),
                                       cuSeqlensDesc.get(),
                                       initialStateDesc.get(),
                                       aLogDesc.get(),
                                       dtBiasDesc.get(),
                                       stateIndicesDesc.get(),
                                       oDesc.get(),
                                       finalStateDesc.get(),
                                       stateCheckpointsDesc.get(),
                                       HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

TEST_F(TestGraphDescriptorKda, KdaAttributesPreserved)
{
    auto qDesc = createFinalizedTensor(K_KDA_TENSOR_Q_UID,
                                       toVec(K_KDA_TENSOR_Q_DIMS),
                                       toVec(K_KDA_TENSOR_Q_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto kDesc = createFinalizedTensor(K_KDA_TENSOR_K_UID,
                                       toVec(K_KDA_TENSOR_K_DIMS),
                                       toVec(K_KDA_TENSOR_K_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto vDesc = createFinalizedTensor(K_KDA_TENSOR_V_UID,
                                       toVec(K_KDA_TENSOR_V_DIMS),
                                       toVec(K_KDA_TENSOR_V_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto gDesc = createFinalizedTensor(K_KDA_TENSOR_G_UID,
                                       toVec(K_KDA_TENSOR_G_DIMS),
                                       toVec(K_KDA_TENSOR_G_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto betaDesc = createFinalizedTensor(K_KDA_TENSOR_BETA_UID,
                                          toVec(K_KDA_TENSOR_BETA_DIMS),
                                          toVec(K_KDA_TENSOR_BETA_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto cuSeqlensDesc = createFinalizedTensor(K_KDA_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
    auto initialStateDesc = createFinalizedTensor(K_KDA_TENSOR_INITIAL_STATE_UID,
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_DIMS),
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_STRIDES),
                                                  HIPDNN_DATA_FLOAT);
    auto aLogDesc = createFinalizedTensor(K_KDA_TENSOR_A_LOG_UID,
                                          toVec(K_KDA_TENSOR_A_LOG_DIMS),
                                          toVec(K_KDA_TENSOR_A_LOG_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto dtBiasDesc = createFinalizedTensor(K_KDA_TENSOR_DT_BIAS_UID,
                                            toVec(K_KDA_TENSOR_DT_BIAS_DIMS),
                                            toVec(K_KDA_TENSOR_DT_BIAS_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto stateIndicesDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_INDICES_UID,
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_DIMS),
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_STRIDES),
                                                  HIPDNN_DATA_INT32);
    auto oDesc = createFinalizedTensor(K_KDA_TENSOR_O_UID,
                                       toVec(K_KDA_TENSOR_O_DIMS),
                                       toVec(K_KDA_TENSOR_O_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto finalStateDesc = createFinalizedTensor(K_KDA_TENSOR_FINAL_STATE_UID,
                                                toVec(K_KDA_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_KDA_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
    auto stateCheckpointsDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_CHECKPOINTS_UID,
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_DIMS),
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_STRIDES),
                                                      HIPDNN_DATA_FLOAT);

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<KdaOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<KdaOperationDescriptor>();

    HipdnnBackendDescriptor* qPtr = qDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&qPtr));
    HipdnnBackendDescriptor* kPtr = kDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_K_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&kPtr));
    HipdnnBackendDescriptor* vPtr = vDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_V_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&vPtr));
    HipdnnBackendDescriptor* gPtr = gDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_G_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&gPtr));
    HipdnnBackendDescriptor* betaPtr = betaDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_BETA_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&betaPtr));
    HipdnnBackendDescriptor* cuSeqlensPtr = cuSeqlensDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_CU_SEQLENS_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&cuSeqlensPtr));
    HipdnnBackendDescriptor* initialStatePtr = initialStateDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_INITIAL_STATE_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&initialStatePtr));
    HipdnnBackendDescriptor* aLogPtr = aLogDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_A_LOG_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&aLogPtr));
    HipdnnBackendDescriptor* dtBiasPtr = dtBiasDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_DT_BIAS_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&dtBiasPtr));
    HipdnnBackendDescriptor* stateIndicesPtr = stateIndicesDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_STATE_INDICES_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&stateIndicesPtr));
    HipdnnBackendDescriptor* oPtr = oDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_O_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&oPtr));
    HipdnnBackendDescriptor* finalStatePtr = finalStateDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_FINAL_STATE_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&finalStatePtr));
    HipdnnBackendDescriptor* stateCheckpointsPtr = stateCheckpointsDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_STATE_CHECKPOINTS_DESC_EXT,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&stateCheckpointsPtr));

    auto scale = 0.35355339F;
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_SCALE_EXT, HIPDNN_TYPE_FLOAT, 1, &scale);

    auto outputFinalState = true;
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_OUTPUT_FINAL_STATE_EXT,
                         HIPDNN_TYPE_BOOLEAN,
                         1,
                         &outputFinalState);

    auto useQkL2norm = true;
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_USE_QK_L2NORM_EXT, HIPDNN_TYPE_BOOLEAN, 1, &useQkL2norm);

    auto checkpointEveryNTokens = static_cast<int64_t>(4);
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_CHECKPOINT_EVERY_N_TOKENS_EXT,
                         HIPDNN_TYPE_INT64,
                         1,
                         &checkpointEveryNTokens);

    auto useBetaSigmoid = true;
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_USE_BETA_SIGMOID_EXT, HIPDNN_TYPE_BOOLEAN, 1, &useBetaSigmoid);

    auto allowNegEigval = true;
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_ALLOW_NEG_EIGVAL_EXT, HIPDNN_TYPE_BOOLEAN, 1, &allowNegEigval);

    auto safeGate = true;
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_SAFE_GATE_EXT, HIPDNN_TYPE_BOOLEAN, 1, &safeGate);

    auto gateDomain = HIPDNN_GATE_DOMAIN_LOG_EXT;
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT, HIPDNN_TYPE_GATE_DOMAIN_EXT, 1, &gateDomain);

    auto gateLowerBound = -5.0F;
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_GATE_LOWER_BOUND_EXT, HIPDNN_TYPE_FLOAT, 1, &gateLowerBound);

    auto batchInvariant = true;
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_BATCH_INVARIANT_EXT, HIPDNN_TYPE_BOOLEAN, 1, &batchInvariant);

    auto overwriteInitialState = false;
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_OVERWRITE_INITIAL_STATE_EXT,
                         HIPDNN_TYPE_BOOLEAN,
                         1,
                         &overwriteInitialState);

    auto computeType = HIPDNN_DATA_FLOAT;
    opDesc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Set operation name
    const std::string opName = "test_kda";
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                         HIPDNN_TYPE_CHAR,
                         static_cast<int64_t>(opName.size()),
                         opName.c_str());
    opDesc->finalize();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {wrapper.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 13);

    // Verify tensors
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_Q_UID),
                 K_KDA_TENSOR_Q_UID,
                 toVec(K_KDA_TENSOR_Q_DIMS),
                 toVec(K_KDA_TENSOR_Q_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_K_UID),
                 K_KDA_TENSOR_K_UID,
                 toVec(K_KDA_TENSOR_K_DIMS),
                 toVec(K_KDA_TENSOR_K_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_V_UID),
                 K_KDA_TENSOR_V_UID,
                 toVec(K_KDA_TENSOR_V_DIMS),
                 toVec(K_KDA_TENSOR_V_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_G_UID),
                 K_KDA_TENSOR_G_UID,
                 toVec(K_KDA_TENSOR_G_DIMS),
                 toVec(K_KDA_TENSOR_G_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_BETA_UID),
                 K_KDA_TENSOR_BETA_UID,
                 toVec(K_KDA_TENSOR_BETA_DIMS),
                 toVec(K_KDA_TENSOR_BETA_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_CU_SEQLENS_UID),
                 K_KDA_TENSOR_CU_SEQLENS_UID,
                 toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS),
                 toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_INITIAL_STATE_UID),
                 K_KDA_TENSOR_INITIAL_STATE_UID,
                 toVec(K_KDA_TENSOR_INITIAL_STATE_DIMS),
                 toVec(K_KDA_TENSOR_INITIAL_STATE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_A_LOG_UID),
                 K_KDA_TENSOR_A_LOG_UID,
                 toVec(K_KDA_TENSOR_A_LOG_DIMS),
                 toVec(K_KDA_TENSOR_A_LOG_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_DT_BIAS_UID),
                 K_KDA_TENSOR_DT_BIAS_UID,
                 toVec(K_KDA_TENSOR_DT_BIAS_DIMS),
                 toVec(K_KDA_TENSOR_DT_BIAS_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_STATE_INDICES_UID),
                 K_KDA_TENSOR_STATE_INDICES_UID,
                 toVec(K_KDA_TENSOR_STATE_INDICES_DIMS),
                 toVec(K_KDA_TENSOR_STATE_INDICES_STRIDES),
                 DataType::INT32);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_O_UID),
                 K_KDA_TENSOR_O_UID,
                 toVec(K_KDA_TENSOR_O_DIMS),
                 toVec(K_KDA_TENSOR_O_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_FINAL_STATE_UID),
                 K_KDA_TENSOR_FINAL_STATE_UID,
                 toVec(K_KDA_TENSOR_FINAL_STATE_DIMS),
                 toVec(K_KDA_TENSOR_FINAL_STATE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_KDA_TENSOR_STATE_CHECKPOINTS_UID),
                 K_KDA_TENSOR_STATE_CHECKPOINTS_UID,
                 toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_DIMS),
                 toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_STRIDES),
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifyKdaNode(*graphT->nodes[0],
                  DataType::FLOAT,
                  K_KDA_TENSOR_Q_UID,
                  K_KDA_TENSOR_K_UID,
                  K_KDA_TENSOR_V_UID,
                  K_KDA_TENSOR_G_UID,
                  K_KDA_TENSOR_BETA_UID,
                  K_KDA_TENSOR_CU_SEQLENS_UID,
                  K_KDA_TENSOR_INITIAL_STATE_UID,
                  K_KDA_TENSOR_A_LOG_UID,
                  K_KDA_TENSOR_DT_BIAS_UID,
                  K_KDA_TENSOR_STATE_INDICES_UID,
                  K_KDA_TENSOR_O_UID,
                  K_KDA_TENSOR_FINAL_STATE_UID,
                  K_KDA_TENSOR_STATE_CHECKPOINTS_UID,
                  0.35355339F,
                  true,
                  true,
                  static_cast<int64_t>(4),
                  true,
                  true,
                  true,
                  GateDomain::LOG,
                  -5.0F,
                  true,
                  false);

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_kda");
}

TEST_F(TestGraphDescriptorKda, OperationNamePreservedInSerialization)
{
    auto qDesc = createFinalizedTensor(K_KDA_TENSOR_Q_UID,
                                       toVec(K_KDA_TENSOR_Q_DIMS),
                                       toVec(K_KDA_TENSOR_Q_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto kDesc = createFinalizedTensor(K_KDA_TENSOR_K_UID,
                                       toVec(K_KDA_TENSOR_K_DIMS),
                                       toVec(K_KDA_TENSOR_K_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto vDesc = createFinalizedTensor(K_KDA_TENSOR_V_UID,
                                       toVec(K_KDA_TENSOR_V_DIMS),
                                       toVec(K_KDA_TENSOR_V_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto gDesc = createFinalizedTensor(K_KDA_TENSOR_G_UID,
                                       toVec(K_KDA_TENSOR_G_DIMS),
                                       toVec(K_KDA_TENSOR_G_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto betaDesc = createFinalizedTensor(K_KDA_TENSOR_BETA_UID,
                                          toVec(K_KDA_TENSOR_BETA_DIMS),
                                          toVec(K_KDA_TENSOR_BETA_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto cuSeqlensDesc = createFinalizedTensor(K_KDA_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
    auto initialStateDesc = createFinalizedTensor(K_KDA_TENSOR_INITIAL_STATE_UID,
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_DIMS),
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_STRIDES),
                                                  HIPDNN_DATA_FLOAT);
    auto aLogDesc = createFinalizedTensor(K_KDA_TENSOR_A_LOG_UID,
                                          toVec(K_KDA_TENSOR_A_LOG_DIMS),
                                          toVec(K_KDA_TENSOR_A_LOG_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto dtBiasDesc = createFinalizedTensor(K_KDA_TENSOR_DT_BIAS_UID,
                                            toVec(K_KDA_TENSOR_DT_BIAS_DIMS),
                                            toVec(K_KDA_TENSOR_DT_BIAS_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto stateIndicesDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_INDICES_UID,
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_DIMS),
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_STRIDES),
                                                  HIPDNN_DATA_INT32);
    auto oDesc = createFinalizedTensor(K_KDA_TENSOR_O_UID,
                                       toVec(K_KDA_TENSOR_O_DIMS),
                                       toVec(K_KDA_TENSOR_O_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto finalStateDesc = createFinalizedTensor(K_KDA_TENSOR_FINAL_STATE_UID,
                                                toVec(K_KDA_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_KDA_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
    auto stateCheckpointsDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_CHECKPOINTS_UID,
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_DIMS),
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_STRIDES),
                                                      HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedKdaOp(qDesc.get(),
                                       kDesc.get(),
                                       vDesc.get(),
                                       gDesc.get(),
                                       betaDesc.get(),
                                       cuSeqlensDesc.get(),
                                       initialStateDesc.get(),
                                       aLogDesc.get(),
                                       dtBiasDesc.get(),
                                       stateIndicesDesc.get(),
                                       oDesc.get(),
                                       finalStateDesc.get(),
                                       stateCheckpointsDesc.get(),
                                       HIPDNN_DATA_FLOAT,
                                       "test_kda_name");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->name, "test_kda_name");
}

TEST_F(TestGraphDescriptorKda, OperationNameRoundTripThroughLifting)
{
    auto qDesc = createFinalizedTensor(K_KDA_TENSOR_Q_UID,
                                       toVec(K_KDA_TENSOR_Q_DIMS),
                                       toVec(K_KDA_TENSOR_Q_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto kDesc = createFinalizedTensor(K_KDA_TENSOR_K_UID,
                                       toVec(K_KDA_TENSOR_K_DIMS),
                                       toVec(K_KDA_TENSOR_K_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto vDesc = createFinalizedTensor(K_KDA_TENSOR_V_UID,
                                       toVec(K_KDA_TENSOR_V_DIMS),
                                       toVec(K_KDA_TENSOR_V_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto gDesc = createFinalizedTensor(K_KDA_TENSOR_G_UID,
                                       toVec(K_KDA_TENSOR_G_DIMS),
                                       toVec(K_KDA_TENSOR_G_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto betaDesc = createFinalizedTensor(K_KDA_TENSOR_BETA_UID,
                                          toVec(K_KDA_TENSOR_BETA_DIMS),
                                          toVec(K_KDA_TENSOR_BETA_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto cuSeqlensDesc = createFinalizedTensor(K_KDA_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
    auto initialStateDesc = createFinalizedTensor(K_KDA_TENSOR_INITIAL_STATE_UID,
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_DIMS),
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_STRIDES),
                                                  HIPDNN_DATA_FLOAT);
    auto aLogDesc = createFinalizedTensor(K_KDA_TENSOR_A_LOG_UID,
                                          toVec(K_KDA_TENSOR_A_LOG_DIMS),
                                          toVec(K_KDA_TENSOR_A_LOG_STRIDES),
                                          HIPDNN_DATA_FLOAT);
    auto dtBiasDesc = createFinalizedTensor(K_KDA_TENSOR_DT_BIAS_UID,
                                            toVec(K_KDA_TENSOR_DT_BIAS_DIMS),
                                            toVec(K_KDA_TENSOR_DT_BIAS_STRIDES),
                                            HIPDNN_DATA_FLOAT);
    auto stateIndicesDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_INDICES_UID,
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_DIMS),
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_STRIDES),
                                                  HIPDNN_DATA_INT32);
    auto oDesc = createFinalizedTensor(K_KDA_TENSOR_O_UID,
                                       toVec(K_KDA_TENSOR_O_DIMS),
                                       toVec(K_KDA_TENSOR_O_STRIDES),
                                       HIPDNN_DATA_FLOAT);
    auto finalStateDesc = createFinalizedTensor(K_KDA_TENSOR_FINAL_STATE_UID,
                                                toVec(K_KDA_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_KDA_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
    auto stateCheckpointsDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_CHECKPOINTS_UID,
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_DIMS),
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_STRIDES),
                                                      HIPDNN_DATA_FLOAT);
    auto opDesc = createFinalizedKdaOp(qDesc.get(),
                                       kDesc.get(),
                                       vDesc.get(),
                                       gDesc.get(),
                                       betaDesc.get(),
                                       cuSeqlensDesc.get(),
                                       initialStateDesc.get(),
                                       aLogDesc.get(),
                                       dtBiasDesc.get(),
                                       stateIndicesDesc.get(),
                                       oDesc.get(),
                                       finalStateDesc.get(),
                                       stateCheckpointsDesc.get(),
                                       HIPDNN_DATA_FLOAT,
                                       "test_kda_lifting");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Serialize the graph
    auto serialized = desc->getSerializedGraph();
    std::vector<uint8_t> bytes(static_cast<const uint8_t*>(serialized.ptr),
                               static_cast<const uint8_t*>(serialized.ptr) + serialized.size);

    // Deserialize into a new GraphDescriptor (lifting path)
    auto liftedWrapper = createDescriptor<GraphDescriptor>();
    auto liftedDesc = liftedWrapper->asDescriptor<GraphDescriptor>();
    liftedDesc->deserializeGraph(bytes.data(), bytes.size());

    hipdnnHandle_t handle = &_mockHandle;
    liftedDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                             HIPDNN_TYPE_HANDLE,
                             1,
                             static_cast<const void*>(&handle));
    liftedDesc->finalize();

    // Re-serialize and verify name survived the round-trip
    auto reSerialized = liftedDesc->getSerializedGraph();
    auto graphT = UnPackGraph(reSerialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->name, "test_kda_lifting");
}

} // namespace
