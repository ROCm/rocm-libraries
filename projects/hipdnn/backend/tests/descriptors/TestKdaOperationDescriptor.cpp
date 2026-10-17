// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/KdaOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/kda_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/KdaConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <optional>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestKdaOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<KdaOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<KdaOperationDescriptor>();
    }

    void setTensors(std::optional<hipdnnBackendAttributeName_t> excluded = std::nullopt) const
    {
        auto desc = getDescriptor();
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT)
        {
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_K_DESC_EXT)
        {
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_KDA_K_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_V_DESC_EXT)
        {
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_KDA_V_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_G_DESC_EXT)
        {
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_KDA_G_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_gDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_BETA_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_BETA_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_betaDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_CU_SEQLENS_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_CU_SEQLENS_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_cuSeqlensDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_INITIAL_STATE_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_INITIAL_STATE_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_initialStateDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_A_LOG_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_A_LOG_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_aLogDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_DT_BIAS_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_DT_BIAS_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_dtBiasDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_STATE_INDICES_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_STATE_INDICES_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_stateIndicesDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_O_DESC_EXT)
        {
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_KDA_O_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_FINAL_STATE_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_FINAL_STATE_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_finalStateDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_STATE_CHECKPOINTS_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_STATE_CHECKPOINTS_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_stateCheckpointsDesc);
        }
    }

    void setKdaParams() const
    {
        auto desc = getDescriptor();

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
    }

    void setRequiredAttributes(std::optional<hipdnnBackendAttributeName_t> excluded
                               = std::nullopt) const
    {
        setTensors(excluded);
        setKdaParams();
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT)
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            getDescriptor()->setAttribute(
                HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
        auto gateDomain = HIPDNN_GATE_DOMAIN_LOG_EXT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT, HIPDNN_TYPE_GATE_DOMAIN_EXT, 1, &gateDomain);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _qDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _kDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _vDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _gDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _betaDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _cuSeqlensDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _initialStateDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _aLogDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dtBiasDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _stateIndicesDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _oDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _finalStateDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _stateCheckpointsDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<KdaOperationDescriptor>();
        _qDesc = createFinalizedTensor(K_KDA_TENSOR_Q_UID,
                                       toVec(K_KDA_TENSOR_Q_DIMS),
                                       toVec(K_KDA_TENSOR_Q_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _kDesc = createFinalizedTensor(K_KDA_TENSOR_K_UID,
                                       toVec(K_KDA_TENSOR_K_DIMS),
                                       toVec(K_KDA_TENSOR_K_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _vDesc = createFinalizedTensor(K_KDA_TENSOR_V_UID,
                                       toVec(K_KDA_TENSOR_V_DIMS),
                                       toVec(K_KDA_TENSOR_V_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _gDesc = createFinalizedTensor(K_KDA_TENSOR_G_UID,
                                       toVec(K_KDA_TENSOR_G_DIMS),
                                       toVec(K_KDA_TENSOR_G_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _betaDesc = createFinalizedTensor(K_KDA_TENSOR_BETA_UID,
                                          toVec(K_KDA_TENSOR_BETA_DIMS),
                                          toVec(K_KDA_TENSOR_BETA_STRIDES),
                                          HIPDNN_DATA_FLOAT);
        _cuSeqlensDesc = createFinalizedTensor(K_KDA_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_KDA_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
        _initialStateDesc = createFinalizedTensor(K_KDA_TENSOR_INITIAL_STATE_UID,
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_DIMS),
                                                  toVec(K_KDA_TENSOR_INITIAL_STATE_STRIDES),
                                                  HIPDNN_DATA_FLOAT);
        _aLogDesc = createFinalizedTensor(K_KDA_TENSOR_A_LOG_UID,
                                          toVec(K_KDA_TENSOR_A_LOG_DIMS),
                                          toVec(K_KDA_TENSOR_A_LOG_STRIDES),
                                          HIPDNN_DATA_FLOAT);
        _dtBiasDesc = createFinalizedTensor(K_KDA_TENSOR_DT_BIAS_UID,
                                            toVec(K_KDA_TENSOR_DT_BIAS_DIMS),
                                            toVec(K_KDA_TENSOR_DT_BIAS_STRIDES),
                                            HIPDNN_DATA_FLOAT);
        _stateIndicesDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_INDICES_UID,
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_DIMS),
                                                  toVec(K_KDA_TENSOR_STATE_INDICES_STRIDES),
                                                  HIPDNN_DATA_INT32);
        _oDesc = createFinalizedTensor(K_KDA_TENSOR_O_UID,
                                       toVec(K_KDA_TENSOR_O_DIMS),
                                       toVec(K_KDA_TENSOR_O_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _finalStateDesc = createFinalizedTensor(K_KDA_TENSOR_FINAL_STATE_UID,
                                                toVec(K_KDA_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_KDA_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
        _stateCheckpointsDesc = createFinalizedTensor(K_KDA_TENSOR_STATE_CHECKPOINTS_UID,
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_DIMS),
                                                      toVec(K_KDA_TENSOR_STATE_CHECKPOINTS_STRIDES),
                                                      HIPDNN_DATA_FLOAT);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _qDesc.reset();
        _kDesc.reset();
        _vDesc.reset();
        _gDesc.reset();
        _betaDesc.reset();
        _cuSeqlensDesc.reset();
        _initialStateDesc.reset();
        _aLogDesc.reset();
        _dtBiasDesc.reset();
        _stateIndicesDesc.reset();
        _oDesc.reset();
        _finalStateDesc.reset();
        _stateCheckpointsDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestKdaOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_KDA_DESCRIPTOR_EXT);
}

TEST_F(TestKdaOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// The parameter is the one required attribute to leave unset. Every other required attribute is
// set, so each case proves the omitted attribute alone is what finalize() rejects.
class TestKdaOperationDescriptorFinalizeFailsWithout
    : public TestKdaOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestKdaOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setRequiredAttributes(GetParam());
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestKdaOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT,
                                           HIPDNN_ATTR_OPERATION_KDA_K_DESC_EXT,
                                           HIPDNN_ATTR_OPERATION_KDA_V_DESC_EXT,
                                           HIPDNN_ATTR_OPERATION_KDA_G_DESC_EXT,
                                           HIPDNN_ATTR_OPERATION_KDA_BETA_DESC_EXT,
                                           HIPDNN_ATTR_OPERATION_KDA_CU_SEQLENS_DESC_EXT,
                                           HIPDNN_ATTR_OPERATION_KDA_O_DESC_EXT,
                                           HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT));

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorQ)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().q_tensor_uid, K_KDA_TENSOR_Q_UID);
    ASSERT_NE(desc->getQDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorK)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_K_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc));

    ASSERT_EQ(desc->getData().k_tensor_uid, K_KDA_TENSOR_K_UID);
    ASSERT_NE(desc->getKDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorV)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_V_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc));

    ASSERT_EQ(desc->getData().v_tensor_uid, K_KDA_TENSOR_V_UID);
    ASSERT_NE(desc->getVDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorG)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_G_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_gDesc));

    ASSERT_EQ(desc->getData().g_tensor_uid, K_KDA_TENSOR_G_UID);
    ASSERT_NE(desc->getGDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorBeta)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_BETA_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_betaDesc));

    ASSERT_EQ(desc->getData().beta_tensor_uid, K_KDA_TENSOR_BETA_UID);
    ASSERT_NE(desc->getBetaDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorCuSeqlens)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_CU_SEQLENS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_cuSeqlensDesc));

    ASSERT_EQ(desc->getData().cu_seqlens_tensor_uid, K_KDA_TENSOR_CU_SEQLENS_UID);
    ASSERT_NE(desc->getCuSeqlensDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorInitialState)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_INITIAL_STATE_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_initialStateDesc));

    ASSERT_EQ(desc->getData().initial_state_tensor_uid, K_KDA_TENSOR_INITIAL_STATE_UID);
    ASSERT_NE(desc->getInitialStateDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorALog)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_A_LOG_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_aLogDesc));

    ASSERT_EQ(desc->getData().a_log_tensor_uid, K_KDA_TENSOR_A_LOG_UID);
    ASSERT_NE(desc->getALogDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorDtBias)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_DT_BIAS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dtBiasDesc));

    ASSERT_EQ(desc->getData().dt_bias_tensor_uid, K_KDA_TENSOR_DT_BIAS_UID);
    ASSERT_NE(desc->getDtBiasDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorStateIndices)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_STATE_INDICES_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_stateIndicesDesc));

    ASSERT_EQ(desc->getData().state_indices_tensor_uid, K_KDA_TENSOR_STATE_INDICES_UID);
    ASSERT_NE(desc->getStateIndicesDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorO)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_O_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc));

    ASSERT_EQ(desc->getData().o_tensor_uid, K_KDA_TENSOR_O_UID);
    ASSERT_NE(desc->getODesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorFinalState)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_FINAL_STATE_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_finalStateDesc));

    ASSERT_EQ(desc->getData().final_state_tensor_uid, K_KDA_TENSOR_FINAL_STATE_UID);
    ASSERT_NE(desc->getFinalStateDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorDescriptorStateCheckpoints)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_STATE_CHECKPOINTS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_stateCheckpointsDesc));

    ASSERT_EQ(desc->getData().state_checkpoints_tensor_uid, K_KDA_TENSOR_STATE_CHECKPOINTS_UID);
    ASSERT_NE(desc->getStateCheckpointsDesc(), nullptr);
}

TEST_F(TestKdaOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestKdaOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT, HIPDNN_TYPE_INT64, 1, &_qDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKdaOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_qDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKdaOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestKdaOperationDescriptor, SetGateDomain)
{
    auto desc = getDescriptor();
    auto gateDomain = HIPDNN_GATE_DOMAIN_LOG_EXT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT, HIPDNN_TYPE_GATE_DOMAIN_EXT, 1, &gateDomain));

    ASSERT_EQ(desc->getData().gate_domain, GateDomain::LOG);
}

TEST_F(TestKdaOperationDescriptor, SetGateDomainWrongElementCount)
{
    auto desc = getDescriptor();
    auto gateDomain = HIPDNN_GATE_DOMAIN_LOG_EXT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT, HIPDNN_TYPE_GATE_DOMAIN_EXT, 2, &gateDomain),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestKdaOperationDescriptor, SetScale)
{
    auto desc = getDescriptor();
    auto scale = 0.35355339F;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_SCALE_EXT, HIPDNN_TYPE_FLOAT, 1, &scale));
    EXPECT_EQ(desc->getData().scale, scale);
}

TEST_F(TestKdaOperationDescriptor, SetOutputFinalState)
{
    auto desc = getDescriptor();
    auto outputFinalState = true;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_OUTPUT_FINAL_STATE_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &outputFinalState));
    EXPECT_EQ(desc->getData().output_final_state, outputFinalState);
}

TEST_F(TestKdaOperationDescriptor, SetUseQkL2norm)
{
    auto desc = getDescriptor();
    auto useQkL2norm = true;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_USE_QK_L2NORM_EXT, HIPDNN_TYPE_BOOLEAN, 1, &useQkL2norm));
    EXPECT_EQ(desc->getData().use_qk_l2norm, useQkL2norm);
}

TEST_F(TestKdaOperationDescriptor, SetCheckpointEveryNTokens)
{
    auto desc = getDescriptor();
    auto checkpointEveryNTokens = static_cast<int64_t>(4);

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_CHECKPOINT_EVERY_N_TOKENS_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &checkpointEveryNTokens));
    EXPECT_EQ(desc->getData().checkpoint_every_n_tokens, checkpointEveryNTokens);
}

TEST_F(TestKdaOperationDescriptor, SetUseBetaSigmoid)
{
    auto desc = getDescriptor();
    auto useBetaSigmoid = true;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_USE_BETA_SIGMOID_EXT, HIPDNN_TYPE_BOOLEAN, 1, &useBetaSigmoid));
    EXPECT_EQ(desc->getData().use_beta_sigmoid, useBetaSigmoid);
}

TEST_F(TestKdaOperationDescriptor, SetAllowNegEigval)
{
    auto desc = getDescriptor();
    auto allowNegEigval = true;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_ALLOW_NEG_EIGVAL_EXT, HIPDNN_TYPE_BOOLEAN, 1, &allowNegEigval));
    EXPECT_EQ(desc->getData().allow_neg_eigval, allowNegEigval);
}

TEST_F(TestKdaOperationDescriptor, SetSafeGate)
{
    auto desc = getDescriptor();
    auto safeGate = true;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_SAFE_GATE_EXT, HIPDNN_TYPE_BOOLEAN, 1, &safeGate));
    EXPECT_EQ(desc->getData().safe_gate, safeGate);
}

TEST_F(TestKdaOperationDescriptor, SetGateLowerBound)
{
    auto desc = getDescriptor();
    auto gateLowerBound = -5.0F;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_GATE_LOWER_BOUND_EXT, HIPDNN_TYPE_FLOAT, 1, &gateLowerBound));
    EXPECT_EQ(desc->getData().gate_lower_bound, gateLowerBound);
}

TEST_F(TestKdaOperationDescriptor, SetBatchInvariant)
{
    auto desc = getDescriptor();
    auto batchInvariant = true;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_BATCH_INVARIANT_EXT, HIPDNN_TYPE_BOOLEAN, 1, &batchInvariant));
    EXPECT_EQ(desc->getData().batch_invariant, batchInvariant);
}

TEST_F(TestKdaOperationDescriptor, SetOverwriteInitialState)
{
    auto desc = getDescriptor();
    auto overwriteInitialState = false;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_KDA_OVERWRITE_INITIAL_STATE_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &overwriteInitialState));
    EXPECT_EQ(desc->getData().overwrite_initial_state, overwriteInitialState);
}

TEST_F(TestKdaOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestKdaOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestKdaOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestKdaOperationDescriptor, SetAttributeUnsupported)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedQ = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedQ)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedQ, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardQ(retrievedQ);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestKdaOperationDescriptor, GetAttributeKdaParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // gate domain
    hipdnnGateDomain_t gateDomain = HIPDNN_GATE_DOMAIN_LOG_EXT;
    int64_t gateDomainCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT,
                                       HIPDNN_TYPE_GATE_DOMAIN_EXT,
                                       1,
                                       &gateDomainCount,
                                       &gateDomain));
    ASSERT_EQ(gateDomainCount, 1);
    EXPECT_EQ(gateDomain, HIPDNN_GATE_DOMAIN_LOG_EXT);

    // scale
    float scale = 0;
    int64_t scaleCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_KDA_SCALE_EXT, HIPDNN_TYPE_FLOAT, 1, &scaleCount, &scale));
    ASSERT_EQ(scaleCount, 1);
    EXPECT_EQ(scale, 0.35355339F);

    // output final state
    bool outputFinalState = false;
    int64_t outputFinalStateCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_OUTPUT_FINAL_STATE_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &outputFinalStateCount,
                                       &outputFinalState));
    ASSERT_EQ(outputFinalStateCount, 1);
    EXPECT_EQ(outputFinalState, true);

    // use qk l2norm
    bool useQkL2norm = false;
    int64_t useQkL2normCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_USE_QK_L2NORM_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &useQkL2normCount,
                                       &useQkL2norm));
    ASSERT_EQ(useQkL2normCount, 1);
    EXPECT_EQ(useQkL2norm, true);

    // checkpoint every n tokens
    int64_t checkpointEveryNTokens = 0;
    int64_t checkpointEveryNTokensCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_CHECKPOINT_EVERY_N_TOKENS_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &checkpointEveryNTokensCount,
                                       &checkpointEveryNTokens));
    ASSERT_EQ(checkpointEveryNTokensCount, 1);
    EXPECT_EQ(checkpointEveryNTokens, static_cast<int64_t>(4));

    // use beta sigmoid
    bool useBetaSigmoid = false;
    int64_t useBetaSigmoidCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_USE_BETA_SIGMOID_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &useBetaSigmoidCount,
                                       &useBetaSigmoid));
    ASSERT_EQ(useBetaSigmoidCount, 1);
    EXPECT_EQ(useBetaSigmoid, true);

    // allow neg eigval
    bool allowNegEigval = false;
    int64_t allowNegEigvalCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_ALLOW_NEG_EIGVAL_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &allowNegEigvalCount,
                                       &allowNegEigval));
    ASSERT_EQ(allowNegEigvalCount, 1);
    EXPECT_EQ(allowNegEigval, true);

    // safe gate
    bool safeGate = false;
    int64_t safeGateCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_SAFE_GATE_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &safeGateCount,
                                       &safeGate));
    ASSERT_EQ(safeGateCount, 1);
    EXPECT_EQ(safeGate, true);

    // gate lower bound
    float gateLowerBound = 0;
    int64_t gateLowerBoundCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_GATE_LOWER_BOUND_EXT,
                                       HIPDNN_TYPE_FLOAT,
                                       1,
                                       &gateLowerBoundCount,
                                       &gateLowerBound));
    ASSERT_EQ(gateLowerBoundCount, 1);
    EXPECT_EQ(gateLowerBound, -5.0F);

    // batch invariant
    bool batchInvariant = false;
    int64_t batchInvariantCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_BATCH_INVARIANT_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &batchInvariantCount,
                                       &batchInvariant));
    ASSERT_EQ(batchInvariantCount, 1);
    EXPECT_EQ(batchInvariant, true);

    // overwrite initial state
    bool overwriteInitialState = false;
    int64_t overwriteInitialStateCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_OVERWRITE_INITIAL_STATE_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &overwriteInitialStateCount,
                                       &overwriteInitialState));
    ASSERT_EQ(overwriteInitialStateCount, 1);
    EXPECT_EQ(overwriteInitialState, false);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &elementCount,
                                       &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestKdaOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Query Mode Tests
// =============================================================================

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorQQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorKQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_K_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorVQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_V_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorGQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_G_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorBetaQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_BETA_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorCuSeqlensQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_CU_SEQLENS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorInitialStateQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_INITIAL_STATE_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorALogQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_A_LOG_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorDtBiasQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_DT_BIAS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorStateIndicesQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_STATE_INDICES_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorOQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_O_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorFinalStateQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_FINAL_STATE_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorStateCheckpointsQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_STATE_CHECKPOINTS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeGateDomainQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT,
                                       HIPDNN_TYPE_GATE_DOMAIN_EXT,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeGateDomainQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT,
                                                  HIPDNN_TYPE_GATE_DOMAIN_EXT,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestKdaOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getQDesc(), nullptr);
    ASSERT_NE(desc->getKDesc(), nullptr);
    ASSERT_NE(desc->getVDesc(), nullptr);
    ASSERT_NE(desc->getGDesc(), nullptr);
    ASSERT_NE(desc->getBetaDesc(), nullptr);
    ASSERT_NE(desc->getCuSeqlensDesc(), nullptr);
    ASSERT_NE(desc->getInitialStateDesc(), nullptr);
    ASSERT_NE(desc->getALogDesc(), nullptr);
    ASSERT_NE(desc->getDtBiasDesc(), nullptr);
    ASSERT_NE(desc->getStateIndicesDesc(), nullptr);
    ASSERT_NE(desc->getODesc(), nullptr);
    ASSERT_NE(desc->getFinalStateDesc(), nullptr);
    ASSERT_NE(desc->getStateCheckpointsDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getQDesc()->getData().uid, K_KDA_TENSOR_Q_UID);
    ASSERT_EQ(desc->getKDesc()->getData().uid, K_KDA_TENSOR_K_UID);
    ASSERT_EQ(desc->getVDesc()->getData().uid, K_KDA_TENSOR_V_UID);
    ASSERT_EQ(desc->getGDesc()->getData().uid, K_KDA_TENSOR_G_UID);
    ASSERT_EQ(desc->getBetaDesc()->getData().uid, K_KDA_TENSOR_BETA_UID);
    ASSERT_EQ(desc->getCuSeqlensDesc()->getData().uid, K_KDA_TENSOR_CU_SEQLENS_UID);
    ASSERT_EQ(desc->getInitialStateDesc()->getData().uid, K_KDA_TENSOR_INITIAL_STATE_UID);
    ASSERT_EQ(desc->getALogDesc()->getData().uid, K_KDA_TENSOR_A_LOG_UID);
    ASSERT_EQ(desc->getDtBiasDesc()->getData().uid, K_KDA_TENSOR_DT_BIAS_UID);
    ASSERT_EQ(desc->getStateIndicesDesc()->getData().uid, K_KDA_TENSOR_STATE_INDICES_UID);
    ASSERT_EQ(desc->getODesc()->getData().uid, K_KDA_TENSOR_O_UID);
    ASSERT_EQ(desc->getFinalStateDesc()->getData().uid, K_KDA_TENSOR_FINAL_STATE_UID);
    ASSERT_EQ(desc->getStateCheckpointsDesc()->getData().uid, K_KDA_TENSOR_STATE_CHECKPOINTS_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestKdaOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("KdaOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("q_uid=" + std::to_string(K_KDA_TENSOR_Q_UID)), std::string::npos);
    ASSERT_NE(str.find("k_uid=" + std::to_string(K_KDA_TENSOR_K_UID)), std::string::npos);
    ASSERT_NE(str.find("v_uid=" + std::to_string(K_KDA_TENSOR_V_UID)), std::string::npos);
    ASSERT_NE(str.find("g_uid=" + std::to_string(K_KDA_TENSOR_G_UID)), std::string::npos);
    ASSERT_NE(str.find("beta_uid=" + std::to_string(K_KDA_TENSOR_BETA_UID)), std::string::npos);
    ASSERT_NE(str.find("cu_seqlens_uid=" + std::to_string(K_KDA_TENSOR_CU_SEQLENS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("initial_state_uid=" + std::to_string(K_KDA_TENSOR_INITIAL_STATE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("a_log_uid=" + std::to_string(K_KDA_TENSOR_A_LOG_UID)), std::string::npos);
    ASSERT_NE(str.find("dt_bias_uid=" + std::to_string(K_KDA_TENSOR_DT_BIAS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("state_indices_uid=" + std::to_string(K_KDA_TENSOR_STATE_INDICES_UID)),
              std::string::npos);
    ASSERT_NE(str.find("o_uid=" + std::to_string(K_KDA_TENSOR_O_UID)), std::string::npos);
    ASSERT_NE(str.find("final_state_uid=" + std::to_string(K_KDA_TENSOR_FINAL_STATE_UID)),
              std::string::npos);
    ASSERT_NE(
        str.find("state_checkpoints_uid=" + std::to_string(K_KDA_TENSOR_STATE_CHECKPOINTS_UID)),
        std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestKdaOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 13);
    ASSERT_EQ(tensors[0]->getData().uid, K_KDA_TENSOR_Q_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_KDA_TENSOR_K_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_KDA_TENSOR_V_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_KDA_TENSOR_G_UID);
    ASSERT_EQ(tensors[4]->getData().uid, K_KDA_TENSOR_BETA_UID);
    ASSERT_EQ(tensors[5]->getData().uid, K_KDA_TENSOR_CU_SEQLENS_UID);
    ASSERT_EQ(tensors[6]->getData().uid, K_KDA_TENSOR_INITIAL_STATE_UID);
    ASSERT_EQ(tensors[7]->getData().uid, K_KDA_TENSOR_A_LOG_UID);
    ASSERT_EQ(tensors[8]->getData().uid, K_KDA_TENSOR_DT_BIAS_UID);
    ASSERT_EQ(tensors[9]->getData().uid, K_KDA_TENSOR_STATE_INDICES_UID);
    ASSERT_EQ(tensors[10]->getData().uid, K_KDA_TENSOR_O_UID);
    ASSERT_EQ(tensors[11]->getData().uid, K_KDA_TENSOR_FINAL_STATE_UID);
    ASSERT_EQ(tensors[12]->getData().uid, K_KDA_TENSOR_STATE_CHECKPOINTS_UID);
}

TEST_F(TestKdaOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::KdaAttributes);

    auto* attrs = node->attributes.AsKdaAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->q_tensor_uid, K_KDA_TENSOR_Q_UID);
    ASSERT_EQ(attrs->k_tensor_uid, K_KDA_TENSOR_K_UID);
    ASSERT_EQ(attrs->v_tensor_uid, K_KDA_TENSOR_V_UID);
    ASSERT_EQ(attrs->g_tensor_uid, K_KDA_TENSOR_G_UID);
    ASSERT_EQ(attrs->beta_tensor_uid, K_KDA_TENSOR_BETA_UID);
    ASSERT_EQ(attrs->cu_seqlens_tensor_uid, K_KDA_TENSOR_CU_SEQLENS_UID);
    ASSERT_EQ(attrs->initial_state_tensor_uid, K_KDA_TENSOR_INITIAL_STATE_UID);
    ASSERT_EQ(attrs->a_log_tensor_uid, K_KDA_TENSOR_A_LOG_UID);
    ASSERT_EQ(attrs->dt_bias_tensor_uid, K_KDA_TENSOR_DT_BIAS_UID);
    ASSERT_EQ(attrs->state_indices_tensor_uid, K_KDA_TENSOR_STATE_INDICES_UID);
    ASSERT_EQ(attrs->o_tensor_uid, K_KDA_TENSOR_O_UID);
    ASSERT_EQ(attrs->final_state_tensor_uid, K_KDA_TENSOR_FINAL_STATE_UID);
    ASSERT_EQ(attrs->state_checkpoints_tensor_uid, K_KDA_TENSOR_STATE_CHECKPOINTS_UID);
    EXPECT_EQ(attrs->gate_domain, GateDomain::LOG);
}

TEST_F(TestKdaOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(
    TestKdaOperationDescriptor,
    GetTensorDescriptorsOrderIsQKVGBetaCuSeqlensInitialStateALogDtBiasStateIndicesOFinalStateStateCheckpoints)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 13);
    // Verify ordering: [Q_DESC_EXT, K_DESC_EXT, V_DESC_EXT, G_DESC_EXT, BETA_DESC_EXT, CU_SEQLENS_DESC_EXT, INITIAL_STATE_DESC_EXT, A_LOG_DESC_EXT, DT_BIAS_DESC_EXT, STATE_INDICES_DESC_EXT, O_DESC_EXT, FINAL_STATE_DESC_EXT, STATE_CHECKPOINTS_DESC_EXT] matches UIDs [2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112]
    EXPECT_EQ(tensors[0], desc->getQDesc());
    EXPECT_EQ(tensors[1], desc->getKDesc());
    EXPECT_EQ(tensors[2], desc->getVDesc());
    EXPECT_EQ(tensors[3], desc->getGDesc());
    EXPECT_EQ(tensors[4], desc->getBetaDesc());
    EXPECT_EQ(tensors[5], desc->getCuSeqlensDesc());
    EXPECT_EQ(tensors[6], desc->getInitialStateDesc());
    EXPECT_EQ(tensors[7], desc->getALogDesc());
    EXPECT_EQ(tensors[8], desc->getDtBiasDesc());
    EXPECT_EQ(tensors[9], desc->getStateIndicesDesc());
    EXPECT_EQ(tensors[10], desc->getODesc());
    EXPECT_EQ(tensors[11], desc->getFinalStateDesc());
    EXPECT_EQ(tensors[12], desc->getStateCheckpointsDesc());
}

TEST_F(TestKdaOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 13);
    ASSERT_EQ(tensors[0]->getData().uid, K_KDA_TENSOR_Q_UID);
}

TEST_F(TestKdaOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _qDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestKdaOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_kda_op";

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       static_cast<int64_t>(name.size()),
                                       name.c_str()));

    // Finalize and verify name round-trips
    setRequiredAttributes();
    desc->finalize();

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(name.size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_kda_op");
}

TEST_F(TestKdaOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
{
    auto desc = getDescriptor();
    const std::string name = "my_op";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(name.size()),
                       name.c_str());
    setRequiredAttributes();
    desc->finalize();

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, static_cast<int64_t>(name.size() + 1));
}

// =============================================================================
// Operation Type Tests
// =============================================================================

TEST_F(TestKdaOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_KDA_EXT);
}

TEST_F(TestKdaOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestKdaOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_kda";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_kda");
}
