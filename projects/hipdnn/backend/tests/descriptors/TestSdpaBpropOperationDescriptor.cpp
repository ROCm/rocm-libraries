// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnDataType.h"
#include "HipdnnDiagonalAlignment.h"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/SdpaBpropOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SdpaBpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestSdpaBpropOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<SdpaBpropOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<SdpaBpropOperationDescriptor>();
    }

    void setAllAttributesExcept(std::initializer_list<hipdnnBackendAttributeName_t> skip = {}) const
    {
        auto desc = getDescriptor();
        auto setIf = [&](hipdnnBackendAttributeName_t attr, auto& tensor) {
            if(std::find(skip.begin(), skip.end(), attr) == skip.end())
            {
                desc->setAttribute(attr, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor);
            }
        };
        // Required input tensors
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT, _qDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT, _kDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT, _vDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT, _oDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT, _doDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT, _statsDesc);
        // Required output tensors
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT, _dqDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT, _dkDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT, _dvDesc);
        // Optional tensors
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SCALE_EXT, _scaleDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_ATTN_MASK_EXT, _attnMaskDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEQ_LEN_Q_EXT, _seqLenQDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEQ_LEN_KV_EXT, _seqLenKvDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEED_EXT, _seedDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_OFFSET_EXT, _offsetDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_MASK_EXT, _dropoutMaskDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_SCALE_EXT, _dropoutScaleDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_SCALE_INV_EXT, _dropoutScaleInvDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DBIAS_EXT, _dbiasDesc);
        // Compute data type
        if(std::find(skip.begin(), skip.end(), HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT) == skip.end())
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            desc->setAttribute(
                HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
    }

    void makeFinalized() const
    {
        setAllAttributesExcept();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    // Required input tensors
    std::unique_ptr<HipdnnBackendDescriptor> _qDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _kDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _vDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _oDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _doDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _statsDesc = nullptr;
    // Required output tensors
    std::unique_ptr<HipdnnBackendDescriptor> _dqDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dkDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dvDesc = nullptr;
    // Optional tensors
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _attnMaskDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _seqLenQDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _seqLenKvDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _seedDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _offsetDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dropoutMaskDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dropoutScaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dropoutScaleInvDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dbiasDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<SdpaBpropOperationDescriptor>();
        _qDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_Q_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
        _kDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_K_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
        _vDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_V_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
        _oDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_O_UID,
                                       toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                                       toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
        _doDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
        _statsDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                                           toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
        _dqDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
        _dkDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
        _dvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                                        toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                                        toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));
        _scaleDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SCALE_UID);
        _attnMaskDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);
        _seqLenQDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID);
        _seqLenKvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID);
        _seedDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_SEED_UID);
        _offsetDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_OFFSET_UID);
        _dropoutMaskDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID);
        _dropoutScaleDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID);
        _dropoutScaleInvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID);
        _dbiasDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DBIAS_UID);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_SDPA_BPROP_DESCRIPTOR_EXT);
}

TEST_F(TestSdpaBpropOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setAllAttributesExcept();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestSdpaBpropOperationDescriptor, DoubleFinalizeThrows)
{
    makeFinalized();
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->finalize());
}

class TestSdpaBpropOperationDescriptorFinalizeFailsWithout
    : public TestSdpaBpropOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestSdpaBpropOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setAllAttributesExcept({GetParam()});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestSdpaBpropOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT,
                                           HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT));

// =============================================================================
// SetAttribute Tests - Tensor Descriptors (required)
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorQ)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc));

    ASSERT_EQ(desc->getData().q_tensor_uid, K_SDPA_BPROP_TENSOR_Q_UID);
    ASSERT_NE(desc->getQDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorK)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc));

    ASSERT_EQ(desc->getData().k_tensor_uid, K_SDPA_BPROP_TENSOR_K_UID);
    ASSERT_NE(desc->getKDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorV)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc));

    ASSERT_EQ(desc->getData().v_tensor_uid, K_SDPA_BPROP_TENSOR_V_UID);
    ASSERT_NE(desc->getVDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorO)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc));

    ASSERT_EQ(desc->getData().o_tensor_uid, K_SDPA_BPROP_TENSOR_O_UID);
    ASSERT_NE(desc->getODesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorDo)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_doDesc));

    ASSERT_EQ(desc->getData().do_tensor_uid, K_SDPA_BPROP_TENSOR_DO_UID);
    ASSERT_NE(desc->getDoDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorStats)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_statsDesc));

    ASSERT_EQ(desc->getData().stats_tensor_uid, K_SDPA_BPROP_TENSOR_STATS_UID);
    ASSERT_NE(desc->getStatsDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorDq)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dqDesc));

    ASSERT_EQ(desc->getData().dq_tensor_uid, K_SDPA_BPROP_TENSOR_DQ_UID);
    ASSERT_NE(desc->getDqDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorDk)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dkDesc));

    ASSERT_EQ(desc->getData().dk_tensor_uid, K_SDPA_BPROP_TENSOR_DK_UID);
    ASSERT_NE(desc->getDkDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorDv)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dvDesc));

    ASSERT_EQ(desc->getData().dv_tensor_uid, K_SDPA_BPROP_TENSOR_DV_UID);
    ASSERT_NE(desc->getDvDesc(), nullptr);
}

// =============================================================================
// SetAttribute Tests - Optional Tensor Descriptors
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));
    ASSERT_TRUE(desc->getData().scale_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().scale_tensor_uid.value(), K_SDPA_BPROP_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorAttnMask)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_ATTN_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_attnMaskDesc));
    ASSERT_TRUE(desc->getData().attn_mask_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().attn_mask_tensor_uid.value(), K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);
    ASSERT_NE(desc->getAttnMaskDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorSeqLenQ)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEQ_LEN_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_seqLenQDesc));
    ASSERT_NE(desc->getSeqLenQDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorSeqLenKv)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEQ_LEN_KV_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_seqLenKvDesc));
    ASSERT_NE(desc->getSeqLenKvDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorSeed)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEED_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_seedDesc));
    ASSERT_NE(desc->getSeedDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorOffset)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_OFFSET_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_offsetDesc));
    ASSERT_NE(desc->getOffsetDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorDropoutMask)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dropoutMaskDesc));
    ASSERT_NE(desc->getDropoutMaskDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorDropoutScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dropoutScaleDesc));
    ASSERT_NE(desc->getDropoutScaleDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorDropoutScaleInv)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_SCALE_INV_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dropoutScaleInvDesc));
    ASSERT_NE(desc->getDropoutScaleInvDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorDescriptorDbias)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DBIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dbiasDesc));
    ASSERT_NE(desc->getDBiasDesc(), nullptr);
}

// =============================================================================
// SetAttribute Tests - Boolean flags
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetCausalMask)
{
    auto desc = getDescriptor();
    bool val = true;
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &val));
    ASSERT_TRUE(desc->getData().causal_mask);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetAlibiMask)
{
    auto desc = getDescriptor();
    bool val = true;
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_ALIBI_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &val));
    ASSERT_TRUE(desc->getData().alibi_mask);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetPaddingMask)
{
    auto desc = getDescriptor();
    bool val = true;
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_PADDING_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &val));
    ASSERT_TRUE(desc->getData().padding_mask);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetCausalMaskBottomRight)
{
    auto desc = getDescriptor();
    bool val = true;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_BOTTOM_RIGHT_EXT, HIPDNN_TYPE_BOOLEAN, 1, &val));
    ASSERT_TRUE(desc->getData().causal_mask_bottom_right);
}

// =============================================================================
// SetAttribute Tests - Optional scalars
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetDropoutProbability)
{
    auto desc = getDescriptor();
    float val = 0.1f;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 1, &val));
    ASSERT_TRUE(desc->getData().dropout_probability.has_value());
    ASSERT_FLOAT_EQ(desc->getData().dropout_probability.value(), 0.1f);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetAttnScaleValue)
{
    auto desc = getDescriptor();
    float val = 0.125f;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_ATTN_SCALE_VALUE_EXT, HIPDNN_TYPE_FLOAT, 1, &val));
    ASSERT_TRUE(desc->getData().attn_scale_value.has_value());
    ASSERT_FLOAT_EQ(desc->getData().attn_scale_value.value(), 0.125f);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetLeftBound)
{
    auto desc = getDescriptor();
    int64_t val = 10;
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_LEFT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &val));
    ASSERT_TRUE(desc->getData().left_bound.has_value());
    ASSERT_EQ(desc->getData().left_bound.value(), 10);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetRightBound)
{
    auto desc = getDescriptor();
    int64_t val = 20;
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_RIGHT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &val));
    ASSERT_TRUE(desc->getData().right_bound.has_value());
    ASSERT_EQ(desc->getData().right_bound.value(), 20);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetDiagonalAlignment)
{
    auto desc = getDescriptor();
    auto val = HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_DIAGONAL_ALIGNMENT_EXT, HIPDNN_TYPE_DIAGONAL_ALIGNMENT, 1, &val));
    ASSERT_EQ(desc->getData().diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetOperationName)
{
    auto desc = getDescriptor();
    const std::string name = "my_bprop_op";
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       static_cast<int64_t>(name.size()),
                                       name.c_str()));
}

// =============================================================================
// SetAttribute Tests - Compute data type
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));
    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

// =============================================================================
// SetAttribute on finalized descriptor should throw
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetAttributeOnFinalizedThrows)
{
    makeFinalized();
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

// =============================================================================
// SetAttribute Negative Coverage
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT, HIPDNN_TYPE_FLOAT, 1, &_qDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_qDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetAttributeUnsupportedNameThrows)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetOptionalTensorWrongTypeThrows)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_BPROP_SCALE_EXT, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetOptionalTensorWrongElementCountThrows)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SCALE_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_scaleDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetOptionalScalarWrongTypeThrows)
{
    auto desc = getDescriptor();
    int64_t val = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_BPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_INT64, 1, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetOptionalScalarWrongElementCountThrows)
{
    auto desc = getDescriptor();
    float val = 0.5f;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_BPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 2, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetDiagonalAlignmentWrongElementCount)
{
    auto desc = getDescriptor();
    auto val = HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_BPROP_DIAGONAL_ALIGNMENT_EXT, HIPDNN_TYPE_DIAGONAL_ALIGNMENT, 2, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// GetAttribute Tests - Required Tensor Descriptors
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorQ)
{
    makeFinalized();
    auto desc = getDescriptor();
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&result)));
    const std::unique_ptr<HipdnnBackendDescriptor> owned(result);
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorK)
{
    makeFinalized();
    auto desc = getDescriptor();
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&result)));
    const std::unique_ptr<HipdnnBackendDescriptor> owned(result);
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorV)
{
    makeFinalized();
    auto desc = getDescriptor();
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&result)));
    const std::unique_ptr<HipdnnBackendDescriptor> owned(result);
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorO)
{
    makeFinalized();
    auto desc = getDescriptor();
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&result)));
    const std::unique_ptr<HipdnnBackendDescriptor> owned(result);
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorDo)
{
    makeFinalized();
    auto desc = getDescriptor();
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&result)));
    const std::unique_ptr<HipdnnBackendDescriptor> owned(result);
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorStats)
{
    makeFinalized();
    auto desc = getDescriptor();
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&result)));
    const std::unique_ptr<HipdnnBackendDescriptor> owned(result);
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorDq)
{
    makeFinalized();
    auto desc = getDescriptor();
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&result)));
    const std::unique_ptr<HipdnnBackendDescriptor> owned(result);
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorDk)
{
    makeFinalized();
    auto desc = getDescriptor();
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&result)));
    const std::unique_ptr<HipdnnBackendDescriptor> owned(result);
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorDv)
{
    makeFinalized();
    auto desc = getDescriptor();
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&result)));
    const std::unique_ptr<HipdnnBackendDescriptor> owned(result);
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

// =============================================================================
// GetAttribute Tests - Optional Tensor Descriptors (set and unset)
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, GetOptionalTensorSetReturnsCount1)
{
    // All optional tensors are set via makeFinalized()
    makeFinalized();
    auto desc = getDescriptor();

    // Scale tensor (set)
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&result)));
    const std::unique_ptr<HipdnnBackendDescriptor> owned(result);
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetOptionalTensorUnsetReturnsCount0)
{
    // Build with required-only attributes, no optional tensors
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_doDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_statsDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dqDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dkDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dvDesc);
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    const std::vector<hipdnnBackendAttributeName_t> optionalTensorAttrs = {
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_SCALE_EXT,
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_ATTN_MASK_EXT,
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEQ_LEN_Q_EXT,
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEQ_LEN_KV_EXT,
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_SEED_EXT,
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_OFFSET_EXT,
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_MASK_EXT,
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_SCALE_EXT,
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DROPOUT_SCALE_INV_EXT,
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DBIAS_EXT,
    };

    for(const auto attr : optionalTensorAttrs)
    {
        int64_t count = 99;
        ASSERT_NO_THROW(
            desc->getAttribute(attr, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, nullptr))
            << "getAttribute failed for attr " << attr;
        EXPECT_EQ(count, 0) << "Expected count 0 for unset optional attr " << attr;
    }
}

// =============================================================================
// GetAttribute Tests - Boolean flags
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetAndGetAllBooleanAttributes)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});

    bool trueVal = true;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_ALIBI_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_PADDING_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_BOTTOM_RIGHT_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->finalize();

    bool result = false;
    int64_t count = 0;

    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_ALIBI_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &count, &result);
    ASSERT_EQ(count, 1);
    EXPECT_TRUE(result);

    result = false;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_PADDING_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &count, &result);
    EXPECT_TRUE(result);

    result = false;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &count, &result);
    EXPECT_TRUE(result);

    result = false;
    desc->getAttribute(HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_BOTTOM_RIGHT_EXT,
                       HIPDNN_TYPE_BOOLEAN,
                       1,
                       &count,
                       &result);
    EXPECT_TRUE(result);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetCausalMask)
{
    auto desc = getDescriptor();
    bool val = true;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &val);
    setAllAttributesExcept();
    desc->finalize();

    bool result = false;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &count, &result));
    ASSERT_EQ(count, 1);
    ASSERT_TRUE(result);
}

// =============================================================================
// GetAttribute Tests - Compute data type
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, GetComputeDataType)
{
    makeFinalized();
    auto desc = getDescriptor();
    hipdnnDataType_t result = HIPDNN_DATA_DOUBLE;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &count, &result));
    ASSERT_EQ(count, 1);
    ASSERT_EQ(result, HIPDNN_DATA_FLOAT);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetAndGetComputeDataTypeHalf)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT});
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t result = HIPDNN_DATA_FLOAT;
    int64_t count = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &count, &result);
    ASSERT_EQ(count, 1);
    ASSERT_EQ(result, HIPDNN_DATA_HALF);
}

// =============================================================================
// GetAttribute Tests - Optional float scalars (set and unset)
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetAndGetDropoutProbability)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});
    float dropoutProb = 0.5f;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 1, &dropoutProb);
    desc->finalize();

    float result = 0.0f;
    int64_t count = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 1, &count, &result);
    ASSERT_EQ(count, 1);
    EXPECT_FLOAT_EQ(result, 0.5f);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetDropoutProbabilityUnsetReturnsCount0)
{
    makeFinalized();
    auto desc = getDescriptor();
    // dropout_probability is not explicitly set via makeFinalized, but setAllAttributesExcept
    // does not set optional scalars either, so it should be unset
    int64_t count = 99;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 1, &count, nullptr);
    ASSERT_EQ(count, 0);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetAndGetAttnScaleValue)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});
    float attnScale = 1.5f;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_ATTN_SCALE_VALUE_EXT, HIPDNN_TYPE_FLOAT, 1, &attnScale);
    desc->finalize();

    float result = 0.0f;
    int64_t count = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_ATTN_SCALE_VALUE_EXT, HIPDNN_TYPE_FLOAT, 1, &count, &result);
    ASSERT_EQ(count, 1);
    EXPECT_FLOAT_EQ(result, 1.5f);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetAttnScaleValueUnsetReturnsCount0)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t count = 99;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_ATTN_SCALE_VALUE_EXT, HIPDNN_TYPE_FLOAT, 1, &count, nullptr);
    ASSERT_EQ(count, 0);
}

// =============================================================================
// GetAttribute Tests - Optional int64 scalars (set and unset)
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetAndGetLeftBound)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});
    int64_t leftBound = 10;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_LEFT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &leftBound);
    desc->finalize();

    int64_t result = 0;
    int64_t count = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_LEFT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &count, &result);
    ASSERT_EQ(count, 1);
    EXPECT_EQ(result, 10);
}

TEST_F(TestSdpaBpropOperationDescriptor, SetAndGetRightBound)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});
    int64_t rightBound = 20;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_RIGHT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &rightBound);
    desc->finalize();

    int64_t result = 0;
    int64_t count = 0;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_RIGHT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &count, &result);
    ASSERT_EQ(count, 1);
    EXPECT_EQ(result, 20);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetLeftBoundUnsetReturnsCount0)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t count = 99;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_LEFT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &count, nullptr);
    ASSERT_EQ(count, 0);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetRightBoundUnsetReturnsCount0)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t count = 99;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_BPROP_RIGHT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &count, nullptr);
    ASSERT_EQ(count, 0);
}

// =============================================================================
// GetAttribute Tests - Diagonal alignment
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetAndGetDiagonalAlignment)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});
    auto diagAlign = HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_DIAGONAL_ALIGNMENT_EXT,
                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                       1,
                       &diagAlign);
    desc->finalize();

    hipdnnDiagonalAlignment_t result = HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;
    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_SDPA_BPROP_DIAGONAL_ALIGNMENT_EXT,
                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                       1,
                       &count,
                       &result);
    ASSERT_EQ(count, 1);
    EXPECT_EQ(result, HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetDiagonalAlignmentDefault)
{
    makeFinalized();
    auto desc = getDescriptor();
    auto result = static_cast<hipdnnDiagonalAlignment_t>(-1);
    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_SDPA_BPROP_DIAGONAL_ALIGNMENT_EXT,
                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                       1,
                       &count,
                       &result);
    ASSERT_EQ(count, 1);
    EXPECT_EQ(result, HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT);
}

// =============================================================================
// GetAttribute Tests - Operation name
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, SetAndGetOperationName)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});
    const std::string name = "sdpa_bprop_test";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(name.size()),
                       name.c_str());
    desc->finalize();

    // Query size first
    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(name.size() + 1));

    // Retrieve
    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "sdpa_bprop_test");
}

// =============================================================================
// GetAttribute on unfinalized descriptor should throw
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, GetAttributeOnUnfinalizedThrows)
{
    auto desc = getDescriptor();
    hipdnnBackendDescriptor_t result = nullptr;
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &count,
                                                  &result),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestSdpaBpropOperationDescriptor, GetAttributeUnsupportedNameThrows)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;
    int64_t count = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &count, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// BuildNode Tests
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, BuildNode)
{
    makeFinalized();
    auto desc = getDescriptor();
    auto node = desc->buildNode();

    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::SdpaBackwardAttributes);

    const auto* attrs = node->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->q_tensor_uid, K_SDPA_BPROP_TENSOR_Q_UID);
    ASSERT_EQ(attrs->k_tensor_uid, K_SDPA_BPROP_TENSOR_K_UID);
    ASSERT_EQ(attrs->v_tensor_uid, K_SDPA_BPROP_TENSOR_V_UID);
    ASSERT_EQ(attrs->o_tensor_uid, K_SDPA_BPROP_TENSOR_O_UID);
    ASSERT_EQ(attrs->do_tensor_uid, K_SDPA_BPROP_TENSOR_DO_UID);
    ASSERT_EQ(attrs->stats_tensor_uid, K_SDPA_BPROP_TENSOR_STATS_UID);
    ASSERT_EQ(attrs->dq_tensor_uid, K_SDPA_BPROP_TENSOR_DQ_UID);
    ASSERT_EQ(attrs->dk_tensor_uid, K_SDPA_BPROP_TENSOR_DK_UID);
    ASSERT_EQ(attrs->dv_tensor_uid, K_SDPA_BPROP_TENSOR_DV_UID);
}

TEST_F(TestSdpaBpropOperationDescriptor, BuildNodeWithNonDefaultScalars)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});

    // Set non-default scalar/enum attributes
    auto diagAlign = HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_DIAGONAL_ALIGNMENT_EXT,
                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                       1,
                       &diagAlign);

    bool trueVal = true;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_CAUSAL_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_ALIBI_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);

    float dropoutProb = 0.3f;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 1, &dropoutProb);

    float attnScale = 0.125f;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_ATTN_SCALE_VALUE_EXT, HIPDNN_TYPE_FLOAT, 1, &attnScale);

    int64_t leftBound = 5;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_LEFT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &leftBound);

    int64_t rightBound = 15;
    desc->setAttribute(HIPDNN_ATTR_SDPA_BPROP_RIGHT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &rightBound);

    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);

    const auto* attrs = node->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
    EXPECT_TRUE(attrs->causal_mask);
    EXPECT_TRUE(attrs->alibi_mask);
    ASSERT_TRUE(attrs->dropout_probability.has_value());
    EXPECT_FLOAT_EQ(attrs->dropout_probability.value(), 0.3f);
    ASSERT_TRUE(attrs->attn_scale_value.has_value());
    EXPECT_FLOAT_EQ(attrs->attn_scale_value.value(), 0.125f);
    ASSERT_TRUE(attrs->left_bound.has_value());
    EXPECT_EQ(attrs->left_bound.value(), 5);
    ASSERT_TRUE(attrs->right_bound.has_value());
    EXPECT_EQ(attrs->right_bound.value(), 15);
}

TEST_F(TestSdpaBpropOperationDescriptor, BuildNodePreservesName)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});
    const std::string name = "bprop_node_name";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(name.size()),
                       name.c_str());
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "bprop_node_name");
}

// =============================================================================
// GetTensorDescriptors Tests
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptors)
{
    makeFinalized();
    auto desc = getDescriptor();
    auto tensors = desc->getTensorDescriptors();

    // 9 required + 10 optional tensors
    ASSERT_EQ(tensors.size(), 19u);

    // Build UID->tensor map and verify every expected UID is present
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> tensorByUid;
    std::unordered_set<int64_t> uids;
    for(const auto& t : tensors)
    {
        ASSERT_NE(t, nullptr);
        const auto uid = t->getData().uid;
        uids.insert(uid);
        tensorByUid[uid] = t;
    }

    // Required tensors with dims and strides verification
    const std::vector<std::tuple<int64_t, std::vector<int64_t>, std::vector<int64_t>>>
        requiredTensorsWithDims = {
            {K_SDPA_BPROP_TENSOR_Q_UID,
             toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES)},
            {K_SDPA_BPROP_TENSOR_K_UID,
             toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_K_STRIDES)},
            {K_SDPA_BPROP_TENSOR_V_UID,
             toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_V_STRIDES)},
            {K_SDPA_BPROP_TENSOR_O_UID,
             toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_O_STRIDES)},
            {K_SDPA_BPROP_TENSOR_DO_UID,
             toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES)},
            {K_SDPA_BPROP_TENSOR_STATS_UID,
             toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES)},
            {K_SDPA_BPROP_TENSOR_DQ_UID,
             toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES)},
            {K_SDPA_BPROP_TENSOR_DK_UID,
             toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES)},
            {K_SDPA_BPROP_TENSOR_DV_UID,
             toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES)},
        };

    for(const auto& [uid, expectedDims, expectedStrides] : requiredTensorsWithDims)
    {
        EXPECT_NE(uids.count(uid), 0u) << "Missing required tensor UID " << uid;
        ASSERT_NE(tensorByUid.count(uid), 0u) << "UID " << uid << " not in map";
        const auto& tensor = tensorByUid[uid];
        EXPECT_EQ(tensor->getData().dims, expectedDims) << "UID " << uid << " has mismatched dims";
        EXPECT_EQ(tensor->getData().strides, expectedStrides)
            << "UID " << uid << " has mismatched strides";
    }

    // Optional tensors (only verify UID presence, no dims/strides)
    const std::vector<int64_t> optionalUids = {
        K_SDPA_BPROP_TENSOR_SCALE_UID,
        K_SDPA_BPROP_TENSOR_ATTN_MASK_UID,
        K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID,
        K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID,
        K_SDPA_BPROP_TENSOR_SEED_UID,
        K_SDPA_BPROP_TENSOR_OFFSET_UID,
        K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID,
        K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID,
        K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID,
        K_SDPA_BPROP_TENSOR_DBIAS_UID,
    };
    for(const auto uid : optionalUids)
    {
        EXPECT_NE(uids.count(uid), 0u) << "Missing optional tensor UID " << uid;
    }
}

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorsRequiredOnly)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DO_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_doDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_STATS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_statsDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DQ_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dqDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DK_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dkDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_BPROP_DV_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dvDesc);
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_BPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 9u);

    // Build UID->tensor map and verify dims/strides for each required tensor
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> tensorByUid;
    std::unordered_set<int64_t> uids;
    for(const auto& t : tensors)
    {
        ASSERT_NE(t, nullptr);
        const auto uid = t->getData().uid;
        uids.insert(uid);
        tensorByUid[uid] = t;
    }

    // Required tensors with dims and strides verification
    const std::vector<std::tuple<int64_t, std::vector<int64_t>, std::vector<int64_t>>>
        requiredTensorsWithDims = {
            {K_SDPA_BPROP_TENSOR_Q_UID,
             toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES)},
            {K_SDPA_BPROP_TENSOR_K_UID,
             toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_K_STRIDES)},
            {K_SDPA_BPROP_TENSOR_V_UID,
             toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_V_STRIDES)},
            {K_SDPA_BPROP_TENSOR_O_UID,
             toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_O_STRIDES)},
            {K_SDPA_BPROP_TENSOR_DO_UID,
             toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES)},
            {K_SDPA_BPROP_TENSOR_STATS_UID,
             toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES)},
            {K_SDPA_BPROP_TENSOR_DQ_UID,
             toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES)},
            {K_SDPA_BPROP_TENSOR_DK_UID,
             toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES)},
            {K_SDPA_BPROP_TENSOR_DV_UID,
             toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
             toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES)},
        };

    for(const auto& [uid, expectedDims, expectedStrides] : requiredTensorsWithDims)
    {
        EXPECT_NE(uids.count(uid), 0u) << "Missing required tensor UID " << uid;
        ASSERT_NE(tensorByUid.count(uid), 0u) << "UID " << uid << " not in map";
        const auto& tensor = tensorByUid[uid];
        EXPECT_EQ(tensor->getData().dims, expectedDims) << "UID " << uid << " has mismatched dims";
        EXPECT_EQ(tensor->getData().strides, expectedStrides)
            << "UID " << uid << " has mismatched strides";
    }

    // No optional UIDs should be present
    const std::vector<int64_t> optionalUids = {
        K_SDPA_BPROP_TENSOR_SCALE_UID,
        K_SDPA_BPROP_TENSOR_ATTN_MASK_UID,
        K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID,
        K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID,
        K_SDPA_BPROP_TENSOR_SEED_UID,
        K_SDPA_BPROP_TENSOR_OFFSET_UID,
        K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID,
        K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID,
        K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID,
        K_SDPA_BPROP_TENSOR_DBIAS_UID,
    };
    for(const auto uid : optionalUids)
    {
        EXPECT_EQ(uids.count(uid), 0u) << "Unexpected optional tensor UID " << uid;
    }
}

TEST_F(TestSdpaBpropOperationDescriptor, GetTensorDescriptorsOrder)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 19u);

    EXPECT_EQ(tensors[0], desc->getQDesc());
    EXPECT_EQ(tensors[1], desc->getKDesc());
    EXPECT_EQ(tensors[2], desc->getVDesc());
    EXPECT_EQ(tensors[3], desc->getODesc());
    EXPECT_EQ(tensors[4], desc->getDoDesc());
    EXPECT_EQ(tensors[5], desc->getStatsDesc());
    EXPECT_EQ(tensors[6], desc->getDqDesc());
    EXPECT_EQ(tensors[7], desc->getDkDesc());
    EXPECT_EQ(tensors[8], desc->getDvDesc());
    EXPECT_EQ(tensors[9], desc->getScaleDesc());
    EXPECT_EQ(tensors[10], desc->getAttnMaskDesc());
    EXPECT_EQ(tensors[11], desc->getSeqLenQDesc());
    EXPECT_EQ(tensors[12], desc->getSeqLenKvDesc());
    EXPECT_EQ(tensors[13], desc->getSeedDesc());
    EXPECT_EQ(tensors[14], desc->getOffsetDesc());
    EXPECT_EQ(tensors[15], desc->getDropoutMaskDesc());
    EXPECT_EQ(tensors[16], desc->getDropoutScaleDesc());
    EXPECT_EQ(tensors[17], desc->getDropoutScaleInvDesc());
    EXPECT_EQ(tensors[18], desc->getDBiasDesc());
}

// =============================================================================
// Accessor Tests - Verify UIDs after finalize
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_NE(desc->getQDesc(), nullptr);
    ASSERT_NE(desc->getKDesc(), nullptr);
    ASSERT_NE(desc->getVDesc(), nullptr);
    ASSERT_NE(desc->getODesc(), nullptr);
    ASSERT_NE(desc->getDoDesc(), nullptr);
    ASSERT_NE(desc->getStatsDesc(), nullptr);
    ASSERT_NE(desc->getDqDesc(), nullptr);
    ASSERT_NE(desc->getDkDesc(), nullptr);
    ASSERT_NE(desc->getDvDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    ASSERT_NE(desc->getAttnMaskDesc(), nullptr);
    ASSERT_NE(desc->getSeqLenQDesc(), nullptr);
    ASSERT_NE(desc->getSeqLenKvDesc(), nullptr);
    ASSERT_NE(desc->getSeedDesc(), nullptr);
    ASSERT_NE(desc->getOffsetDesc(), nullptr);
    ASSERT_NE(desc->getDropoutMaskDesc(), nullptr);
    ASSERT_NE(desc->getDropoutScaleDesc(), nullptr);
    ASSERT_NE(desc->getDropoutScaleInvDesc(), nullptr);
    ASSERT_NE(desc->getDBiasDesc(), nullptr);

    ASSERT_EQ(desc->getQDesc()->getData().uid, K_SDPA_BPROP_TENSOR_Q_UID);
    ASSERT_EQ(desc->getKDesc()->getData().uid, K_SDPA_BPROP_TENSOR_K_UID);
    ASSERT_EQ(desc->getVDesc()->getData().uid, K_SDPA_BPROP_TENSOR_V_UID);
    ASSERT_EQ(desc->getODesc()->getData().uid, K_SDPA_BPROP_TENSOR_O_UID);
    ASSERT_EQ(desc->getDoDesc()->getData().uid, K_SDPA_BPROP_TENSOR_DO_UID);
    ASSERT_EQ(desc->getStatsDesc()->getData().uid, K_SDPA_BPROP_TENSOR_STATS_UID);
    ASSERT_EQ(desc->getDqDesc()->getData().uid, K_SDPA_BPROP_TENSOR_DQ_UID);
    ASSERT_EQ(desc->getDkDesc()->getData().uid, K_SDPA_BPROP_TENSOR_DK_UID);
    ASSERT_EQ(desc->getDvDesc()->getData().uid, K_SDPA_BPROP_TENSOR_DV_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_SDPA_BPROP_TENSOR_SCALE_UID);
    ASSERT_EQ(desc->getAttnMaskDesc()->getData().uid, K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);
    ASSERT_EQ(desc->getSeqLenQDesc()->getData().uid, K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID);
    ASSERT_EQ(desc->getSeqLenKvDesc()->getData().uid, K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID);
    ASSERT_EQ(desc->getSeedDesc()->getData().uid, K_SDPA_BPROP_TENSOR_SEED_UID);
    ASSERT_EQ(desc->getOffsetDesc()->getData().uid, K_SDPA_BPROP_TENSOR_OFFSET_UID);
    ASSERT_EQ(desc->getDropoutMaskDesc()->getData().uid, K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID);
    ASSERT_EQ(desc->getDropoutScaleDesc()->getData().uid, K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID);
    ASSERT_EQ(desc->getDropoutScaleInvDesc()->getData().uid,
              K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID);
    ASSERT_EQ(desc->getDBiasDesc()->getData().uid, K_SDPA_BPROP_TENSOR_DBIAS_UID);
}

// =============================================================================
// ToString Tests
// =============================================================================

TEST_F(TestSdpaBpropOperationDescriptor, ToString)
{
    makeFinalized();
    auto desc = getDescriptor();
    auto str = desc->toString();
    ASSERT_FALSE(str.empty());
    ASSERT_NE(str.find("SdpaBpropOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("q_uid="), std::string::npos);
    ASSERT_NE(str.find("dq_uid="), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// fromNode() Tests
// =============================================================================

class TestSdpaBpropOperationFromNode : public ::testing::Test
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

        // Required tensors
        makeTensor(K_SDPA_BPROP_TENSOR_Q_UID,
                   toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                   toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
        makeTensor(K_SDPA_BPROP_TENSOR_K_UID,
                   toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                   toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
        makeTensor(K_SDPA_BPROP_TENSOR_V_UID,
                   toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                   toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
        makeTensor(K_SDPA_BPROP_TENSOR_O_UID,
                   toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                   toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
        makeTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                   toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                   toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
        makeTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                   toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                   toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
        makeTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                   toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                   toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
        makeTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                   toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                   toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
        makeTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                   toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                   toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));

        // Optional tensors
        auto makeScalarTensor = [this](int64_t uid) {
            TensorAttributesT attrs;
            attrs.uid = uid;
            attrs.data_type = DataType::FLOAT;
            attrs.dims = toVec(K_SDPA_BPROP_TENSOR_SCALAR_DIMS);
            attrs.strides = toVec(K_SDPA_BPROP_TENSOR_SCALAR_STRIDES);
            _tensorMap[uid] = TensorDescriptor::fromFlatBuffer(attrs);
        };

        makeScalarTensor(K_SDPA_BPROP_TENSOR_SCALE_UID);
        makeScalarTensor(K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);
        makeScalarTensor(K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID);
        makeScalarTensor(K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID);
        makeScalarTensor(K_SDPA_BPROP_TENSOR_SEED_UID);
        makeScalarTensor(K_SDPA_BPROP_TENSOR_OFFSET_UID);
        makeScalarTensor(K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID);
        makeScalarTensor(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID);
        makeScalarTensor(K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID);
        makeScalarTensor(K_SDPA_BPROP_TENSOR_DBIAS_UID);
    }

    static SdpaBackwardAttributesT createRequiredOnlyAttrs()
    {
        SdpaBackwardAttributesT attrs;
        attrs.q_tensor_uid = K_SDPA_BPROP_TENSOR_Q_UID;
        attrs.k_tensor_uid = K_SDPA_BPROP_TENSOR_K_UID;
        attrs.v_tensor_uid = K_SDPA_BPROP_TENSOR_V_UID;
        attrs.o_tensor_uid = K_SDPA_BPROP_TENSOR_O_UID;
        attrs.do_tensor_uid = K_SDPA_BPROP_TENSOR_DO_UID;
        attrs.stats_tensor_uid = K_SDPA_BPROP_TENSOR_STATS_UID;
        attrs.dq_tensor_uid = K_SDPA_BPROP_TENSOR_DQ_UID;
        attrs.dk_tensor_uid = K_SDPA_BPROP_TENSOR_DK_UID;
        attrs.dv_tensor_uid = K_SDPA_BPROP_TENSOR_DV_UID;
        return attrs;
    }

    static SdpaBackwardAttributesT createAllAttrs()
    {
        auto attrs = createRequiredOnlyAttrs();
        attrs.scale_tensor_uid = K_SDPA_BPROP_TENSOR_SCALE_UID;
        attrs.attn_mask_tensor_uid = K_SDPA_BPROP_TENSOR_ATTN_MASK_UID;
        attrs.seq_len_q_tensor_uid = K_SDPA_BPROP_TENSOR_SEQ_LEN_Q_UID;
        attrs.seq_len_kv_tensor_uid = K_SDPA_BPROP_TENSOR_SEQ_LEN_KV_UID;
        attrs.seed_tensor_uid = K_SDPA_BPROP_TENSOR_SEED_UID;
        attrs.offset_tensor_uid = K_SDPA_BPROP_TENSOR_OFFSET_UID;
        attrs.dropout_mask_tensor_uid = K_SDPA_BPROP_TENSOR_DROPOUT_MASK_UID;
        attrs.dropout_scale_tensor_uid = K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_UID;
        attrs.dropout_scale_inv_tensor_uid = K_SDPA_BPROP_TENSOR_DROPOUT_SCALE_INV_UID;
        attrs.dbias_tensor_uid = K_SDPA_BPROP_TENSOR_DBIAS_UID;
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

TEST_F(TestSdpaBpropOperationFromNode, FromNodeRequiredOnly)
{
    auto node = createRequiredOnlyNode();
    auto desc = SdpaBpropOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getComputeDataType(), DataType::FLOAT);

    EXPECT_EQ(desc->getData().q_tensor_uid, K_SDPA_BPROP_TENSOR_Q_UID);
    EXPECT_EQ(desc->getData().k_tensor_uid, K_SDPA_BPROP_TENSOR_K_UID);
    EXPECT_EQ(desc->getData().v_tensor_uid, K_SDPA_BPROP_TENSOR_V_UID);
    EXPECT_EQ(desc->getData().o_tensor_uid, K_SDPA_BPROP_TENSOR_O_UID);
    EXPECT_EQ(desc->getData().do_tensor_uid, K_SDPA_BPROP_TENSOR_DO_UID);
    EXPECT_EQ(desc->getData().stats_tensor_uid, K_SDPA_BPROP_TENSOR_STATS_UID);
    EXPECT_EQ(desc->getData().dq_tensor_uid, K_SDPA_BPROP_TENSOR_DQ_UID);
    EXPECT_EQ(desc->getData().dk_tensor_uid, K_SDPA_BPROP_TENSOR_DK_UID);
    EXPECT_EQ(desc->getData().dv_tensor_uid, K_SDPA_BPROP_TENSOR_DV_UID);

    // Optional tensors should be null
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

    // Optional scalars should be unset
    EXPECT_FALSE(desc->getData().dropout_probability.has_value());
    EXPECT_FALSE(desc->getData().attn_scale_value.has_value());
    EXPECT_FALSE(desc->getData().left_bound.has_value());
    EXPECT_FALSE(desc->getData().right_bound.has_value());

    // Bool defaults
    EXPECT_FALSE(desc->getData().alibi_mask);
    EXPECT_FALSE(desc->getData().causal_mask);
    EXPECT_FALSE(desc->getData().padding_mask);
    EXPECT_FALSE(desc->getData().causal_mask_bottom_right);
}

TEST_F(TestSdpaBpropOperationFromNode, FromNodeAllFields)
{
    auto node = createAllNode();
    auto desc = SdpaBpropOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensors
    EXPECT_NE(desc->getQDesc(), nullptr);
    EXPECT_NE(desc->getKDesc(), nullptr);
    EXPECT_NE(desc->getVDesc(), nullptr);
    EXPECT_NE(desc->getODesc(), nullptr);
    EXPECT_NE(desc->getDoDesc(), nullptr);
    EXPECT_NE(desc->getStatsDesc(), nullptr);
    EXPECT_NE(desc->getDqDesc(), nullptr);
    EXPECT_NE(desc->getDkDesc(), nullptr);
    EXPECT_NE(desc->getDvDesc(), nullptr);

    // Optional tensors
    EXPECT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_NE(desc->getAttnMaskDesc(), nullptr);
    EXPECT_NE(desc->getSeqLenQDesc(), nullptr);
    EXPECT_NE(desc->getSeqLenKvDesc(), nullptr);
    EXPECT_NE(desc->getSeedDesc(), nullptr);
    EXPECT_NE(desc->getOffsetDesc(), nullptr);
    EXPECT_NE(desc->getDropoutMaskDesc(), nullptr);
    EXPECT_NE(desc->getDropoutScaleDesc(), nullptr);
    EXPECT_NE(desc->getDropoutScaleInvDesc(), nullptr);
    EXPECT_NE(desc->getDBiasDesc(), nullptr);

    // Scalar fields
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

TEST_F(TestSdpaBpropOperationFromNode, FromNodePreservesNameAndComputeType)
{
    auto node = createRequiredOnlyNode(DataType::HALF);
    node.name = "my_bprop_node";
    auto desc = SdpaBpropOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->getComputeDataType(), DataType::HALF);

    auto rebuiltNode = desc->buildNode();
    EXPECT_EQ(rebuiltNode->name, "my_bprop_node");
    EXPECT_EQ(rebuiltNode->compute_data_type, DataType::HALF);
}

TEST_F(TestSdpaBpropOperationFromNode, FromNodeFailsMissingRequiredTensor)
{
    _tensorMap.erase(K_SDPA_BPROP_TENSOR_Q_UID);
    auto node = createRequiredOnlyNode();

    ASSERT_THROW_HIPDNN_STATUS(SdpaBpropOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestSdpaBpropOperationFromNode, FromNodeOptionalTensorUidSetButMissingInMap)
{
    // Set scale_tensor_uid in the node attributes, but remove it from tensor map
    _tensorMap.erase(K_SDPA_BPROP_TENSOR_SCALE_UID);

    auto allAttrs = createAllAttrs();
    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(allAttrs);

    // The implementation uses findOptional which returns nullptr when UID is set
    // but not in the map. This is handled gracefully: the optional desc is just null.
    auto desc = SdpaBpropOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getScaleDesc(), nullptr);
}

TEST_F(TestSdpaBpropOperationFromNode, FromNodeRoundTrip)
{
    auto node = createAllNode();
    auto desc = SdpaBpropOperationDescriptor::fromNode(node, _tensorMap);

    auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::SdpaBackwardAttributes);

    const auto* attrs = rebuiltNode->attributes.AsSdpaBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->q_tensor_uid, K_SDPA_BPROP_TENSOR_Q_UID);
    EXPECT_EQ(attrs->k_tensor_uid, K_SDPA_BPROP_TENSOR_K_UID);
    EXPECT_EQ(attrs->v_tensor_uid, K_SDPA_BPROP_TENSOR_V_UID);
    EXPECT_EQ(attrs->o_tensor_uid, K_SDPA_BPROP_TENSOR_O_UID);
    EXPECT_EQ(attrs->do_tensor_uid, K_SDPA_BPROP_TENSOR_DO_UID);
    EXPECT_EQ(attrs->stats_tensor_uid, K_SDPA_BPROP_TENSOR_STATS_UID);
    EXPECT_EQ(attrs->dq_tensor_uid, K_SDPA_BPROP_TENSOR_DQ_UID);
    EXPECT_EQ(attrs->dk_tensor_uid, K_SDPA_BPROP_TENSOR_DK_UID);
    EXPECT_EQ(attrs->dv_tensor_uid, K_SDPA_BPROP_TENSOR_DV_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_SDPA_BPROP_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->attn_mask_tensor_uid, K_SDPA_BPROP_TENSOR_ATTN_MASK_UID);
    EXPECT_TRUE(attrs->causal_mask);
    EXPECT_TRUE(attrs->alibi_mask);
    ASSERT_TRUE(attrs->dropout_probability.has_value());
    EXPECT_FLOAT_EQ(attrs->dropout_probability.value(), 0.5f);
    EXPECT_EQ(attrs->diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
}

TEST_F(TestSdpaBpropOperationFromNode, FromNodeTensorReferencesMatchMap)
{
    auto node = createRequiredOnlyNode();
    auto desc = SdpaBpropOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getQDesc(), _tensorMap[K_SDPA_BPROP_TENSOR_Q_UID]);
    EXPECT_EQ(desc->getKDesc(), _tensorMap[K_SDPA_BPROP_TENSOR_K_UID]);
    EXPECT_EQ(desc->getVDesc(), _tensorMap[K_SDPA_BPROP_TENSOR_V_UID]);
    EXPECT_EQ(desc->getODesc(), _tensorMap[K_SDPA_BPROP_TENSOR_O_UID]);
    EXPECT_EQ(desc->getDoDesc(), _tensorMap[K_SDPA_BPROP_TENSOR_DO_UID]);
    EXPECT_EQ(desc->getStatsDesc(), _tensorMap[K_SDPA_BPROP_TENSOR_STATS_UID]);
    EXPECT_EQ(desc->getDqDesc(), _tensorMap[K_SDPA_BPROP_TENSOR_DQ_UID]);
    EXPECT_EQ(desc->getDkDesc(), _tensorMap[K_SDPA_BPROP_TENSOR_DK_UID]);
    EXPECT_EQ(desc->getDvDesc(), _tensorMap[K_SDPA_BPROP_TENSOR_DV_UID]);
}
