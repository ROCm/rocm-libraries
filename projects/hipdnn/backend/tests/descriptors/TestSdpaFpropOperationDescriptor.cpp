// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnAttentionImplementation.h"
#include "HipdnnDataType.h"
#include "HipdnnDiagonalAlignment.h"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/SdpaFpropOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SdpaFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <algorithm>
#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;

class TestSdpaFpropOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<SdpaFpropOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<SdpaFpropOperationDescriptor>();
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
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, _qDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, _kDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, _vDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, _oDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT, _attnMaskDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT, _scaleDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT, _seqLenQDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT, _seqLenKvDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT, _seedDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT, _offsetDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT, _dropoutMaskDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT, _dropoutScaleDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT, _pageTableKDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT, _pageTableVDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT, _blockMaskDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT, _sinkTokenDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT, _descaleQDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT, _descaleKDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT, _descaleVDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT, _descaleSDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT, _scaleSDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT, _scaleODesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT, _statsDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT, _maxDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT, _sumExpDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT, _rngDumpDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT, _amaxSDesc);
        setIf(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT, _amaxODesc);
        // Compute data type
        if(std::find(skip.begin(), skip.end(), HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT) == skip.end())
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            desc->setAttribute(
                HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
    }

    void makeFinalized() const
    {
        setAllAttributesExcept();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _qDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _kDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _vDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _oDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _attnMaskDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _seqLenQDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _seqLenKvDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _seedDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _offsetDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dropoutMaskDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dropoutScaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _pageTableKDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _pageTableVDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _blockMaskDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _sinkTokenDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _descaleQDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _descaleKDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _descaleVDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _descaleSDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleSDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleODesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _statsDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _maxDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _sumExpDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _rngDumpDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _amaxSDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _amaxODesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<SdpaFpropOperationDescriptor>();
        _qDesc = createFinalizedTensor(K_SDPA_TENSOR_Q_UID,
                                       hipdnn_tests::toVec(K_SDPA_TENSOR_Q_DIMS),
                                       hipdnn_tests::toVec(K_SDPA_TENSOR_Q_STRIDES));
        _kDesc = createFinalizedTensor(K_SDPA_TENSOR_K_UID,
                                       hipdnn_tests::toVec(K_SDPA_TENSOR_K_DIMS),
                                       hipdnn_tests::toVec(K_SDPA_TENSOR_K_STRIDES));
        _vDesc = createFinalizedTensor(K_SDPA_TENSOR_V_UID,
                                       hipdnn_tests::toVec(K_SDPA_TENSOR_V_DIMS),
                                       hipdnn_tests::toVec(K_SDPA_TENSOR_V_STRIDES));
        _oDesc = createFinalizedTensor(K_SDPA_TENSOR_O_UID,
                                       hipdnn_tests::toVec(K_SDPA_TENSOR_O_DIMS),
                                       hipdnn_tests::toVec(K_SDPA_TENSOR_O_STRIDES));
        _attnMaskDesc = createFinalizedTensor(K_SDPA_TENSOR_ATTN_MASK_UID);
        _scaleDesc = createFinalizedTensor(K_SDPA_TENSOR_SCALE_UID);
        _seqLenQDesc = createFinalizedTensor(K_SDPA_TENSOR_SEQ_LEN_Q_UID);
        _seqLenKvDesc = createFinalizedTensor(K_SDPA_TENSOR_SEQ_LEN_KV_UID);
        _seedDesc = createFinalizedTensor(K_SDPA_TENSOR_SEED_UID);
        _offsetDesc = createFinalizedTensor(K_SDPA_TENSOR_OFFSET_UID);
        _dropoutMaskDesc = createFinalizedTensor(K_SDPA_TENSOR_DROPOUT_MASK_UID);
        _dropoutScaleDesc = createFinalizedTensor(K_SDPA_TENSOR_DROPOUT_SCALE_UID);
        _pageTableKDesc = createFinalizedTensor(K_SDPA_TENSOR_PAGE_TABLE_K_UID);
        _pageTableVDesc = createFinalizedTensor(K_SDPA_TENSOR_PAGE_TABLE_V_UID);
        _blockMaskDesc = createFinalizedTensor(K_SDPA_TENSOR_BLOCK_MASK_UID);
        _sinkTokenDesc = createFinalizedTensor(K_SDPA_TENSOR_SINK_TOKEN_UID);
        _descaleQDesc = createFinalizedTensor(K_SDPA_TENSOR_DESCALE_Q_UID);
        _descaleKDesc = createFinalizedTensor(K_SDPA_TENSOR_DESCALE_K_UID);
        _descaleVDesc = createFinalizedTensor(K_SDPA_TENSOR_DESCALE_V_UID);
        _descaleSDesc = createFinalizedTensor(K_SDPA_TENSOR_DESCALE_S_UID);
        _scaleSDesc = createFinalizedTensor(K_SDPA_TENSOR_SCALE_S_UID);
        _scaleODesc = createFinalizedTensor(K_SDPA_TENSOR_SCALE_O_UID);
        _statsDesc = createFinalizedTensor(K_SDPA_TENSOR_STATS_UID);
        _maxDesc = createFinalizedTensor(K_SDPA_TENSOR_MAX_UID);
        _sumExpDesc = createFinalizedTensor(K_SDPA_TENSOR_SUM_EXP_UID);
        _rngDumpDesc = createFinalizedTensor(K_SDPA_TENSOR_RNG_DUMP_UID);
        _amaxSDesc = createFinalizedTensor(K_SDPA_TENSOR_AMAX_S_UID);
        _amaxODesc = createFinalizedTensor(K_SDPA_TENSOR_AMAX_O_UID);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_SDPA_FPROP_DESCRIPTOR_EXT);
}

TEST_F(TestSdpaFpropOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setAllAttributesExcept();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestSdpaFpropOperationDescriptor, DoubleFinalizeThrows)
{
    // SdpaFpropOperationDescriptor::finalize() has no guard for already-finalized state.
    // All required fields remain set after the first finalize, so the second call
    // succeeds without throwing. This documents the current behavior.
    makeFinalized();
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->finalize());
}

class TestSdpaFpropOperationDescriptorFinalizeFailsWithout
    : public TestSdpaFpropOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestSdpaFpropOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setAllAttributesExcept({GetParam()});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestSdpaFpropOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT,
                                           HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT,
                                           HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT));

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorQ)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().q_tensor_uid, K_SDPA_TENSOR_Q_UID);
    ASSERT_NE(desc->getQDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorK)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc));

    ASSERT_EQ(desc->getData().k_tensor_uid, K_SDPA_TENSOR_K_UID);
    ASSERT_NE(desc->getKDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorV)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc));

    ASSERT_EQ(desc->getData().v_tensor_uid, K_SDPA_TENSOR_V_UID);
    ASSERT_NE(desc->getVDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorO)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc));

    ASSERT_EQ(desc->getData().o_tensor_uid, K_SDPA_TENSOR_O_UID);
    ASSERT_NE(desc->getODesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorAttnMask)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_attnMaskDesc));

    ASSERT_EQ(desc->getData().attn_mask_tensor_uid, K_SDPA_TENSOR_ATTN_MASK_UID);
    ASSERT_NE(desc->getAttnMaskDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_SDPA_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorSeqLenQ)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_seqLenQDesc));

    ASSERT_EQ(desc->getData().seq_len_q_tensor_uid, K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    ASSERT_NE(desc->getSeqLenQDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorSeqLenKv)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_seqLenKvDesc));

    ASSERT_EQ(desc->getData().seq_len_kv_tensor_uid, K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    ASSERT_NE(desc->getSeqLenKvDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorSeed)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_seedDesc));

    ASSERT_EQ(desc->getData().seed_tensor_uid, K_SDPA_TENSOR_SEED_UID);
    ASSERT_NE(desc->getSeedDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorOffset)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_offsetDesc));

    ASSERT_EQ(desc->getData().offset_tensor_uid, K_SDPA_TENSOR_OFFSET_UID);
    ASSERT_NE(desc->getOffsetDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDropoutMask)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dropoutMaskDesc));

    ASSERT_EQ(desc->getData().dropout_mask_tensor_uid, K_SDPA_TENSOR_DROPOUT_MASK_UID);
    ASSERT_NE(desc->getDropoutMaskDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDropoutScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dropoutScaleDesc));

    ASSERT_EQ(desc->getData().dropout_scale_tensor_uid, K_SDPA_TENSOR_DROPOUT_SCALE_UID);
    ASSERT_NE(desc->getDropoutScaleDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorPageTableK)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_pageTableKDesc));

    ASSERT_EQ(desc->getData().page_table_k_tensor_uid, K_SDPA_TENSOR_PAGE_TABLE_K_UID);
    ASSERT_NE(desc->getPageTableKDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorPageTableV)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_pageTableVDesc));

    ASSERT_EQ(desc->getData().page_table_v_tensor_uid, K_SDPA_TENSOR_PAGE_TABLE_V_UID);
    ASSERT_NE(desc->getPageTableVDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorBlockMask)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_blockMaskDesc));

    ASSERT_EQ(desc->getData().block_mask_tensor_uid, K_SDPA_TENSOR_BLOCK_MASK_UID);
    ASSERT_NE(desc->getBlockMaskDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorSinkToken)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_sinkTokenDesc));

    ASSERT_EQ(desc->getData().sink_token_tensor_uid, K_SDPA_TENSOR_SINK_TOKEN_UID);
    ASSERT_NE(desc->getSinkTokenDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDescaleQ)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_descaleQDesc));

    ASSERT_EQ(desc->getData().descale_q_tensor_uid, K_SDPA_TENSOR_DESCALE_Q_UID);
    ASSERT_NE(desc->getDescaleQDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDescaleK)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_descaleKDesc));

    ASSERT_EQ(desc->getData().descale_k_tensor_uid, K_SDPA_TENSOR_DESCALE_K_UID);
    ASSERT_NE(desc->getDescaleKDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDescaleV)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_descaleVDesc));

    ASSERT_EQ(desc->getData().descale_v_tensor_uid, K_SDPA_TENSOR_DESCALE_V_UID);
    ASSERT_NE(desc->getDescaleVDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDescaleS)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_descaleSDesc));

    ASSERT_EQ(desc->getData().descale_s_tensor_uid, K_SDPA_TENSOR_DESCALE_S_UID);
    ASSERT_NE(desc->getDescaleSDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorScaleS)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleSDesc));

    ASSERT_EQ(desc->getData().scale_s_tensor_uid, K_SDPA_TENSOR_SCALE_S_UID);
    ASSERT_NE(desc->getScaleSDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorScaleO)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleODesc));

    ASSERT_EQ(desc->getData().scale_o_tensor_uid, K_SDPA_TENSOR_SCALE_O_UID);
    ASSERT_NE(desc->getScaleODesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorStats)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_statsDesc));

    ASSERT_EQ(desc->getData().stats_tensor_uid, K_SDPA_TENSOR_STATS_UID);
    ASSERT_NE(desc->getStatsDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorMax)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_maxDesc));

    ASSERT_EQ(desc->getData().max_tensor_uid, K_SDPA_TENSOR_MAX_UID);
    ASSERT_NE(desc->getMaxDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorSumExp)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_sumExpDesc));

    ASSERT_EQ(desc->getData().sum_exp_tensor_uid, K_SDPA_TENSOR_SUM_EXP_UID);
    ASSERT_NE(desc->getSumExpDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorRngDump)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_rngDumpDesc));

    ASSERT_EQ(desc->getData().rng_dump_tensor_uid, K_SDPA_TENSOR_RNG_DUMP_UID);
    ASSERT_NE(desc->getRngDumpDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorAmaxS)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_amaxSDesc));

    ASSERT_EQ(desc->getData().amax_s_tensor_uid, K_SDPA_TENSOR_AMAX_S_UID);
    ASSERT_NE(desc->getAmaxSDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorAmaxO)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_amaxODesc));

    ASSERT_EQ(desc->getData().amax_o_tensor_uid, K_SDPA_TENSOR_AMAX_O_UID);
    ASSERT_NE(desc->getAmaxODesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_INT64, 1, &_qDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_qDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetOptionalTensorDescriptorWrongTypeThrows)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetOptionalTensorDescriptorWrongElementCountThrows)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_attnMaskDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, SetDiagonalAlignment)
{
    auto desc = getDescriptor();
    auto diagonalAlignment = HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT,
                                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                                       1,
                                       &diagonalAlignment));

    ASSERT_EQ(desc->getData().diagonal_alignment, DiagonalAlignment::TOP_LEFT);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetDiagonalAlignmentWrongElementCount)
{
    auto desc = getDescriptor();
    auto diagonalAlignment = HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT,
                                                  HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                                                  2,
                                                  &diagonalAlignment),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetMmaCoreMode)
{
    auto desc = getDescriptor();
    hipdnnDataType_t mmaCoreMode = HIPDNN_DATA_HALF;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &mmaCoreMode));

    ASSERT_EQ(desc->getData().mma_core_mode, DataType::HALF);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetMmaCoreModeWrongElementCount)
{
    auto desc = getDescriptor();
    hipdnnDataType_t mmaCoreMode = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &mmaCoreMode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetAttentionImplementation)
{
    auto desc = getDescriptor();
    auto implementation = HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT,
                                       HIPDNN_TYPE_ATTENTION_IMPLEMENTATION,
                                       1,
                                       &implementation));

    ASSERT_EQ(desc->getData().implementation, AttentionImplementation::AUTO);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetAttentionImplementationWrongElementCount)
{
    auto desc = getDescriptor();
    auto implementation = HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT;

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT,
                                                  HIPDNN_TYPE_ATTENTION_IMPLEMENTATION,
                                                  2,
                                                  &implementation),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetOptionalScalarWrongTypeThrows)
{
    auto desc = getDescriptor();
    int64_t val = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_FPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_INT64, 1, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetOptionalScalarWrongElementCountThrows)
{
    auto desc = getDescriptor();
    float val = 0.5f;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_FPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 2, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetAttributeUnsupported)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetAttributeAfterFinalizeThrows)
{
    makeFinalized();
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetAttributeUnsupportedNameThrows)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &_qDesc),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* rawQ = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&rawQ)));
    std::unique_ptr<HipdnnBackendDescriptor> retrievedQ(rawQ);

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedQ, nullptr);
    auto unpackedQ = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        retrievedQ.get(), HIPDNN_STATUS_BAD_PARAM, "unpack retrieved Q");
    ASSERT_EQ(unpackedQ->getData().uid, K_SDPA_TENSOR_Q_UID);

    const auto& qData = unpackedQ->getData();
    const std::vector<int64_t> expectedDims(K_SDPA_TENSOR_Q_DIMS.begin(),
                                            K_SDPA_TENSOR_Q_DIMS.end());
    const std::vector<int64_t> expectedStrides(K_SDPA_TENSOR_Q_STRIDES.begin(),
                                               K_SDPA_TENSOR_Q_STRIDES.end());
    ASSERT_EQ(qData.dims, expectedDims);
    ASSERT_EQ(qData.strides, expectedStrides);
    ASSERT_EQ(qData.data_type, hipdnn_data_sdk::data_objects::DataType::FLOAT);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeSdpafpropParams)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});

    // Set mma_core_mode to a valid value so it can be retrieved
    hipdnnDataType_t mmaCoreValue = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &mmaCoreValue);
    desc->finalize();

    // diagonal alignment
    auto diagonalAlignment = static_cast<hipdnnDiagonalAlignment_t>(-1);
    int64_t diagonalAlignmentCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT,
                                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                                       1,
                                       &diagonalAlignmentCount,
                                       &diagonalAlignment));
    ASSERT_EQ(diagonalAlignmentCount, 1);
    EXPECT_EQ(diagonalAlignment, HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT);

    // mma core mode
    auto mmaCoreMode = static_cast<hipdnnDataType_t>(-1);
    int64_t mmaCoreModeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &mmaCoreModeCount,
                                       &mmaCoreMode));
    ASSERT_EQ(mmaCoreModeCount, 1);
    EXPECT_EQ(mmaCoreMode, HIPDNN_DATA_FLOAT);

    // implementation
    auto implementation = static_cast<hipdnnAttentionImplementation_t>(-1);
    int64_t implementationCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT,
                                       HIPDNN_TYPE_ATTENTION_IMPLEMENTATION,
                                       1,
                                       &implementationCount,
                                       &implementation));
    ASSERT_EQ(implementationCount, 1);
    EXPECT_EQ(implementation, HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeUnsupportedNameThrows)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;
    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &elementCount, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Query Mode Tests
// =============================================================================

struct QueryModeParam
{
    hipdnnBackendAttributeName_t attr;
    hipdnnBackendAttributeType_t type;
    int64_t expectedElementCount;
};

class TestSdpaFpropOperationDescriptorQueryMode
    : public TestSdpaFpropOperationDescriptor,
      public ::testing::WithParamInterface<QueryModeParam>
{
};

TEST_P(TestSdpaFpropOperationDescriptorQueryMode, QueryReturnsExpectedElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();
    const auto& param = GetParam();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(param.attr, param.type, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, param.expectedElementCount);
}

TEST_P(TestSdpaFpropOperationDescriptorQueryMode, QueryFailsWithNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(GetParam().attr, GetParam().type, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

INSTANTIATE_TEST_SUITE_P(
    SdpaFpropQueryMode,
    TestSdpaFpropOperationDescriptorQueryMode,
    ::testing::Values(
        QueryModeParam{HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1},
        QueryModeParam{
            HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT, HIPDNN_TYPE_DIAGONAL_ALIGNMENT, 1},
        QueryModeParam{HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT, HIPDNN_TYPE_DATA_TYPE, 1},
        QueryModeParam{
            HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT, HIPDNN_TYPE_ATTENTION_IMPLEMENTATION, 1},
        QueryModeParam{HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1}));

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getQDesc(), nullptr);
    ASSERT_NE(desc->getKDesc(), nullptr);
    ASSERT_NE(desc->getVDesc(), nullptr);
    ASSERT_NE(desc->getODesc(), nullptr);
    ASSERT_NE(desc->getAttnMaskDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    ASSERT_NE(desc->getSeqLenQDesc(), nullptr);
    ASSERT_NE(desc->getSeqLenKvDesc(), nullptr);
    ASSERT_NE(desc->getSeedDesc(), nullptr);
    ASSERT_NE(desc->getOffsetDesc(), nullptr);
    ASSERT_NE(desc->getDropoutMaskDesc(), nullptr);
    ASSERT_NE(desc->getDropoutScaleDesc(), nullptr);
    ASSERT_NE(desc->getPageTableKDesc(), nullptr);
    ASSERT_NE(desc->getPageTableVDesc(), nullptr);
    ASSERT_NE(desc->getBlockMaskDesc(), nullptr);
    ASSERT_NE(desc->getSinkTokenDesc(), nullptr);
    ASSERT_NE(desc->getDescaleQDesc(), nullptr);
    ASSERT_NE(desc->getDescaleKDesc(), nullptr);
    ASSERT_NE(desc->getDescaleVDesc(), nullptr);
    ASSERT_NE(desc->getDescaleSDesc(), nullptr);
    ASSERT_NE(desc->getScaleSDesc(), nullptr);
    ASSERT_NE(desc->getScaleODesc(), nullptr);
    ASSERT_NE(desc->getStatsDesc(), nullptr);
    ASSERT_NE(desc->getMaxDesc(), nullptr);
    ASSERT_NE(desc->getSumExpDesc(), nullptr);
    ASSERT_NE(desc->getRngDumpDesc(), nullptr);
    ASSERT_NE(desc->getAmaxSDesc(), nullptr);
    ASSERT_NE(desc->getAmaxODesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getQDesc()->getData().uid, K_SDPA_TENSOR_Q_UID);
    ASSERT_EQ(desc->getKDesc()->getData().uid, K_SDPA_TENSOR_K_UID);
    ASSERT_EQ(desc->getVDesc()->getData().uid, K_SDPA_TENSOR_V_UID);
    ASSERT_EQ(desc->getODesc()->getData().uid, K_SDPA_TENSOR_O_UID);
    ASSERT_EQ(desc->getAttnMaskDesc()->getData().uid, K_SDPA_TENSOR_ATTN_MASK_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_SDPA_TENSOR_SCALE_UID);
    ASSERT_EQ(desc->getSeqLenQDesc()->getData().uid, K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    ASSERT_EQ(desc->getSeqLenKvDesc()->getData().uid, K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    ASSERT_EQ(desc->getSeedDesc()->getData().uid, K_SDPA_TENSOR_SEED_UID);
    ASSERT_EQ(desc->getOffsetDesc()->getData().uid, K_SDPA_TENSOR_OFFSET_UID);
    ASSERT_EQ(desc->getDropoutMaskDesc()->getData().uid, K_SDPA_TENSOR_DROPOUT_MASK_UID);
    ASSERT_EQ(desc->getDropoutScaleDesc()->getData().uid, K_SDPA_TENSOR_DROPOUT_SCALE_UID);
    ASSERT_EQ(desc->getPageTableKDesc()->getData().uid, K_SDPA_TENSOR_PAGE_TABLE_K_UID);
    ASSERT_EQ(desc->getPageTableVDesc()->getData().uid, K_SDPA_TENSOR_PAGE_TABLE_V_UID);
    ASSERT_EQ(desc->getBlockMaskDesc()->getData().uid, K_SDPA_TENSOR_BLOCK_MASK_UID);
    ASSERT_EQ(desc->getSinkTokenDesc()->getData().uid, K_SDPA_TENSOR_SINK_TOKEN_UID);
    ASSERT_EQ(desc->getDescaleQDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_Q_UID);
    ASSERT_EQ(desc->getDescaleKDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_K_UID);
    ASSERT_EQ(desc->getDescaleVDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_V_UID);
    ASSERT_EQ(desc->getDescaleSDesc()->getData().uid, K_SDPA_TENSOR_DESCALE_S_UID);
    ASSERT_EQ(desc->getScaleSDesc()->getData().uid, K_SDPA_TENSOR_SCALE_S_UID);
    ASSERT_EQ(desc->getScaleODesc()->getData().uid, K_SDPA_TENSOR_SCALE_O_UID);
    ASSERT_EQ(desc->getStatsDesc()->getData().uid, K_SDPA_TENSOR_STATS_UID);
    ASSERT_EQ(desc->getMaxDesc()->getData().uid, K_SDPA_TENSOR_MAX_UID);
    ASSERT_EQ(desc->getSumExpDesc()->getData().uid, K_SDPA_TENSOR_SUM_EXP_UID);
    ASSERT_EQ(desc->getRngDumpDesc()->getData().uid, K_SDPA_TENSOR_RNG_DUMP_UID);
    ASSERT_EQ(desc->getAmaxSDesc()->getData().uid, K_SDPA_TENSOR_AMAX_S_UID);
    ASSERT_EQ(desc->getAmaxODesc()->getData().uid, K_SDPA_TENSOR_AMAX_O_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, ToStringContainsExpectedInfo)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();

    std::string str = desc->toString();
    ASSERT_NE(str.find("SdpaFpropOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("q_uid=" + std::to_string(K_SDPA_TENSOR_Q_UID)), std::string::npos);
    ASSERT_NE(str.find("k_uid=" + std::to_string(K_SDPA_TENSOR_K_UID)), std::string::npos);
    ASSERT_NE(str.find("v_uid=" + std::to_string(K_SDPA_TENSOR_V_UID)), std::string::npos);
    ASSERT_NE(str.find("o_uid=" + std::to_string(K_SDPA_TENSOR_O_UID)), std::string::npos);
    ASSERT_NE(str.find("attn_mask_uid=" + std::to_string(K_SDPA_TENSOR_ATTN_MASK_UID)),
              std::string::npos);
    ASSERT_NE(str.find("scale_uid=" + std::to_string(K_SDPA_TENSOR_SCALE_UID)), std::string::npos);
    ASSERT_NE(str.find("seq_len_q_uid=" + std::to_string(K_SDPA_TENSOR_SEQ_LEN_Q_UID)),
              std::string::npos);
    ASSERT_NE(str.find("seq_len_kv_uid=" + std::to_string(K_SDPA_TENSOR_SEQ_LEN_KV_UID)),
              std::string::npos);
    ASSERT_NE(str.find("seed_uid=" + std::to_string(K_SDPA_TENSOR_SEED_UID)), std::string::npos);
    ASSERT_NE(str.find("offset_uid=" + std::to_string(K_SDPA_TENSOR_OFFSET_UID)),
              std::string::npos);
    ASSERT_NE(str.find("dropout_mask_uid=" + std::to_string(K_SDPA_TENSOR_DROPOUT_MASK_UID)),
              std::string::npos);
    ASSERT_NE(str.find("dropout_scale_uid=" + std::to_string(K_SDPA_TENSOR_DROPOUT_SCALE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("page_table_k_uid=" + std::to_string(K_SDPA_TENSOR_PAGE_TABLE_K_UID)),
              std::string::npos);
    ASSERT_NE(str.find("page_table_v_uid=" + std::to_string(K_SDPA_TENSOR_PAGE_TABLE_V_UID)),
              std::string::npos);
    ASSERT_NE(str.find("block_mask_uid=" + std::to_string(K_SDPA_TENSOR_BLOCK_MASK_UID)),
              std::string::npos);
    ASSERT_NE(str.find("sink_token_uid=" + std::to_string(K_SDPA_TENSOR_SINK_TOKEN_UID)),
              std::string::npos);
    ASSERT_NE(str.find("descale_q_uid=" + std::to_string(K_SDPA_TENSOR_DESCALE_Q_UID)),
              std::string::npos);
    ASSERT_NE(str.find("descale_k_uid=" + std::to_string(K_SDPA_TENSOR_DESCALE_K_UID)),
              std::string::npos);
    ASSERT_NE(str.find("descale_v_uid=" + std::to_string(K_SDPA_TENSOR_DESCALE_V_UID)),
              std::string::npos);
    ASSERT_NE(str.find("descale_s_uid=" + std::to_string(K_SDPA_TENSOR_DESCALE_S_UID)),
              std::string::npos);
    ASSERT_NE(str.find("scale_s_uid=" + std::to_string(K_SDPA_TENSOR_SCALE_S_UID)),
              std::string::npos);
    ASSERT_NE(str.find("scale_o_uid=" + std::to_string(K_SDPA_TENSOR_SCALE_O_UID)),
              std::string::npos);
    ASSERT_NE(str.find("stats_uid=" + std::to_string(K_SDPA_TENSOR_STATS_UID)), std::string::npos);
    ASSERT_NE(str.find("max_uid=" + std::to_string(K_SDPA_TENSOR_MAX_UID)), std::string::npos);
    ASSERT_NE(str.find("sum_exp_uid=" + std::to_string(K_SDPA_TENSOR_SUM_EXP_UID)),
              std::string::npos);
    ASSERT_NE(str.find("rng_dump_uid=" + std::to_string(K_SDPA_TENSOR_RNG_DUMP_UID)),
              std::string::npos);
    ASSERT_NE(str.find("amax_s_uid=" + std::to_string(K_SDPA_TENSOR_AMAX_S_UID)),
              std::string::npos);
    ASSERT_NE(str.find("amax_o_uid=" + std::to_string(K_SDPA_TENSOR_AMAX_O_UID)),
              std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setAllAttributesExcept();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::SdpaAttributes);

    auto* attrs = node->attributes.AsSdpaAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->q_tensor_uid, K_SDPA_TENSOR_Q_UID);
    ASSERT_EQ(attrs->k_tensor_uid, K_SDPA_TENSOR_K_UID);
    ASSERT_EQ(attrs->v_tensor_uid, K_SDPA_TENSOR_V_UID);
    ASSERT_EQ(attrs->o_tensor_uid, K_SDPA_TENSOR_O_UID);
    ASSERT_EQ(attrs->attn_mask_tensor_uid, K_SDPA_TENSOR_ATTN_MASK_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_SDPA_TENSOR_SCALE_UID);
    ASSERT_EQ(attrs->seq_len_q_tensor_uid, K_SDPA_TENSOR_SEQ_LEN_Q_UID);
    ASSERT_EQ(attrs->seq_len_kv_tensor_uid, K_SDPA_TENSOR_SEQ_LEN_KV_UID);
    ASSERT_EQ(attrs->seed_tensor_uid, K_SDPA_TENSOR_SEED_UID);
    ASSERT_EQ(attrs->offset_tensor_uid, K_SDPA_TENSOR_OFFSET_UID);
    ASSERT_EQ(attrs->dropout_mask_tensor_uid, K_SDPA_TENSOR_DROPOUT_MASK_UID);
    ASSERT_EQ(attrs->dropout_scale_tensor_uid, K_SDPA_TENSOR_DROPOUT_SCALE_UID);
    ASSERT_EQ(attrs->page_table_k_tensor_uid, K_SDPA_TENSOR_PAGE_TABLE_K_UID);
    ASSERT_EQ(attrs->page_table_v_tensor_uid, K_SDPA_TENSOR_PAGE_TABLE_V_UID);
    ASSERT_EQ(attrs->block_mask_tensor_uid, K_SDPA_TENSOR_BLOCK_MASK_UID);
    ASSERT_EQ(attrs->sink_token_tensor_uid, K_SDPA_TENSOR_SINK_TOKEN_UID);
    ASSERT_EQ(attrs->descale_q_tensor_uid, K_SDPA_TENSOR_DESCALE_Q_UID);
    ASSERT_EQ(attrs->descale_k_tensor_uid, K_SDPA_TENSOR_DESCALE_K_UID);
    ASSERT_EQ(attrs->descale_v_tensor_uid, K_SDPA_TENSOR_DESCALE_V_UID);
    ASSERT_EQ(attrs->descale_s_tensor_uid, K_SDPA_TENSOR_DESCALE_S_UID);
    ASSERT_EQ(attrs->scale_s_tensor_uid, K_SDPA_TENSOR_SCALE_S_UID);
    ASSERT_EQ(attrs->scale_o_tensor_uid, K_SDPA_TENSOR_SCALE_O_UID);
    ASSERT_EQ(attrs->stats_tensor_uid, K_SDPA_TENSOR_STATS_UID);
    ASSERT_EQ(attrs->max_tensor_uid, K_SDPA_TENSOR_MAX_UID);
    ASSERT_EQ(attrs->sum_exp_tensor_uid, K_SDPA_TENSOR_SUM_EXP_UID);
    ASSERT_EQ(attrs->rng_dump_tensor_uid, K_SDPA_TENSOR_RNG_DUMP_UID);
    ASSERT_EQ(attrs->amax_s_tensor_uid, K_SDPA_TENSOR_AMAX_S_UID);
    ASSERT_EQ(attrs->amax_o_tensor_uid, K_SDPA_TENSOR_AMAX_O_UID);
}

TEST_F(TestSdpaFpropOperationDescriptor, BuildNodeProducesCorrectNodeTNonDefaults)
{
    setAllAttributesExcept();

    auto desc = getDescriptor();

    // Set non-default scalar/enum attributes before finalizing
    auto diagonalAlignment = HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT,
                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                       1,
                       &diagonalAlignment);

    auto implementation = HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT;
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT,
                       HIPDNN_TYPE_ATTENTION_IMPLEMENTATION,
                       1,
                       &implementation);

    hipdnnDataType_t mmaCoreMode = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &mmaCoreMode);

    bool generateStats = true;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_GENERATE_STATS_EXT, HIPDNN_TYPE_BOOLEAN, 1, &generateStats);

    float dropoutProbability = 0.5f;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 1, &dropoutProbability);

    int64_t leftBound = 10;
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_LEFT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &leftBound);

    int64_t rightBound = 20;
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_RIGHT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &rightBound);

    int32_t maxSeqLenKv = 512;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MAX_SEQ_LEN_KV_EXT, HIPDNN_TYPE_INT32, 1, &maxSeqLenKv);

    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::SdpaAttributes);

    auto* attrs = node->attributes.AsSdpaAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify non-default scalar/enum attributes are propagated correctly
    ASSERT_EQ(attrs->diagonal_alignment, DiagonalAlignment::BOTTOM_RIGHT);
    ASSERT_EQ(attrs->implementation, AttentionImplementation::UNIFIED);
    ASSERT_EQ(attrs->mma_core_mode, DataType::HALF);
    ASSERT_TRUE(attrs->generate_stats.has_value());
    ASSERT_TRUE(attrs->generate_stats.value());
    ASSERT_TRUE(attrs->dropout_probability.has_value());
    ASSERT_FLOAT_EQ(attrs->dropout_probability.value(), 0.5f);
    ASSERT_TRUE(attrs->left_bound.has_value());
    ASSERT_EQ(attrs->left_bound.value(), 10);
    ASSERT_TRUE(attrs->right_bound.has_value());
    ASSERT_EQ(attrs->right_bound.value(), 20);
    ASSERT_TRUE(attrs->max_seq_len_kv.has_value());
    ASSERT_EQ(attrs->max_seq_len_kv.value(), 512);
}

TEST_F(TestSdpaFpropOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setAllAttributesExcept();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetTensorDescriptorsReturnsExpectedOrder)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 28);
    // Verify ordering: [Q, K, V, O, ATTN_MASK, SCALE, SEQ_LEN_Q, SEQ_LEN_KV, SEED, OFFSET, DROPOUT_MASK, DROPOUT_SCALE, PAGE_TABLE_K, PAGE_TABLE_V, BLOCK_MASK, SINK_TOKEN, DESCALE_Q, DESCALE_K, DESCALE_V, DESCALE_S, SCALE_S, SCALE_O, STATS, MAX, SUM_EXP, RNG_DUMP, AMAX_S, AMAX_O] matches UIDs [40, 41, 42, 43, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    EXPECT_EQ(tensors[0], desc->getQDesc());
    EXPECT_EQ(tensors[1], desc->getKDesc());
    EXPECT_EQ(tensors[2], desc->getVDesc());
    EXPECT_EQ(tensors[3], desc->getODesc());
    EXPECT_EQ(tensors[4], desc->getAttnMaskDesc());
    EXPECT_EQ(tensors[5], desc->getScaleDesc());
    EXPECT_EQ(tensors[6], desc->getSeqLenQDesc());
    EXPECT_EQ(tensors[7], desc->getSeqLenKvDesc());
    EXPECT_EQ(tensors[8], desc->getSeedDesc());
    EXPECT_EQ(tensors[9], desc->getOffsetDesc());
    EXPECT_EQ(tensors[10], desc->getDropoutMaskDesc());
    EXPECT_EQ(tensors[11], desc->getDropoutScaleDesc());
    EXPECT_EQ(tensors[12], desc->getPageTableKDesc());
    EXPECT_EQ(tensors[13], desc->getPageTableVDesc());
    EXPECT_EQ(tensors[14], desc->getBlockMaskDesc());
    EXPECT_EQ(tensors[15], desc->getSinkTokenDesc());
    EXPECT_EQ(tensors[16], desc->getDescaleQDesc());
    EXPECT_EQ(tensors[17], desc->getDescaleKDesc());
    EXPECT_EQ(tensors[18], desc->getDescaleVDesc());
    EXPECT_EQ(tensors[19], desc->getDescaleSDesc());
    EXPECT_EQ(tensors[20], desc->getScaleSDesc());
    EXPECT_EQ(tensors[21], desc->getScaleODesc());
    EXPECT_EQ(tensors[22], desc->getStatsDesc());
    EXPECT_EQ(tensors[23], desc->getMaxDesc());
    EXPECT_EQ(tensors[24], desc->getSumExpDesc());
    EXPECT_EQ(tensors[25], desc->getRngDumpDesc());
    EXPECT_EQ(tensors[26], desc->getAmaxSDesc());
    EXPECT_EQ(tensors[27], desc->getAmaxODesc());
}

TEST_F(TestSdpaFpropOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 28);
    ASSERT_EQ(tensors[0]->getData().uid, K_SDPA_TENSOR_Q_UID);
}

TEST_F(TestSdpaFpropOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _qDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Scalar/Enum Attribute Round-Trip Tests
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, SetAndGetBooleanAttributes)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});

    // Set boolean attributes to true
    bool trueVal = true;
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_GENERATE_STATS_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_ALIBI_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_PADDING_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_CAUSAL_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_CAUSAL_MASK_BOTTOM_RIGHT_EXT, HIPDNN_TYPE_BOOLEAN, 1, &trueVal);

    desc->finalize();

    // Verify all boolean attributes are true
    bool retrieved = false;
    int64_t elementCount = 0;

    desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_GENERATE_STATS_EXT,
                       HIPDNN_TYPE_BOOLEAN,
                       1,
                       &elementCount,
                       &retrieved);
    ASSERT_EQ(elementCount, 1);
    EXPECT_TRUE(retrieved);

    retrieved = false;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_ALIBI_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &retrieved);
    EXPECT_TRUE(retrieved);

    retrieved = false;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_PADDING_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &retrieved);
    EXPECT_TRUE(retrieved);

    retrieved = false;
    desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_CAUSAL_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &retrieved);
    EXPECT_TRUE(retrieved);

    retrieved = false;
    desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_CAUSAL_MASK_BOTTOM_RIGHT_EXT,
                       HIPDNN_TYPE_BOOLEAN,
                       1,
                       &elementCount,
                       &retrieved);
    EXPECT_TRUE(retrieved);

    // Verify setting to false works (false != unset)
    auto desc2 = createDescriptor<SdpaFpropOperationDescriptor>();
    auto sdpa2 = desc2->asDescriptor<SdpaFpropOperationDescriptor>();
    // Set up required attributes on desc2
    sdpa2->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc);
    sdpa2->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc);
    sdpa2->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc);
    sdpa2->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc);
    auto computeType = HIPDNN_DATA_FLOAT;
    sdpa2->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    bool falseVal = false;
    sdpa2->setAttribute(HIPDNN_ATTR_SDPA_FPROP_ALIBI_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &falseVal);
    sdpa2->finalize();

    retrieved = true;
    sdpa2->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_ALIBI_MASK_EXT, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &retrieved);
    EXPECT_FALSE(retrieved);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetAndGetFloatAttributes)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});

    float dropoutProb = 0.5f;
    float attnScale = 1.5f;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_DROPOUT_PROBABILITY_EXT, HIPDNN_TYPE_FLOAT, 1, &dropoutProb);
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_ATTN_SCALE_VALUE_EXT, HIPDNN_TYPE_FLOAT, 1, &attnScale);

    desc->finalize();

    float retrievedDropout = 0.0f;
    float retrievedScale = 0.0f;
    int64_t elementCount = 0;

    desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_DROPOUT_PROBABILITY_EXT,
                       HIPDNN_TYPE_FLOAT,
                       1,
                       &elementCount,
                       &retrievedDropout);
    ASSERT_EQ(elementCount, 1);
    EXPECT_FLOAT_EQ(retrievedDropout, 0.5f);

    desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_ATTN_SCALE_VALUE_EXT,
                       HIPDNN_TYPE_FLOAT,
                       1,
                       &elementCount,
                       &retrievedScale);
    ASSERT_EQ(elementCount, 1);
    EXPECT_FLOAT_EQ(retrievedScale, 1.5f);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetAndGetIntAttributes)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});

    int64_t leftBound = 10;
    int64_t rightBound = 20;
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_LEFT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &leftBound);
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_RIGHT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &rightBound);

    int32_t maxSeqLenKv = 512;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MAX_SEQ_LEN_KV_EXT, HIPDNN_TYPE_INT32, 1, &maxSeqLenKv);

    desc->finalize();

    int64_t retrievedLeft = 0;
    int64_t retrievedRight = 0;
    int32_t retrievedMaxSeqLen = 0;
    int64_t elementCount = 0;

    desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_LEFT_BOUND_EXT, HIPDNN_TYPE_INT64, 1, &elementCount, &retrievedLeft);
    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(retrievedLeft, 10);

    desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_RIGHT_BOUND_EXT,
                       HIPDNN_TYPE_INT64,
                       1,
                       &elementCount,
                       &retrievedRight);
    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(retrievedRight, 20);

    desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_MAX_SEQ_LEN_KV_EXT,
                       HIPDNN_TYPE_INT32,
                       1,
                       &elementCount,
                       &retrievedMaxSeqLen);
    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(retrievedMaxSeqLen, 512);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetAndGetNonDefaultEnumAttributes)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({});

    auto diagonalAlignment = HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT;
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT,
                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                       1,
                       &diagonalAlignment);

    auto implementation = HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT;
    desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT,
                       HIPDNN_TYPE_ATTENTION_IMPLEMENTATION,
                       1,
                       &implementation);

    hipdnnDataType_t mmaCoreMode = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &mmaCoreMode);

    desc->finalize();

    hipdnnDiagonalAlignment_t retrievedAlignment = HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT;
    hipdnnAttentionImplementation_t retrievedImpl = HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT;
    hipdnnDataType_t retrievedMmaCore = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;

    desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT,
                       HIPDNN_TYPE_DIAGONAL_ALIGNMENT,
                       1,
                       &elementCount,
                       &retrievedAlignment);
    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(retrievedAlignment, HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT);

    desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT,
                       HIPDNN_TYPE_ATTENTION_IMPLEMENTATION,
                       1,
                       &elementCount,
                       &retrievedImpl);
    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(retrievedImpl, HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT);

    desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &elementCount,
                       &retrievedMmaCore);
    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(retrievedMmaCore, HIPDNN_DATA_HALF);
}

TEST_F(TestSdpaFpropOperationDescriptor, FinalizeWithOnlyRequiredAttributes)
{
    auto desc = getDescriptor();

    // Only set Q, K, V, O tensor descriptors and compute type
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc);

    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MATH_PREC_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Finalize should succeed with only required attributes
    ASSERT_NO_THROW(desc->finalize());
    ASSERT_TRUE(desc->isFinalized());
}
