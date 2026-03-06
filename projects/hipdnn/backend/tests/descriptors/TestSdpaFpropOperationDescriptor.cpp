// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
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

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;

class TestSdpaFpropOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<SdpaFpropOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<SdpaFpropOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_attnMaskDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_scaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_seqLenQDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_seqLenKvDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_seedDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_offsetDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dropoutMaskDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dropoutScaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_pageTableKDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_pageTableVDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_blockMaskDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_sinkTokenDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_descaleQDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_descaleKDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_descaleVDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_descaleSDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_scaleSDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_scaleODesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_statsDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_maxDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_sumExpDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_rngDumpDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_amaxSDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_amaxODesc);
    }

    void setSdpaFpropParams() const
    {
        auto desc = getDescriptor();
    }

    void setRequiredAttributes() const
    {
        setTensors();
        setSdpaFpropParams();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_SDPA_FPROP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
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
        _qDesc = createFinalizedTensor(40, {2, 4, 128, 64}, {32768, 8192, 64, 1});
        _kDesc = createFinalizedTensor(41, {2, 4, 128, 64}, {32768, 8192, 64, 1});
        _vDesc = createFinalizedTensor(42, {2, 4, 128, 64}, {32768, 8192, 64, 1});
        _oDesc = createFinalizedTensor(43, {2, 4, 128, 64}, {32768, 8192, 64, 1});
        _attnMaskDesc = createFinalizedTensor(5);
        _scaleDesc = createFinalizedTensor(6);
        _seqLenQDesc = createFinalizedTensor(7);
        _seqLenKvDesc = createFinalizedTensor(8);
        _seedDesc = createFinalizedTensor(9);
        _offsetDesc = createFinalizedTensor(10);
        _dropoutMaskDesc = createFinalizedTensor(11);
        _dropoutScaleDesc = createFinalizedTensor(12);
        _pageTableKDesc = createFinalizedTensor(13);
        _pageTableVDesc = createFinalizedTensor(14);
        _blockMaskDesc = createFinalizedTensor(15);
        _sinkTokenDesc = createFinalizedTensor(16);
        _descaleQDesc = createFinalizedTensor(17);
        _descaleKDesc = createFinalizedTensor(18);
        _descaleVDesc = createFinalizedTensor(19);
        _descaleSDesc = createFinalizedTensor(20);
        _scaleSDesc = createFinalizedTensor(21);
        _scaleODesc = createFinalizedTensor(22);
        _statsDesc = createFinalizedTensor(23);
        _maxDesc = createFinalizedTensor(24);
        _sumExpDesc = createFinalizedTensor(25);
        _rngDumpDesc = createFinalizedTensor(26);
        _amaxSDesc = createFinalizedTensor(27);
        _amaxODesc = createFinalizedTensor(28);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _qDesc.reset();
        _kDesc.reset();
        _vDesc.reset();
        _oDesc.reset();
        _attnMaskDesc.reset();
        _scaleDesc.reset();
        _seqLenQDesc.reset();
        _seqLenKvDesc.reset();
        _seedDesc.reset();
        _offsetDesc.reset();
        _dropoutMaskDesc.reset();
        _dropoutScaleDesc.reset();
        _pageTableKDesc.reset();
        _pageTableVDesc.reset();
        _blockMaskDesc.reset();
        _sinkTokenDesc.reset();
        _descaleQDesc.reset();
        _descaleKDesc.reset();
        _descaleVDesc.reset();
        _descaleSDesc.reset();
        _scaleSDesc.reset();
        _scaleODesc.reset();
        _statsDesc.reset();
        _maxDesc.reset();
        _sumExpDesc.reset();
        _rngDumpDesc.reset();
        _amaxSDesc.reset();
        _amaxODesc.reset();
        _unfinalizedTensor.reset();
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
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestSdpaFpropOperationDescriptor, FinalizeFailsWithoutQTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_attnMaskDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_seqLenQDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_seqLenKvDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_seedDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_offsetDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dropoutMaskDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dropoutScaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_pageTableKDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_pageTableVDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_blockMaskDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_sinkTokenDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleQDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleKDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleVDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleODesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_statsDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_maxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_sumExpDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_rngDumpDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_amaxSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_amaxODesc);
    setSdpaFpropParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, FinalizeFailsWithoutKTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_attnMaskDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_seqLenQDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_seqLenKvDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_seedDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_offsetDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dropoutMaskDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dropoutScaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_pageTableKDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_pageTableVDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_blockMaskDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_sinkTokenDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleQDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleKDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleVDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleODesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_statsDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_maxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_sumExpDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_rngDumpDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_amaxSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_amaxODesc);
    setSdpaFpropParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, FinalizeFailsWithoutVTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_attnMaskDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_seqLenQDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_seqLenKvDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_seedDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_offsetDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dropoutMaskDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dropoutScaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_pageTableKDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_pageTableVDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_blockMaskDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_sinkTokenDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleQDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleKDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleVDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleODesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_statsDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_maxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_sumExpDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_rngDumpDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_amaxSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_amaxODesc);
    setSdpaFpropParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, FinalizeFailsWithoutOTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_attnMaskDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_seqLenQDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_seqLenKvDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_seedDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_offsetDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dropoutMaskDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dropoutScaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_pageTableKDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_pageTableVDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_blockMaskDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_sinkTokenDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleQDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleKDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleVDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_descaleSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleODesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_statsDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_maxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_sumExpDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_rngDumpDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_amaxSDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_amaxODesc);
    setSdpaFpropParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    setSdpaFpropParams();
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorQ)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_qDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().q_tensor_uid, 40);
    ASSERT_NE(desc->getQDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorK)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_kDesc));

    ASSERT_EQ(desc->getData().k_tensor_uid, 41);
    ASSERT_NE(desc->getKDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorV)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_vDesc));

    ASSERT_EQ(desc->getData().v_tensor_uid, 42);
    ASSERT_NE(desc->getVDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorO)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_oDesc));

    ASSERT_EQ(desc->getData().o_tensor_uid, 43);
    ASSERT_NE(desc->getODesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorAttnMask)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_attnMaskDesc));

    ASSERT_EQ(desc->getData().attn_mask_tensor_uid, 5);
    ASSERT_NE(desc->getAttnMaskDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, 6);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorSeqLenQ)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_seqLenQDesc));

    ASSERT_EQ(desc->getData().seq_len_q_tensor_uid, 7);
    ASSERT_NE(desc->getSeqLenQDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorSeqLenKv)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_seqLenKvDesc));

    ASSERT_EQ(desc->getData().seq_len_kv_tensor_uid, 8);
    ASSERT_NE(desc->getSeqLenKvDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorSeed)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_seedDesc));

    ASSERT_EQ(desc->getData().seed_tensor_uid, 9);
    ASSERT_NE(desc->getSeedDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorOffset)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_offsetDesc));

    ASSERT_EQ(desc->getData().offset_tensor_uid, 10);
    ASSERT_NE(desc->getOffsetDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDropoutMask)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dropoutMaskDesc));

    ASSERT_EQ(desc->getData().dropout_mask_tensor_uid, 11);
    ASSERT_NE(desc->getDropoutMaskDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDropoutScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dropoutScaleDesc));

    ASSERT_EQ(desc->getData().dropout_scale_tensor_uid, 12);
    ASSERT_NE(desc->getDropoutScaleDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorPageTableK)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_pageTableKDesc));

    ASSERT_EQ(desc->getData().page_table_k_tensor_uid, 13);
    ASSERT_NE(desc->getPageTableKDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorPageTableV)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_pageTableVDesc));

    ASSERT_EQ(desc->getData().page_table_v_tensor_uid, 14);
    ASSERT_NE(desc->getPageTableVDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorBlockMask)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_blockMaskDesc));

    ASSERT_EQ(desc->getData().block_mask_tensor_uid, 15);
    ASSERT_NE(desc->getBlockMaskDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorSinkToken)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_sinkTokenDesc));

    ASSERT_EQ(desc->getData().sink_token_tensor_uid, 16);
    ASSERT_NE(desc->getSinkTokenDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDescaleQ)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_descaleQDesc));

    ASSERT_EQ(desc->getData().descale_q_tensor_uid, 17);
    ASSERT_NE(desc->getDescaleQDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDescaleK)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_descaleKDesc));

    ASSERT_EQ(desc->getData().descale_k_tensor_uid, 18);
    ASSERT_NE(desc->getDescaleKDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDescaleV)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_descaleVDesc));

    ASSERT_EQ(desc->getData().descale_v_tensor_uid, 19);
    ASSERT_NE(desc->getDescaleVDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorDescaleS)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_descaleSDesc));

    ASSERT_EQ(desc->getData().descale_s_tensor_uid, 20);
    ASSERT_NE(desc->getDescaleSDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorScaleS)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleSDesc));

    ASSERT_EQ(desc->getData().scale_s_tensor_uid, 21);
    ASSERT_NE(desc->getScaleSDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorScaleO)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleODesc));

    ASSERT_EQ(desc->getData().scale_o_tensor_uid, 22);
    ASSERT_NE(desc->getScaleODesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorStats)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_statsDesc));

    ASSERT_EQ(desc->getData().stats_tensor_uid, 23);
    ASSERT_NE(desc->getStatsDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorMax)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_maxDesc));

    ASSERT_EQ(desc->getData().max_tensor_uid, 24);
    ASSERT_NE(desc->getMaxDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorSumExp)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_sumExpDesc));

    ASSERT_EQ(desc->getData().sum_exp_tensor_uid, 25);
    ASSERT_NE(desc->getSumExpDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorRngDump)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_rngDumpDesc));

    ASSERT_EQ(desc->getData().rng_dump_tensor_uid, 26);
    ASSERT_NE(desc->getRngDumpDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorAmaxS)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_amaxSDesc));

    ASSERT_EQ(desc->getData().amax_s_tensor_uid, 27);
    ASSERT_NE(desc->getAmaxSDesc(), nullptr);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetTensorDescriptorAmaxO)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_amaxODesc));

    ASSERT_EQ(desc->getData().amax_o_tensor_uid, 28);
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

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, SetDiagonalAlignment)
{
    auto desc = getDescriptor();
    auto diagonalAlignment = static_cast<int64_t>(DiagonalAlignment::TOP_LEFT);

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT, HIPDNN_TYPE_INT64, 1, &diagonalAlignment));

    ASSERT_EQ(desc->getData().diagonal_alignment, DiagonalAlignment::TOP_LEFT);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetDiagonalAlignmentWrongElementCount)
{
    auto desc = getDescriptor();
    int64_t diagonalAlignment = 0;

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT,
                                                  HIPDNN_TYPE_INT64,
                                                  2,
                                                  &diagonalAlignment),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetDataType)
{
    auto desc = getDescriptor();
    auto mmaCoreMode = static_cast<int64_t>(DataType::UNSET);

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT, HIPDNN_TYPE_INT64, 1, &mmaCoreMode));

    ASSERT_EQ(desc->getData().mma_core_mode, DataType::UNSET);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    int64_t mmaCoreMode = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT, HIPDNN_TYPE_INT64, 2, &mmaCoreMode),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetAttentionImplementation)
{
    auto desc = getDescriptor();
    auto implementation = static_cast<int64_t>(AttentionImplementation::AUTO);

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT, HIPDNN_TYPE_INT64, 1, &implementation));

    ASSERT_EQ(desc->getData().implementation, AttentionImplementation::AUTO);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetAttentionImplementationWrongElementCount)
{
    auto desc = getDescriptor();
    int64_t implementation = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT, HIPDNN_TYPE_INT64, 2, &implementation),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestSdpaFpropOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SDPA_FPROP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
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

// =============================================================================
// GetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedQ = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       &retrievedQ));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedQ, nullptr);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeSdpafpropParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // diagonal alignment
    int64_t diagonalAlignment = -1;
    int64_t diagonalAlignmentCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &diagonalAlignmentCount,
                                       &diagonalAlignment));
    ASSERT_EQ(diagonalAlignmentCount, 1);
    EXPECT_EQ(diagonalAlignment, static_cast<int64_t>(DiagonalAlignment::TOP_LEFT));

    // mma core mode
    int64_t mmaCoreMode = -1;
    int64_t mmaCoreModeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &mmaCoreModeCount,
                                       &mmaCoreMode));
    ASSERT_EQ(mmaCoreModeCount, 1);
    EXPECT_EQ(mmaCoreMode, static_cast<int64_t>(DataType::UNSET));

    // implementation
    int64_t implementation = -1;
    int64_t implementationCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &implementationCount,
                                       &implementation));
    ASSERT_EQ(implementationCount, 1);
    EXPECT_EQ(implementation, static_cast<int64_t>(AttentionImplementation::AUTO));
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

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

// =============================================================================
// GetAttribute Query Mode Tests
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorQQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorKQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_K_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorVQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_V_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorOQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_O_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorAttnMaskQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_ATTN_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorSeqLenQQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorSeqLenKvQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEQ_LEN_KV_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorSeedQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SEED_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorOffsetQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_OFFSET_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorDropoutMaskQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorDropoutScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DROPOUT_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorPageTableKQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_K_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorPageTableVQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_PAGE_TABLE_V_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorBlockMaskQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_BLOCK_MASK_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorSinkTokenQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SINK_TOKEN_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorDescaleQQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_Q_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorDescaleKQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_K_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorDescaleVQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_V_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorDescaleSQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_DESCALE_S_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorScaleSQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_S_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorScaleOQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SCALE_O_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorStatsQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_STATS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorMaxQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_MAX_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorSumExpQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_SUM_EXP_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorRngDumpQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_RNG_DUMP_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorAmaxSQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_S_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorAmaxOQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_AMAX_O_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeDiagonalAlignmentQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT,
                                       HIPDNN_TYPE_INT64,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeDataTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeAttentionImplementationQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_SDPA_FPROP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_SDPA_FPROP_Q_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeDiagonalAlignmentQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_SDPA_FPROP_DIAGONAL_ALIGNMENT_EXT, HIPDNN_TYPE_INT64, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestSdpaFpropOperationDescriptor, GetAttributeDataTypeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_SDPA_FPROP_MMA_CORE_MODE_EXT, HIPDNN_TYPE_INT64, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestSdpaFpropOperationDescriptor,
       GetAttributeAttentionImplementationQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_SDPA_FPROP_IMPLEMENTATION_EXT, HIPDNN_TYPE_INT64, 0, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

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
    ASSERT_EQ(desc->getQDesc()->getData().uid, 40);
    ASSERT_EQ(desc->getKDesc()->getData().uid, 41);
    ASSERT_EQ(desc->getVDesc()->getData().uid, 42);
    ASSERT_EQ(desc->getODesc()->getData().uid, 43);
    ASSERT_EQ(desc->getAttnMaskDesc()->getData().uid, 5);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, 6);
    ASSERT_EQ(desc->getSeqLenQDesc()->getData().uid, 7);
    ASSERT_EQ(desc->getSeqLenKvDesc()->getData().uid, 8);
    ASSERT_EQ(desc->getSeedDesc()->getData().uid, 9);
    ASSERT_EQ(desc->getOffsetDesc()->getData().uid, 10);
    ASSERT_EQ(desc->getDropoutMaskDesc()->getData().uid, 11);
    ASSERT_EQ(desc->getDropoutScaleDesc()->getData().uid, 12);
    ASSERT_EQ(desc->getPageTableKDesc()->getData().uid, 13);
    ASSERT_EQ(desc->getPageTableVDesc()->getData().uid, 14);
    ASSERT_EQ(desc->getBlockMaskDesc()->getData().uid, 15);
    ASSERT_EQ(desc->getSinkTokenDesc()->getData().uid, 16);
    ASSERT_EQ(desc->getDescaleQDesc()->getData().uid, 17);
    ASSERT_EQ(desc->getDescaleKDesc()->getData().uid, 18);
    ASSERT_EQ(desc->getDescaleVDesc()->getData().uid, 19);
    ASSERT_EQ(desc->getDescaleSDesc()->getData().uid, 20);
    ASSERT_EQ(desc->getScaleSDesc()->getData().uid, 21);
    ASSERT_EQ(desc->getScaleODesc()->getData().uid, 22);
    ASSERT_EQ(desc->getStatsDesc()->getData().uid, 23);
    ASSERT_EQ(desc->getMaxDesc()->getData().uid, 24);
    ASSERT_EQ(desc->getSumExpDesc()->getData().uid, 25);
    ASSERT_EQ(desc->getRngDumpDesc()->getData().uid, 26);
    ASSERT_EQ(desc->getAmaxSDesc()->getData().uid, 27);
    ASSERT_EQ(desc->getAmaxODesc()->getData().uid, 28);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    std::string str = desc->toString();
    ASSERT_NE(str.find("SdpaFpropOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("q_uid=40"), std::string::npos);
    ASSERT_NE(str.find("k_uid=41"), std::string::npos);
    ASSERT_NE(str.find("v_uid=42"), std::string::npos);
    ASSERT_NE(str.find("o_uid=43"), std::string::npos);
    ASSERT_NE(str.find("attn_mask_uid=5"), std::string::npos);
    ASSERT_NE(str.find("scale_uid=6"), std::string::npos);
    ASSERT_NE(str.find("seq_len_q_uid=7"), std::string::npos);
    ASSERT_NE(str.find("seq_len_kv_uid=8"), std::string::npos);
    ASSERT_NE(str.find("seed_uid=9"), std::string::npos);
    ASSERT_NE(str.find("offset_uid=10"), std::string::npos);
    ASSERT_NE(str.find("dropout_mask_uid=11"), std::string::npos);
    ASSERT_NE(str.find("dropout_scale_uid=12"), std::string::npos);
    ASSERT_NE(str.find("page_table_k_uid=13"), std::string::npos);
    ASSERT_NE(str.find("page_table_v_uid=14"), std::string::npos);
    ASSERT_NE(str.find("block_mask_uid=15"), std::string::npos);
    ASSERT_NE(str.find("sink_token_uid=16"), std::string::npos);
    ASSERT_NE(str.find("descale_q_uid=17"), std::string::npos);
    ASSERT_NE(str.find("descale_k_uid=18"), std::string::npos);
    ASSERT_NE(str.find("descale_v_uid=19"), std::string::npos);
    ASSERT_NE(str.find("descale_s_uid=20"), std::string::npos);
    ASSERT_NE(str.find("scale_s_uid=21"), std::string::npos);
    ASSERT_NE(str.find("scale_o_uid=22"), std::string::npos);
    ASSERT_NE(str.find("stats_uid=23"), std::string::npos);
    ASSERT_NE(str.find("max_uid=24"), std::string::npos);
    ASSERT_NE(str.find("sum_exp_uid=25"), std::string::npos);
    ASSERT_NE(str.find("rng_dump_uid=26"), std::string::npos);
    ASSERT_NE(str.find("amax_s_uid=27"), std::string::npos);
    ASSERT_NE(str.find("amax_o_uid=28"), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestSdpaFpropOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 28);
    ASSERT_EQ(tensors[0]->getData().uid, 40);
    ASSERT_EQ(tensors[1]->getData().uid, 41);
    ASSERT_EQ(tensors[2]->getData().uid, 42);
    ASSERT_EQ(tensors[3]->getData().uid, 43);
    ASSERT_EQ(tensors[4]->getData().uid, 5);
    ASSERT_EQ(tensors[5]->getData().uid, 6);
    ASSERT_EQ(tensors[6]->getData().uid, 7);
    ASSERT_EQ(tensors[7]->getData().uid, 8);
    ASSERT_EQ(tensors[8]->getData().uid, 9);
    ASSERT_EQ(tensors[9]->getData().uid, 10);
    ASSERT_EQ(tensors[10]->getData().uid, 11);
    ASSERT_EQ(tensors[11]->getData().uid, 12);
    ASSERT_EQ(tensors[12]->getData().uid, 13);
    ASSERT_EQ(tensors[13]->getData().uid, 14);
    ASSERT_EQ(tensors[14]->getData().uid, 15);
    ASSERT_EQ(tensors[15]->getData().uid, 16);
    ASSERT_EQ(tensors[16]->getData().uid, 17);
    ASSERT_EQ(tensors[17]->getData().uid, 18);
    ASSERT_EQ(tensors[18]->getData().uid, 19);
    ASSERT_EQ(tensors[19]->getData().uid, 20);
    ASSERT_EQ(tensors[20]->getData().uid, 21);
    ASSERT_EQ(tensors[21]->getData().uid, 22);
    ASSERT_EQ(tensors[22]->getData().uid, 23);
    ASSERT_EQ(tensors[23]->getData().uid, 24);
    ASSERT_EQ(tensors[24]->getData().uid, 25);
    ASSERT_EQ(tensors[25]->getData().uid, 26);
    ASSERT_EQ(tensors[26]->getData().uid, 27);
    ASSERT_EQ(tensors[27]->getData().uid, 28);
}

TEST_F(TestSdpaFpropOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::SdpaAttributes);

    auto* attrs = node->attributes.AsSdpaAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->q_tensor_uid, 40);
    ASSERT_EQ(attrs->k_tensor_uid, 41);
    ASSERT_EQ(attrs->v_tensor_uid, 42);
    ASSERT_EQ(attrs->o_tensor_uid, 43);
    ASSERT_EQ(attrs->attn_mask_tensor_uid, 5);
    ASSERT_EQ(attrs->scale_tensor_uid, 6);
    ASSERT_EQ(attrs->seq_len_q_tensor_uid, 7);
    ASSERT_EQ(attrs->seq_len_kv_tensor_uid, 8);
    ASSERT_EQ(attrs->seed_tensor_uid, 9);
    ASSERT_EQ(attrs->offset_tensor_uid, 10);
    ASSERT_EQ(attrs->dropout_mask_tensor_uid, 11);
    ASSERT_EQ(attrs->dropout_scale_tensor_uid, 12);
    ASSERT_EQ(attrs->page_table_k_tensor_uid, 13);
    ASSERT_EQ(attrs->page_table_v_tensor_uid, 14);
    ASSERT_EQ(attrs->block_mask_tensor_uid, 15);
    ASSERT_EQ(attrs->sink_token_tensor_uid, 16);
    ASSERT_EQ(attrs->descale_q_tensor_uid, 17);
    ASSERT_EQ(attrs->descale_k_tensor_uid, 18);
    ASSERT_EQ(attrs->descale_v_tensor_uid, 19);
    ASSERT_EQ(attrs->descale_s_tensor_uid, 20);
    ASSERT_EQ(attrs->scale_s_tensor_uid, 21);
    ASSERT_EQ(attrs->scale_o_tensor_uid, 22);
    ASSERT_EQ(attrs->stats_tensor_uid, 23);
    ASSERT_EQ(attrs->max_tensor_uid, 24);
    ASSERT_EQ(attrs->sum_exp_tensor_uid, 25);
    ASSERT_EQ(attrs->rng_dump_tensor_uid, 26);
    ASSERT_EQ(attrs->amax_s_tensor_uid, 27);
    ASSERT_EQ(attrs->amax_o_tensor_uid, 28);
}

TEST_F(TestSdpaFpropOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_SDPA_FPROP_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(
    TestSdpaFpropOperationDescriptor,
    GetTensorDescriptorsOrderIsQKVOAttnMaskScaleSeqLenQSeqLenKvSeedOffsetDropoutMaskDropoutScalePageTableKPageTableVBlockMaskSinkTokenDescaleQDescaleKDescaleVDescaleSScaleSScaleOStatsMaxSumExpRngDumpAmaxSAmaxO)
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

    auto graphOp = _wrapper->tryAsInterface<IGraphOperation>();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 28);
    ASSERT_EQ(tensors[0]->getData().uid, 40);
}

TEST_F(TestSdpaFpropOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _qDesc->tryAsInterface<IGraphOperation>();
    EXPECT_EQ(graphOp, nullptr);
}
