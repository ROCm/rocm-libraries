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
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;

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
                                       hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_Q_DIMS),
                                       hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_Q_STRIDES));
        _kDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_K_UID,
                                       hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_K_DIMS),
                                       hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_K_STRIDES));
        _vDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_V_UID,
                                       hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_V_DIMS),
                                       hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_V_STRIDES));
        _oDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_O_UID,
                                       hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_O_DIMS),
                                       hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_O_STRIDES));
        _doDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DO_UID,
                                        hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_DO_DIMS),
                                        hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_DO_STRIDES));
        _statsDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_STATS_UID,
                                           hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_STATS_DIMS),
                                           hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_STATS_STRIDES));
        _dqDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DQ_UID,
                                        hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_DQ_DIMS),
                                        hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_DQ_STRIDES));
        _dkDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DK_UID,
                                        hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_DK_DIMS),
                                        hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_DK_STRIDES));
        _dvDesc = createFinalizedTensor(K_SDPA_BPROP_TENSOR_DV_UID,
                                        hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_DV_DIMS),
                                        hipdnn_tests::toVec(K_SDPA_BPROP_TENSOR_DV_STRIDES));
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
// SetAttribute Tests - Tensor Descriptors
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
// GetAttribute Tests
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
    ASSERT_EQ(count, 1);
    ASSERT_NE(result, nullptr);
}

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
}
