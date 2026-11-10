// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/MoeRoutingOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_routing_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MoeRoutingConstants.hpp>
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

class TestMoeRoutingOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<MoeRoutingOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<MoeRoutingOperationDescriptor>();
    }

    void setTensors(std::optional<hipdnnBackendAttributeName_t> excluded = std::nullopt) const
    {
        auto desc = getDescriptor();
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_topkIdsDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_firstTokenOffsetDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_tokenIndexDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_tokenKsDesc);
        }
    }

    void setMoeRoutingParams() const
    {
        auto desc = getDescriptor();

        auto numExperts = static_cast<int64_t>(4);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_MOE_ROUTING_NUM_EXPERTS_EXT, HIPDNN_TYPE_INT64, 1, &numExperts);
    }

    void setRequiredAttributes(std::optional<hipdnnBackendAttributeName_t> excluded
                               = std::nullopt) const
    {
        setTensors(excluded);
        setMoeRoutingParams();
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _topkIdsDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _firstTokenOffsetDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _tokenIndexDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _tokenKsDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<MoeRoutingOperationDescriptor>();
        _topkIdsDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES),
                                             HIPDNN_DATA_INT32);
        _firstTokenOffsetDesc
            = createFinalizedTensor(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                    toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                    toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                    HIPDNN_DATA_INT32);
        _tokenIndexDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID,
                                                toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS),
                                                toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES),
                                                HIPDNN_DATA_INT32);
        _tokenKsDesc = createFinalizedTensor(K_MOE_ROUTING_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _topkIdsDesc.reset();
        _firstTokenOffsetDesc.reset();
        _tokenIndexDesc.reset();
        _tokenKsDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestMoeRoutingOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_MOE_ROUTING_DESCRIPTOR_EXT);
}

TEST_F(TestMoeRoutingOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// The parameter is the one required attribute to leave unset. Every other required attribute is
// set, so each case proves the omitted attribute alone is what finalize() rejects.
class TestMoeRoutingOperationDescriptorFinalizeFailsWithout
    : public TestMoeRoutingOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestMoeRoutingOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setRequiredAttributes(GetParam());
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(
    RequiredAttributes,
    TestMoeRoutingOperationDescriptorFinalizeFailsWithout,
    ::testing::Values(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                      HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                      HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT,
                      HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT));

TEST_F(TestMoeRoutingOperationDescriptor, FinalizeFailsWithoutNumExperts)
{
    setTensors();
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeRoutingOperationDescriptor, FinalizeAcceptsInt64FirstTokenOffset)
{
    auto offset64 = createFinalizedTensor(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                          toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                          toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                          HIPDNN_DATA_INT64);
    setRequiredAttributes(HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT);
    getDescriptor()->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                  1,
                                  &offset64);
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

TEST_F(TestMoeRoutingOperationDescriptor, FinalizeFailsWithFloatFirstTokenOffset)
{
    auto offsetFloat = createFinalizedTensor(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                             toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                             toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                             HIPDNN_DATA_FLOAT);
    setRequiredAttributes(HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT);
    getDescriptor()->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                  1,
                                  &offsetFloat);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestMoeRoutingOperationDescriptor, SetTensorDescriptorTopkIds)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_topkIdsDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().topk_ids_tensor_uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    ASSERT_NE(desc->getTopkIdsDesc(), nullptr);
}

TEST_F(TestMoeRoutingOperationDescriptor, SetTensorDescriptorFirstTokenOffset)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_firstTokenOffsetDesc));

    ASSERT_EQ(desc->getData().first_token_offset_tensor_uid,
              K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
}

TEST_F(TestMoeRoutingOperationDescriptor, SetTensorDescriptorTokenIndex)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_tokenIndexDesc));

    ASSERT_EQ(desc->getData().token_index_tensor_uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    ASSERT_NE(desc->getTokenIndexDesc(), nullptr);
}

TEST_F(TestMoeRoutingOperationDescriptor, SetTensorDescriptorTokenKs)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_tokenKsDesc));

    ASSERT_EQ(desc->getData().token_ks_tensor_uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
    ASSERT_NE(desc->getTokenKsDesc(), nullptr);
}

TEST_F(TestMoeRoutingOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_unfinalizedTensor),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestMoeRoutingOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                           HIPDNN_TYPE_INT64,
                           1,
                           &_topkIdsDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeRoutingOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           2,
                           &_topkIdsDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeRoutingOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestMoeRoutingOperationDescriptor, SetNumExperts)
{
    auto desc = getDescriptor();
    auto numExperts = static_cast<int64_t>(4);

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_ROUTING_NUM_EXPERTS_EXT, HIPDNN_TYPE_INT64, 1, &numExperts));
    EXPECT_EQ(desc->getData().num_experts, numExperts);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestMoeRoutingOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_topkIdsDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestMoeRoutingOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedTopkIds = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedTopkIds)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedTopkIds, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardTopkIds(retrievedTopkIds);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeMoeRoutingParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // num experts
    int64_t numExperts = 0;
    int64_t numExpertsCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_NUM_EXPERTS_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &numExpertsCount,
                                       &numExperts));
    ASSERT_EQ(numExpertsCount, 1);
    EXPECT_EQ(numExperts, static_cast<int64_t>(4));
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr,
                           &dummy),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeTensorTopkIdsQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeTensorFirstTokenOffsetQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           0,
                           &elementCount,
                           nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeTensorTokenIndexQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeTensorTokenKsQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           0,
                           nullptr,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestMoeRoutingOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getTopkIdsDesc(), nullptr);
    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
    ASSERT_NE(desc->getTokenIndexDesc(), nullptr);
    ASSERT_NE(desc->getTokenKsDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getTopkIdsDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    ASSERT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_EQ(desc->getTokenIndexDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    ASSERT_EQ(desc->getTokenKsDesc()->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestMoeRoutingOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("MoeRoutingOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("topk_ids_uid=" + std::to_string(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("first_token_offset_uid="
                       + std::to_string(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID)),
              std::string::npos);
    ASSERT_NE(str.find("token_index_uid=" + std::to_string(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID)),
              std::string::npos);
    ASSERT_NE(str.find("token_ks_uid=" + std::to_string(K_MOE_ROUTING_TENSOR_TOKEN_KS_UID)),
              std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestMoeRoutingOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    ASSERT_EQ(tensors[0]->getData().uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
}

TEST_F(TestMoeRoutingOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->attributes.type, NodeAttributes::MoeRoutingAttributes);

    auto* attrs = node->attributes.AsMoeRoutingAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->topk_ids_tensor_uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    ASSERT_EQ(attrs->first_token_offset_tensor_uid, K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_EQ(attrs->token_index_tensor_uid, K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    ASSERT_EQ(attrs->token_ks_tensor_uid, K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
}

TEST_F(TestMoeRoutingOperationDescriptor,
       GetTensorDescriptorsOrderIsTopkIdsFirstTokenOffsetTokenIndexTokenKs)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    // Verify ordering: [TOPK_IDS_DESC_EXT, FIRST_TOKEN_OFFSET_DESC_EXT, TOKEN_INDEX_DESC_EXT, TOKEN_KS_DESC_EXT] matches UIDs [9700, 9701, 9702, 9703]
    EXPECT_EQ(tensors[0], desc->getTopkIdsDesc());
    EXPECT_EQ(tensors[1], desc->getFirstTokenOffsetDesc());
    EXPECT_EQ(tensors[2], desc->getTokenIndexDesc());
    EXPECT_EQ(tensors[3], desc->getTokenKsDesc());
}

TEST_F(TestMoeRoutingOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    ASSERT_EQ(tensors[0]->getData().uid, K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
}

TEST_F(TestMoeRoutingOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _topkIdsDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestMoeRoutingOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_moerouting_op";

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
    EXPECT_STREQ(buffer.data(), "test_moerouting_op");
}

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_MOE_ROUTING_EXT);
}

TEST_F(TestMoeRoutingOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeRoutingOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_moerouting";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_moerouting");
}
