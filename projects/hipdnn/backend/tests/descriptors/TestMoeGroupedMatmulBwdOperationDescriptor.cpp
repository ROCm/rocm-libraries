// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/MoeGroupedMatmulBwdOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MoeGroupedMatmulBwdConstants.hpp>
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

class TestMoeGroupedMatmulBwdOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<MoeGroupedMatmulBwdOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<MoeGroupedMatmulBwdOperationDescriptor>();
    }

    void setTensors(std::optional<hipdnnBackendAttributeName_t> excluded = std::nullopt) const
    {
        auto desc = getDescriptor();
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_doutputDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_tokenDesc);
        }
        if(!excluded
           || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_firstTokenOffsetDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_dweightDesc);
        }
    }

    // Sets exactly one tensor attribute, leaving the other three unset.
    void setTensor(hipdnnBackendAttributeName_t attribute) const
    {
        auto desc = getDescriptor();
        if(attribute == HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC)
        {
            desc->setAttribute(attribute, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_doutputDesc);
        }
        else if(attribute == HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC)
        {
            desc->setAttribute(attribute, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_tokenDesc);
        }
        else if(attribute == HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC)
        {
            desc->setAttribute(
                attribute, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_firstTokenOffsetDesc);
        }
        else if(attribute == HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC)
        {
            desc->setAttribute(attribute, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dweightDesc);
        }
    }

    void setRequiredAttributes(std::optional<hipdnnBackendAttributeName_t> excluded
                               = std::nullopt) const
    {
        setTensors(excluded);
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC)
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            getDescriptor()->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                                          HIPDNN_TYPE_DATA_TYPE,
                                          1,
                                          &computeType);
        }
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _doutputDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _tokenDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _firstTokenOffsetDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dweightDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<MoeGroupedMatmulBwdOperationDescriptor>();
        _doutputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_STRIDES),
                                             HIPDNN_DATA_FLOAT);
        _tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
        _firstTokenOffsetDesc = createFinalizedTensor(
            K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID,
            toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
            toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
            HIPDNN_DATA_INT32);
        _dweightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_STRIDES),
                                             HIPDNN_DATA_FLOAT);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _doutputDesc.reset();
        _tokenDesc.reset();
        _firstTokenOffsetDesc.reset();
        _dweightDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// Every other required attribute is set, so each case proves the omitted attribute alone is
// what finalize() rejects.
class TestMoeGroupedMatmulBwdOperationDescriptorFinalizeFailsWithout
    : public TestMoeGroupedMatmulBwdOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestMoeGroupedMatmulBwdOperationDescriptorFinalizeFailsWithout, RequiredAttribute)
{
    setRequiredAttributes(GetParam());
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(
    RequiredAttributes,
    TestMoeGroupedMatmulBwdOperationDescriptorFinalizeFailsWithout,
    ::testing::Values(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                      HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
                      HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
                      HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
                      HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC));

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, FinalizeFailsWithNonInt32FirstTokenOffset)
{
    auto desc = getDescriptor();
    auto wrongTypeFirstTokenOffsetDesc
        = createFinalizedTensor(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                toVec(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                HIPDNN_DATA_FLOAT);

    // Set everything except first_token_offset, then supply the wrong-typed one in its place.
    setRequiredAttributes(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &wrongTypeFirstTokenOffsetDesc);

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

namespace
{

// Each case sets exactly one tensor attribute and checks it lands in the matching data field
// and accessor, so a cross-wired attribute would surface as a UID mismatch.
struct SetTensorCase
{
    hipdnnBackendAttributeName_t attribute;
    const char* name;
    int64_t expectedUid;
    int64_t MoeGroupedMatmulBwdAttributesT::*uidField;
    std::shared_ptr<TensorDescriptor> (MoeGroupedMatmulBwdOperationDescriptor::*getter)() const;
};

} // namespace

class TestMoeGroupedMatmulBwdOperationDescriptorSetTensor
    : public TestMoeGroupedMatmulBwdOperationDescriptor,
      public ::testing::WithParamInterface<SetTensorCase>
{
};

TEST_P(TestMoeGroupedMatmulBwdOperationDescriptorSetTensor, StoresTensorDescriptor)
{
    const auto& tc = GetParam();
    auto desc = getDescriptor();

    ASSERT_NO_THROW(setTensor(tc.attribute));

    // Verify UID extracted via getData()
    ASSERT_EQ((desc->getData()).*(tc.uidField), tc.expectedUid);
    ASSERT_NE((desc.get()->*(tc.getter))(), nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    AllTensors,
    TestMoeGroupedMatmulBwdOperationDescriptorSetTensor,
    ::testing::Values(SetTensorCase{HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                                    "Doutput",
                                    K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID,
                                    &MoeGroupedMatmulBwdAttributesT::doutput_tensor_uid,
                                    &MoeGroupedMatmulBwdOperationDescriptor::getDoutputDesc},
                      SetTensorCase{HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
                                    "Token",
                                    K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID,
                                    &MoeGroupedMatmulBwdAttributesT::token_tensor_uid,
                                    &MoeGroupedMatmulBwdOperationDescriptor::getTokenDesc},
                      SetTensorCase{
                          HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
                          "FirstTokenOffset",
                          K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID,
                          &MoeGroupedMatmulBwdAttributesT::first_token_offset_tensor_uid,
                          &MoeGroupedMatmulBwdOperationDescriptor::getFirstTokenOffsetDesc},
                      SetTensorCase{HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
                                    "Dweight",
                                    K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID,
                                    &MoeGroupedMatmulBwdAttributesT::dweight_tensor_uid,
                                    &MoeGroupedMatmulBwdOperationDescriptor::getDweightDesc}),
    [](const ::testing::TestParamInfo<SetTensorCase>& info) { return info.param.name; });

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_unfinalizedTensor),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                           HIPDNN_TYPE_INT64,
                           1,
                           &_doutputDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           2,
                           &_doutputDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                           HIPDNN_TYPE_DATA_TYPE,
                           2,
                           &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_doutputDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedDoutput = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedDoutput)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedDoutput, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardDoutput(retrievedDoutput);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
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

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr,
                           &dummy),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, GetAttributeUnsupported)
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

namespace
{

struct QueryAttributeParam
{
    hipdnnBackendAttributeName_t attributeName;
    hipdnnBackendAttributeType_t attributeType;
};

} // namespace

class TestMoeGroupedMatmulBwdOperationDescriptorQuery
    : public TestMoeGroupedMatmulBwdOperationDescriptor,
      public ::testing::WithParamInterface<QueryAttributeParam>
{
};

TEST_P(TestMoeGroupedMatmulBwdOperationDescriptorQuery, ReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();
    const auto param = GetParam();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(
        desc->getAttribute(param.attributeName, param.attributeType, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

INSTANTIATE_TEST_SUITE_P(
    QueryableAttributes,
    TestMoeGroupedMatmulBwdOperationDescriptorQuery,
    ::testing::Values(QueryAttributeParam{HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR},
                      QueryAttributeParam{HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR},
                      QueryAttributeParam{
                          HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
                          HIPDNN_TYPE_BACKEND_DESCRIPTOR},
                      QueryAttributeParam{HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
                                          HIPDNN_TYPE_BACKEND_DESCRIPTOR},
                      QueryAttributeParam{HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                                          HIPDNN_TYPE_DATA_TYPE}));

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           0,
                           nullptr,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getDoutputDesc(), nullptr);
    ASSERT_NE(desc->getTokenDesc(), nullptr);
    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
    ASSERT_NE(desc->getDweightDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getDoutputDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    ASSERT_EQ(desc->getTokenDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    ASSERT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_EQ(desc->getDweightDesc()->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("MoeGroupedMatmulBwdOperationDescriptor"), std::string::npos);
    ASSERT_NE(
        str.find("doutput_uid=" + std::to_string(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID)),
        std::string::npos);
    ASSERT_NE(str.find("token_uid=" + std::to_string(K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID)),
              std::string::npos);
    ASSERT_NE(str.find("first_token_offset_uid="
                       + std::to_string(K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID)),
              std::string::npos);
    ASSERT_NE(
        str.find("dweight_uid=" + std::to_string(K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID)),
        std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    ASSERT_EQ(tensors[0]->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::MoeGroupedMatmulBwdAttributes);

    auto* attrs = node->attributes.AsMoeGroupedMatmulBwdAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->doutput_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
    ASSERT_EQ(attrs->token_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_TOKEN_UID);
    ASSERT_EQ(attrs->first_token_offset_tensor_uid,
              K_MOE_GROUPED_MATMUL_BWD_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_EQ(attrs->dweight_tensor_uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DWEIGHT_UID);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor,
       GetTensorDescriptorsOrderIsDoutputTokenFirstTokenOffsetDweight)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    // Verify ordering: [DOUTPUT_DESC, TOKEN_DESC, FIRST_TOKEN_OFFSET_DESC, DWEIGHT_DESC] matches UIDs [1910, 1911, 1912, 1913]
    EXPECT_EQ(tensors[0], desc->getDoutputDesc());
    EXPECT_EQ(tensors[1], desc->getTokenDesc());
    EXPECT_EQ(tensors[2], desc->getFirstTokenOffsetDesc());
    EXPECT_EQ(tensors[3], desc->getDweightDesc());
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    ASSERT_EQ(tensors[0]->getData().uid, K_MOE_GROUPED_MATMUL_BWD_TENSOR_DOUTPUT_UID);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _doutputDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_moegroupedmatmulbwd_op";

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
    EXPECT_STREQ(buffer.data(), "test_moegroupedmatmulbwd_op");
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_MOE_GROUPED_MATMUL_BWD_EXT);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeGroupedMatmulBwdOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_moegroupedmatmulbwd";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_moegroupedmatmulbwd");
}
