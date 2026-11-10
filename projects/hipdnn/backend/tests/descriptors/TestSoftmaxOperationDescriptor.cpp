// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/SoftmaxOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/softmax_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/SoftmaxConstants.hpp>
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

class TestSoftmaxOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<SoftmaxOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<SoftmaxOperationDescriptor>();
    }

    void setTensors(std::optional<hipdnnBackendAttributeName_t> excluded = std::nullopt) const
    {
        auto desc = getDescriptor();
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC)
        {
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_pDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC)
        {
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_sDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_statsDesc);
        }
    }

    void setSoftmaxParams() const
    {
        auto desc = getDescriptor();
        auto statsUseLog2 = true;
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2, HIPDNN_TYPE_BOOLEAN, 1, &statsUseLog2);
    }

    void setRequiredAttributes(std::optional<hipdnnBackendAttributeName_t> excluded
                               = std::nullopt) const
    {
        setTensors(excluded);
        setSoftmaxParams();
        if(!excluded || *excluded != HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT)
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            getDescriptor()->setAttribute(
                HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _pDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _sDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _statsDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<SoftmaxOperationDescriptor>();
        _pDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_P_UID,
                                       toVec(K_SOFTMAX_TENSOR_P_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_P_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _sDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_S_UID,
                                       toVec(K_SOFTMAX_TENSOR_S_DIMS),
                                       toVec(K_SOFTMAX_TENSOR_S_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _statsDesc = createFinalizedTensor(K_SOFTMAX_TENSOR_STATS_UID,
                                           toVec(K_SOFTMAX_TENSOR_STATS_DIMS),
                                           toVec(K_SOFTMAX_TENSOR_STATS_STRIDES),
                                           HIPDNN_DATA_FLOAT);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _pDesc.reset();
        _sDesc.reset();
        _statsDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestSoftmaxOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR);
}

TEST_F(TestSoftmaxOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// The parameter is the one required attribute to leave unset. Every other required attribute is
// set, so each case proves the omitted attribute alone is what finalize() rejects.
class TestSoftmaxOperationDescriptorFinalizeFailsWithout
    : public TestSoftmaxOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestSoftmaxOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setRequiredAttributes(GetParam());
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestSoftmaxOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                                           HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC,
                                           HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT));

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestSoftmaxOperationDescriptor, SetTensorDescriptorP)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_pDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().p_tensor_uid, K_SOFTMAX_TENSOR_P_UID);
    ASSERT_NE(desc->getPDesc(), nullptr);
}

TEST_F(TestSoftmaxOperationDescriptor, SetTensorDescriptorS)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_sDesc));

    ASSERT_EQ(desc->getData().s_tensor_uid, K_SOFTMAX_TENSOR_S_UID);
    ASSERT_NE(desc->getSDesc(), nullptr);
}

TEST_F(TestSoftmaxOperationDescriptor, SetTensorDescriptorStats)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_statsDesc));

    ASSERT_EQ(desc->getData().stats_tensor_uid, K_SOFTMAX_TENSOR_STATS_UID);
    ASSERT_NE(desc->getStatsDesc(), nullptr);
}

TEST_F(TestSoftmaxOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestSoftmaxOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC, HIPDNN_TYPE_INT64, 1, &_pDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSoftmaxOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_pDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestSoftmaxOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestSoftmaxOperationDescriptor, SetStatsUseLog2)
{
    auto desc = getDescriptor();
    auto statsUseLog2 = true;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2, HIPDNN_TYPE_BOOLEAN, 1, &statsUseLog2));
    EXPECT_EQ(desc->getData().stats_use_log2, statsUseLog2);
}

TEST_F(TestSoftmaxOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestSoftmaxOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestSoftmaxOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_pDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestSoftmaxOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedP = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedP)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedP, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardP(retrievedP);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeSoftmaxParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // stats use log2
    bool statsUseLog2 = false;
    int64_t statsUseLog2Count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &statsUseLog2Count,
                                       &statsUseLog2));
    ASSERT_EQ(statsUseLog2Count, 1);
    EXPECT_EQ(statsUseLog2, static_cast<bool>(true));
}

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeTensorPQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeTensorSQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeTensorStatsQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestSoftmaxOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getPDesc(), nullptr);
    ASSERT_NE(desc->getSDesc(), nullptr);
    ASSERT_NE(desc->getStatsDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getPDesc()->getData().uid, K_SOFTMAX_TENSOR_P_UID);
    ASSERT_EQ(desc->getSDesc()->getData().uid, K_SOFTMAX_TENSOR_S_UID);
    ASSERT_EQ(desc->getStatsDesc()->getData().uid, K_SOFTMAX_TENSOR_STATS_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestSoftmaxOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("SoftmaxOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("p_uid=" + std::to_string(K_SOFTMAX_TENSOR_P_UID)), std::string::npos);
    ASSERT_NE(str.find("s_uid=" + std::to_string(K_SOFTMAX_TENSOR_S_UID)), std::string::npos);
    ASSERT_NE(str.find("stats_uid=" + std::to_string(K_SOFTMAX_TENSOR_STATS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestSoftmaxOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_SOFTMAX_TENSOR_P_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_SOFTMAX_TENSOR_S_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_SOFTMAX_TENSOR_STATS_UID);
}

TEST_F(TestSoftmaxOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::SoftmaxAttributes);

    auto* attrs = node->attributes.AsSoftmaxAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->p_tensor_uid, K_SOFTMAX_TENSOR_P_UID);
    ASSERT_EQ(attrs->s_tensor_uid, K_SOFTMAX_TENSOR_S_UID);
    ASSERT_EQ(attrs->stats_tensor_uid, K_SOFTMAX_TENSOR_STATS_UID);
}

TEST_F(TestSoftmaxOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestSoftmaxOperationDescriptor, GetTensorDescriptorsOrderIsPSStats)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    // Verify ordering: [XDESC, YDESC, STATS_DESC] matches UIDs [9200, 9201, 9202]
    EXPECT_EQ(tensors[0], desc->getPDesc());
    EXPECT_EQ(tensors[1], desc->getSDesc());
    EXPECT_EQ(tensors[2], desc->getStatsDesc());
}

TEST_F(TestSoftmaxOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_SOFTMAX_TENSOR_P_UID);
}

TEST_F(TestSoftmaxOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _pDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestSoftmaxOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_softmax_op";

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
    EXPECT_STREQ(buffer.data(), "test_softmax_op");
}

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_SOFTMAX_EXT);
}

TEST_F(TestSoftmaxOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestSoftmaxOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_softmax";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_softmax");
}
