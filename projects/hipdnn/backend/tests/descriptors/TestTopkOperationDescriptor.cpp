// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "descriptors/TopkOperationDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/topk_attributes_generated.h>
#include <hipdnn_test_sdk/constants/TopkConstants.hpp>
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

class TestTopkOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<TopkOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<TopkOperationDescriptor>();
    }

    void setTensors(std::optional<hipdnnBackendAttributeName_t> excluded = std::nullopt) const
    {
        auto desc = getDescriptor();
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT)
        {
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_indicesDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_valuesDesc);
        }
    }

    void setTopkParams() const
    {
        auto desc = getDescriptor();

        auto k = static_cast<int64_t>(4);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_K_EXT, HIPDNN_TYPE_INT64, 1, &k);
        auto largest = false;
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_TOPK_LARGEST_EXT, HIPDNN_TYPE_BOOLEAN, 1, &largest);
    }

    void setRequiredAttributes(std::optional<hipdnnBackendAttributeName_t> excluded
                               = std::nullopt) const
    {
        setTensors(excluded);
        setTopkParams();
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _indicesDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _valuesDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<TopkOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_TOPK_TENSOR_X_UID,
                                       toVec(K_TOPK_TENSOR_X_DIMS),
                                       toVec(K_TOPK_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _indicesDesc = createFinalizedTensor(K_TOPK_TENSOR_INDICES_UID,
                                             toVec(K_TOPK_TENSOR_INDICES_DIMS),
                                             toVec(K_TOPK_TENSOR_INDICES_STRIDES),
                                             HIPDNN_DATA_INT64);
        _valuesDesc = createFinalizedTensor(K_TOPK_TENSOR_VALUES_UID,
                                            toVec(K_TOPK_TENSOR_VALUES_DIMS),
                                            toVec(K_TOPK_TENSOR_VALUES_STRIDES),
                                            HIPDNN_DATA_FLOAT);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _indicesDesc.reset();
        _valuesDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestTopkOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_TOPK_DESCRIPTOR_EXT);
}

TEST_F(TestTopkOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// The parameter is the one required attribute to leave unset. Every other required attribute is
// set, so each case proves the omitted attribute alone is what finalize() rejects.
class TestTopkOperationDescriptorFinalizeFailsWithout
    : public TestTopkOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestTopkOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setRequiredAttributes(GetParam());
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestTopkOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                                           HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT));

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestTopkOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, K_TOPK_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestTopkOperationDescriptor, SetTensorDescriptorIndices)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_indicesDesc));

    ASSERT_EQ(desc->getData().indices_tensor_uid, K_TOPK_TENSOR_INDICES_UID);
    ASSERT_NE(desc->getIndicesDesc(), nullptr);
}

TEST_F(TestTopkOperationDescriptor, SetTensorDescriptorValues)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_valuesDesc));

    ASSERT_EQ(desc->getData().values_tensor_uid, K_TOPK_TENSOR_VALUES_UID);
    ASSERT_NE(desc->getValuesDesc(), nullptr);
}

TEST_F(TestTopkOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestTopkOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTopkOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTopkOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestTopkOperationDescriptor, SetK)
{
    auto desc = getDescriptor();
    auto k = static_cast<int64_t>(4);

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_TOPK_K_EXT, HIPDNN_TYPE_INT64, 1, &k));
    EXPECT_EQ(desc->getData().k, k);
}

TEST_F(TestTopkOperationDescriptor, SetLargest)
{
    auto desc = getDescriptor();
    auto largest = false;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_TOPK_LARGEST_EXT, HIPDNN_TYPE_BOOLEAN, 1, &largest));
    EXPECT_EQ(desc->getData().largest, largest);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestTopkOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestTopkOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestTopkOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedX)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardX(retrievedX);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestTopkOperationDescriptor, GetAttributeTopkParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // k
    int64_t k = 0;
    int64_t kCount = 0;
    ASSERT_NO_THROW(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_K_EXT, HIPDNN_TYPE_INT64, 1, &kCount, &k));
    ASSERT_EQ(kCount, 1);
    EXPECT_EQ(k, static_cast<int64_t>(4));

    // largest
    bool largest = false;
    int64_t largestCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TOPK_LARGEST_EXT, HIPDNN_TYPE_BOOLEAN, 1, &largestCount, &largest));
    ASSERT_EQ(largestCount, 1);
    EXPECT_EQ(largest, static_cast<bool>(false));
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestTopkOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestTopkOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestTopkOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestTopkOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestTopkOperationDescriptor, GetAttributeTensorIndicesQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestTopkOperationDescriptor, GetAttributeTensorValuesQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestTopkOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestTopkOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getIndicesDesc(), nullptr);
    ASSERT_NE(desc->getValuesDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_TOPK_TENSOR_X_UID);
    ASSERT_EQ(desc->getIndicesDesc()->getData().uid, K_TOPK_TENSOR_INDICES_UID);
    ASSERT_EQ(desc->getValuesDesc()->getData().uid, K_TOPK_TENSOR_VALUES_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestTopkOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("TopkOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_TOPK_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("indices_uid=" + std::to_string(K_TOPK_TENSOR_INDICES_UID)),
              std::string::npos);
    ASSERT_NE(str.find("values_uid=" + std::to_string(K_TOPK_TENSOR_VALUES_UID)),
              std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestTopkOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_TOPK_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_TOPK_TENSOR_INDICES_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_TOPK_TENSOR_VALUES_UID);
}

TEST_F(TestTopkOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->attributes.type, NodeAttributes::TopkAttributes);

    auto* attrs = node->attributes.AsTopkAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_TOPK_TENSOR_X_UID);
    ASSERT_EQ(attrs->indices_tensor_uid, K_TOPK_TENSOR_INDICES_UID);
    ASSERT_EQ(attrs->values_tensor_uid, K_TOPK_TENSOR_VALUES_UID);
}

TEST_F(TestTopkOperationDescriptor, GetTensorDescriptorsOrderIsXIndicesValues)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    // Verify ordering: [XDESC_EXT, INDICES_DESC_EXT, VALUES_DESC_EXT] matches UIDs [9300, 9301, 9302]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getIndicesDesc());
    EXPECT_EQ(tensors[2], desc->getValuesDesc());
}

TEST_F(TestTopkOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_TOPK_TENSOR_X_UID);
}

TEST_F(TestTopkOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestTopkOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_topk_op";

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
    EXPECT_STREQ(buffer.data(), "test_topk_op");
}

TEST_F(TestTopkOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestTopkOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_TOPK_EXT);
}

TEST_F(TestTopkOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestTopkOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_topk";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_topk");
}
