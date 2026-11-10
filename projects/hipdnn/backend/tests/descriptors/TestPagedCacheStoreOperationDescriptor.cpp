// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/PagedCacheStoreOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/paged_cache_store_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/PagedCacheStoreConstants.hpp>
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

class TestPagedCacheStoreOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<PagedCacheStoreOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<PagedCacheStoreOperationDescriptor>();
    }

    void setTensors(std::optional<hipdnnBackendAttributeName_t> excluded = std::nullopt) const
    {
        auto desc = getDescriptor();
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_xDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_slotMappingDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_containerDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SCALE_DESC_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SCALE_DESC_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_scaleDesc);
        }
    }

    void setRequiredAttributes(std::optional<hipdnnBackendAttributeName_t> excluded
                               = std::nullopt) const
    {
        setTensors(excluded);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _slotMappingDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _containerDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<PagedCacheStoreOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_X_UID,
                                       toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS),
                                       toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _slotMappingDesc
            = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID,
                                    toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS),
                                    toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES),
                                    HIPDNN_DATA_INT32);
        _containerDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID,
                                               toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS),
                                               toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES),
                                               HIPDNN_DATA_FLOAT);
        _scaleDesc = createFinalizedTensor(K_PAGED_CACHE_STORE_TENSOR_SCALE_UID,
                                           toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_DIMS),
                                           toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_STRIDES),
                                           HIPDNN_DATA_FLOAT);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _slotMappingDesc.reset();
        _containerDesc.reset();
        _scaleDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestPagedCacheStoreOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_PAGED_CACHE_STORE_DESCRIPTOR_EXT);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// The parameter is the one required attribute to leave unset. Every other required attribute is
// set, so each case proves the omitted attribute alone is what finalize() rejects.
class TestPagedCacheStoreOperationDescriptorFinalizeFailsWithout
    : public TestPagedCacheStoreOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestPagedCacheStoreOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setRequiredAttributes(GetParam());
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(
    RequiredAttributes,
    TestPagedCacheStoreOperationDescriptorFinalizeFailsWithout,
    ::testing::Values(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                      HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT,
                      HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT));

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestPagedCacheStoreOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, SetTensorDescriptorSlotMapping)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_slotMappingDesc));

    ASSERT_EQ(desc->getData().slot_mapping_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    ASSERT_NE(desc->getSlotMappingDesc(), nullptr);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, SetTensorDescriptorContainer)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_containerDesc));

    ASSERT_EQ(desc->getData().container_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    ASSERT_NE(desc->getContainerDesc(), nullptr);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SCALE_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_xDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestPagedCacheStoreOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_xDesc),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
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

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeTensorSlotMappingQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           0,
                           &elementCount,
                           nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeTensorContainerQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeTensorScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SCALE_DESC_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestPagedCacheStoreOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getSlotMappingDesc(), nullptr);
    ASSERT_NE(desc->getContainerDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    ASSERT_EQ(desc->getSlotMappingDesc()->getData().uid,
              K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    ASSERT_EQ(desc->getContainerDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestPagedCacheStoreOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("PagedCacheStoreOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_PAGED_CACHE_STORE_TENSOR_X_UID)),
              std::string::npos);
    ASSERT_NE(
        str.find("slot_mapping_uid=" + std::to_string(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID)),
        std::string::npos);
    ASSERT_NE(str.find("container_uid=" + std::to_string(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID)),
              std::string::npos);
    ASSERT_NE(str.find("scale_uid=" + std::to_string(K_PAGED_CACHE_STORE_TENSOR_SCALE_UID)),
              std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestPagedCacheStoreOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    ASSERT_EQ(tensors[0]->getData().uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->attributes.type, NodeAttributes::PagedCacheStoreAttributes);

    auto* attrs = node->attributes.AsPagedCacheStoreAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    ASSERT_EQ(attrs->slot_mapping_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    ASSERT_EQ(attrs->container_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
}

TEST_F(TestPagedCacheStoreOperationDescriptor,
       GetTensorDescriptorsOrderIsXSlotMappingContainerScale)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    // Verify ordering: [XDESC_EXT, SLOT_MAPPING_DESC_EXT, CONTAINER_DESC_EXT, SCALE_DESC_EXT] matches UIDs [9500, 9501, 9502, 9503]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getSlotMappingDesc());
    EXPECT_EQ(tensors[2], desc->getContainerDesc());
    EXPECT_EQ(tensors[3], desc->getScaleDesc());
}

TEST_F(TestPagedCacheStoreOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    ASSERT_EQ(tensors[0]->getData().uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestPagedCacheStoreOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_pagedcachestore_op";

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
    EXPECT_STREQ(buffer.data(), "test_pagedcachestore_op");
}

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_PAGED_CACHE_STORE_EXT);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestPagedCacheStoreOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_pagedcachestore";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_pagedcachestore");
}
