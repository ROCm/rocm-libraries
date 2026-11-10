// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/NodeFactory.hpp"
#include "descriptors/PagedCacheStoreOperationDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/paged_cache_store_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/PagedCacheStoreConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_backend::test_utilities::verifyTensorDescriptor;
using hipdnn_tests::toVec;

// =============================================================================
// PagedCacheStoreOperationDescriptor::fromNode() Tests
// =============================================================================

class TestPagedCacheStoreOperationFromNode : public ::testing::Test
{
protected:
    std::unordered_map<int64_t, std::shared_ptr<TensorDescriptor>> _tensorMap;

    void SetUp() override
    {
        TensorAttributesT xAttrs;
        xAttrs.uid = K_PAGED_CACHE_STORE_TENSOR_X_UID;
        xAttrs.data_type = DataType::FLOAT;
        xAttrs.dims = toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS);
        xAttrs.strides = toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES);

        _tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID] = TensorDescriptor::fromFlatBuffer(xAttrs);
        TensorAttributesT slotMappingAttrs;
        slotMappingAttrs.uid = K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID;
        slotMappingAttrs.data_type = DataType::INT32;
        slotMappingAttrs.dims = toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS);
        slotMappingAttrs.strides = toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES);

        _tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]
            = TensorDescriptor::fromFlatBuffer(slotMappingAttrs);
        TensorAttributesT containerAttrs;
        containerAttrs.uid = K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID;
        containerAttrs.data_type = DataType::FLOAT;
        containerAttrs.dims = toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS);
        containerAttrs.strides = toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES);

        _tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]
            = TensorDescriptor::fromFlatBuffer(containerAttrs);
        TensorAttributesT scaleAttrs;
        scaleAttrs.uid = K_PAGED_CACHE_STORE_TENSOR_SCALE_UID;
        scaleAttrs.data_type = DataType::FLOAT;
        scaleAttrs.dims = toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_DIMS);
        scaleAttrs.strides = toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_STRIDES);

        _tensorMap[K_PAGED_CACHE_STORE_TENSOR_SCALE_UID]
            = TensorDescriptor::fromFlatBuffer(scaleAttrs);
    }

    static hipdnn_flatbuffers_sdk::data_objects::PagedCacheStoreAttributesT
        createStandardPagedCacheStoreAttrs()
    {
        hipdnn_flatbuffers_sdk::data_objects::PagedCacheStoreAttributesT attrs;
        attrs.x_tensor_uid = K_PAGED_CACHE_STORE_TENSOR_X_UID;
        attrs.slot_mapping_tensor_uid = K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID;
        attrs.container_tensor_uid = K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID;
        attrs.scale_tensor_uid = K_PAGED_CACHE_STORE_TENSOR_SCALE_UID;
        return attrs;
    }

    static NodeT createStandardNode(DataType computeType = DataType::FLOAT)
    {
        NodeT node;
        node.compute_data_type = computeType;
        node.attributes.Set(createStandardPagedCacheStoreAttrs());
        return node;
    }
};

TEST_F(TestPagedCacheStoreOperationFromNode, CreatesValidFinalizedDescriptor)
{
    auto node = createStandardNode();
    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_PAGED_CACHE_STORE_DESCRIPTOR_EXT);
    EXPECT_EQ(desc->getData().x_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
}

TEST_F(TestPagedCacheStoreOperationFromNode, NodeFactoryDelegatesCorrectly)
{
    auto node = createStandardNode();

    // NodeFactory::createOperationFromNode delegates to fromNode internally.
    // Verify the delegation produces a valid, correctly-typed descriptor.
    auto graphOp = NodeFactory::createOperationFromNode(node, _tensorMap);
    ASSERT_NE(graphOp, nullptr);

    // Verify the factory dispatched to the correct operation type, then static_cast.
    // Cannot use dynamic_pointer_cast: backend tests compile with -fno-rtti.
    auto* op = graphOp->asGraphOperation();
    ASSERT_NE(op, nullptr);
    const auto rebuiltNode = op->buildNode();
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::PagedCacheStoreAttributes);
    auto desc = std::static_pointer_cast<PagedCacheStoreOperationDescriptor>(graphOp);
    ASSERT_TRUE(desc->isFinalized());

    // Verify all attributes are correctly populated via the delegated path
    EXPECT_EQ(desc->getData().x_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().slot_mapping_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    EXPECT_EQ(desc->getData().container_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    EXPECT_EQ(desc->getData().scale_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    EXPECT_EQ(desc->getSlotMappingDesc()->getData().uid,
              K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    EXPECT_EQ(desc->getContainerDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
}

TEST_F(TestPagedCacheStoreOperationFromNode, SetsTensorReferences)
{
    auto node = createStandardNode();
    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    ASSERT_NE(desc->getSlotMappingDesc(), nullptr);
    EXPECT_EQ(desc->getSlotMappingDesc()->getData().uid,
              K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    ASSERT_NE(desc->getContainerDesc(), nullptr);
    EXPECT_EQ(desc->getContainerDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
}

TEST_F(TestPagedCacheStoreOperationFromNode, TensorReferencesMatchTensorMap)
{
    auto node = createStandardNode();
    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);

    EXPECT_EQ(desc->getXDesc(), _tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]);
    EXPECT_EQ(desc->getSlotMappingDesc(), _tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]);
    EXPECT_EQ(desc->getContainerDesc(), _tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]);
    EXPECT_EQ(desc->getScaleDesc(), _tensorMap[K_PAGED_CACHE_STORE_TENSOR_SCALE_UID]);
}

TEST_F(TestPagedCacheStoreOperationFromNode, SetsTensorReferencesWithFullValues)
{
    auto node = createStandardNode();
    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);

    ASSERT_NE(desc->getXDesc(), nullptr);
    EXPECT_EQ(desc->getXDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    EXPECT_EQ(desc->getXDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getXDesc()->getData().dims, (std::vector<int64_t>{4, 2, 8}));
    EXPECT_EQ(desc->getXDesc()->getData().strides, (std::vector<int64_t>{16, 8, 1}));

    ASSERT_NE(desc->getSlotMappingDesc(), nullptr);
    EXPECT_EQ(desc->getSlotMappingDesc()->getData().uid,
              K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    EXPECT_EQ(desc->getSlotMappingDesc()->getData().data_type, DataType::INT32);
    EXPECT_EQ(desc->getSlotMappingDesc()->getData().dims, (std::vector<int64_t>{4}));
    EXPECT_EQ(desc->getSlotMappingDesc()->getData().strides, (std::vector<int64_t>{1}));

    ASSERT_NE(desc->getContainerDesc(), nullptr);
    EXPECT_EQ(desc->getContainerDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    EXPECT_EQ(desc->getContainerDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getContainerDesc()->getData().dims, (std::vector<int64_t>{3, 2, 4, 8}));
    EXPECT_EQ(desc->getContainerDesc()->getData().strides, (std::vector<int64_t>{64, 32, 8, 1}));

    ASSERT_NE(desc->getScaleDesc(), nullptr);
    EXPECT_EQ(desc->getScaleDesc()->getData().uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
    EXPECT_EQ(desc->getScaleDesc()->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getScaleDesc()->getData().dims, (std::vector<int64_t>{1}));
    EXPECT_EQ(desc->getScaleDesc()->getData().strides, (std::vector<int64_t>{1}));
}

// The parameter is the UID of the one required tensor to drop from the tensor map.
class TestPagedCacheStoreOperationFromNodeMissingTensor
    : public TestPagedCacheStoreOperationFromNode,
      public ::testing::WithParamInterface<int64_t>
{
};

TEST_P(TestPagedCacheStoreOperationFromNodeMissingTensor, FailsWithMissingTensor)
{
    _tensorMap.erase(GetParam());
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

INSTANTIATE_TEST_SUITE_P(RequiredTensors,
                         TestPagedCacheStoreOperationFromNodeMissingTensor,
                         ::testing::Values(K_PAGED_CACHE_STORE_TENSOR_X_UID,
                                           K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID,
                                           K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID));

TEST_F(TestPagedCacheStoreOperationFromNode, SucceedsWithOnlyRequiredTensors)
{
    auto attrs = createStandardPagedCacheStoreAttrs();
    attrs.scale_tensor_uid = flatbuffers::nullopt;

    NodeT node;
    node.compute_data_type = DataType::FLOAT;
    node.attributes.Set(attrs);

    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);
    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());

    // Required tensor getters are non-null
    EXPECT_NE(desc->getXDesc(), nullptr);
    EXPECT_NE(desc->getSlotMappingDesc(), nullptr);
    EXPECT_NE(desc->getContainerDesc(), nullptr);
    // Optional tensor getters are null
    EXPECT_EQ(desc->getScaleDesc(), nullptr);

    const auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3u);
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getSlotMappingDesc());
    EXPECT_EQ(tensors[2], desc->getContainerDesc());
}

TEST_F(TestPagedCacheStoreOperationFromNode, FailsWhenOptionalScaleUidSetButTensorMissing)
{
    _tensorMap.erase(K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
    auto node = createStandardNode();

    ASSERT_THROW_HIPDNN_STATUS(PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap),
                               HIPDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestPagedCacheStoreOperationFromNode, GetTensorDescriptorsReturnsAllTensors)
{
    auto node = createStandardNode();
    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    EXPECT_EQ(tensors[0]->getData().uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    EXPECT_EQ(tensors[1]->getData().uid, K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    EXPECT_EQ(tensors[2]->getData().uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    EXPECT_EQ(tensors[3]->getData().uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
}

TEST_F(TestPagedCacheStoreOperationFromNode, BuildNodeRoundTrip)
{
    auto node = createStandardNode();
    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);

    const auto rebuiltNode = desc->buildNode();
    ASSERT_NE(rebuiltNode, nullptr);
    ASSERT_EQ(rebuiltNode->attributes.type, NodeAttributes::PagedCacheStoreAttributes);

    const auto* rebuiltAttrs = rebuiltNode->attributes.AsPagedCacheStoreAttributes();
    ASSERT_NE(rebuiltAttrs, nullptr);
    EXPECT_EQ(rebuiltAttrs->x_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    EXPECT_EQ(rebuiltAttrs->slot_mapping_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    EXPECT_EQ(rebuiltAttrs->container_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    EXPECT_EQ(rebuiltAttrs->scale_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
}

TEST_F(TestPagedCacheStoreOperationFromNode, GetAttributeWorksAfterFromNode)
{
    auto node = createStandardNode();
    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);

    // Verify x tensor
    hipdnn_backend::ScopedDescriptor xScoped;
    int64_t xCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &xCount,
                       static_cast<void*>(xScoped.getPtr()));
    ASSERT_EQ(xCount, 1);
    ASSERT_NE(xScoped.get(), nullptr);
    verifyTensorDescriptor(
        xScoped.get(), K_PAGED_CACHE_STORE_TENSOR_X_UID, HIPDNN_DATA_FLOAT, {4, 2, 8}, {16, 8, 1});

    // Verify slot_mapping tensor
    hipdnn_backend::ScopedDescriptor slotMappingScoped;
    int64_t slotMappingCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &slotMappingCount,
                       static_cast<void*>(slotMappingScoped.getPtr()));
    ASSERT_EQ(slotMappingCount, 1);
    ASSERT_NE(slotMappingScoped.get(), nullptr);
    verifyTensorDescriptor(slotMappingScoped.get(),
                           K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID,
                           HIPDNN_DATA_INT32,
                           {4},
                           {1});

    // Verify container tensor
    hipdnn_backend::ScopedDescriptor containerScoped;
    int64_t containerCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &containerCount,
                       static_cast<void*>(containerScoped.getPtr()));
    ASSERT_EQ(containerCount, 1);
    ASSERT_NE(containerScoped.get(), nullptr);
    verifyTensorDescriptor(containerScoped.get(),
                           K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID,
                           HIPDNN_DATA_FLOAT,
                           {3, 2, 4, 8},
                           {64, 32, 8, 1});

    // Verify scale tensor (optional)
    hipdnn_backend::ScopedDescriptor scaleScoped;
    int64_t scaleCount = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SCALE_DESC_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleCount,
                       static_cast<void*>(scaleScoped.getPtr()));
    ASSERT_EQ(scaleCount, 1);
    ASSERT_NE(scaleScoped.get(), nullptr);
    verifyTensorDescriptor(
        scaleScoped.get(), K_PAGED_CACHE_STORE_TENSOR_SCALE_UID, HIPDNN_DATA_FLOAT, {1}, {1});

    // Verify operation type
    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t opTypeCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &opTypeCount, &opType);
    ASSERT_EQ(opTypeCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_PAGED_CACHE_STORE_EXT);
}

TEST_F(TestPagedCacheStoreOperationFromNode, NamePreservedFromNode)
{
    auto node = createStandardNode();
    node.name = "test_pagedcachestore_1";

    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(std::string("test_pagedcachestore_1").size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_pagedcachestore_1");
}

TEST_F(TestPagedCacheStoreOperationFromNode, EmptyNamePreservedFromNode)
{
    auto node = createStandardNode();
    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, 1);
}

TEST_F(TestPagedCacheStoreOperationFromNode, BuildNodePreservesName)
{
    auto node = createStandardNode();
    node.name = "test_build_name";

    auto desc = PagedCacheStoreOperationDescriptor::fromNode(node, _tensorMap);
    const auto rebuiltNode = desc->buildNode();

    ASSERT_NE(rebuiltNode, nullptr);
    EXPECT_EQ(rebuiltNode->name, "test_build_name");
}
