// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/paged_cache_store_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/PagedCacheStoreConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationPagedCacheStoreDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT
        buildAndDeserialize(PagedCacheStoreAttributes& attrs, bool assignUids = true)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("PagedCacheStoreIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            x->set_uid(K_PAGED_CACHE_STORE_TENSOR_X_UID);
        }
        x->set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS))
            .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES));

        auto slotMapping = std::make_shared<TensorAttributes>();
        if(assignUids)
        {
            slotMapping->set_uid(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
        }
        slotMapping->set_name("slot_mapping").set_data_type(DataType::INT32);
        slotMapping->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS))
            .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES));

        auto container = std::make_shared<TensorAttributes>();
        container->set_data_type(DataType::FLOAT);
        container->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS))
            .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES));
        graph->paged_cache_store(x, slotMapping, container, attrs);
        if(assignUids)
        {
            container->set_uid(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
        }
        container->set_output(true).set_name("container");

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationPagedCacheStoreDescriptorLowering, PagedCacheStoreLoweringRoundTrip)
{
    PagedCacheStoreAttributes attrs;
    attrs.set_name("test_op");

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 3u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_PAGED_CACHE_STORE_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->dims,
              toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->strides,
              toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->name, "x");
    ASSERT_NE(tensorMap.count(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID), 0u);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->dims,
              toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->strides,
              toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->data_type,
              DataTypeSdk::INT32);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->name, "slot_mapping");
    ASSERT_NE(tensorMap.count(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID), 0u);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->dims,
              toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->strides,
              toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->name, "container");

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    // The descriptor has no compute-type attribute, so the lowered node carries none.
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::UNSET);
    EXPECT_EQ(node->attributes.type, NodeAttrType::PagedCacheStoreAttributes);

    auto* opNode = node->attributes.AsPagedCacheStoreAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->x_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_X_UID);
    EXPECT_EQ(opNode->slot_mapping_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    EXPECT_EQ(opNode->container_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");
}

// The optional scale tensor is lowered with its UID; absent, the field stays unset.
TEST_F(IntegrationPagedCacheStoreDescriptorLowering, ScaleTensorPreservedInLowering)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_PAGED_CACHE_STORE_TENSOR_SCALE_UID)
        .set_name("scale")
        .set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_STRIDES));

    PagedCacheStoreAttributes attrs;
    attrs.set_name("test_scale");
    attrs.set_scale(scale);

    auto graphT = buildAndDeserialize(attrs);
    ASSERT_EQ(graphT.tensors.size(), 4u);
    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsPagedCacheStoreAttributes();
    ASSERT_NE(opNode, nullptr);
    ASSERT_TRUE(opNode->scale_tensor_uid.has_value());
    EXPECT_EQ(*opNode->scale_tensor_uid, K_PAGED_CACHE_STORE_TENSOR_SCALE_UID);
}

TEST_F(IntegrationPagedCacheStoreDescriptorLowering, AbsentScaleStaysUnset)
{
    PagedCacheStoreAttributes attrs;
    attrs.set_name("test_no_scale");

    auto graphT = buildAndDeserialize(attrs);
    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsPagedCacheStoreAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_FALSE(opNode->scale_tensor_uid.has_value());
}

// Verifies that lowering assigns unique UIDs when callers omit them.
TEST_F(IntegrationPagedCacheStoreDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    PagedCacheStoreAttributes attrs;
    attrs.set_name("test_auto_uid");

    auto graphT = buildAndDeserialize(attrs, false);

    ASSERT_EQ(graphT.tensors.size(), 3u);
    std::unordered_set<int64_t> uids;
    for(const auto& tensor : graphT.tensors)
    {
        uids.insert(tensor->uid);
    }
    ASSERT_EQ(uids.size(), 3u);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    const auto* opNode = graphT.nodes[0]->attributes.AsPagedCacheStoreAttributes();
    ASSERT_NE(opNode, nullptr);
    EXPECT_TRUE(uids.count(opNode->x_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->slot_mapping_tensor_uid) > 0);
    EXPECT_TRUE(uids.count(opNode->container_tensor_uid) > 0);
}

} // namespace
