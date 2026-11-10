// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/PagedCacheStoreNode.hpp>
#include <hipdnn_test_sdk/constants/PagedCacheStoreConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;
using hipdnn_tests::toVec;

namespace
{

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationPagedCacheStoreDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard PagedCacheStore graph for round-trip testing.
    static PagedCacheStoreAttributes createAttributes(DataType operationComputeDataType
                                                      = DataType::NOT_SET)
    {
        PagedCacheStoreAttributes attrs;
        attrs.set_name("test_op");
        if(operationComputeDataType != DataType::NOT_SET)
        {
            attrs.set_compute_data_type(operationComputeDataType);
        }
        return attrs;
    }

    static std::shared_ptr<TestableGraphLifting> buildGraph(PagedCacheStoreAttributes attrs)
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("PagedCacheStoreLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_PAGED_CACHE_STORE_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS))
            .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES));

        auto slotMapping = std::make_shared<TensorAttributes>();
        slotMapping->set_uid(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID)
            .set_name("slot_mapping")
            .set_data_type(DataType::INT32);
        slotMapping->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS))
            .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES));

        auto container = std::make_shared<TensorAttributes>();
        container->set_data_type(DataType::FLOAT);
        container->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS))
            .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES));
        graph->paged_cache_store(x, slotMapping, container, std::move(attrs));
        container->set_uid(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID)
            .set_output(true)
            .set_name("container");

        return graph;
    }
};

// Builds a standard PagedCacheStore graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationPagedCacheStoreDescriptorLifting, BasicPagedCacheStoreRoundTrip)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_PAGED_CACHE_STORE_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->get_uid(),
              K_PAGED_CACHE_STORE_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->get_dim(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->get_stride(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->get_name(), "x");

    // Verify slot_mapping tensor
    ASSERT_NE(tensorMap.count(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID), 0u);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->get_uid(),
              K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->get_dim(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->get_stride(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->get_data_type(),
              DataType::INT32);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->get_name(), "slot_mapping");

    // Verify container tensor
    ASSERT_NE(tensorMap.count(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID), 0u);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->get_uid(),
              K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->get_dim(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->get_stride(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->get_data_type(),
              DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->get_name(), "container");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<PagedCacheStoreNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a PagedCacheStoreNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// The optional scale tensor survives lowering and lifting.
TEST_F(IntegrationPagedCacheStoreDescriptorLifting, ScaleTensorPreservedInLiftingRoundTrip)
{
    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_PAGED_CACHE_STORE_TENSOR_SCALE_UID)
        .set_name("scale")
        .set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_STRIDES));

    auto attrs = createAttributes();
    attrs.set_scale(scale);
    auto originalGraph = buildGraph(std::move(attrs));

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);
    ASSERT_NE(tensorMap.count(K_PAGED_CACHE_STORE_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SCALE_UID]->get_data_type(), DataType::FLOAT);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<PagedCacheStoreNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    ASSERT_NE(opNode->attributes.get_scale(), nullptr);
    EXPECT_EQ(opNode->attributes.get_scale().get(),
              tensorMap[K_PAGED_CACHE_STORE_TENSOR_SCALE_UID].get());
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationPagedCacheStoreDescriptorLifting, PagedCacheStoreTensorSharingPreserved)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<PagedCacheStoreNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_PAGED_CACHE_STORE_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify slot_mapping tensor sharing
    EXPECT_EQ(opNode->attributes.get_slot_mapping()->get_uid(),
              K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID].get(),
              opNode->attributes.get_slot_mapping().get());
    // Verify container tensor sharing
    EXPECT_EQ(opNode->attributes.get_container()->get_uid(),
              K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID].get(),
              opNode->attributes.get_container().get());
}

// Builds a PagedCacheStore graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationPagedCacheStoreDescriptorLifting, PagedCacheStoreLiftWithoutFinalization)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<PagedCacheStoreNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_PAGED_CACHE_STORE_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->get_dim(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->get_stride(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_X_UID]->get_name(), "x");
    ASSERT_NE(tensorMap.count(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID), 0u);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->get_dim(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->get_stride(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_UID]->get_name(), "slot_mapping");
    ASSERT_NE(tensorMap.count(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID), 0u);
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->get_dim(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->get_stride(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES));
    EXPECT_EQ(tensorMap[K_PAGED_CACHE_STORE_TENSOR_CONTAINER_UID]->get_name(), "container");
}

// Builds a PagedCacheStore graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationPagedCacheStoreDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PagedCacheStoreAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS))
        .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES));

    auto slotMapping = std::make_shared<TensorAttributes>();
    slotMapping->set_name("slot_mapping").set_data_type(DataType::INT32);
    slotMapping->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS))
        .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES));

    PagedCacheStoreAttributes attrs;
    attrs.set_name("test_auto_uid");

    auto container = std::make_shared<TensorAttributes>();
    container->set_data_type(DataType::FLOAT);
    container->set_dim(toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS))
        .set_stride(toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES));
    graph->paged_cache_store(x, slotMapping, container, attrs);
    container->set_output(true).set_name("container");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify the tensor map has the expected number of tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    // Verify all UIDs are positive and distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        EXPECT_GE(uid, 0)
            << "Auto-assigned UID should be non-negative"; // NOLINT(readability-implicit-bool-conversion)
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    ASSERT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "Found duplicate auto-assigned UIDs"; // NOLINT(readability-implicit-bool-conversion)

    // Verify sub-node tensor UIDs are distinct via the node attributes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<PagedCacheStoreNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_x(), nullptr);
    nodeUids.insert(opNode->attributes.get_x()->get_uid());
    ASSERT_NE(opNode->attributes.get_slot_mapping(), nullptr);
    nodeUids.insert(opNode->attributes.get_slot_mapping()->get_uid());
    ASSERT_NE(opNode->attributes.get_container(), nullptr);
    nodeUids.insert(opNode->attributes.get_container()->get_uid());
    ASSERT_EQ(nodeUids.size(), 3u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_x()->get_dim(), toVec(K_PAGED_CACHE_STORE_TENSOR_X_DIMS));
    EXPECT_EQ(opNode->attributes.get_x()->get_stride(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_X_STRIDES));
    EXPECT_EQ(opNode->attributes.get_x()->get_name(), "x");
    EXPECT_EQ(opNode->attributes.get_slot_mapping()->get_dim(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_DIMS));
    EXPECT_EQ(opNode->attributes.get_slot_mapping()->get_stride(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_SLOT_MAPPING_STRIDES));
    EXPECT_EQ(opNode->attributes.get_slot_mapping()->get_name(), "slot_mapping");
    EXPECT_EQ(opNode->attributes.get_container()->get_dim(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_DIMS));
    EXPECT_EQ(opNode->attributes.get_container()->get_stride(),
              toVec(K_PAGED_CACHE_STORE_TENSOR_CONTAINER_STRIDES));
    EXPECT_EQ(opNode->attributes.get_container()->get_name(), "container");
}

} // namespace
