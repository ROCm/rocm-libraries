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
#include <hipdnn_frontend/node/MoeRoutingNode.hpp>
#include <hipdnn_test_sdk/constants/MoeRoutingConstants.hpp>
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
class IntegrationMoeRoutingDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard MoeRouting graph for round-trip testing.
    static MoeRoutingAttributes createAttributes(DataType operationComputeDataType
                                                 = DataType::NOT_SET)
    {
        MoeRoutingAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_num_experts(4);
        if(operationComputeDataType != DataType::NOT_SET)
        {
            attrs.set_compute_data_type(operationComputeDataType);
        }
        return attrs;
    }

    static std::shared_ptr<TestableGraphLifting> buildGraph(MoeRoutingAttributes attrs)
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("MoeRoutingLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto topkIds = std::make_shared<TensorAttributes>();
        topkIds->set_uid(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID)
            .set_name("topk_ids")
            .set_data_type(DataType::INT32);
        topkIds->set_dim(toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS))
            .set_stride(toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES));

        auto results = graph->moe_routing(topkIds, std::move(attrs));
        results[0]
            ->set_uid(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID)
            .set_output(true)
            .set_name("first_token_offset");
        results[1]
            ->set_uid(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID)
            .set_output(true)
            .set_name("token_index");
        results[2]
            ->set_uid(K_MOE_ROUTING_TENSOR_TOKEN_KS_UID)
            .set_output(true)
            .set_name("token_ks");

        return graph;
    }
};

// Builds a standard MoeRouting graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationMoeRoutingDescriptorLifting, BasicMoeRoutingRoundTrip)
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
    ASSERT_EQ(tensorMap.size(), 4u);

    // Verify topk_ids tensor
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->get_uid(),
              K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->get_data_type(), DataType::INT32);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->get_name(), "topk_ids");

    // Verify first_token_offset tensor
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_uid(),
              K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_data_type(),
              DataType::INT32);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_name(),
              "first_token_offset");

    // Verify token_index tensor
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->get_uid(),
              K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->get_data_type(), DataType::INT32);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->get_name(), "token_index");

    // Verify token_ks tensor
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_TOKEN_KS_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->get_uid(),
              K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->get_data_type(), DataType::INT32);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->get_name(), "token_ks");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<MoeRoutingNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a MoeRoutingNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(opNode->attributes.get_num_experts(), 4);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationMoeRoutingDescriptorLifting, MoeRoutingTensorSharingPreserved)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<MoeRoutingNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify topk_ids tensor sharing
    EXPECT_EQ(opNode->attributes.get_topk_ids()->get_uid(), K_MOE_ROUTING_TENSOR_TOPK_IDS_UID);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID].get(),
              opNode->attributes.get_topk_ids().get());
    // Verify first_token_offset tensor sharing
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_uid(),
              K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID].get(),
              opNode->attributes.get_first_token_offset().get());
    // Verify token_index tensor sharing
    EXPECT_EQ(opNode->attributes.get_token_index()->get_uid(),
              K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID].get(),
              opNode->attributes.get_token_index().get());
    // Verify token_ks tensor sharing
    EXPECT_EQ(opNode->attributes.get_token_ks()->get_uid(), K_MOE_ROUTING_TENSOR_TOKEN_KS_UID);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID].get(),
              opNode->attributes.get_token_ks().get());
}

// Builds a MoeRouting graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationMoeRoutingDescriptorLifting, MoeRoutingLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<MoeRoutingNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    EXPECT_EQ(opNode->attributes.get_num_experts(), 4);
    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);

    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_TOPK_IDS_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOPK_IDS_UID]->get_name(), "topk_ids");
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_UID]->get_name(),
              "first_token_offset");
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_INDEX_UID]->get_name(), "token_index");
    ASSERT_NE(tensorMap.count(K_MOE_ROUTING_TENSOR_TOKEN_KS_UID), 0u);
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES));
    EXPECT_EQ(tensorMap[K_MOE_ROUTING_TENSOR_TOKEN_KS_UID]->get_name(), "token_ks");
}

// Builds a MoeRouting graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationMoeRoutingDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("MoeRoutingAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto topkIds = std::make_shared<TensorAttributes>();
    topkIds->set_name("topk_ids").set_data_type(DataType::INT32);
    topkIds->set_dim(toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS))
        .set_stride(toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES));

    MoeRoutingAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_num_experts(4);

    auto results = graph->moe_routing(topkIds, attrs);
    results[0]->set_output(true).set_name("first_token_offset");
    results[1]->set_output(true).set_name("token_index");
    results[2]->set_output(true).set_name("token_ks");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify the tensor map has the expected number of tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);

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

    auto* opNode = dynamic_cast<MoeRoutingNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_topk_ids(), nullptr);
    nodeUids.insert(opNode->attributes.get_topk_ids()->get_uid());
    ASSERT_NE(opNode->attributes.get_first_token_offset(), nullptr);
    nodeUids.insert(opNode->attributes.get_first_token_offset()->get_uid());
    ASSERT_NE(opNode->attributes.get_token_index(), nullptr);
    nodeUids.insert(opNode->attributes.get_token_index()->get_uid());
    ASSERT_NE(opNode->attributes.get_token_ks(), nullptr);
    nodeUids.insert(opNode->attributes.get_token_ks()->get_uid());
    ASSERT_EQ(nodeUids.size(), 4u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_topk_ids()->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_DIMS));
    EXPECT_EQ(opNode->attributes.get_topk_ids()->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_TOPK_IDS_STRIDES));
    EXPECT_EQ(opNode->attributes.get_topk_ids()->get_name(), "topk_ids");
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_DIMS));
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_FIRST_TOKEN_OFFSET_STRIDES));
    EXPECT_EQ(opNode->attributes.get_first_token_offset()->get_name(), "first_token_offset");
    EXPECT_EQ(opNode->attributes.get_token_index()->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_DIMS));
    EXPECT_EQ(opNode->attributes.get_token_index()->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_INDEX_STRIDES));
    EXPECT_EQ(opNode->attributes.get_token_index()->get_name(), "token_index");
    EXPECT_EQ(opNode->attributes.get_token_ks()->get_dim(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_DIMS));
    EXPECT_EQ(opNode->attributes.get_token_ks()->get_stride(),
              toVec(K_MOE_ROUTING_TENSOR_TOKEN_KS_STRIDES));
    EXPECT_EQ(opNode->attributes.get_token_ks()->get_name(), "token_ks");
}

} // namespace
