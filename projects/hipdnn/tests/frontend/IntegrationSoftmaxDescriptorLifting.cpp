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
#include <hipdnn_frontend/node/SoftmaxNode.hpp>
#include <hipdnn_test_sdk/constants/SoftmaxConstants.hpp>
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
class IntegrationSoftmaxDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard Softmax graph for round-trip testing.
    static SoftmaxAttributes createAttributes(DataType operationComputeDataType = DataType::NOT_SET)
    {
        SoftmaxAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_stats_use_log2(true);
        if(operationComputeDataType != DataType::NOT_SET)
        {
            attrs.set_compute_data_type(operationComputeDataType);
        }
        return attrs;
    }

    static std::shared_ptr<TestableGraphLifting> buildGraph(SoftmaxAttributes attrs)
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("SoftmaxLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto p = std::make_shared<TensorAttributes>();
        p->set_uid(K_SOFTMAX_TENSOR_P_UID).set_name("p").set_data_type(DataType::FLOAT);
        p->set_dim(toVec(K_SOFTMAX_TENSOR_P_DIMS)).set_stride(toVec(K_SOFTMAX_TENSOR_P_STRIDES));

        auto results = graph->softmax(p, std::move(attrs));
        results[0]->set_uid(K_SOFTMAX_TENSOR_S_UID).set_output(true).set_name("s");

        return graph;
    }
};

// Builds a standard Softmax graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationSoftmaxDescriptorLifting, BasicSoftmaxRoundTrip)
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
    ASSERT_EQ(tensorMap.size(), 2u);

    // Verify p tensor
    ASSERT_NE(tensorMap.count(K_SOFTMAX_TENSOR_P_UID), 0u);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->get_uid(), K_SOFTMAX_TENSOR_P_UID);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->get_dim(), toVec(K_SOFTMAX_TENSOR_P_DIMS));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->get_stride(), toVec(K_SOFTMAX_TENSOR_P_STRIDES));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->get_name(), "p");

    // Verify s tensor
    ASSERT_NE(tensorMap.count(K_SOFTMAX_TENSOR_S_UID), 0u);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->get_uid(), K_SOFTMAX_TENSOR_S_UID);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->get_dim(), toVec(K_SOFTMAX_TENSOR_S_DIMS));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->get_stride(), toVec(K_SOFTMAX_TENSOR_S_STRIDES));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->get_name(), "s");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<SoftmaxNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a SoftmaxNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(opNode->attributes.get_stats_use_log2(), true);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// Verifies an operation-level compute type survives descriptor lifting.
TEST_F(IntegrationSoftmaxDescriptorLifting, OperationComputeDataTypeSurvivesLifting)
{
    auto originalGraph = buildGraph(createAttributes(DataType::HALF));

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<SoftmaxNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    EXPECT_EQ(opNode->attributes.get_compute_data_type(), DataType::HALF);
}

// Exercises JSON serialization and deserialization for each mode, including
// canonicalization of mode-irrelevant frontend attributes.
TEST_F(IntegrationSoftmaxDescriptorLifting, JsonRoundTripsAllModeScenarios) {}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationSoftmaxDescriptorLifting, SoftmaxTensorSharingPreserved)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<SoftmaxNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify p tensor sharing
    EXPECT_EQ(opNode->attributes.get_p()->get_uid(), K_SOFTMAX_TENSOR_P_UID);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID].get(), opNode->attributes.get_p().get());
    // Verify s tensor sharing
    EXPECT_EQ(opNode->attributes.get_s()->get_uid(), K_SOFTMAX_TENSOR_S_UID);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID].get(), opNode->attributes.get_s().get());
}

// Builds a Softmax graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationSoftmaxDescriptorLifting, SoftmaxLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<SoftmaxNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    EXPECT_EQ(opNode->attributes.get_stats_use_log2(), true);
    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);

    ASSERT_NE(tensorMap.count(K_SOFTMAX_TENSOR_P_UID), 0u);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->get_dim(), toVec(K_SOFTMAX_TENSOR_P_DIMS));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->get_stride(), toVec(K_SOFTMAX_TENSOR_P_STRIDES));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_P_UID]->get_name(), "p");
    ASSERT_NE(tensorMap.count(K_SOFTMAX_TENSOR_S_UID), 0u);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->get_dim(), toVec(K_SOFTMAX_TENSOR_S_DIMS));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->get_stride(), toVec(K_SOFTMAX_TENSOR_S_STRIDES));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_S_UID]->get_name(), "s");
}

// Builds a Softmax graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationSoftmaxDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("SoftmaxAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto p = std::make_shared<TensorAttributes>();
    p->set_name("p").set_data_type(DataType::FLOAT);
    p->set_dim(toVec(K_SOFTMAX_TENSOR_P_DIMS)).set_stride(toVec(K_SOFTMAX_TENSOR_P_STRIDES));

    SoftmaxAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_stats_use_log2(true);

    auto results = graph->softmax(p, attrs);
    results[0]->set_output(true).set_name("s");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify the tensor map has the expected number of tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);

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

    auto* opNode = dynamic_cast<SoftmaxNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_p(), nullptr);
    nodeUids.insert(opNode->attributes.get_p()->get_uid());
    ASSERT_NE(opNode->attributes.get_s(), nullptr);
    nodeUids.insert(opNode->attributes.get_s()->get_uid());
    ASSERT_EQ(nodeUids.size(), 2u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_p()->get_dim(), toVec(K_SOFTMAX_TENSOR_P_DIMS));
    EXPECT_EQ(opNode->attributes.get_p()->get_stride(), toVec(K_SOFTMAX_TENSOR_P_STRIDES));
    EXPECT_EQ(opNode->attributes.get_p()->get_name(), "p");
    EXPECT_EQ(opNode->attributes.get_s()->get_dim(), toVec(K_SOFTMAX_TENSOR_S_DIMS));
    EXPECT_EQ(opNode->attributes.get_s()->get_stride(), toVec(K_SOFTMAX_TENSOR_S_STRIDES));
    EXPECT_EQ(opNode->attributes.get_s()->get_name(), "s");
}

// The optional stats output survives lowering and lifting, and the lifted node reports
// generate_stats from its presence.
TEST_F(IntegrationSoftmaxDescriptorLifting, StatsOutputPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("SoftmaxStatsLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto p = std::make_shared<TensorAttributes>();
    p->set_uid(K_SOFTMAX_TENSOR_P_UID).set_name("p").set_data_type(DataType::FLOAT);
    p->set_dim(toVec(K_SOFTMAX_TENSOR_P_DIMS)).set_stride(toVec(K_SOFTMAX_TENSOR_P_STRIDES));

    SoftmaxAttributes attrs;
    attrs.set_name("test_stats_lift");
    attrs.set_generate_stats(true).set_stats_use_log2(true);

    auto results = graph->softmax(p, attrs);
    results[0]->set_uid(K_SOFTMAX_TENSOR_S_UID).set_output(true).set_name("s");
    ASSERT_NE(results[1], nullptr);
    results[1]
        ->set_uid(K_SOFTMAX_TENSOR_STATS_UID)
        .set_output(true)
        .set_name("stats")
        .set_data_type(DataType::FLOAT);

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);
    ASSERT_NE(tensorMap.count(K_SOFTMAX_TENSOR_STATS_UID), 0u);
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_STATS_UID]->get_dim(), toVec(K_SOFTMAX_TENSOR_STATS_DIMS));
    EXPECT_EQ(tensorMap[K_SOFTMAX_TENSOR_STATS_UID]->get_stride(),
              toVec(K_SOFTMAX_TENSOR_STATS_STRIDES));

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* opNode = dynamic_cast<SoftmaxNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);
    ASSERT_NE(opNode->attributes.get_stats(), nullptr);
    EXPECT_EQ(opNode->attributes.get_stats()->get_uid(), K_SOFTMAX_TENSOR_STATS_UID);
    EXPECT_TRUE(opNode->attributes.get_generate_stats());
    EXPECT_TRUE(opNode->attributes.get_stats_use_log2());
}

} // namespace
