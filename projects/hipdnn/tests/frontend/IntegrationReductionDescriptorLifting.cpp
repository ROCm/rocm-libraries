// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <set>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/ReductionNode.hpp>
#include <hipdnn_test_sdk/constants/ReductionConstants.hpp>
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
class IntegrationReductionDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard Reduction graph for round-trip testing.
    static std::shared_ptr<TestableGraphLifting> buildGraph()
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("ReductionLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_REDUCTION_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_REDUCTION_TENSOR_X_DIMS))
            .set_stride(toVec(K_REDUCTION_TENSOR_X_STRIDES));

        ReductionAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_mode(ReductionMode::ADD);

        auto y = graph->reduction(x, attrs);
        y->set_uid(K_REDUCTION_TENSOR_Y_UID).set_output(true).set_name("y");
        y->set_dim(toVec(K_REDUCTION_TENSOR_Y_DIMS))
            .set_stride(toVec(K_REDUCTION_TENSOR_Y_STRIDES));

        return graph;
    }
};

// Builds a standard Reduction graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationReductionDescriptorLifting, BasicReductionRoundTrip)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_REDUCTION_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->get_uid(), K_REDUCTION_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->get_dim(), toVec(K_REDUCTION_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->get_stride(),
              toVec(K_REDUCTION_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->get_name(), "x");

    // Verify y tensor
    ASSERT_NE(tensorMap.count(K_REDUCTION_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->get_uid(), K_REDUCTION_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->get_dim(), toVec(K_REDUCTION_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->get_stride(),
              toVec(K_REDUCTION_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->get_name(), "y");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<ReductionNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a ReductionNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_mode(), ReductionMode::ADD);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationReductionDescriptorLifting, ReductionTensorSharingPreserved)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ReductionNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_REDUCTION_TENSOR_X_UID);
    EXPECT_EQ(opNode->attributes.get_x()->get_name(), "x");
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify y tensor sharing
    EXPECT_EQ(opNode->attributes.get_y()->get_uid(), K_REDUCTION_TENSOR_Y_UID);
    EXPECT_EQ(opNode->attributes.get_y()->get_name(), "y");
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID].get(), opNode->attributes.get_y().get());
}

// Builds a Reduction graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationReductionDescriptorLifting, ReductionLiftWithoutFinalization)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ReductionNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_mode(), ReductionMode::ADD);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);

    ASSERT_NE(tensorMap.count(K_REDUCTION_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->get_dim(), toVec(K_REDUCTION_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->get_stride(),
              toVec(K_REDUCTION_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_X_UID]->get_name(), "x");
    ASSERT_NE(tensorMap.count(K_REDUCTION_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->get_dim(), toVec(K_REDUCTION_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->get_stride(),
              toVec(K_REDUCTION_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_REDUCTION_TENSOR_Y_UID]->get_name(), "y");
}

// Verifies that the optional is_deterministic attribute survives a lifting round-trip.
TEST_F(IntegrationReductionDescriptorLifting, IsDeterministicPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("ReductionIsDeterministicLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_REDUCTION_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_REDUCTION_TENSOR_X_DIMS)).set_stride(toVec(K_REDUCTION_TENSOR_X_STRIDES));

    ReductionAttributes attrs;
    attrs.set_name("test_is_deterministic");
    attrs.set_mode(ReductionMode::ADD);
    attrs.set_is_deterministic(true);

    auto y = graph->reduction(x, attrs);
    y->set_uid(K_REDUCTION_TENSOR_Y_UID).set_output(true).set_name("y");
    y->set_dim(toVec(K_REDUCTION_TENSOR_Y_DIMS)).set_stride(toVec(K_REDUCTION_TENSOR_Y_STRIDES));

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ReductionNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a ReductionNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_TRUE(opNode->attributes.get_is_deterministic());
}

// Builds a Reduction graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationReductionDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("ReductionAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_REDUCTION_TENSOR_X_DIMS)).set_stride(toVec(K_REDUCTION_TENSOR_X_STRIDES));

    ReductionAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_mode(ReductionMode::ADD);

    auto y = graph->reduction(x, attrs);
    y->set_output(true).set_name("y");
    y->set_dim(toVec(K_REDUCTION_TENSOR_Y_DIMS)).set_stride(toVec(K_REDUCTION_TENSOR_Y_STRIDES));

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify the tensor map has the expected number of tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);

    // Verify all UIDs are distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    ASSERT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "Found duplicate auto-assigned UIDs"; // NOLINT(readability-implicit-bool-conversion)

    // Verify sub-node tensor UIDs are distinct via the node attributes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ReductionNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_x(), nullptr);
    nodeUids.insert(opNode->attributes.get_x()->get_uid());
    ASSERT_NE(opNode->attributes.get_y(), nullptr);
    nodeUids.insert(opNode->attributes.get_y()->get_uid());
    ASSERT_EQ(nodeUids.size(), 2u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_x()->get_dim(), toVec(K_REDUCTION_TENSOR_X_DIMS));
    EXPECT_EQ(opNode->attributes.get_x()->get_stride(), toVec(K_REDUCTION_TENSOR_X_STRIDES));
    EXPECT_EQ(opNode->attributes.get_y()->get_dim(), toVec(K_REDUCTION_TENSOR_Y_DIMS));
    EXPECT_EQ(opNode->attributes.get_y()->get_stride(), toVec(K_REDUCTION_TENSOR_Y_STRIDES));
}

} // namespace
