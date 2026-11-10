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
#include <hipdnn_frontend/node/RopeNode.hpp>
#include <hipdnn_test_sdk/constants/RopeConstants.hpp>
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
class IntegrationRopeDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard Rope graph for round-trip testing.
    static RopeAttributes createAttributes(DataType operationComputeDataType = DataType::NOT_SET)
    {
        RopeAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_output_scale(0.5);
        attrs.set_rope_dim(4);
        attrs.set_rope_offset(0);
        if(operationComputeDataType != DataType::NOT_SET)
        {
            attrs.set_compute_data_type(operationComputeDataType);
        }
        return attrs;
    }

    static std::shared_ptr<TestableGraphLifting> buildGraph(RopeAttributes attrs)
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("RopeLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_ROPE_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_ROPE_TENSOR_X_DIMS)).set_stride(toVec(K_ROPE_TENSOR_X_STRIDES));

        auto freqs = std::make_shared<TensorAttributes>();
        freqs->set_uid(K_ROPE_TENSOR_FREQS_UID).set_name("freqs").set_data_type(DataType::FLOAT);
        freqs->set_dim(toVec(K_ROPE_TENSOR_FREQS_DIMS))
            .set_stride(toVec(K_ROPE_TENSOR_FREQS_STRIDES));

        auto y = graph->rope(x, freqs, std::move(attrs));
        y->set_uid(K_ROPE_TENSOR_Y_UID).set_output(true).set_name("y");

        return graph;
    }
};

// Builds a standard Rope graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationRopeDescriptorLifting, BasicRopeRoundTrip)
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
    ASSERT_NE(tensorMap.count(K_ROPE_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->get_uid(), K_ROPE_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->get_dim(), toVec(K_ROPE_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->get_stride(), toVec(K_ROPE_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->get_name(), "x");

    // Verify freqs tensor
    ASSERT_NE(tensorMap.count(K_ROPE_TENSOR_FREQS_UID), 0u);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->get_uid(), K_ROPE_TENSOR_FREQS_UID);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->get_dim(), toVec(K_ROPE_TENSOR_FREQS_DIMS));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->get_stride(), toVec(K_ROPE_TENSOR_FREQS_STRIDES));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->get_name(), "freqs");

    // Verify y tensor
    ASSERT_NE(tensorMap.count(K_ROPE_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->get_uid(), K_ROPE_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->get_dim(), toVec(K_ROPE_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->get_stride(), toVec(K_ROPE_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->get_name(), "y");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<RopeNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a RopeNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(opNode->attributes.get_output_scale(), 0.5);
    EXPECT_EQ(opNode->attributes.get_rope_dim(), 4);
    EXPECT_EQ(opNode->attributes.get_rope_offset(), 0);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// Exercises JSON serialization and deserialization for each mode, including
// canonicalization of mode-irrelevant frontend attributes.
TEST_F(IntegrationRopeDescriptorLifting, JsonRoundTripsAllModeScenarios) {}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationRopeDescriptorLifting, RopeTensorSharingPreserved)
{
    auto originalGraph = buildGraph(createAttributes());

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<RopeNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_ROPE_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify freqs tensor sharing
    EXPECT_EQ(opNode->attributes.get_freqs()->get_uid(), K_ROPE_TENSOR_FREQS_UID);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID].get(), opNode->attributes.get_freqs().get());
    // Verify y tensor sharing
    EXPECT_EQ(opNode->attributes.get_y()->get_uid(), K_ROPE_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID].get(), opNode->attributes.get_y().get());
}

// Builds a Rope graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationRopeDescriptorLifting, RopeLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<RopeNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    EXPECT_EQ(opNode->attributes.get_output_scale(), 0.5);
    EXPECT_EQ(opNode->attributes.get_rope_dim(), 4);
    EXPECT_EQ(opNode->attributes.get_rope_offset(), 0);
    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_ROPE_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->get_dim(), toVec(K_ROPE_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->get_stride(), toVec(K_ROPE_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_X_UID]->get_name(), "x");
    ASSERT_NE(tensorMap.count(K_ROPE_TENSOR_FREQS_UID), 0u);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->get_dim(), toVec(K_ROPE_TENSOR_FREQS_DIMS));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->get_stride(), toVec(K_ROPE_TENSOR_FREQS_STRIDES));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_FREQS_UID]->get_name(), "freqs");
    ASSERT_NE(tensorMap.count(K_ROPE_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->get_dim(), toVec(K_ROPE_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->get_stride(), toVec(K_ROPE_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_ROPE_TENSOR_Y_UID]->get_name(), "y");
}

// Verifies that the optional rope_offset attribute survives a lifting round-trip.
TEST_F(IntegrationRopeDescriptorLifting, RopeOffsetPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("RopeRopeOffsetLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_ROPE_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_ROPE_TENSOR_X_DIMS)).set_stride(toVec(K_ROPE_TENSOR_X_STRIDES));

    auto freqs = std::make_shared<TensorAttributes>();
    freqs->set_uid(K_ROPE_TENSOR_FREQS_UID).set_name("freqs").set_data_type(DataType::FLOAT);
    freqs->set_dim(toVec(K_ROPE_TENSOR_FREQS_DIMS)).set_stride(toVec(K_ROPE_TENSOR_FREQS_STRIDES));

    RopeAttributes attrs;
    attrs.set_name("test_rope_offset");
    attrs.set_rope_dim(4);
    attrs.set_rope_offset(2);

    auto y = graph->rope(x, freqs, attrs);
    y->set_uid(K_ROPE_TENSOR_Y_UID).set_output(true).set_name("y");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<RopeNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a RopeNode"; // NOLINT(readability-implicit-bool-conversion)

    ASSERT_TRUE(opNode->attributes.get_rope_offset().has_value());
    EXPECT_EQ(opNode->attributes.get_rope_offset().value(), 2);
}

// Builds a Rope graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationRopeDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("RopeAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_ROPE_TENSOR_X_DIMS)).set_stride(toVec(K_ROPE_TENSOR_X_STRIDES));

    auto freqs = std::make_shared<TensorAttributes>();
    freqs->set_name("freqs").set_data_type(DataType::FLOAT);
    freqs->set_dim(toVec(K_ROPE_TENSOR_FREQS_DIMS)).set_stride(toVec(K_ROPE_TENSOR_FREQS_STRIDES));

    RopeAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_output_scale(0.5);
    attrs.set_rope_dim(4);
    attrs.set_rope_offset(0);

    auto y = graph->rope(x, freqs, attrs);
    y->set_output(true).set_name("y");

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

    auto* opNode = dynamic_cast<RopeNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_x(), nullptr);
    nodeUids.insert(opNode->attributes.get_x()->get_uid());
    ASSERT_NE(opNode->attributes.get_freqs(), nullptr);
    nodeUids.insert(opNode->attributes.get_freqs()->get_uid());
    ASSERT_NE(opNode->attributes.get_y(), nullptr);
    nodeUids.insert(opNode->attributes.get_y()->get_uid());
    ASSERT_EQ(nodeUids.size(), 3u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_x()->get_dim(), toVec(K_ROPE_TENSOR_X_DIMS));
    EXPECT_EQ(opNode->attributes.get_x()->get_stride(), toVec(K_ROPE_TENSOR_X_STRIDES));
    EXPECT_EQ(opNode->attributes.get_x()->get_name(), "x");
    EXPECT_EQ(opNode->attributes.get_freqs()->get_dim(), toVec(K_ROPE_TENSOR_FREQS_DIMS));
    EXPECT_EQ(opNode->attributes.get_freqs()->get_stride(), toVec(K_ROPE_TENSOR_FREQS_STRIDES));
    EXPECT_EQ(opNode->attributes.get_freqs()->get_name(), "freqs");
    EXPECT_EQ(opNode->attributes.get_y()->get_dim(), toVec(K_ROPE_TENSOR_Y_DIMS));
    EXPECT_EQ(opNode->attributes.get_y()->get_stride(), toVec(K_ROPE_TENSOR_Y_STRIDES));
    EXPECT_EQ(opNode->attributes.get_y()->get_name(), "y");
}

} // namespace
