// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <set>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/ReductionNode.hpp>
#include <hipdnn_test_sdk/constants/ReductionConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Exposes protected Graph methods for lifting integration tests
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph;
    using Graph::deserialize_via_backend;
    using Graph::fromBackendDescriptor;
    using Graph::get_raw_graph_descriptor;

    const std::vector<std::shared_ptr<INode>>& getSubNodes() const
    {
        return _sub_nodes;
    }
};

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationReductionDescriptorLifting : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnDestroy(_handle);
        }
    }

    /// Builds a standard Reduction graph for round-trip testing.
    static std::shared_ptr<TestableGraph> buildGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
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

    hipdnnHandle_t _handle = nullptr;
};

// Builds a standard Reduction graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationReductionDescriptorLifting, BasicReductionRoundTrip)
{
    auto originalGraph = buildGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

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
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<ReductionNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ReductionNode";

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

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

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
// all fields survive the FlatBuffer-direct path.
TEST_F(IntegrationReductionDescriptorLifting, ReductionLiftWithoutFinalization)
{
    auto originalGraph = buildGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to binary via the frontend
    auto data = originalGraph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create a backend graph descriptor from serialized bytes (no handle, no finalize)
    const detail::ScopedHipdnnBackendDescriptor graphDesc(data.data(), data.size());
    ASSERT_TRUE(graphDesc.valid()) << "Failed to create backend graph descriptor";

    // Lift into a new graph via fromBackendDescriptor
    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(graphDesc.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

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
    auto graph = std::make_shared<TestableGraph>();
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

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ReductionNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ReductionNode";

    EXPECT_TRUE(opNode->attributes.get_is_deterministic());
}

// Builds a Reduction graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationReductionDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
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

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

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
        << "Found duplicate auto-assigned UIDs";

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
    ASSERT_EQ(nodeUids.size(), 2u) << "Node tensor UIDs are not all distinct";

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_x()->get_dim(), toVec(K_REDUCTION_TENSOR_X_DIMS));
    EXPECT_EQ(opNode->attributes.get_x()->get_stride(), toVec(K_REDUCTION_TENSOR_X_STRIDES));
    EXPECT_EQ(opNode->attributes.get_y()->get_dim(), toVec(K_REDUCTION_TENSOR_Y_DIMS));
    EXPECT_EQ(opNode->attributes.get_y()->get_stride(), toVec(K_REDUCTION_TENSOR_Y_STRIDES));
}

} // namespace
