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
#include <hipdnn_frontend/node/RMSNormBackwardNode.hpp>
#include <hipdnn_test_sdk/constants/RMSNormBackwardConstants.hpp>
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
class IntegrationRMSNormBackwardDescriptorLifting : public ::testing::Test
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

    /// Builds a standard RMSNormBackward graph for round-trip testing.
    static std::shared_ptr<TestableGraph> buildGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("RMSNormBackwardLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_RMSNORMBACKWARD_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS))
            .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_RMSNORMBACKWARD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
            .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_RMSNORMBACKWARD_TENSOR_SCALE_UID)
            .set_name("scale")
            .set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS))
            .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));

        RMSNormBackwardAttributes attrs;
        attrs.set_name("test_op");

        auto results = graph->rmsnorm_backward(dy, x, scale, attrs);
        results[0]->set_uid(K_RMSNORMBACKWARD_TENSOR_DX_UID).set_output(true).set_name("dx");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a standard RMSNormBackward graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationRMSNormBackwardDescriptorLifting, BasicRMSNormBackwardRoundTrip)
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
    ASSERT_EQ(tensorMap.size(), 5u);

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_data_type(), DataType::FLOAT);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_uid(), K_RMSNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);

    // Verify scale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_data_type(), DataType::FLOAT);

    // Verify dx tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_data_type(), DataType::FLOAT);

    // Verify dscale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_uid(),
              K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_data_type(), DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<RMSNormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a RMSNormBackwardNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationRMSNormBackwardDescriptorLifting, RMSNormBackwardTensorSharingPreserved)
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

    auto* opNode = dynamic_cast<RMSNormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify dy tensor sharing
    EXPECT_EQ(opNode->attributes.get_dy()->get_uid(), K_RMSNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID].get(), opNode->attributes.get_dy().get());
    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_RMSNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify scale tensor sharing
    EXPECT_EQ(opNode->attributes.get_scale()->get_uid(), K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID].get(),
              opNode->attributes.get_scale().get());
    // Verify dx tensor sharing
    EXPECT_EQ(opNode->attributes.get_dx()->get_uid(), K_RMSNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID].get(), opNode->attributes.get_dx().get());
}

// Builds a RMSNormBackward graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the FlatBuffer-direct path.
TEST_F(IntegrationRMSNormBackwardDescriptorLifting, RMSNormBackwardLiftWithoutFinalization)
{
    auto originalGraph = buildGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to binary via the frontend
    auto data = originalGraph->toBinary();
    ASSERT_FALSE(data.empty());

    // Create a backend graph descriptor from serialized bytes (no handle, no finalize)
    const detail::ScopedHipdnnBackendDescriptor graphDesc(data.data(), data.size());
    ASSERT_TRUE(graphDesc.valid())
        << "Failed to create backend graph descriptor"; // NOLINT(readability-implicit-bool-conversion)

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

    auto* opNode = dynamic_cast<RMSNormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 5u);

    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
}

// Builds a RMSNormBackward graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationRMSNormBackwardDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("RMSNormBackwardAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));

    RMSNormBackwardAttributes attrs;
    attrs.set_name("test_auto_uid");

    auto results = graph->rmsnorm_backward(dy, x, scale, attrs);
    results[0]->set_output(true).set_name("dx");

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
    ASSERT_EQ(tensorMap.size(), 5u);

    // Verify all UIDs are positive and distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        EXPECT_GT(uid, 0)
            << "Auto-assigned UID should be positive"; // NOLINT(readability-implicit-bool-conversion)
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    ASSERT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "Found duplicate auto-assigned UIDs"; // NOLINT(readability-implicit-bool-conversion)

    // Verify sub-node tensor UIDs are distinct via the node attributes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<RMSNormBackwardNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_dy(), nullptr);
    nodeUids.insert(opNode->attributes.get_dy()->get_uid());
    ASSERT_NE(opNode->attributes.get_x(), nullptr);
    nodeUids.insert(opNode->attributes.get_x()->get_uid());
    ASSERT_NE(opNode->attributes.get_scale(), nullptr);
    nodeUids.insert(opNode->attributes.get_scale()->get_uid());
    ASSERT_NE(opNode->attributes.get_dx(), nullptr);
    nodeUids.insert(opNode->attributes.get_dx()->get_uid());
    ASSERT_EQ(nodeUids.size(), 5u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_dy()->get_dim(), toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(opNode->attributes.get_dy()->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(opNode->attributes.get_x()->get_dim(), toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(opNode->attributes.get_x()->get_stride(), toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(opNode->attributes.get_scale()->get_dim(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(opNode->attributes.get_scale()->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(opNode->attributes.get_dx()->get_dim(), toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(opNode->attributes.get_dx()->get_stride(),
              toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
}

} // namespace
