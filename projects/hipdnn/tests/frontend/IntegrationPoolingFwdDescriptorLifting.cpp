// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/PoolingFwdNode.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

#include <hipdnn_test_sdk/constants/PoolingFwdConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants::integration;
using hipdnn_tests::toVec;

namespace
{

// Exposes protected Graph methods for lifting integration tests
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph;
    using Graph::deserialize_via_backend;
    using Graph::get_raw_graph_descriptor;

    const std::vector<std::shared_ptr<INode>>& getSubNodes() const
    {
        return _sub_nodes;
    }
};

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationPoolingFwdDescriptorLifting : public ::testing::Test
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

    /// Builds a standard PoolingFwd graph for round-trip testing.
    static std::shared_ptr<TestableGraph> buildGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("PoolingFwdLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_POOL_FWD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_POOL_FWD_TENSOR_X_DIMS)).set_stride(toVec(K_POOL_FWD_TENSOR_X_STRIDES));

        PoolingFwdAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_pooling_mode(PoolingMode::MAX);
        attrs.set_pre_padding({1, 1});
        attrs.set_post_padding({1, 1});
        attrs.set_stride({2, 2});
        attrs.set_window({3, 3});

        auto y = graph->pooling_fwd(x, attrs);
        y->set_uid(K_POOL_FWD_TENSOR_Y_UID).set_output(true).set_name("y");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a standard PoolingFwd graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationPoolingFwdDescriptorLifting, BasicPoolingFwdRoundTrip)
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
    ASSERT_GE(tensorMap.size(), 2u);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_POOL_FWD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->get_name(), "x");
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->get_uid(), K_POOL_FWD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->get_dim(), toVec(K_POOL_FWD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->get_stride(), toVec(K_POOL_FWD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);

    // Verify y tensor
    ASSERT_NE(tensorMap.count(K_POOL_FWD_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->get_name(), "y");
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->get_uid(), K_POOL_FWD_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->get_dim(), toVec(K_POOL_FWD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->get_stride(), toVec(K_POOL_FWD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->get_data_type(), DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<PoolingFwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a PoolingFwdNode";

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_pooling_mode(), PoolingMode::MAX);

    // Verify pre_padding
    EXPECT_EQ(opNode->attributes.get_pre_padding(), std::vector<int64_t>({1, 1}));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), std::vector<int64_t>({1, 1}));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), std::vector<int64_t>({2, 2}));
    // Verify window
    EXPECT_EQ(opNode->attributes.get_window(), std::vector<int64_t>({3, 3}));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationPoolingFwdDescriptorLifting, PoolingFwdTensorSharingPreserved)
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

    auto* opNode = dynamic_cast<PoolingFwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_POOL_FWD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify y tensor sharing
    EXPECT_EQ(opNode->attributes.get_y()->get_uid(), K_POOL_FWD_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID].get(), opNode->attributes.get_y().get());
}

// Builds a PoolingFwd graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the FlatBuffer-direct path.
TEST_F(IntegrationPoolingFwdDescriptorLifting, PoolingFwdLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<PoolingFwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_pooling_mode(), PoolingMode::MAX);

    // Verify pre_padding
    EXPECT_EQ(opNode->attributes.get_pre_padding(), std::vector<int64_t>({1, 1}));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), std::vector<int64_t>({1, 1}));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), std::vector<int64_t>({2, 2}));
    // Verify window
    EXPECT_EQ(opNode->attributes.get_window(), std::vector<int64_t>({3, 3}));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);

    ASSERT_NE(tensorMap.count(K_POOL_FWD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->get_dim(), toVec(K_POOL_FWD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->get_stride(), toVec(K_POOL_FWD_TENSOR_X_STRIDES));
    ASSERT_NE(tensorMap.count(K_POOL_FWD_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->get_dim(), toVec(K_POOL_FWD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->get_stride(), toVec(K_POOL_FWD_TENSOR_X_STRIDES));
}

// Verifies that the optional generate_index attribute survives a lifting round-trip.
TEST_F(IntegrationPoolingFwdDescriptorLifting, GenerateIndexPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("PoolingFwdGenerateIndexLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_POOL_FWD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_POOL_FWD_TENSOR_X_DIMS)).set_stride(toVec(K_POOL_FWD_TENSOR_X_STRIDES));

    PoolingFwdAttributes attrs;
    attrs.set_name("test_generate_index");
    attrs.set_pooling_mode(PoolingMode::MAX);
    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});
    attrs.set_generate_index(true);

    auto y = graph->pooling_fwd(x, attrs);
    y->set_uid(K_POOL_FWD_TENSOR_Y_UID).set_output(true).set_name("y");

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

    auto* opNode = dynamic_cast<PoolingFwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a PoolingFwdNode";

    ASSERT_TRUE(opNode->attributes.get_generate_index().has_value());
    EXPECT_EQ(opNode->attributes.get_generate_index().value(), true);
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the round trip and are all distinct.
TEST_F(IntegrationPoolingFwdDescriptorLifting, PoolingFwdAutoAssignedUidsPreserved)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidPoolingFwdLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_POOL_FWD_TENSOR_X_DIMS)).set_stride(toVec(K_POOL_FWD_TENSOR_X_STRIDES));

    PoolingFwdAttributes attrs;
    attrs.set_name("auto_uid_pool_op");
    attrs.set_pooling_mode(PoolingMode::MAX);
    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

    auto y = graph->pooling_fwd(x, attrs);
    y->set_output(true).set_name("y");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u) << "Expected 2 tensors in lifted graph";

    // Collect all UIDs and verify they are distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    EXPECT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "All auto-assigned UIDs must be distinct";

    // Verify the node references tensors with auto-assigned UIDs
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<PoolingFwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify tensor UIDs are distinct
    auto xUid = opNode->attributes.get_x()->get_uid();
    auto yUid = opNode->attributes.get_y()->get_uid();

    EXPECT_NE(xUid, yUid);

    // Verify tensor dims survived the round trip
    EXPECT_EQ(tensorMap[xUid]->get_dim(), toVec(K_POOL_FWD_TENSOR_X_DIMS));
}

} // namespace
