// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
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

// -- Test constants --
constexpr int64_t K_TEST_X_UID = 70;
constexpr int64_t K_TEST_MEAN_UID = 71;
constexpr int64_t K_TEST_INV_VARIANCE_UID = 72;
constexpr int64_t K_TEST_SCALE_UID = 73;
constexpr int64_t K_TEST_BIAS_UID = 74;
constexpr int64_t K_TEST_Y_UID = 75;

constexpr std::array<int64_t, 4> K_TEST_SPATIAL_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_TEST_SPATIAL_STRIDES = {65536, 1024, 32, 1};
constexpr std::array<int64_t, 4> K_TEST_CHANNEL_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_TEST_CHANNEL_STRIDES = {64, 1, 1, 1};

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationBatchnormInferenceDescriptorLifting : public ::testing::Test
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

    /// Builds a standard BatchnormInference graph for round-trip testing.
    static std::shared_ptr<TestableGraph> buildGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("BatchnormInferenceLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_TEST_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_TEST_SPATIAL_DIMS)).set_stride(toVec(K_TEST_SPATIAL_STRIDES));

        auto mean = std::make_shared<TensorAttributes>();
        mean->set_uid(K_TEST_MEAN_UID).set_name("mean").set_data_type(DataType::FLOAT);
        mean->set_dim(toVec(K_TEST_CHANNEL_DIMS)).set_stride(toVec(K_TEST_CHANNEL_STRIDES));

        auto invVariance = std::make_shared<TensorAttributes>();
        invVariance->set_uid(K_TEST_INV_VARIANCE_UID)
            .set_name("inv_variance")
            .set_data_type(DataType::FLOAT);
        invVariance->set_dim(toVec(K_TEST_CHANNEL_DIMS)).set_stride(toVec(K_TEST_CHANNEL_STRIDES));

        auto scale = std::make_shared<TensorAttributes>();
        scale->set_uid(K_TEST_SCALE_UID).set_name("scale").set_data_type(DataType::FLOAT);
        scale->set_dim(toVec(K_TEST_CHANNEL_DIMS)).set_stride(toVec(K_TEST_CHANNEL_STRIDES));

        auto bias = std::make_shared<TensorAttributes>();
        bias->set_uid(K_TEST_BIAS_UID).set_name("bias").set_data_type(DataType::FLOAT);
        bias->set_dim(toVec(K_TEST_CHANNEL_DIMS)).set_stride(toVec(K_TEST_CHANNEL_STRIDES));

        BatchnormInferenceAttributes attrs;
        attrs.set_name("test_op");

        auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, attrs);
        y->set_uid(K_TEST_Y_UID).set_output(true).set_name("y");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a standard BatchnormInference graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationBatchnormInferenceDescriptorLifting, BasicBatchnormInferenceRoundTrip)
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
    ASSERT_GE(tensorMap.size(), 6u);

    // Verify x tensor (spatial dims)
    ASSERT_NE(tensorMap.count(K_TEST_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_uid(), K_TEST_X_UID);
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_dim(), toVec(K_TEST_SPATIAL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_stride(), toVec(K_TEST_SPATIAL_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_data_type(), DataType::FLOAT);

    // Verify mean tensor (channel dims)
    ASSERT_NE(tensorMap.count(K_TEST_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_MEAN_UID]->get_uid(), K_TEST_MEAN_UID);
    EXPECT_EQ(tensorMap[K_TEST_MEAN_UID]->get_dim(), toVec(K_TEST_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_MEAN_UID]->get_stride(), toVec(K_TEST_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_MEAN_UID]->get_data_type(), DataType::FLOAT);

    // Verify inv_variance tensor (channel dims)
    ASSERT_NE(tensorMap.count(K_TEST_INV_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_INV_VARIANCE_UID]->get_uid(), K_TEST_INV_VARIANCE_UID);
    EXPECT_EQ(tensorMap[K_TEST_INV_VARIANCE_UID]->get_dim(), toVec(K_TEST_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_INV_VARIANCE_UID]->get_stride(), toVec(K_TEST_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_INV_VARIANCE_UID]->get_data_type(), DataType::FLOAT);

    // Verify scale tensor (channel dims)
    ASSERT_NE(tensorMap.count(K_TEST_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_SCALE_UID]->get_uid(), K_TEST_SCALE_UID);
    EXPECT_EQ(tensorMap[K_TEST_SCALE_UID]->get_dim(), toVec(K_TEST_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_SCALE_UID]->get_stride(), toVec(K_TEST_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_SCALE_UID]->get_data_type(), DataType::FLOAT);

    // Verify bias tensor (channel dims)
    ASSERT_NE(tensorMap.count(K_TEST_BIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_BIAS_UID]->get_uid(), K_TEST_BIAS_UID);
    EXPECT_EQ(tensorMap[K_TEST_BIAS_UID]->get_dim(), toVec(K_TEST_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_BIAS_UID]->get_stride(), toVec(K_TEST_CHANNEL_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_BIAS_UID]->get_data_type(), DataType::FLOAT);

    // Verify y tensor (spatial dims, inferred from x)
    ASSERT_NE(tensorMap.count(K_TEST_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_Y_UID]->get_uid(), K_TEST_Y_UID);
    EXPECT_EQ(tensorMap[K_TEST_Y_UID]->get_dim(), toVec(K_TEST_SPATIAL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_Y_UID]->get_stride(), toVec(K_TEST_SPATIAL_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_Y_UID]->get_data_type(), DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<BatchnormInferenceNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a BatchnormInferenceNode";

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationBatchnormInferenceDescriptorLifting, BatchnormInferenceTensorSharingPreserved)
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

    auto* opNode = dynamic_cast<BatchnormInferenceNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_TEST_X_UID);
    EXPECT_EQ(tensorMap[K_TEST_X_UID].get(), opNode->attributes.get_x().get());
    // Verify mean tensor sharing
    EXPECT_EQ(opNode->attributes.get_mean()->get_uid(), K_TEST_MEAN_UID);
    EXPECT_EQ(tensorMap[K_TEST_MEAN_UID].get(), opNode->attributes.get_mean().get());
    // Verify inv_variance tensor sharing
    EXPECT_EQ(opNode->attributes.get_inv_variance()->get_uid(), K_TEST_INV_VARIANCE_UID);
    EXPECT_EQ(tensorMap[K_TEST_INV_VARIANCE_UID].get(),
              opNode->attributes.get_inv_variance().get());
    // Verify scale tensor sharing
    EXPECT_EQ(opNode->attributes.get_scale()->get_uid(), K_TEST_SCALE_UID);
    EXPECT_EQ(tensorMap[K_TEST_SCALE_UID].get(), opNode->attributes.get_scale().get());
    // Verify bias tensor sharing
    EXPECT_EQ(opNode->attributes.get_bias()->get_uid(), K_TEST_BIAS_UID);
    EXPECT_EQ(tensorMap[K_TEST_BIAS_UID].get(), opNode->attributes.get_bias().get());
    // Verify y tensor sharing
    EXPECT_EQ(opNode->attributes.get_y()->get_uid(), K_TEST_Y_UID);
    EXPECT_EQ(tensorMap[K_TEST_Y_UID].get(), opNode->attributes.get_y().get());
}

// Builds a BatchnormInference graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the FlatBuffer-direct path.
TEST_F(IntegrationBatchnormInferenceDescriptorLifting, BatchnormInferenceLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<BatchnormInferenceNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_GE(tensorMap.size(), 6u);

    ASSERT_NE(tensorMap.count(K_TEST_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_dim(), toVec(K_TEST_SPATIAL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_stride(), toVec(K_TEST_SPATIAL_STRIDES));
    ASSERT_NE(tensorMap.count(K_TEST_MEAN_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_MEAN_UID]->get_dim(), toVec(K_TEST_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_MEAN_UID]->get_stride(), toVec(K_TEST_CHANNEL_STRIDES));
    ASSERT_NE(tensorMap.count(K_TEST_INV_VARIANCE_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_INV_VARIANCE_UID]->get_dim(), toVec(K_TEST_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_INV_VARIANCE_UID]->get_stride(), toVec(K_TEST_CHANNEL_STRIDES));
    ASSERT_NE(tensorMap.count(K_TEST_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_SCALE_UID]->get_dim(), toVec(K_TEST_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_SCALE_UID]->get_stride(), toVec(K_TEST_CHANNEL_STRIDES));
    ASSERT_NE(tensorMap.count(K_TEST_BIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_BIAS_UID]->get_dim(), toVec(K_TEST_CHANNEL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_BIAS_UID]->get_stride(), toVec(K_TEST_CHANNEL_STRIDES));
    ASSERT_NE(tensorMap.count(K_TEST_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_Y_UID]->get_dim(), toVec(K_TEST_SPATIAL_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_Y_UID]->get_stride(), toVec(K_TEST_SPATIAL_STRIDES));
}

} // namespace
