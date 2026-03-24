// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
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
constexpr int64_t K_TEST_X_UID = 20;
constexpr int64_t K_TEST_DY_UID = 21;
constexpr int64_t K_TEST_DW_UID = 22;

constexpr std::array<int64_t, 4> K_TEST_X_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_TEST_X_STRIDES = {3072, 1024, 32, 1};
constexpr std::array<int64_t, 4> K_TEST_DY_DIMS = {1, 64, 32, 32};
constexpr std::array<int64_t, 4> K_TEST_DY_STRIDES = {65536, 1024, 32, 1};
constexpr std::array<int64_t, 4> K_TEST_DW_DIMS = {64, 3, 3, 3};
constexpr std::array<int64_t, 4> K_TEST_DW_STRIDES = {27, 9, 3, 1};

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationConvolutionWgradDescriptorLifting : public ::testing::Test
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

    /// Builds a standard ConvolutionWrw graph for round-trip testing.
    static std::shared_ptr<TestableGraph> buildGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("ConvolutionWrwLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_TEST_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_TEST_DY_DIMS)).set_stride(toVec(K_TEST_DY_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_TEST_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_TEST_X_DIMS)).set_stride(toVec(K_TEST_X_STRIDES));

        ConvWgradAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);
        attrs.set_pre_padding({1, 1});
        attrs.set_post_padding({1, 1});
        attrs.set_stride({1, 1});
        attrs.set_dilation({1, 1});

        auto dw = graph->conv_wgrad(dy, x, attrs);
        dw->set_uid(K_TEST_DW_UID).set_output(true).set_name("dw");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a standard ConvolutionWrw graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationConvolutionWgradDescriptorLifting, BasicConvolutionWrwRoundTrip)
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
    ASSERT_GE(tensorMap.size(), 3u);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_TEST_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_uid(), K_TEST_X_UID);
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_dim(), toVec(K_TEST_X_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_stride(), toVec(K_TEST_X_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_data_type(), DataType::FLOAT);

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_TEST_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_DY_UID]->get_uid(), K_TEST_DY_UID);
    EXPECT_EQ(tensorMap[K_TEST_DY_UID]->get_dim(), toVec(K_TEST_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_DY_UID]->get_stride(), toVec(K_TEST_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_DY_UID]->get_data_type(), DataType::FLOAT);

    // Verify dw tensor (dims/strides are inferred by infer_properties)
    ASSERT_NE(tensorMap.count(K_TEST_DW_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_DW_UID]->get_uid(), K_TEST_DW_UID);
    EXPECT_EQ(tensorMap[K_TEST_DW_UID]->get_dim(), toVec(K_TEST_DW_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_DW_UID]->get_stride(), toVec(K_TEST_DW_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_DW_UID]->get_data_type(), DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ConvolutionWgradNode";

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_convolution_mode(), ConvolutionMode::CONVOLUTION);

    // Verify pre_padding
    EXPECT_EQ(opNode->attributes.get_pre_padding(), std::vector<int64_t>({1, 1}));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), std::vector<int64_t>({1, 1}));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), std::vector<int64_t>({1, 1}));
    // Verify dilation
    EXPECT_EQ(opNode->attributes.get_dilation(), std::vector<int64_t>({1, 1}));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationConvolutionWgradDescriptorLifting, ConvolutionWrwTensorSharingPreserved)
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

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_TEST_X_UID);
    EXPECT_EQ(tensorMap[K_TEST_X_UID].get(), opNode->attributes.get_x().get());
    // Verify dy tensor sharing
    EXPECT_EQ(opNode->attributes.get_dy()->get_uid(), K_TEST_DY_UID);
    EXPECT_EQ(tensorMap[K_TEST_DY_UID].get(), opNode->attributes.get_dy().get());
    // Verify dw tensor sharing
    EXPECT_EQ(opNode->attributes.get_dw()->get_uid(), K_TEST_DW_UID);
    EXPECT_EQ(tensorMap[K_TEST_DW_UID].get(), opNode->attributes.get_dw().get());
}

// Builds a ConvolutionWrw graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the FlatBuffer-direct path.
TEST_F(IntegrationConvolutionWgradDescriptorLifting, ConvolutionWrwLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_convolution_mode(), ConvolutionMode::CONVOLUTION);

    // Verify pre_padding
    EXPECT_EQ(opNode->attributes.get_pre_padding(), std::vector<int64_t>({1, 1}));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), std::vector<int64_t>({1, 1}));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), std::vector<int64_t>({1, 1}));
    // Verify dilation
    EXPECT_EQ(opNode->attributes.get_dilation(), std::vector<int64_t>({1, 1}));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_GE(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_TEST_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_dim(), toVec(K_TEST_X_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->get_stride(), toVec(K_TEST_X_STRIDES));
    ASSERT_NE(tensorMap.count(K_TEST_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_DY_UID]->get_dim(), toVec(K_TEST_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_DY_UID]->get_stride(), toVec(K_TEST_DY_STRIDES));
    ASSERT_NE(tensorMap.count(K_TEST_DW_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_DW_UID]->get_dim(), toVec(K_TEST_DW_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_DW_UID]->get_stride(), toVec(K_TEST_DW_STRIDES));
}

} // namespace
