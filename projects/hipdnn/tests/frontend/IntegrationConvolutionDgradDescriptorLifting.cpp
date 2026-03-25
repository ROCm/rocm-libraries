// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"
#include <hipdnn_test_sdk/constants/ConvDgradConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants::dgrad;

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
class IntegrationConvolutionBwdDescriptorLifting : public ::testing::Test
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

    /// Builds a standard ConvolutionBwd graph for round-trip testing.
    static std::shared_ptr<TestableGraph> buildGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("ConvolutionBwdLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

        auto w = std::make_shared<TensorAttributes>();
        w->set_uid(K_TENSOR_W_UID).set_name("w").set_data_type(DataType::FLOAT);
        w->set_dim(toVec(K_TENSOR_W_DIMS)).set_stride(toVec(K_TENSOR_W_STRIDES));

        ConvDgradAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);
        attrs.set_pre_padding({1, 1});
        attrs.set_post_padding({1, 1});
        attrs.set_stride({1, 1});
        attrs.set_dilation({1, 1});

        auto dx = graph->conv_dgrad(dy, w, attrs);
        dx->set_uid(K_TENSOR_DX_UID).set_output(true).set_name("dx");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a standard ConvolutionBwd graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationConvolutionBwdDescriptorLifting, BasicConvolutionBwdRoundTrip)
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

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_uid(), K_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_dim(), toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_stride(), toVec(K_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_data_type(), DataType::FLOAT);

    // Verify w tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_W_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_uid(), K_TENSOR_W_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_dim(), toVec(K_TENSOR_W_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_stride(), toVec(K_TENSOR_W_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_data_type(), DataType::FLOAT);

    // Verify dx tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_uid(), K_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_dim(), toVec(K_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_stride(), toVec(K_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_data_type(), DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<ConvolutionDgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ConvolutionDgradNode";

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
TEST_F(IntegrationConvolutionBwdDescriptorLifting, ConvolutionBwdTensorSharingPreserved)
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

    auto* opNode = dynamic_cast<ConvolutionDgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify dy tensor sharing
    EXPECT_EQ(opNode->attributes.get_dy()->get_uid(), K_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID].get(), opNode->attributes.get_dy().get());
    // Verify w tensor sharing
    EXPECT_EQ(opNode->attributes.get_w()->get_uid(), K_TENSOR_W_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID].get(), opNode->attributes.get_w().get());
    // Verify dx tensor sharing
    EXPECT_EQ(opNode->attributes.get_dx()->get_uid(), K_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID].get(), opNode->attributes.get_dx().get());
}

// Builds a ConvolutionBwd graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the FlatBuffer-direct path.
TEST_F(IntegrationConvolutionBwdDescriptorLifting, ConvolutionBwdLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<ConvolutionDgradNode*>(subNodes[0].get());
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

    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_dim(), toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_stride(), toVec(K_TENSOR_DY_STRIDES));
    ASSERT_NE(tensorMap.count(K_TENSOR_W_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_dim(), toVec(K_TENSOR_W_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_stride(), toVec(K_TENSOR_W_STRIDES));
    ASSERT_NE(tensorMap.count(K_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_dim(), toVec(K_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_stride(), toVec(K_TENSOR_DX_STRIDES));
}

} // namespace
