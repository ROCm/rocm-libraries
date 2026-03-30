// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_test_sdk/constants/ConvWgradConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants::conv_wgrad;
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
        dy->set_uid(K_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_TENSOR_X_DIMS)).set_stride(toVec(K_TENSOR_X_STRIDES));

        ConvWgradAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);
        attrs.set_pre_padding(toVec(K_CONV_PADDING));
        attrs.set_post_padding(toVec(K_CONV_PADDING));
        attrs.set_stride(toVec(K_CONV_STRIDE));
        attrs.set_dilation(toVec(K_CONV_DILATION));

        auto dw = graph->conv_wgrad(dy, x, attrs);
        dw->set_uid(K_TENSOR_DW_UID).set_output(true).set_name("dw");

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
    ASSERT_EQ(tensorMap.size(), 3u);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_uid(), K_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_dim(), toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_stride(), toVec(K_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_name(), "x");

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_uid(), K_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_dim(), toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_stride(), toVec(K_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_name(), "dy");

    // Verify dw tensor (dims/strides are inferred by infer_properties)
    ASSERT_NE(tensorMap.count(K_TENSOR_DW_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_uid(), K_TENSOR_DW_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_dim(), toVec(K_TENSOR_DW_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_stride(), toVec(K_TENSOR_DW_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_name(), "dw");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ConvolutionWgradNode";

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_convolution_mode(), ConvolutionMode::CONVOLUTION);

    // Verify pre_padding
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_CONV_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_CONV_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    // Verify dilation
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_CONV_DILATION));

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
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify dy tensor sharing
    EXPECT_EQ(opNode->attributes.get_dy()->get_uid(), K_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID].get(), opNode->attributes.get_dy().get());
    // Verify dw tensor sharing
    EXPECT_EQ(opNode->attributes.get_dw()->get_uid(), K_TENSOR_DW_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID].get(), opNode->attributes.get_dw().get());

    // Verify tensor names
    EXPECT_EQ(opNode->attributes.get_x()->get_name(), "x");
    EXPECT_EQ(opNode->attributes.get_dy()->get_name(), "dy");
    EXPECT_EQ(opNode->attributes.get_dw()->get_name(), "dw");
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
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_CONV_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_CONV_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    // Verify dilation
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_CONV_DILATION));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_dim(), toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_stride(), toVec(K_TENSOR_X_STRIDES));
    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_dim(), toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_stride(), toVec(K_TENSOR_DY_STRIDES));
    ASSERT_NE(tensorMap.count(K_TENSOR_DW_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_dim(), toVec(K_TENSOR_DW_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DW_UID]->get_stride(), toVec(K_TENSOR_DW_STRIDES));
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the lifting round trip and are all distinct.
TEST_F(IntegrationConvolutionWgradDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    constexpr std::array<int64_t, 4> K_AUTO_X_DIMS = {1, 4, 8, 8};
    constexpr std::array<int64_t, 4> K_AUTO_X_STRIDES = {256, 64, 8, 1};
    constexpr std::array<int64_t, 4> K_AUTO_DY_DIMS = {1, 16, 6, 6};
    constexpr std::array<int64_t, 4> K_AUTO_DY_STRIDES = {576, 36, 6, 1};

    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidWgradLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_AUTO_X_DIMS)).set_stride(toVec(K_AUTO_X_STRIDES));

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_AUTO_DY_DIMS)).set_stride(toVec(K_AUTO_DY_STRIDES));

    ConvWgradAttributes convAttrs;
    convAttrs.set_name("auto_uid_wgrad_op");
    convAttrs.set_pre_padding({1, 1});
    convAttrs.set_post_padding({1, 1});
    convAttrs.set_stride({1, 1});
    convAttrs.set_dilation({1, 1});
    convAttrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);

    auto dw = graph->conv_wgrad(dy, x, convAttrs);
    dw->set_output(true).set_name("dw");

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
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors in lifted graph";

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

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify tensor dims survived the round trip
    auto xUid = opNode->attributes.get_x()->get_uid();
    auto dyUid = opNode->attributes.get_dy()->get_uid();
    auto dwUid = opNode->attributes.get_dw()->get_uid();

    EXPECT_NE(xUid, dyUid);
    EXPECT_NE(xUid, dwUid);
    EXPECT_NE(dyUid, dwUid);

    EXPECT_EQ(tensorMap[xUid]->get_dim(), toVec(K_AUTO_X_DIMS));
    EXPECT_EQ(tensorMap[dyUid]->get_dim(), toVec(K_AUTO_DY_DIMS));
}

// Builds a conv wgrad graph with asymmetric padding (pre_padding={1,1},
// post_padding={2,2}), lowers, lifts, and verifies asymmetric padding
// values survive the round trip.
TEST_F(IntegrationConvolutionWgradDescriptorLifting, AsymmetricPaddingPreservedInLiftingRoundTrip)
{
    constexpr std::array<int64_t, 2> K_ASYM_PRE_PADDING = {1, 1};
    constexpr std::array<int64_t, 2> K_ASYM_POST_PADDING = {2, 2};

    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AsymPaddingWgradLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(K_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_X_DIMS)).set_stride(toVec(K_TENSOR_X_STRIDES));

    ConvWgradAttributes convAttrs;
    convAttrs.set_name("asym_wgrad_op");
    convAttrs.set_pre_padding(toVec(K_ASYM_PRE_PADDING));
    convAttrs.set_post_padding(toVec(K_ASYM_POST_PADDING));
    convAttrs.set_stride({1, 1});
    convAttrs.set_dilation({1, 1});
    convAttrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);

    auto dw = graph->conv_wgrad(dy, x, convAttrs);
    dw->set_uid(K_TENSOR_DW_UID).set_output(true).set_name("dw");

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

    auto* opNode = dynamic_cast<ConvolutionWgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ConvolutionWgradNode";

    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_ASYM_PRE_PADDING));
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_ASYM_POST_PADDING));
    EXPECT_EQ(opNode->attributes.get_stride(), std::vector<int64_t>({1, 1}));
    EXPECT_EQ(opNode->attributes.get_dilation(), std::vector<int64_t>({1, 1}));
}

} // namespace
