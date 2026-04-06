// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"
#include <hipdnn_test_sdk/constants/ConvDgradConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_data_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_data_sdk::data_objects::NodeAttributes;
using ConvModeSdk = hipdnn_data_sdk::data_objects::ConvMode;

namespace
{

// Exposes protected Graph methods for testing
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::get_raw_graph_descriptor;
};

using namespace hipdnn_tests::constants::dgrad::integration;

// -- Test constants for AutoAssignedUidsPreservedInRoundTrip --

constexpr std::array<int64_t, 4> K_AUTO_DY_DIMS = {1, 16, 6, 6};
constexpr std::array<int64_t, 4> K_AUTO_DY_STRIDES = {576, 36, 6, 1};
constexpr std::array<int64_t, 4> K_AUTO_W_DIMS = {16, 3, 3, 3};
constexpr std::array<int64_t, 4> K_AUTO_W_STRIDES = {27, 9, 3, 1};

constexpr std::array<int64_t, 2> K_AUTO_PADDING = {0, 0};
constexpr std::array<int64_t, 2> K_AUTO_STRIDE = {1, 1};
constexpr std::array<int64_t, 2> K_AUTO_DILATION = {1, 1};

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationConvolutionDgradDescriptorLowering : public ::testing::Test
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

    hipdnnHandle_t _handle = nullptr;
};

// Builds a conv_dgrad graph via the frontend API, lowers it to the backend
// via build_operation_graph_via_descriptors, retrieves the serialized graph,
// and verifies all tensor and operation attributes match the values set
// in the frontend.
TEST_F(IntegrationConvolutionDgradDescriptorLowering, ConvDgradGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestConvDgradGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(K_TENSOR_DY_UID).set_name("DY").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_TENSOR_W_UID).set_name("W").set_data_type(DataType::FLOAT);
    w->set_dim(toVec(K_TENSOR_W_DIMS)).set_stride(toVec(K_TENSOR_W_STRIDES));

    ConvDgradAttributes convAttrs;
    convAttrs.set_name("conv_dgrad_op");
    convAttrs.set_pre_padding(toVec(K_CONV_PRE_PADDING));
    convAttrs.set_post_padding(toVec(K_CONV_POST_PADDING));
    convAttrs.set_stride(toVec(K_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_CONV_DILATION));
    convAttrs.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    auto dx = graph->conv_dgrad(dy, w, convAttrs);
    dx->set_uid(K_TENSOR_DX_UID).set_output(true).set_name("DX");

    // -- Validate and lower --
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // -- Retrieve serialized graph --
    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    // -- Deserialize into GraphT --
    auto graphFb = hipdnn_data_sdk::data_objects::GetGraph(serializedData.data());
    ASSERT_NE(graphFb, nullptr);
    hipdnn_data_sdk::data_objects::GraphT graphT;
    graphFb->UnPackTo(&graphT);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors --
    ASSERT_EQ(graphT.tensors.size(), 3u);

    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    // Verify DY tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    auto* dyT = tensorMap[K_TENSOR_DY_UID];
    EXPECT_EQ(dyT->name, "DY");
    EXPECT_EQ(dyT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(dyT->dims, toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(dyT->strides, toVec(K_TENSOR_DY_STRIDES));
    EXPECT_FALSE(dyT->virtual_);

    // Verify W tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_W_UID), 0u);
    auto* wT = tensorMap[K_TENSOR_W_UID];
    EXPECT_EQ(wT->name, "W");
    EXPECT_EQ(wT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(wT->dims, toVec(K_TENSOR_W_DIMS));
    EXPECT_EQ(wT->strides, toVec(K_TENSOR_W_STRIDES));
    EXPECT_FALSE(wT->virtual_);

    // Verify DX tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_DX_UID), 0u);
    auto* dxT = tensorMap[K_TENSOR_DX_UID];
    EXPECT_EQ(dxT->name, "DX");
    EXPECT_EQ(dxT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(dxT->virtual_);

    // -- Verify conv bwd operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::ConvolutionBwdAttributes);

    auto* convBwd = node->attributes.AsConvolutionBwdAttributes();
    ASSERT_NE(convBwd, nullptr);

    EXPECT_EQ(convBwd->dy_tensor_uid, K_TENSOR_DY_UID);
    EXPECT_EQ(convBwd->w_tensor_uid, K_TENSOR_W_UID);
    EXPECT_EQ(convBwd->dx_tensor_uid, K_TENSOR_DX_UID);
    EXPECT_EQ(convBwd->pre_padding, toVec(K_CONV_PRE_PADDING));
    EXPECT_EQ(convBwd->post_padding, toVec(K_CONV_POST_PADDING));
    EXPECT_EQ(convBwd->stride, toVec(K_CONV_STRIDE));
    EXPECT_EQ(convBwd->dilation, toVec(K_CONV_DILATION));
    EXPECT_EQ(convBwd->conv_mode, ConvModeSdk::CROSS_CORRELATION);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationConvolutionDgradDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidDgradGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("DY").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_AUTO_DY_DIMS)).set_stride(toVec(K_AUTO_DY_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_name("W").set_data_type(DataType::FLOAT);
    w->set_dim(toVec(K_AUTO_W_DIMS)).set_stride(toVec(K_AUTO_W_STRIDES));

    ConvDgradAttributes convAttrs;
    convAttrs.set_padding(toVec(K_AUTO_PADDING));
    convAttrs.set_stride(toVec(K_AUTO_STRIDE));
    convAttrs.set_dilation(toVec(K_AUTO_DILATION));

    auto dx = graph->conv_dgrad(dy, w, convAttrs);
    dx->set_output(true);

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve serialized graph
    auto rawDesc = graph->get_raw_graph_descriptor();
    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnn_data_sdk::data_objects::GraphT graphT;
    hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    // All tensors should have been auto-assigned unique UIDs
    ASSERT_EQ(graphT.tensors.size(), 3u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 3u) << "Tensor UIDs are not unique";

    // The conv bwd operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* convBwd = graphT.nodes[0]->attributes.AsConvolutionBwdAttributes();
    ASSERT_NE(convBwd, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(convBwd->dy_tensor_uid) > 0)
        << "DY tensor UID " << convBwd->dy_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(convBwd->w_tensor_uid) > 0)
        << "W tensor UID " << convBwd->w_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(convBwd->dx_tensor_uid) > 0)
        << "DX tensor UID " << convBwd->dx_tensor_uid << " not found in graph tensors";

    // All three tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids
        = {convBwd->dy_tensor_uid, convBwd->w_tensor_uid, convBwd->dx_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 3u) << "Conv bwd node tensor UIDs are not distinct";
}

} // namespace
