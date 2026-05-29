// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using ConvModeSdk = hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;

namespace
{

// -- Test constants for AutoAssignedUidsPreservedInRoundTrip --

constexpr std::array<int64_t, 4> K_AUTO_X_DIMS = {1, 3, 8, 8};
constexpr std::array<int64_t, 4> K_AUTO_X_STRIDES = {192, 64, 8, 1};
constexpr std::array<int64_t, 4> K_AUTO_W_DIMS = {16, 3, 3, 3};
constexpr std::array<int64_t, 4> K_AUTO_W_STRIDES = {27, 9, 3, 1};

constexpr std::array<int64_t, 2> K_AUTO_PADDING = {0, 0};
constexpr std::array<int64_t, 2> K_AUTO_STRIDE = {1, 1};
constexpr std::array<int64_t, 2> K_AUTO_DILATION = {1, 1};

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationConvFpropDescriptorLowering : public IntegrationTestFixture
{
};

// Builds a conv_fprop graph via the frontend API, lowers it to the backend
// via build_operation_graph_via_descriptors, retrieves the serialized graph,
// and verifies all tensor and operation attributes match the values set
// in the frontend.
TEST_F(IntegrationConvFpropDescriptorLowering, ConvFpropGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestConvGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_FPROP_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_FPROP_TENSOR_X_DIMS)).set_stride(toVec(K_FPROP_TENSOR_X_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_FPROP_TENSOR_W_UID).set_name("W").set_data_type(DataType::FLOAT);
    w->set_dim(toVec(K_FPROP_TENSOR_W_DIMS)).set_stride(toVec(K_FPROP_TENSOR_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_name("conv_fprop_op");
    convAttrs.set_pre_padding(toVec(K_FPROP_CONV_PADDING));
    convAttrs.set_post_padding(toVec(K_FPROP_CONV_PADDING));
    convAttrs.set_stride(toVec(K_FPROP_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_FPROP_CONV_DILATION));
    convAttrs.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    auto y = graph->conv_fprop(x, w, convAttrs);
    y->set_uid(K_FPROP_TENSOR_Y_UID).set_output(true).set_name("Y");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors --
    ASSERT_EQ(graphT.tensors.size(), 3u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_FPROP_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(xT->dims, toVec(K_FPROP_TENSOR_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_FPROP_TENSOR_X_STRIDES));
    EXPECT_FALSE(xT->virtual_);

    // Verify W tensor
    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_W_UID), 0u);
    auto* wT = tensorMap[K_FPROP_TENSOR_W_UID];
    EXPECT_EQ(wT->name, "W");
    EXPECT_EQ(wT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(wT->dims, toVec(K_FPROP_TENSOR_W_DIMS));
    EXPECT_EQ(wT->strides, toVec(K_FPROP_TENSOR_W_STRIDES));
    EXPECT_FALSE(wT->virtual_);

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_Y_UID), 0u);
    auto* yT = tensorMap[K_FPROP_TENSOR_Y_UID];
    EXPECT_EQ(yT->name, "Y");
    EXPECT_EQ(yT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(yT->virtual_);

    // -- Verify conv operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->name, "conv_fprop_op");
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::ConvolutionFwdAttributes);

    auto* convFwd = node->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(convFwd, nullptr);

    EXPECT_EQ(convFwd->x_tensor_uid, K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(convFwd->w_tensor_uid, K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(convFwd->y_tensor_uid, K_FPROP_TENSOR_Y_UID);
    EXPECT_EQ(convFwd->pre_padding, toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convFwd->post_padding, toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convFwd->stride, toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(convFwd->dilation, toVec(K_FPROP_CONV_DILATION));
    EXPECT_EQ(convFwd->conv_mode, ConvModeSdk::CROSS_CORRELATION);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationConvFpropDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("AutoUidGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_AUTO_X_DIMS)).set_stride(toVec(K_AUTO_X_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_name("W").set_data_type(DataType::FLOAT);
    w->set_dim(toVec(K_AUTO_W_DIMS)).set_stride(toVec(K_AUTO_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_padding(toVec(K_AUTO_PADDING));
    convAttrs.set_stride(toVec(K_AUTO_STRIDE));
    convAttrs.set_dilation(toVec(K_AUTO_DILATION));

    auto y = graph->conv_fprop(x, w, convAttrs);
    y->set_output(true);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // All tensors should have been auto-assigned unique UIDs
    // (auto-assignment starts from 0, so UID 0 is valid)
    ASSERT_EQ(graphT.tensors.size(), 3u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 3u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)

    // The conv operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* convFwd = graphT.nodes[0]->attributes.AsConvolutionFwdAttributes();
    ASSERT_NE(convFwd, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(convFwd->x_tensor_uid) > 0)
        << "X tensor UID " << convFwd->x_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(convFwd->w_tensor_uid) > 0)
        << "W tensor UID " << convFwd->w_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(convFwd->y_tensor_uid) > 0)
        << "Y tensor UID " << convFwd->y_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All three tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids
        = {convFwd->x_tensor_uid, convFwd->w_tensor_uid, convFwd->y_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 3u)
        << "Conv node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}

} // namespace
