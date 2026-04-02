// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants::integration;

namespace
{

// Exposes protected Graph methods for testing
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

class IntegrationConvFpropDescriptorLifting : public ::testing::Test
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

    // Builds a standard conv fprop graph for round-trip testing
    static std::shared_ptr<TestableGraph>
        buildConvFpropGraph(DataType computeType = DataType::FLOAT,
                            DataType intermediateType = DataType::FLOAT,
                            DataType ioType = DataType::FLOAT)
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("ConvFpropLiftingTestGraph")
            .set_compute_data_type(computeType)
            .set_intermediate_data_type(intermediateType)
            .set_io_data_type(ioType);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_TENSOR_X_DIMS)).set_stride(toVec(K_TENSOR_X_STRIDES));

        auto w = std::make_shared<TensorAttributes>();
        w->set_uid(K_TENSOR_W_UID).set_name("W").set_data_type(DataType::FLOAT);
        w->set_dim(toVec(K_TENSOR_W_DIMS)).set_stride(toVec(K_TENSOR_W_STRIDES));

        ConvFpropAttributes convAttrs;
        convAttrs.set_name("conv_fprop_op");
        convAttrs.set_pre_padding(toVec(K_CONV_PRE_PADDING));
        convAttrs.set_post_padding(toVec(K_CONV_POST_PADDING));
        convAttrs.set_stride(toVec(K_CONV_STRIDE));
        convAttrs.set_dilation(toVec(K_CONV_DILATION));
        convAttrs.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

        auto y = graph->conv_fprop(x, w, convAttrs);
        y->set_uid(K_TENSOR_Y_UID).set_output(true).set_name("Y");

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a standard conv fprop graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and convolution parameters.
TEST_F(IntegrationConvFpropDescriptorLifting, BasicConvFpropRoundTrip)
{
    auto originalGraph = buildConvFpropGraph();

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
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors (X, W, Y) in lifted graph";

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_X_UID), 0u);
    auto liftedX = tensorMap[K_TENSOR_X_UID];
    EXPECT_EQ(liftedX->get_uid(), K_TENSOR_X_UID);
    EXPECT_EQ(liftedX->get_name(), "X");
    EXPECT_EQ(liftedX->get_dim(), toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(liftedX->get_stride(), toVec(K_TENSOR_X_STRIDES));
    EXPECT_EQ(liftedX->get_data_type(), DataType::FLOAT);

    // Verify W tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_W_UID), 0u);
    auto liftedW = tensorMap[K_TENSOR_W_UID];
    EXPECT_EQ(liftedW->get_uid(), K_TENSOR_W_UID);
    EXPECT_EQ(liftedW->get_name(), "W");
    EXPECT_EQ(liftedW->get_dim(), toVec(K_TENSOR_W_DIMS));
    EXPECT_EQ(liftedW->get_stride(), toVec(K_TENSOR_W_STRIDES));
    EXPECT_EQ(liftedW->get_data_type(), DataType::FLOAT);

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_Y_UID), 0u);
    auto liftedY = tensorMap[K_TENSOR_Y_UID];
    EXPECT_EQ(liftedY->get_uid(), K_TENSOR_Y_UID);
    EXPECT_EQ(liftedY->get_name(), "Y");
    EXPECT_EQ(liftedY->get_dim(), toVec(K_TENSOR_Y_DIMS));
    EXPECT_EQ(liftedY->get_stride(), toVec(K_TENSOR_Y_STRIDES));
    EXPECT_EQ(liftedY->get_data_type(), DataType::FLOAT);

    // Verify 1 sub-node of the correct type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr) << "Expected a ConvolutionFpropNode";

    // Verify convolution parameters
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_CONV_PRE_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_CONV_POST_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_CONV_DILATION));
    EXPECT_EQ(convNode->attributes.get_convolution_mode(), ConvolutionMode::CROSS_CORRELATION);
    EXPECT_EQ(convNode->attributes.get_name(), "conv_fprop_op");
}

// Builds a conv fprop graph with asymmetric padding (pre_padding={2,0},
// post_padding={0,3}), lowers, lifts, and verifies asymmetric padding
// values survive the round trip.
TEST_F(IntegrationConvFpropDescriptorLifting, ConvFpropAsymmetricPaddingPreserved)
{
    constexpr std::array<int64_t, 2> K_ASYM_PRE_PADDING = {2, 0};
    constexpr std::array<int64_t, 2> K_ASYM_POST_PADDING = {0, 3};

    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AsymmetricPaddingLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_X_DIMS)).set_stride(toVec(K_TENSOR_X_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_TENSOR_W_UID).set_name("W").set_data_type(DataType::FLOAT);
    w->set_dim(toVec(K_TENSOR_W_DIMS)).set_stride(toVec(K_TENSOR_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_name("asym_conv_op");
    convAttrs.set_pre_padding(toVec(K_ASYM_PRE_PADDING));
    convAttrs.set_post_padding(toVec(K_ASYM_POST_PADDING));
    convAttrs.set_stride(toVec(K_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_CONV_DILATION));
    convAttrs.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    auto y = graph->conv_fprop(x, w, convAttrs);
    y->set_uid(K_TENSOR_Y_UID).set_output(true).set_name("Y");

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

    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr) << "Expected a ConvolutionFpropNode";

    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_ASYM_PRE_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_ASYM_POST_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_CONV_DILATION));

    // Verify Y tensor dims were inferred correctly with asymmetric padding
    // X={2,3,14,14}, W={8,3,3,3}, stride={2,2}, pre_pad={2,0}, post_pad={0,3}
    // H_out = (14 + 2 + 0 - 3) / 2 + 1 = 7 (rounded up)
    // W_out = (14 + 0 + 3 - 3) / 2 + 1 = 8
    const std::vector<int64_t> expectedYDims = {2, 8, 7, 8};
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_NE(tensorMap.count(K_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_Y_UID]->get_dim(), expectedYDims);
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationConvFpropDescriptorLifting, ConvFpropTensorSharingPreserved)
{
    auto originalGraph = buildConvFpropGraph();

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

    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr);

    // Verify UIDs match
    EXPECT_EQ(convNode->attributes.get_x()->get_uid(), K_TENSOR_X_UID);
    EXPECT_EQ(convNode->attributes.get_w()->get_uid(), K_TENSOR_W_UID);
    EXPECT_EQ(convNode->attributes.get_y()->get_uid(), K_TENSOR_Y_UID);

    // Verify tensor names
    EXPECT_EQ(convNode->attributes.get_x()->get_name(), "X");
    EXPECT_EQ(convNode->attributes.get_w()->get_name(), "W");
    EXPECT_EQ(convNode->attributes.get_y()->get_name(), "Y");

    // Verify pointer equality: tensor map and node attributes share the same objects
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID].get(), convNode->attributes.get_x().get());
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID].get(), convNode->attributes.get_w().get());
    EXPECT_EQ(tensorMap[K_TENSOR_Y_UID].get(), convNode->attributes.get_y().get());
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the round trip and are all distinct.
TEST_F(IntegrationConvFpropDescriptorLifting, ConvFpropAutoAssignedUidsPreserved)
{
    // Use different dims to distinguish from standard constants
    constexpr std::array<int64_t, 4> K_AUTO_X_DIMS = {1, 4, 8, 8};
    constexpr std::array<int64_t, 4> K_AUTO_X_STRIDES = {256, 64, 8, 1};
    constexpr std::array<int64_t, 4> K_AUTO_W_DIMS = {16, 4, 3, 3};
    constexpr std::array<int64_t, 4> K_AUTO_W_STRIDES = {36, 9, 3, 1};

    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_AUTO_X_DIMS)).set_stride(toVec(K_AUTO_X_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_name("W").set_data_type(DataType::FLOAT);
    w->set_dim(toVec(K_AUTO_W_DIMS)).set_stride(toVec(K_AUTO_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_name("auto_uid_conv_op");
    convAttrs.set_pre_padding(toVec(K_CONV_PRE_PADDING));
    convAttrs.set_post_padding(toVec(K_CONV_POST_PADDING));
    convAttrs.set_stride(toVec(K_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_CONV_DILATION));
    convAttrs.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    auto y = graph->conv_fprop(x, w, convAttrs);
    y->set_output(true).set_name("Y");

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

    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr);

    // Verify tensor dims survived the round trip
    auto xUid = convNode->attributes.get_x()->get_uid();
    auto wUid = convNode->attributes.get_w()->get_uid();
    auto yUid = convNode->attributes.get_y()->get_uid();

    EXPECT_NE(xUid, wUid);
    EXPECT_NE(xUid, yUid);
    EXPECT_NE(wUid, yUid);

    EXPECT_EQ(tensorMap[xUid]->get_dim(), toVec(K_AUTO_X_DIMS));
    EXPECT_EQ(tensorMap[wUid]->get_dim(), toVec(K_AUTO_W_DIMS));
}

// Builds a conv fprop graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all conv fprop fields survive the FlatBuffer-direct path.
TEST_F(IntegrationConvFpropDescriptorLifting, ConvFpropLiftWithoutFinalization)
{
    auto originalGraph = buildConvFpropGraph();

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

    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr);

    // Verify convolution parameters
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_CONV_PRE_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_CONV_POST_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_CONV_DILATION));
    EXPECT_EQ(convNode->attributes.get_convolution_mode(), ConvolutionMode::CROSS_CORRELATION);
    EXPECT_EQ(convNode->attributes.get_name(), "conv_fprop_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_dim(), toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_stride(), toVec(K_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_X_UID]->get_name(), "X");

    ASSERT_NE(tensorMap.count(K_TENSOR_W_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_dim(), toVec(K_TENSOR_W_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_stride(), toVec(K_TENSOR_W_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_W_UID]->get_name(), "W");

    ASSERT_NE(tensorMap.count(K_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_Y_UID]->get_dim(), toVec(K_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_Y_UID]->get_stride(), toVec(K_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_Y_UID]->get_name(), "Y");
}

} // namespace
