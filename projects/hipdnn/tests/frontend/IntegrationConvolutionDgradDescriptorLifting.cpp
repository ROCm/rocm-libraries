// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_test_sdk/constants/ConvDgradConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;

namespace
{
// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationConvolutionBwdDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard ConvolutionBwd graph for round-trip testing.
    static std::shared_ptr<TestableGraphLifting> buildGraph()
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("ConvolutionBwdLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_DGRAD_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_DGRAD_TENSOR_DY_DIMS)).set_stride(toVec(K_DGRAD_TENSOR_DY_STRIDES));

        auto w = std::make_shared<TensorAttributes>();
        w->set_uid(K_DGRAD_TENSOR_W_UID).set_name("w").set_data_type(DataType::FLOAT);
        w->set_dim(toVec(K_DGRAD_TENSOR_W_DIMS)).set_stride(toVec(K_DGRAD_TENSOR_W_STRIDES));

        ConvDgradAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);
        attrs.set_pre_padding(toVec(K_DGRAD_CONV_PADDING));
        attrs.set_post_padding(toVec(K_DGRAD_CONV_PADDING));
        attrs.set_stride(toVec(K_DGRAD_CONV_STRIDE));
        attrs.set_dilation(toVec(K_DGRAD_CONV_DILATION));

        auto dx = graph->conv_dgrad(dy, w, attrs);
        dx->set_uid(K_DGRAD_TENSOR_DX_UID).set_output(true).set_name("dx");

        return graph;
    }
};

// Builds a standard ConvolutionBwd graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationConvolutionBwdDescriptorLifting, BasicConvolutionBwdRoundTrip)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_DGRAD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DY_UID]->get_uid(), K_DGRAD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DY_UID]->get_dim(), toVec(K_DGRAD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DY_UID]->get_stride(), toVec(K_DGRAD_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DY_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DY_UID]->get_name(), "dy");

    // Verify w tensor
    ASSERT_NE(tensorMap.count(K_DGRAD_TENSOR_W_UID), 0u);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_W_UID]->get_uid(), K_DGRAD_TENSOR_W_UID);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_W_UID]->get_dim(), toVec(K_DGRAD_TENSOR_W_DIMS));
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_W_UID]->get_stride(), toVec(K_DGRAD_TENSOR_W_STRIDES));
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_W_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_W_UID]->get_name(), "w");

    // Verify dx tensor
    ASSERT_NE(tensorMap.count(K_DGRAD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DX_UID]->get_uid(), K_DGRAD_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DX_UID]->get_dim(), toVec(K_DGRAD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DX_UID]->get_stride(), toVec(K_DGRAD_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DX_UID]->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DX_UID]->get_name(), "dx");

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<ConvolutionDgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a ConvolutionDgradNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_convolution_mode(), ConvolutionMode::CONVOLUTION);

    // Verify pre_padding
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_DGRAD_CONV_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_DGRAD_CONV_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_DGRAD_CONV_STRIDE));
    // Verify dilation
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_DGRAD_CONV_DILATION));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationConvolutionBwdDescriptorLifting, ConvolutionBwdTensorSharingPreserved)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ConvolutionDgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify dy tensor sharing
    EXPECT_EQ(opNode->attributes.get_dy()->get_uid(), K_DGRAD_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DY_UID].get(), opNode->attributes.get_dy().get());
    // Verify w tensor sharing
    EXPECT_EQ(opNode->attributes.get_w()->get_uid(), K_DGRAD_TENSOR_W_UID);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_W_UID].get(), opNode->attributes.get_w().get());
    // Verify dx tensor sharing
    EXPECT_EQ(opNode->attributes.get_dx()->get_uid(), K_DGRAD_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DX_UID].get(), opNode->attributes.get_dx().get());
}

// Builds a ConvolutionBwd graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationConvolutionBwdDescriptorLifting, ConvolutionBwdLiftWithoutFinalization)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

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
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_DGRAD_CONV_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_DGRAD_CONV_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_DGRAD_CONV_STRIDE));
    // Verify dilation
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_DGRAD_CONV_DILATION));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_DGRAD_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DY_UID]->get_dim(), toVec(K_DGRAD_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DY_UID]->get_stride(), toVec(K_DGRAD_TENSOR_DY_STRIDES));
    ASSERT_NE(tensorMap.count(K_DGRAD_TENSOR_W_UID), 0u);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_W_UID]->get_dim(), toVec(K_DGRAD_TENSOR_W_DIMS));
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_W_UID]->get_stride(), toVec(K_DGRAD_TENSOR_W_STRIDES));
    ASSERT_NE(tensorMap.count(K_DGRAD_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DX_UID]->get_dim(), toVec(K_DGRAD_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_DGRAD_TENSOR_DX_UID]->get_stride(), toVec(K_DGRAD_TENSOR_DX_STRIDES));
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the lifting round trip and are all distinct.
TEST_F(IntegrationConvolutionBwdDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    constexpr std::array<int64_t, 4> K_AUTO_DY_DIMS = {1, 64, 32, 32};
    constexpr std::array<int64_t, 4> K_AUTO_DY_STRIDES = {65536, 1024, 32, 1};
    constexpr std::array<int64_t, 4> K_AUTO_W_DIMS = {64, 3, 3, 3};
    constexpr std::array<int64_t, 4> K_AUTO_W_STRIDES = {27, 9, 3, 1};

    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("AutoUidDgradLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_AUTO_DY_DIMS)).set_stride(toVec(K_AUTO_DY_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_name("w").set_data_type(DataType::FLOAT);
    w->set_dim(toVec(K_AUTO_W_DIMS)).set_stride(toVec(K_AUTO_W_STRIDES));

    ConvDgradAttributes convAttrs;
    convAttrs.set_name("auto_uid_dgrad_op");
    convAttrs.set_pre_padding({1, 1});
    convAttrs.set_post_padding({1, 1});
    convAttrs.set_stride({1, 1});
    convAttrs.set_dilation({1, 1});
    convAttrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);

    auto dx = graph->conv_dgrad(dy, w, convAttrs);
    dx->set_output(true).set_name("dx");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u)
        << "Expected 3 tensors in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    // Collect all UIDs and verify they are distinct
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    EXPECT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "All auto-assigned UIDs must be distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify the node references tensors with auto-assigned UIDs
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ConvolutionDgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify tensor dims survived the round trip
    auto dyUid = opNode->attributes.get_dy()->get_uid();
    auto wUid = opNode->attributes.get_w()->get_uid();
    auto dxUid = opNode->attributes.get_dx()->get_uid();

    EXPECT_NE(dyUid, wUid);
    EXPECT_NE(dyUid, dxUid);
    EXPECT_NE(wUid, dxUid);

    EXPECT_EQ(tensorMap[dyUid]->get_dim(), toVec(K_AUTO_DY_DIMS));
    EXPECT_EQ(tensorMap[wUid]->get_dim(), toVec(K_AUTO_W_DIMS));
}

// Builds a conv dgrad graph with asymmetric padding (pre_padding={1,0},
// post_padding={0,1}), lowers, lifts, and verifies asymmetric padding
// values survive the round trip.
TEST_F(IntegrationConvolutionBwdDescriptorLifting, AsymmetricPaddingPreservedInLiftingRoundTrip)
{
    constexpr std::array<int64_t, 2> K_ASYM_PRE_PADDING = {1, 0};
    constexpr std::array<int64_t, 2> K_ASYM_POST_PADDING = {0, 1};

    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("AsymPaddingDgradLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(K_DGRAD_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_DGRAD_TENSOR_DY_DIMS)).set_stride(toVec(K_DGRAD_TENSOR_DY_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_DGRAD_TENSOR_W_UID).set_name("w").set_data_type(DataType::FLOAT);
    w->set_dim(toVec(K_DGRAD_TENSOR_W_DIMS)).set_stride(toVec(K_DGRAD_TENSOR_W_STRIDES));

    ConvDgradAttributes convAttrs;
    convAttrs.set_name("asym_dgrad_op");
    convAttrs.set_pre_padding(toVec(K_ASYM_PRE_PADDING));
    convAttrs.set_post_padding(toVec(K_ASYM_POST_PADDING));
    convAttrs.set_stride({1, 1});
    convAttrs.set_dilation({1, 1});
    convAttrs.set_convolution_mode(ConvolutionMode::CONVOLUTION);

    auto dx = graph->conv_dgrad(dy, w, convAttrs);
    dx->set_uid(K_DGRAD_TENSOR_DX_UID).set_output(true).set_name("dx");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ConvolutionDgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a ConvolutionDgradNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_ASYM_PRE_PADDING));
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_ASYM_POST_PADDING));
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_DGRAD_CONV_STRIDE));
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_DGRAD_CONV_DILATION));
}

// Exercises the JSON serialize/deserialize path with a handle (full finalization)
// for a conv backward data graph.
TEST_F(IntegrationConvolutionBwdDescriptorLifting, JsonRoundTripWithHandle)
{
    auto originalGraph = buildGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to JSON (auto-lowers internally)
    std::string jsonData;
    result = originalGraph->serialize(jsonData);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_FALSE(jsonData.empty());

    // Deserialize from JSON with handle
    auto liftedGraph = std::make_shared<TestableGraphLifting>();
    result = liftedGraph->deserialize(_handle, jsonData);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level attributes
    EXPECT_EQ(liftedGraph->get_name(), "ConvolutionBwdLiftingTestGraph");
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors (dy, w, dx)";

    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_DGRAD_TENSOR_DY_UID,
                                      "dy",
                                      toVec(K_DGRAD_TENSOR_DY_DIMS),
                                      toVec(K_DGRAD_TENSOR_DY_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_DGRAD_TENSOR_W_UID,
                                      "w",
                                      toVec(K_DGRAD_TENSOR_W_DIMS),
                                      toVec(K_DGRAD_TENSOR_W_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_DGRAD_TENSOR_DX_UID,
                                      "dx",
                                      toVec(K_DGRAD_TENSOR_DX_DIMS),
                                      toVec(K_DGRAD_TENSOR_DX_STRIDES),
                                      DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<ConvolutionDgradNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ConvolutionDgradNode";

    // Verify convolution parameters
    EXPECT_EQ(opNode->attributes.get_convolution_mode(), ConvolutionMode::CONVOLUTION);
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_DGRAD_CONV_PADDING));
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_DGRAD_CONV_PADDING));
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_DGRAD_CONV_STRIDE));
    EXPECT_EQ(opNode->attributes.get_dilation(), toVec(K_DGRAD_CONV_DILATION));
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

} // namespace
