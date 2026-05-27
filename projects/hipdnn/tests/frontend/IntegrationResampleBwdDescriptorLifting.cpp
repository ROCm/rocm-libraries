// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <set>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/ResampleBwdNode.hpp>
#include <hipdnn_test_sdk/constants/ResampleBwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;
using hipdnn_tests::toVec;

namespace
{

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationResampleBwdDescriptorLifting : public IntegrationTestFixture
{
protected:
    /// Builds a standard ResampleBwd graph for round-trip testing.
    static std::shared_ptr<TestableGraphLifting> buildGraph()
    {
        auto graph = std::make_shared<TestableGraphLifting>();
        graph->set_name("ResampleBwdLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

        ResampleBwdAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_resample_mode(ResampleMode::MAXPOOL);
        attrs.set_pre_padding(toVec(K_PRE_PADDING));
        attrs.set_post_padding(toVec(K_POST_PADDING));
        attrs.set_stride(toVec(K_STRIDE));
        attrs.set_window(toVec(K_WINDOW));

        auto dx = graph->resample_bwd(dy, attrs);
        dx->set_uid(K_TENSOR_DX_UID).set_output(true).set_name("dx");
        dx->set_dim(toVec(K_TENSOR_DX_DIMS)).set_stride(toVec(K_TENSOR_DX_STRIDES));

        return graph;
    }
};

// Builds a standard ResampleBwd graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationResampleBwdDescriptorLifting, BasicResampleBwdRoundTrip)
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
    ASSERT_EQ(tensorMap.size(), 2u);

    // Verify dy tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_uid(), K_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_dim(), toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_stride(), toVec(K_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_data_type(), DataType::FLOAT);

    // Verify dx tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_uid(), K_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_dim(), toVec(K_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_stride(), toVec(K_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_data_type(), DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    auto* opNode = dynamic_cast<ResampleBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr)
        << "Expected a ResampleBwdNode"; // NOLINT(readability-implicit-bool-conversion)

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_resample_mode(), ResampleMode::MAXPOOL);

    // Verify pre_padding
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_PRE_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_POST_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_STRIDE));
    // Verify window
    EXPECT_EQ(opNode->attributes.get_window(), toVec(K_WINDOW));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationResampleBwdDescriptorLifting, ResampleBwdTensorSharingPreserved)
{
    auto originalGraph = buildGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ResampleBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify dy tensor sharing
    EXPECT_EQ(opNode->attributes.get_dy()->get_uid(), K_TENSOR_DY_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID].get(), opNode->attributes.get_dy().get());
    // Verify dx tensor sharing
    EXPECT_EQ(opNode->attributes.get_dx()->get_uid(), K_TENSOR_DX_UID);
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID].get(), opNode->attributes.get_dx().get());
}

// Builds a ResampleBwd graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the backend C API serialization path.
TEST_F(IntegrationResampleBwdDescriptorLifting, ResampleBwdLiftWithoutFinalization)
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

    auto* opNode = dynamic_cast<ResampleBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_resample_mode(), ResampleMode::MAXPOOL);

    // Verify pre_padding
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_PRE_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_POST_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_STRIDE));
    // Verify window
    EXPECT_EQ(opNode->attributes.get_window(), toVec(K_WINDOW));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);

    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_dim(), toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->get_stride(), toVec(K_TENSOR_DY_STRIDES));
    ASSERT_NE(tensorMap.count(K_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_dim(), toVec(K_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->get_stride(), toVec(K_TENSOR_DX_STRIDES));
}

// Builds a ResampleBwd graph without calling set_uid() on any tensor,
// lowers to backend, lifts, and verifies all auto-assigned UIDs are
// distinct and survive the round-trip.
TEST_F(IntegrationResampleBwdDescriptorLifting, AutoAssignedUidsPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("ResampleBwdAutoUidLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

    ResampleBwdAttributes attrs;
    attrs.set_name("test_auto_uid");
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_pre_padding(toVec(K_PRE_PADDING));
    attrs.set_post_padding(toVec(K_POST_PADDING));
    attrs.set_stride(toVec(K_STRIDE));
    attrs.set_window(toVec(K_WINDOW));

    auto dx = graph->resample_bwd(dy, attrs);
    dx->set_output(true).set_name("dx");
    dx->set_dim(toVec(K_TENSOR_DX_DIMS)).set_stride(toVec(K_TENSOR_DX_STRIDES));

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify the tensor map has the expected number of tensors
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);

    // Verify all UIDs are distinct. Auto-assignment starts from 0, so UID 0 is valid.
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& [uid, tensor] : tensorMap)
    {
        uids.push_back(uid);
    }
    std::sort(uids.begin(), uids.end());
    ASSERT_EQ(std::adjacent_find(uids.begin(), uids.end()), uids.end())
        << "Found duplicate auto-assigned UIDs"; // NOLINT(readability-implicit-bool-conversion)

    // Verify sub-node tensor UIDs are distinct via the node attributes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ResampleBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    std::set<int64_t> nodeUids;
    ASSERT_NE(opNode->attributes.get_dy(), nullptr);
    nodeUids.insert(opNode->attributes.get_dy()->get_uid());
    ASSERT_NE(opNode->attributes.get_dx(), nullptr);
    nodeUids.insert(opNode->attributes.get_dx()->get_uid());
    ASSERT_EQ(nodeUids.size(), 2u)
        << "Node tensor UIDs are not all distinct"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor dims survived the round trip
    EXPECT_EQ(opNode->attributes.get_dy()->get_dim(), toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(opNode->attributes.get_dy()->get_stride(), toVec(K_TENSOR_DY_STRIDES));
    EXPECT_EQ(opNode->attributes.get_dx()->get_dim(), toVec(K_TENSOR_DX_DIMS));
    EXPECT_EQ(opNode->attributes.get_dx()->get_stride(), toVec(K_TENSOR_DX_STRIDES));
}

TEST_F(IntegrationResampleBwdDescriptorLifting, GenerateIndexPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<hipdnn_tests::TestableGraphLifting>();
    graph->set_name("ResampleBwdGenerateIndexLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(K_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

    ResampleBwdAttributes attrs;
    attrs.set_name("test_generate_index");
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_padding_mode(PaddingMode::ZERO_PAD);
    attrs.set_pre_padding(toVec(K_PRE_PADDING));
    attrs.set_post_padding(toVec(K_POST_PADDING));
    attrs.set_stride(toVec(K_STRIDE));
    attrs.set_window(toVec(K_WINDOW));
    attrs.set_generate_index(true);

    auto dx = graph->resample_bwd(dy, attrs);
    dx->set_uid(K_TENSOR_DX_UID).set_output(true).set_name("dx");
    dx->set_dim(toVec(K_TENSOR_DX_DIMS)).set_stride(toVec(K_TENSOR_DX_STRIDES));

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<hipdnn_tests::TestableGraphLifting>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ResampleBwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ResampleBwdNode";

    ASSERT_TRUE(opNode->attributes.get_generate_index().has_value());
    EXPECT_EQ(opNode->attributes.get_generate_index().value(), true);
}

} // namespace
