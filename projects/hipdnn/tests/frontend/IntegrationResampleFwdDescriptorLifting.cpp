// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/ResampleFwdNode.hpp>
#include <hipdnn_test_sdk/constants/ResampleFwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Lifts a frontend graph via build_operation_graph(handle), then
// reconstructs it with fromBackendDescriptor() for verification.
class IntegrationResampleFwdDescriptorLifting : public hipdnn_tests::IntegrationTestFixture
{
protected:
    /// Builds a standard ResampleFwd graph for round-trip testing.
    static std::shared_ptr<hipdnn_tests::TestableGraphLifting> buildGraph()
    {
        auto graph = std::make_shared<hipdnn_tests::TestableGraphLifting>();
        graph->set_name("ResampleFwdLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_RESAMPLE_FWD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS))
            .set_stride(toVec(K_RESAMPLE_FWD_TENSOR_X_STRIDES));

        ResampleFwdAttributes attrs;
        attrs.set_name("test_op");
        attrs.set_resample_mode(ResampleMode::MAXPOOL);
        attrs.set_padding_mode(PaddingMode::ZERO_PAD);
        attrs.set_pre_padding(toVec(K_RESAMPLE_FWD_PRE_PADDING));
        attrs.set_post_padding(toVec(K_RESAMPLE_FWD_POST_PADDING));
        attrs.set_stride(toVec(K_RESAMPLE_FWD_STRIDE));
        attrs.set_window(toVec(K_RESAMPLE_FWD_WINDOW));

        auto y = graph->resample_fwd(x, attrs);
        y->set_uid(K_RESAMPLE_FWD_TENSOR_Y_UID).set_output(true).set_name("y");

        return graph;
    }
};

// Builds a standard ResampleFwd graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and operation parameters.
TEST_F(IntegrationResampleFwdDescriptorLifting, BasicResampleFwdRoundTrip)
{
    auto originalGraph = buildGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<hipdnn_tests::TestableGraphLifting>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);

    // Verify x tensor
    ASSERT_NE(tensorMap.count(K_RESAMPLE_FWD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->get_name(), "x");
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->get_uid(), K_RESAMPLE_FWD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->get_dim(),
              toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->get_stride(),
              toVec(K_RESAMPLE_FWD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->get_data_type(), DataType::FLOAT);

    // Verify y tensor
    ASSERT_NE(tensorMap.count(K_RESAMPLE_FWD_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->get_name(), "y");
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->get_uid(), K_RESAMPLE_FWD_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->get_dim(),
              toVec(K_RESAMPLE_FWD_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->get_stride(),
              toVec(K_RESAMPLE_FWD_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->get_data_type(), DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* opNode = dynamic_cast<ResampleFwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ResampleFwdNode";

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_resample_mode(), ResampleMode::MAXPOOL);

    // Verify padding_mode
    EXPECT_EQ(opNode->attributes.get_padding_mode(), PaddingMode::ZERO_PAD);

    // Verify pre_padding
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_RESAMPLE_FWD_PRE_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_RESAMPLE_FWD_POST_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_RESAMPLE_FWD_STRIDE));
    // Verify window
    EXPECT_EQ(opNode->attributes.get_window(), toVec(K_RESAMPLE_FWD_WINDOW));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationResampleFwdDescriptorLifting, ResampleFwdTensorSharingPreserved)
{
    auto originalGraph = buildGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<hipdnn_tests::TestableGraphLifting>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto tensorMap = liftedGraph->getTensorsByUid();

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ResampleFwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify x tensor sharing
    EXPECT_EQ(opNode->attributes.get_x()->get_uid(), K_RESAMPLE_FWD_TENSOR_X_UID);
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID].get(), opNode->attributes.get_x().get());
    // Verify y tensor sharing
    EXPECT_EQ(opNode->attributes.get_y()->get_uid(), K_RESAMPLE_FWD_TENSOR_Y_UID);
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID].get(), opNode->attributes.get_y().get());
}

// Builds a ResampleFwd graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all fields survive the FlatBuffer-direct path.
TEST_F(IntegrationResampleFwdDescriptorLifting, ResampleFwdLiftWithoutFinalization)
{
    auto originalGraph = buildGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Serialize to binary via the frontend
    auto [data, binErr] = originalGraph->to_binary();
    ASSERT_FALSE(data.empty());

    // Create a backend graph descriptor from serialized bytes (no handle, no finalize)
    const detail::ScopedHipdnnBackendDescriptor graphDesc(data.data(), data.size());
    ASSERT_TRUE(graphDesc.valid()) << "Failed to create backend graph descriptor";

    // Lift into a new graph via fromBackendDescriptor
    auto liftedGraph = std::make_shared<hipdnn_tests::TestableGraphLifting>();
    result = liftedGraph->fromBackendDescriptor(graphDesc.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 operation node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* opNode = dynamic_cast<ResampleFwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify mode
    EXPECT_EQ(opNode->attributes.get_resample_mode(), ResampleMode::MAXPOOL);

    // Verify padding_mode
    EXPECT_EQ(opNode->attributes.get_padding_mode(), PaddingMode::ZERO_PAD);

    // Verify pre_padding
    EXPECT_EQ(opNode->attributes.get_pre_padding(), toVec(K_RESAMPLE_FWD_PRE_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->attributes.get_post_padding(), toVec(K_RESAMPLE_FWD_POST_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->attributes.get_stride(), toVec(K_RESAMPLE_FWD_STRIDE));
    // Verify window
    EXPECT_EQ(opNode->attributes.get_window(), toVec(K_RESAMPLE_FWD_WINDOW));

    // Verify operation name
    EXPECT_EQ(opNode->attributes.get_name(), "test_op");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);

    ASSERT_NE(tensorMap.count(K_RESAMPLE_FWD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->get_dim(),
              toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->get_stride(),
              toVec(K_RESAMPLE_FWD_TENSOR_X_STRIDES));
    ASSERT_NE(tensorMap.count(K_RESAMPLE_FWD_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->get_dim(),
              toVec(K_RESAMPLE_FWD_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->get_stride(),
              toVec(K_RESAMPLE_FWD_TENSOR_Y_STRIDES));
}

// Verifies that the optional generate_index attribute survives a lifting round-trip.
TEST_F(IntegrationResampleFwdDescriptorLifting, GenerateIndexPreservedInLiftingRoundTrip)
{
    auto graph = std::make_shared<hipdnn_tests::TestableGraphLifting>();
    graph->set_name("ResampleFwdGenerateIndexLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_RESAMPLE_FWD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RESAMPLE_FWD_TENSOR_X_STRIDES));

    ResampleFwdAttributes attrs;
    attrs.set_name("test_generate_index");
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_padding_mode(PaddingMode::ZERO_PAD);
    attrs.set_pre_padding(toVec(K_RESAMPLE_FWD_PRE_PADDING));
    attrs.set_post_padding(toVec(K_RESAMPLE_FWD_POST_PADDING));
    attrs.set_stride(toVec(K_RESAMPLE_FWD_STRIDE));
    attrs.set_window(toVec(K_RESAMPLE_FWD_WINDOW));
    attrs.set_generate_index(true);

    auto y = graph->resample_fwd(x, attrs);
    y->set_uid(K_RESAMPLE_FWD_TENSOR_Y_UID).set_output(true).set_name("y");

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

    auto* opNode = dynamic_cast<ResampleFwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr) << "Expected a ResampleFwdNode";

    ASSERT_TRUE(opNode->attributes.get_generate_index().has_value());
    EXPECT_EQ(opNode->attributes.get_generate_index().value(), true);
}

// TODO: Add IndexTensorPreservedInLiftingRoundTrip test once resample_fwd_with_index
// API is available on the Graph class. The index tensor parameter from the schema
// needs coverage in the lifting round-trip.

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the round trip and are all distinct.
TEST_F(IntegrationResampleFwdDescriptorLifting, ResampleFwdAutoAssignedUidsPreserved)
{
    auto graph = std::make_shared<hipdnn_tests::TestableGraphLifting>();
    graph->set_name("AutoUidResampleFwdLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RESAMPLE_FWD_TENSOR_X_STRIDES));

    ResampleFwdAttributes attrs;
    attrs.set_name("auto_uid_pool_op");
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_padding_mode(PaddingMode::ZERO_PAD);
    attrs.set_pre_padding(toVec(K_RESAMPLE_FWD_PRE_PADDING));
    attrs.set_post_padding(toVec(K_RESAMPLE_FWD_POST_PADDING));
    attrs.set_stride(toVec(K_RESAMPLE_FWD_STRIDE));
    attrs.set_window(toVec(K_RESAMPLE_FWD_WINDOW));

    auto y = graph->resample_fwd(x, attrs);
    y->set_output(true).set_name("y");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<hipdnn_tests::TestableGraphLifting>();
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

    auto* opNode = dynamic_cast<ResampleFwdNode*>(subNodes[0].get());
    ASSERT_NE(opNode, nullptr);

    // Verify tensor UIDs are distinct
    auto xUid = opNode->attributes.get_x()->get_uid();
    auto yUid = opNode->attributes.get_y()->get_uid();

    EXPECT_NE(xUid, yUid);

    // Verify tensor dims survived the round trip
    EXPECT_EQ(tensorMap[xUid]->get_dim(), toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS));
}

} // namespace
