// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_fwd_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/ResampleFwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using ResampleModeSdk = hipdnn_flatbuffers_sdk::data_objects::ResampleMode;
using PaddingModeSdk = hipdnn_flatbuffers_sdk::data_objects::PaddingMode;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationResampleFwdDescriptorLowering : public hipdnn_tests::IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(ResampleFwdAttributes& attrs)
    {
        auto graph = std::make_shared<hipdnn_tests::TestableGraphLowering>();
        graph->set_name("ResampleFwdIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_RESAMPLE_FWD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS))
            .set_stride(toVec(K_RESAMPLE_FWD_TENSOR_X_STRIDES));

        auto y = graph->resample_fwd(x, attrs);
        y->set_uid(K_RESAMPLE_FWD_TENSOR_Y_UID).set_output(true).set_name("y");

        auto result = graph->validate();
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph_via_descriptors(_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        auto rawDesc = graph->get_raw_graph_descriptor();
        EXPECT_NE(rawDesc, nullptr);

        size_t serializedSize = 0;
        EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
                  HIPDNN_STATUS_SUCCESS);

        std::vector<uint8_t> serializedData(serializedSize);
        EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                      rawDesc, serializedSize, &serializedSize, serializedData.data()),
                  HIPDNN_STATUS_SUCCESS);

        hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;
        hipdnn_flatbuffers_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);
        return graphT;
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationResampleFwdDescriptorLowering, ResampleFwdLoweringRoundTrip)
{
    ResampleFwdAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_padding_mode(PaddingMode::ZERO_PAD);
    attrs.set_pre_padding(toVec(K_RESAMPLE_FWD_PRE_PADDING));
    attrs.set_post_padding(toVec(K_RESAMPLE_FWD_POST_PADDING));
    attrs.set_stride(toVec(K_RESAMPLE_FWD_STRIDE));
    attrs.set_window(toVec(K_RESAMPLE_FWD_WINDOW));

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_GE(graphT.tensors.size(), 2u);

    // Verify tensor attributes
    std::unordered_map<int64_t, const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT*>
        tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }
    ASSERT_NE(tensorMap.count(K_RESAMPLE_FWD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->name, "x");
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->dims, toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->strides,
              toVec(K_RESAMPLE_FWD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_X_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_RESAMPLE_FWD_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->name, "y");
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->dims, toVec(K_RESAMPLE_FWD_TENSOR_Y_DIMS));
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->strides,
              toVec(K_RESAMPLE_FWD_TENSOR_Y_STRIDES));
    EXPECT_EQ(tensorMap[K_RESAMPLE_FWD_TENSOR_Y_UID]->data_type, DataTypeSdk::FLOAT);

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::ResampleFwdAttributes);

    auto* opNode = node->attributes.AsResampleFwdAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->x_tensor_uid, K_RESAMPLE_FWD_TENSOR_X_UID);
    EXPECT_EQ(opNode->y_tensor_uid, K_RESAMPLE_FWD_TENSOR_Y_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    // Verify mode
    EXPECT_EQ(opNode->resample_mode, ResampleModeSdk::MAXPOOL);

    // Verify padding_mode
    EXPECT_EQ(opNode->padding_mode, PaddingModeSdk::ZERO_PAD);

    // Verify pre_padding
    EXPECT_EQ(opNode->pre_padding, toVec(K_RESAMPLE_FWD_PRE_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->post_padding, toVec(K_RESAMPLE_FWD_POST_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->stride, toVec(K_RESAMPLE_FWD_STRIDE));
    // Verify window
    EXPECT_EQ(opNode->window, toVec(K_RESAMPLE_FWD_WINDOW));
}

// Verifies that the optional generate_index attribute survives lowering round-trip.
TEST_F(IntegrationResampleFwdDescriptorLowering, GenerateIndexPreservedInRoundTrip)
{
    ResampleFwdAttributes attrs;
    attrs.set_name("test_generate_index");
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_padding_mode(PaddingMode::ZERO_PAD);
    attrs.set_pre_padding(toVec(K_RESAMPLE_FWD_PRE_PADDING));
    attrs.set_post_padding(toVec(K_RESAMPLE_FWD_POST_PADDING));
    attrs.set_stride(toVec(K_RESAMPLE_FWD_STRIDE));
    attrs.set_window(toVec(K_RESAMPLE_FWD_WINDOW));
    attrs.set_generate_index(true);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsResampleFwdAttributes();
    ASSERT_NE(opNode, nullptr);

    ASSERT_TRUE(opNode->generate_index.has_value());
    EXPECT_EQ(opNode->generate_index.value(), true);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationResampleFwdDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<hipdnn_tests::TestableGraphLowering>();
    graph->set_name("AutoUidResampleFwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RESAMPLE_FWD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RESAMPLE_FWD_TENSOR_X_STRIDES));

    ResampleFwdAttributes attrs;
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_padding_mode(PaddingMode::ZERO_PAD);
    attrs.set_pre_padding(toVec(K_RESAMPLE_FWD_PRE_PADDING));
    attrs.set_post_padding(toVec(K_RESAMPLE_FWD_POST_PADDING));
    attrs.set_stride(toVec(K_RESAMPLE_FWD_STRIDE));
    attrs.set_window(toVec(K_RESAMPLE_FWD_WINDOW));

    auto y = graph->resample_fwd(x, attrs);
    y->set_output(true);

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

    hipdnn_flatbuffers_sdk::data_objects::GraphT graphT;
    hipdnn_flatbuffers_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    // All tensors should have been auto-assigned unique UIDs
    // (auto-assignment starts from 0, so UID 0 is valid)
    ASSERT_EQ(graphT.tensors.size(), 2u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 2u) << "Tensor UIDs are not unique";

    // The resample operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsResampleFwdAttributes();
    ASSERT_NE(opNode, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(opNode->x_tensor_uid) > 0)
        << "X tensor UID " << opNode->x_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(opNode->y_tensor_uid) > 0)
        << "Y tensor UID " << opNode->y_tensor_uid << " not found in graph tensors";

    // Both tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids = {opNode->x_tensor_uid, opNode->y_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 2u) << "ResampleFwd node tensor UIDs are not distinct";
}

} // namespace
