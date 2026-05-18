// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/ResampleBwdConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using ResampleModeSdk = hipdnn_flatbuffers_sdk::data_objects::ResampleMode;
using PaddingModeSdk = hipdnn_flatbuffers_sdk::data_objects::PaddingMode;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationResampleBwdDescriptorLowering : public IntegrationTestFixture
{
protected:
    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(ResampleBwdAttributes& attrs)
    {
        auto graph = std::make_shared<TestableGraphLowering>();
        graph->set_name("ResampleBwdIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto dy = std::make_shared<TensorAttributes>();
        dy->set_uid(K_TENSOR_DY_UID).set_name("dy").set_data_type(DataType::FLOAT);
        dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

        auto dx = graph->resample_bwd(dy, attrs);
        dx->set_uid(K_TENSOR_DX_UID).set_output(true).set_name("dx");

        return lowerAndDeserialize(*graph, _handle);
    }
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationResampleBwdDescriptorLowering, ResampleBwdLoweringRoundTrip)
{
    ResampleBwdAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_pre_padding(toVec(K_PRE_PADDING));
    attrs.set_post_padding(toVec(K_POST_PADDING));
    attrs.set_stride(toVec(K_STRIDE));
    attrs.set_window(toVec(K_WINDOW));

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 2u);

    // Verify tensor attributes
    auto tensorMap = buildTensorMap(graphT);
    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->dims, toVec(K_TENSOR_DY_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->strides, toVec(K_TENSOR_DY_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_DY_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_TENSOR_DX_UID), 0u);
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->dims, toVec(K_TENSOR_DX_DIMS));
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->strides, toVec(K_TENSOR_DX_STRIDES));
    EXPECT_EQ(tensorMap[K_TENSOR_DX_UID]->data_type, DataTypeSdk::FLOAT);

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::ResampleBwdAttributes);

    auto* opNode = node->attributes.AsResampleBwdAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->dy_tensor_uid, K_TENSOR_DY_UID);
    EXPECT_EQ(opNode->dx_tensor_uid, K_TENSOR_DX_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    // Verify mode
    EXPECT_EQ(opNode->resample_mode, ResampleModeSdk::MAXPOOL);

    // Verify pre_padding
    EXPECT_EQ(opNode->pre_padding, toVec(K_PRE_PADDING));
    // Verify post_padding
    EXPECT_EQ(opNode->post_padding, toVec(K_POST_PADDING));
    // Verify stride
    EXPECT_EQ(opNode->stride, toVec(K_STRIDE));
    // Verify window
    EXPECT_EQ(opNode->window, toVec(K_WINDOW));
}

TEST_F(IntegrationResampleBwdDescriptorLowering, GenerateIndexPreservedInRoundTrip)
{
    ResampleBwdAttributes attrs;
    attrs.set_name("test_generate_index");
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_padding_mode(PaddingMode::ZERO_PAD);
    attrs.set_pre_padding(toVec(K_PRE_PADDING));
    attrs.set_post_padding(toVec(K_POST_PADDING));
    attrs.set_stride(toVec(K_STRIDE));
    attrs.set_window(toVec(K_WINDOW));
    attrs.set_generate_index(true);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsResampleBwdAttributes();
    ASSERT_NE(opNode, nullptr);

    ASSERT_TRUE(opNode->generate_index.has_value());
    EXPECT_EQ(opNode->generate_index.value(), true);
}

TEST_F(IntegrationResampleBwdDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<hipdnn_tests::TestableGraphLowering>();
    graph->set_name("AutoUidResampleBwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("dy").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_TENSOR_DY_DIMS)).set_stride(toVec(K_TENSOR_DY_STRIDES));

    ResampleBwdAttributes attrs;
    attrs.set_resample_mode(ResampleMode::MAXPOOL);
    attrs.set_padding_mode(PaddingMode::ZERO_PAD);
    attrs.set_pre_padding(toVec(K_PRE_PADDING));
    attrs.set_post_padding(toVec(K_POST_PADDING));
    attrs.set_stride(toVec(K_STRIDE));
    attrs.set_window(toVec(K_WINDOW));

    auto dx = graph->resample_bwd(dy, attrs);
    dx->set_output(true);

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

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

    ASSERT_EQ(graphT.tensors.size(), 2u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 2u) << "Tensor UIDs are not unique";

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsResampleBwdAttributes();
    ASSERT_NE(opNode, nullptr);

    EXPECT_TRUE(uids.count(opNode->dy_tensor_uid) > 0)
        << "DY tensor UID " << opNode->dy_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(opNode->dx_tensor_uid) > 0)
        << "DX tensor UID " << opNode->dx_tensor_uid << " not found in graph tensors";

    const std::unordered_set<int64_t> nodeUids = {opNode->dy_tensor_uid, opNode->dx_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 2u) << "ResampleBwd node tensor UIDs are not distinct";
}

} // namespace
