// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pooling_fwd_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

#include <hipdnn_test_sdk/constants/PoolingFwdConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants::integration;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using PoolingModeSdk = hipdnn_flatbuffers_sdk::data_objects::PoolingMode;
using PaddingModeSdk = hipdnn_flatbuffers_sdk::data_objects::PaddingMode;

namespace
{

// Exposes protected Graph methods for testing
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::get_raw_graph_descriptor;
};

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationPoolingFwdDescriptorLowering : public ::testing::Test
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

    /// Builds and lowers a graph, returning the deserialized GraphT.
    /// Callers set up attrs before calling; this creates tensors, calls the
    /// graph method, validates, lowers, serializes, and deserializes.
    hipdnn_flatbuffers_sdk::data_objects::GraphT buildAndDeserialize(PoolingFwdAttributes& attrs)
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("PoolingFwdIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_POOL_FWD_TENSOR_X_UID).set_name("x").set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_POOL_FWD_TENSOR_X_DIMS)).set_stride(toVec(K_POOL_FWD_TENSOR_X_STRIDES));

        auto y = graph->pooling_fwd(x, attrs);
        y->set_uid(K_POOL_FWD_TENSOR_Y_UID).set_output(true).set_name("y");

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

    hipdnnHandle_t _handle = nullptr;
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationPoolingFwdDescriptorLowering, PoolingFwdLoweringRoundTrip)
{
    PoolingFwdAttributes attrs;
    attrs.set_name("test_op");
    attrs.set_pooling_mode(PoolingMode::MAX);
    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

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
    ASSERT_NE(tensorMap.count(K_POOL_FWD_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->name, "x");
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->dims, toVec(K_POOL_FWD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->strides, toVec(K_POOL_FWD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_X_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_POOL_FWD_TENSOR_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->name, "y");
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->dims, toVec(K_POOL_FWD_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->strides, toVec(K_POOL_FWD_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_POOL_FWD_TENSOR_Y_UID]->data_type, DataTypeSdk::FLOAT);

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::PoolingFwdAttributes);

    auto* opNode = node->attributes.AsPoolingFwdAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->x_tensor_uid, K_POOL_FWD_TENSOR_X_UID);
    EXPECT_EQ(opNode->y_tensor_uid, K_POOL_FWD_TENSOR_Y_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    // Verify mode
    EXPECT_EQ(opNode->pooling_mode, PoolingModeSdk::MAX_POOLING);

    // Verify pre_padding
    {
        const std::vector<int64_t> expectedPrePadding = {1, 1};
        EXPECT_EQ(opNode->pre_padding, expectedPrePadding);
    }
    // Verify post_padding
    {
        const std::vector<int64_t> expectedPostPadding = {1, 1};
        EXPECT_EQ(opNode->post_padding, expectedPostPadding);
    }
    // Verify stride
    {
        const std::vector<int64_t> expectedStride = {2, 2};
        EXPECT_EQ(opNode->stride, expectedStride);
    }
    // Verify window
    {
        const std::vector<int64_t> expectedWindow = {3, 3};
        EXPECT_EQ(opNode->window, expectedWindow);
    }
}

// Verifies that the optional generate_index attribute survives lowering round-trip.
TEST_F(IntegrationPoolingFwdDescriptorLowering, GenerateIndexPreservedInRoundTrip)
{
    PoolingFwdAttributes attrs;
    attrs.set_name("test_generate_index");
    attrs.set_pooling_mode(PoolingMode::MAX);
    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});
    attrs.set_generate_index(true);

    auto graphT = buildAndDeserialize(attrs);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsPoolingFwdAttributes();
    ASSERT_NE(opNode, nullptr);

    ASSERT_TRUE(opNode->generate_index.has_value());
    EXPECT_EQ(opNode->generate_index.value(), true);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationPoolingFwdDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidPoolingFwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_POOL_FWD_TENSOR_X_DIMS)).set_stride(toVec(K_POOL_FWD_TENSOR_X_STRIDES));

    PoolingFwdAttributes attrs;
    attrs.set_pooling_mode(PoolingMode::MAX);
    attrs.set_pre_padding({1, 1});
    attrs.set_post_padding({1, 1});
    attrs.set_stride({2, 2});
    attrs.set_window({3, 3});

    auto y = graph->pooling_fwd(x, attrs);
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

    // The pooling operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* opNode = graphT.nodes[0]->attributes.AsPoolingFwdAttributes();
    ASSERT_NE(opNode, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(opNode->x_tensor_uid) > 0)
        << "X tensor UID " << opNode->x_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(opNode->y_tensor_uid) > 0)
        << "Y tensor UID " << opNode->y_tensor_uid << " not found in graph tensors";

    // Both tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids = {opNode->x_tensor_uid, opNode->y_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 2u) << "PoolingFwd node tensor UIDs are not distinct";
}

} // namespace
