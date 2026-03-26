// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/data_objects/pooling_fwd_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_data_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_data_sdk::data_objects::NodeAttributes;
using PoolingModeSdk = hipdnn_data_sdk::data_objects::PoolingMode;

namespace
{

// Exposes protected Graph methods for testing
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::get_raw_graph_descriptor;
};

// -- Test constants --
constexpr int64_t K_TEST_X_UID = 40;
constexpr int64_t K_TEST_Y_UID = 41;

constexpr std::array<int64_t, 4> K_TEST_DIMS = {1, 3, 32, 32};
constexpr std::array<int64_t, 4> K_TEST_STRIDES = {3072, 1024, 32, 1};

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
    hipdnn_data_sdk::data_objects::GraphT buildAndDeserialize(
        PoolingFwdAttributes& attrs)
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("PoolingFwdIntegrationTest")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_TEST_X_UID)
            .set_name("x")
            .set_data_type(DataType::FLOAT);
        x->set_dim(toVec(K_TEST_DIMS)).set_stride(toVec(K_TEST_STRIDES));

        auto y = graph->pooling_fwd(
            x,
            attrs);
        y->set_uid(K_TEST_Y_UID)
            .set_output(true)
            .set_name("y");

        auto result = graph->validate();
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph->build_operation_graph_via_descriptors(_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        auto rawDesc = graph->get_raw_graph_descriptor();
        EXPECT_NE(rawDesc, nullptr);

        size_t serializedSize = 0;
        EXPECT_EQ(
            hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
            HIPDNN_STATUS_SUCCESS);

        std::vector<uint8_t> serializedData(serializedSize);
        EXPECT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                      rawDesc, serializedSize, &serializedSize, serializedData.data()),
                  HIPDNN_STATUS_SUCCESS);

        hipdnn_data_sdk::data_objects::GraphT graphT;
        hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);
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
    attrs.set_window_size({3, 3});

    auto graphT = buildAndDeserialize(attrs);

    // Verify tensors
    ASSERT_GE(graphT.tensors.size(), 2u);

    // Verify tensor attributes
    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }
    ASSERT_NE(tensorMap.count(K_TEST_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->dims, toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->strides, toVec(K_TEST_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_X_UID]->data_type, DataTypeSdk::FLOAT);
    ASSERT_NE(tensorMap.count(K_TEST_Y_UID), 0u);
    EXPECT_EQ(tensorMap[K_TEST_Y_UID]->dims, toVec(K_TEST_DIMS));
    EXPECT_EQ(tensorMap[K_TEST_Y_UID]->strides, toVec(K_TEST_STRIDES));
    EXPECT_EQ(tensorMap[K_TEST_Y_UID]->data_type, DataTypeSdk::FLOAT);

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::PoolingFwdAttributes);

    auto* opNode = node->attributes.AsPoolingFwdAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->x_tensor_uid, K_TEST_X_UID);
    EXPECT_EQ(opNode->y_tensor_uid, K_TEST_Y_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "test_op");

    // Verify mode
    EXPECT_EQ(opNode->pooling_mode, PoolingModeSdk::MAX);

    // Verify pre_padding
    {
        std::vector<int64_t> const expectedPrePadding = {1, 1};
        EXPECT_EQ(opNode->pre_padding, expectedPrePadding);
    }
    // Verify post_padding
    {
        std::vector<int64_t> const expectedPostPadding = {1, 1};
        EXPECT_EQ(opNode->post_padding, expectedPostPadding);
    }
    // Verify stride
    {
        std::vector<int64_t> const expectedStride = {2, 2};
        EXPECT_EQ(opNode->stride, expectedStride);
    }
    // Verify window_size
    {
        std::vector<int64_t> const expectedWindowSize = {3, 3};
        EXPECT_EQ(opNode->window_size, expectedWindowSize);
    }
}

} // namespace
