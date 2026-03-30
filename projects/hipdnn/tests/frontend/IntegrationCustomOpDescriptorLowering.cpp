// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/custom_op_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/CustomOpConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using DataTypeSdk = hipdnn_data_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_data_sdk::data_objects::NodeAttributes;

namespace
{

class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::get_raw_graph_descriptor;
};

class IntegrationCustomOpDescriptorLowering : public ::testing::Test
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

TEST_F(IntegrationCustomOpDescriptorLowering, CustomOpGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestCustomOpGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // Create input tensors
    auto input0 = std::make_shared<TensorAttributes>();
    input0->set_uid(K_CUSTOM_OP_INPUT_UID_0).set_name("input0").set_data_type(DataType::FLOAT);
    input0->set_dim({2, 3}).set_stride({3, 1});

    auto input1 = std::make_shared<TensorAttributes>();
    input1->set_uid(K_CUSTOM_OP_INPUT_UID_1).set_name("input1").set_data_type(DataType::FLOAT);
    input1->set_dim({2, 3}).set_stride({3, 1});

    // Build CustomOp attributes
    CustomOpAttributes attrs;
    attrs.set_custom_op_id(K_CUSTOM_OP_ID)
        .set_data(K_CUSTOM_OP_OPAQUE_DATA)
        .set_name("test_custom_op");

    auto outputs = graph->custom_op({input0, input1}, 1, attrs);
    ASSERT_EQ(outputs.size(), 1);
    outputs[0]->set_uid(K_CUSTOM_OP_OUTPUT_UID_0).set_output(true).set_name("output0");
    outputs[0]->set_dim({2, 3}).set_stride({3, 1}).set_data_type(DataType::FLOAT);

    // Validate and lower
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve serialized graph
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

    // Deserialize into GraphT
    auto graphFb = hipdnn_data_sdk::data_objects::GetGraph(serializedData.data());
    ASSERT_NE(graphFb, nullptr);
    hipdnn_data_sdk::data_objects::GraphT graphT;
    graphFb->UnPackTo(&graphT);

    // Verify graph-level attributes
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // Verify tensors
    ASSERT_EQ(graphT.tensors.size(), 3u);

    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_INPUT_UID_0), 0u);
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_0]->name, "input0");
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_0]->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_0]->dims, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_0]->strides, (std::vector<int64_t>{3, 1}));

    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_INPUT_UID_1), 0u);
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_1]->name, "input1");
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_1]->dims, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_1]->strides, (std::vector<int64_t>{3, 1}));

    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_OUTPUT_UID_0), 0u);
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_OUTPUT_UID_0]->name, "output0");
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_OUTPUT_UID_0]->dims, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_OUTPUT_UID_0]->strides, (std::vector<int64_t>{3, 1}));

    // Verify custom op operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->name, "test_custom_op");
    EXPECT_EQ(node->attributes.type, NodeAttrType::CustomOpAttributes);

    auto* customOp = node->attributes.AsCustomOpAttributes();
    ASSERT_NE(customOp, nullptr);

    EXPECT_EQ(customOp->custom_op_id, K_CUSTOM_OP_ID);
    ASSERT_EQ(customOp->input_tensor_uids.size(), 2u);
    EXPECT_EQ(customOp->input_tensor_uids[0], K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_EQ(customOp->input_tensor_uids[1], K_CUSTOM_OP_INPUT_UID_1);
    ASSERT_EQ(customOp->output_tensor_uids.size(), 1u);
    EXPECT_EQ(customOp->output_tensor_uids[0], K_CUSTOM_OP_OUTPUT_UID_0);

    EXPECT_EQ(customOp->data, K_CUSTOM_OP_OPAQUE_DATA);
}

// Verifies minimal custom op with 1 input, 1 output, and no opaque data payload.
TEST_F(IntegrationCustomOpDescriptorLowering, MinimalCustomOpRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("MinimalCustomOpGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto input0 = std::make_shared<TensorAttributes>();
    input0->set_uid(K_CUSTOM_OP_INPUT_UID_0).set_name("input0").set_data_type(DataType::FLOAT);
    input0->set_dim({2, 3}).set_stride({3, 1});

    CustomOpAttributes attrs;
    attrs.set_custom_op_id("test.minimal");

    auto outputs = graph->custom_op({input0}, 1, attrs);
    ASSERT_EQ(outputs.size(), 1);
    outputs[0]->set_uid(K_CUSTOM_OP_OUTPUT_UID_0).set_output(true).set_name("output0");
    outputs[0]->set_dim({2, 3}).set_stride({3, 1}).set_data_type(DataType::FLOAT);

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve and deserialize
    auto rawDesc = graph->get_raw_graph_descriptor();
    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnn_data_sdk::data_objects::GraphT graphT;
    hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    ASSERT_EQ(graphT.tensors.size(), 2u);
    ASSERT_EQ(graphT.nodes.size(), 1u);

    auto* customOp = graphT.nodes[0]->attributes.AsCustomOpAttributes();
    ASSERT_NE(customOp, nullptr);

    EXPECT_EQ(customOp->custom_op_id, "test.minimal");
    ASSERT_EQ(customOp->input_tensor_uids.size(), 1u);
    ASSERT_EQ(customOp->output_tensor_uids.size(), 1u);
    EXPECT_TRUE(customOp->data.empty());
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationCustomOpDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidCustomOpGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    // Create tensors without explicit UIDs (auto-assigned)
    auto input0 = std::make_shared<TensorAttributes>();
    input0->set_name("input0").set_data_type(DataType::FLOAT);
    input0->set_dim({2, 3}).set_stride({3, 1});

    auto input1 = std::make_shared<TensorAttributes>();
    input1->set_name("input1").set_data_type(DataType::FLOAT);
    input1->set_dim({2, 3}).set_stride({3, 1});

    CustomOpAttributes attrs;
    attrs.set_custom_op_id("test.op").set_data({0xDE, 0xAD});

    auto outputs = graph->custom_op({input0, input1}, 1, attrs);
    ASSERT_EQ(outputs.size(), 1);
    outputs[0]->set_output(true);
    outputs[0]->set_dim({2, 3}).set_stride({3, 1}).set_data_type(DataType::FLOAT);

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

    // The custom op node should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* customOp = graphT.nodes[0]->attributes.AsCustomOpAttributes();
    ASSERT_NE(customOp, nullptr);

    // Input tensor UIDs in the node should match tensors in the graph
    for(auto inputUid : customOp->input_tensor_uids)
    {
        EXPECT_TRUE(uids.count(inputUid) > 0)
            << "Input tensor UID " << inputUid << " not found in graph tensors";
    }
    for(auto outputUid : customOp->output_tensor_uids)
    {
        EXPECT_TRUE(uids.count(outputUid) > 0)
            << "Output tensor UID " << outputUid << " not found in graph tensors";
    }

    // All tensor UIDs referenced by the node should be distinct
    std::unordered_set<int64_t> nodeUids;
    nodeUids.insert(customOp->input_tensor_uids.begin(), customOp->input_tensor_uids.end());
    nodeUids.insert(customOp->output_tensor_uids.begin(), customOp->output_tensor_uids.end());
    EXPECT_EQ(nodeUids.size(), 3u) << "CustomOp node tensor UIDs are not distinct";
}

// Verifies that a custom op with zero inputs and zero outputs can be lowered
// through the descriptor path (e.g. a side-effect-only custom op).
TEST_F(IntegrationCustomOpDescriptorLowering, ZeroInputZeroOutputCustomOpRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("ZeroIOCustomOpGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    CustomOpAttributes attrs;
    attrs.set_custom_op_id("test.zero_io").set_name("zero_io_op");

    auto outputs = graph->custom_op({}, 0, attrs);
    ASSERT_EQ(outputs.size(), 0u);

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve and deserialize
    auto rawDesc = graph->get_raw_graph_descriptor();
    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnn_data_sdk::data_objects::GraphT graphT;
    hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    ASSERT_EQ(graphT.tensors.size(), 0u);
    ASSERT_EQ(graphT.nodes.size(), 1u);

    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->name, "zero_io_op");
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);

    auto* customOp = node->attributes.AsCustomOpAttributes();
    ASSERT_NE(customOp, nullptr);

    EXPECT_EQ(customOp->custom_op_id, "test.zero_io");
    EXPECT_TRUE(customOp->input_tensor_uids.empty());
    EXPECT_TRUE(customOp->output_tensor_uids.empty());
}

} // namespace
