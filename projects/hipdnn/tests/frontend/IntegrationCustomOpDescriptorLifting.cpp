// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_set>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_frontend/node/CustomOpNode.hpp>
#include <hipdnn_test_sdk/constants/CustomOpConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;

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

class IntegrationCustomOpDescriptorLifting : public ::testing::Test
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

    // Builds a zero-input custom op graph for round-trip testing
    static std::shared_ptr<TestableGraph> buildZeroInputCustomOpGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("ZeroInputCustomOpLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        CustomOpAttributes attrs;
        attrs.set_name("zero_input_custom_op")
            .set_custom_op_id(K_CUSTOM_OP_ID)
            .set_data(K_CUSTOM_OP_OPAQUE_DATA);

        auto outputs = graph->custom_op({}, 1, attrs);
        EXPECT_EQ(outputs.size(), 1u);
        outputs[0]
            ->set_uid(K_CUSTOM_OP_OUTPUT_UID_0)
            .set_output(true)
            .set_name("output0")
            .set_dim({2, 3})
            .set_stride({3, 1})
            .set_data_type(DataType::FLOAT);

        return graph;
    }

    // Builds a zero-output custom op graph for round-trip testing
    static std::shared_ptr<TestableGraph> buildZeroOutputCustomOpGraph()
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("ZeroOutputCustomOpLiftingTestGraph")
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_io_data_type(DataType::FLOAT);

        auto input0 = std::make_shared<TensorAttributes>();
        input0->set_uid(K_CUSTOM_OP_INPUT_UID_0).set_name("input0").set_data_type(DataType::FLOAT);
        input0->set_dim({2, 3}).set_stride({3, 1});

        auto input1 = std::make_shared<TensorAttributes>();
        input1->set_uid(K_CUSTOM_OP_INPUT_UID_1).set_name("input1").set_data_type(DataType::FLOAT);
        input1->set_dim({2, 3}).set_stride({3, 1});

        CustomOpAttributes attrs;
        attrs.set_name("zero_output_custom_op")
            .set_custom_op_id(K_CUSTOM_OP_ID)
            .set_data(K_CUSTOM_OP_OPAQUE_DATA);

        auto outputs = graph->custom_op({input0, input1}, 0, attrs);
        EXPECT_EQ(outputs.size(), 0u);

        return graph;
    }

    // Builds a standard custom op graph for round-trip testing
    static std::shared_ptr<TestableGraph> buildCustomOpGraph(DataType computeType = DataType::FLOAT,
                                                             DataType intermediateType
                                                             = DataType::FLOAT,
                                                             DataType ioType = DataType::FLOAT)
    {
        auto graph = std::make_shared<TestableGraph>();
        graph->set_name("CustomOpLiftingTestGraph")
            .set_compute_data_type(computeType)
            .set_intermediate_data_type(intermediateType)
            .set_io_data_type(ioType);

        auto input0 = std::make_shared<TensorAttributes>();
        input0->set_uid(K_CUSTOM_OP_INPUT_UID_0).set_name("input0").set_data_type(DataType::FLOAT);
        input0->set_dim({2, 3}).set_stride({3, 1});

        auto input1 = std::make_shared<TensorAttributes>();
        input1->set_uid(K_CUSTOM_OP_INPUT_UID_1).set_name("input1").set_data_type(DataType::FLOAT);
        input1->set_dim({2, 3}).set_stride({3, 1});

        CustomOpAttributes attrs;
        attrs.set_name("custom_op_test")
            .set_custom_op_id(K_CUSTOM_OP_ID)
            .set_data(K_CUSTOM_OP_OPAQUE_DATA);

        auto outputs = graph->custom_op({input0, input1}, 1, attrs);
        EXPECT_EQ(outputs.size(), 1u);
        outputs[0]
            ->set_uid(K_CUSTOM_OP_OUTPUT_UID_0)
            .set_output(true)
            .set_name("output0")
            .set_dim({2, 3})
            .set_stride({3, 1})
            .set_data_type(DataType::FLOAT);

        return graph;
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds a standard custom op graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and performs comprehensive field-by-field
// validation of graph data types, tensor attributes, and custom op parameters.
TEST_F(IntegrationCustomOpDescriptorLifting, BasicCustomOpRoundTrip)
{
    auto originalGraph = buildCustomOpGraph();

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
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors in lifted graph";

    // Verify input0 tensor
    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_INPUT_UID_0), 0u);
    auto liftedInput0 = tensorMap[K_CUSTOM_OP_INPUT_UID_0];
    EXPECT_EQ(liftedInput0->get_uid(), K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_EQ(liftedInput0->get_name(), "input0");
    EXPECT_EQ(liftedInput0->get_dim(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(liftedInput0->get_stride(), (std::vector<int64_t>{3, 1}));
    EXPECT_EQ(liftedInput0->get_data_type(), DataType::FLOAT);

    // Verify input1 tensor
    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_INPUT_UID_1), 0u);
    auto liftedInput1 = tensorMap[K_CUSTOM_OP_INPUT_UID_1];
    EXPECT_EQ(liftedInput1->get_uid(), K_CUSTOM_OP_INPUT_UID_1);
    EXPECT_EQ(liftedInput1->get_name(), "input1");
    EXPECT_EQ(liftedInput1->get_dim(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(liftedInput1->get_stride(), (std::vector<int64_t>{3, 1}));

    // Verify output0 tensor
    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_OUTPUT_UID_0), 0u);
    auto liftedOutput0 = tensorMap[K_CUSTOM_OP_OUTPUT_UID_0];
    EXPECT_EQ(liftedOutput0->get_uid(), K_CUSTOM_OP_OUTPUT_UID_0);
    EXPECT_EQ(liftedOutput0->get_name(), "output0");
    EXPECT_EQ(liftedOutput0->get_dim(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(liftedOutput0->get_stride(), (std::vector<int64_t>{3, 1}));

    // Verify 1 sub-node of the correct type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* customOpNode = dynamic_cast<CustomOpNode*>(subNodes[0].get());
    ASSERT_NE(customOpNode, nullptr) << "Expected a CustomOpNode";

    // Verify custom op parameters
    EXPECT_EQ(customOpNode->attributes.get_custom_op_id(), K_CUSTOM_OP_ID);
    EXPECT_EQ(customOpNode->attributes.get_data(), K_CUSTOM_OP_OPAQUE_DATA);
    EXPECT_EQ(customOpNode->attributes.get_name(), "custom_op_test");
    EXPECT_EQ(customOpNode->attributes.get_compute_data_type(), DataType::FLOAT);

    // Verify tensor references
    ASSERT_EQ(customOpNode->attributes.get_inputs().size(), 2u);
    EXPECT_EQ(customOpNode->attributes.get_inputs()[0]->get_uid(), K_CUSTOM_OP_INPUT_UID_0);
    EXPECT_EQ(customOpNode->attributes.get_inputs()[1]->get_uid(), K_CUSTOM_OP_INPUT_UID_1);
    ASSERT_EQ(customOpNode->attributes.get_outputs().size(), 1u);
    EXPECT_EQ(customOpNode->attributes.get_outputs()[0]->get_uid(), K_CUSTOM_OP_OUTPUT_UID_0);
}

// After lifting, verifies tensor objects in the node attributes are the same
// shared_ptr instances as in the tensor map (pointer equality).
TEST_F(IntegrationCustomOpDescriptorLifting, CustomOpTensorSharingPreserved)
{
    auto originalGraph = buildCustomOpGraph();

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

    auto* customOpNode = dynamic_cast<CustomOpNode*>(subNodes[0].get());
    ASSERT_NE(customOpNode, nullptr);

    // Verify pointer equality: tensor map and node attributes share the same objects
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_0].get(),
              customOpNode->attributes.get_inputs()[0].get());
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_1].get(),
              customOpNode->attributes.get_inputs()[1].get());
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_OUTPUT_UID_0].get(),
              customOpNode->attributes.get_outputs()[0].get());
}

// Creates tensors without explicit set_uid(), verifies that auto-assigned UIDs
// survive the round trip and are all distinct.
TEST_F(IntegrationCustomOpDescriptorLifting, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidCustomOpLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Create tensors without explicit UIDs (auto-assigned)
    auto input0 = std::make_shared<TensorAttributes>();
    input0->set_name("input0").set_data_type(DataType::FLOAT);
    input0->set_dim({2, 3}).set_stride({3, 1});

    auto input1 = std::make_shared<TensorAttributes>();
    input1->set_name("input1").set_data_type(DataType::FLOAT);
    input1->set_dim({2, 3}).set_stride({3, 1});

    CustomOpAttributes attrs;
    attrs.set_name("auto_uid_custom_op")
        .set_custom_op_id(K_CUSTOM_OP_ID)
        .set_data(K_CUSTOM_OP_OPAQUE_DATA);

    auto outputs = graph->custom_op({input0, input1}, 1, attrs);
    ASSERT_EQ(outputs.size(), 1u);
    outputs[0]->set_output(true).set_name("output0");
    outputs[0]->set_dim({2, 3}).set_stride({3, 1}).set_data_type(DataType::FLOAT);

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

    auto* customOpNode = dynamic_cast<CustomOpNode*>(subNodes[0].get());
    ASSERT_NE(customOpNode, nullptr);

    // Verify all tensor UIDs are unique across inputs and outputs
    std::unordered_set<int64_t> nodeUids;
    for(const auto& tensor : customOpNode->attributes.get_inputs())
    {
        nodeUids.insert(tensor->get_uid());
    }
    for(const auto& tensor : customOpNode->attributes.get_outputs())
    {
        nodeUids.insert(tensor->get_uid());
    }
    EXPECT_EQ(nodeUids.size(), 3u) << "CustomOp node tensor UIDs are not distinct";
}

// Builds a custom op graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// all custom op fields survive the FlatBuffer-direct path.
TEST_F(IntegrationCustomOpDescriptorLifting, CustomOpLiftWithoutFinalization)
{
    auto originalGraph = buildCustomOpGraph();

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

    auto* customOpNode = dynamic_cast<CustomOpNode*>(subNodes[0].get());
    ASSERT_NE(customOpNode, nullptr);

    // Verify custom op parameters
    EXPECT_EQ(customOpNode->attributes.get_custom_op_id(), K_CUSTOM_OP_ID);
    EXPECT_EQ(customOpNode->attributes.get_data(), K_CUSTOM_OP_OPAQUE_DATA);
    EXPECT_EQ(customOpNode->attributes.get_name(), "custom_op_test");

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_INPUT_UID_0), 0u);
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_0]->get_dim(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_0]->get_stride(), (std::vector<int64_t>{3, 1}));
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_0]->get_name(), "input0");

    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_INPUT_UID_1), 0u);
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_1]->get_dim(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_1]->get_stride(), (std::vector<int64_t>{3, 1}));
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_INPUT_UID_1]->get_name(), "input1");

    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_OUTPUT_UID_0), 0u);
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_OUTPUT_UID_0]->get_dim(), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_OUTPUT_UID_0]->get_stride(), (std::vector<int64_t>{3, 1}));
    EXPECT_EQ(tensorMap[K_CUSTOM_OP_OUTPUT_UID_0]->get_name(), "output0");
}

// Builds a custom op graph with 1 input and 2 outputs, verifies the round-trip
// preserves tensor counts and UIDs correctly.
TEST_F(IntegrationCustomOpDescriptorLifting, SingleInputTwoOutputsRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("SingleInputTwoOutputsTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto input0 = std::make_shared<TensorAttributes>();
    input0->set_uid(K_CUSTOM_OP_INPUT_UID_0).set_name("input0").set_data_type(DataType::FLOAT);
    input0->set_dim({2, 3}).set_stride({3, 1});

    CustomOpAttributes attrs;
    attrs.set_name("single_input_two_outputs")
        .set_custom_op_id(K_CUSTOM_OP_ID)
        .set_data(K_CUSTOM_OP_OPAQUE_DATA);

    auto outputs = graph->custom_op({input0}, 2, attrs);
    ASSERT_EQ(outputs.size(), 2u);
    outputs[0]
        ->set_uid(K_CUSTOM_OP_OUTPUT_UID_0)
        .set_output(true)
        .set_name("output0")
        .set_dim({2, 3})
        .set_stride({3, 1})
        .set_data_type(DataType::FLOAT);
    outputs[1]
        ->set_uid(K_CUSTOM_OP_OUTPUT_UID_1)
        .set_output(true)
        .set_name("output1")
        .set_dim({2, 3})
        .set_stride({3, 1})
        .set_data_type(DataType::FLOAT);

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify 3 tensors: 1 input + 2 outputs
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_INPUT_UID_0), 0u);
    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_OUTPUT_UID_0), 0u);
    ASSERT_NE(tensorMap.count(K_CUSTOM_OP_OUTPUT_UID_1), 0u);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* customOpNode = dynamic_cast<CustomOpNode*>(subNodes[0].get());
    ASSERT_NE(customOpNode, nullptr);

    ASSERT_EQ(customOpNode->attributes.get_inputs().size(), 1u);
    EXPECT_EQ(customOpNode->attributes.get_inputs()[0]->get_uid(), K_CUSTOM_OP_INPUT_UID_0);
    ASSERT_EQ(customOpNode->attributes.get_outputs().size(), 2u);
    EXPECT_EQ(customOpNode->attributes.get_outputs()[0]->get_uid(), K_CUSTOM_OP_OUTPUT_UID_0);
    EXPECT_EQ(customOpNode->attributes.get_outputs()[1]->get_uid(), K_CUSTOM_OP_OUTPUT_UID_1);
    EXPECT_EQ(customOpNode->attributes.get_name(), "single_input_two_outputs");
}

TEST_F(IntegrationCustomOpDescriptorLifting, ZeroInputCustomOpRoundTrip)
{
    auto originalGraph = buildZeroInputCustomOpGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* customOpNode = dynamic_cast<CustomOpNode*>(subNodes[0].get());
    ASSERT_NE(customOpNode, nullptr);

    ASSERT_EQ(customOpNode->attributes.get_inputs().size(), 0u);
    ASSERT_EQ(customOpNode->attributes.get_outputs().size(), 1u);
    EXPECT_EQ(customOpNode->attributes.get_outputs()[0]->get_uid(), K_CUSTOM_OP_OUTPUT_UID_0);
}

TEST_F(IntegrationCustomOpDescriptorLifting, ZeroOutputCustomOpRoundTrip)
{
    auto originalGraph = buildZeroOutputCustomOpGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = originalGraph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto rawDesc = originalGraph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    auto liftedGraph = std::make_shared<TestableGraph>();
    result = liftedGraph->fromBackendDescriptor(rawDesc);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* customOpNode = dynamic_cast<CustomOpNode*>(subNodes[0].get());
    ASSERT_NE(customOpNode, nullptr);

    ASSERT_EQ(customOpNode->attributes.get_inputs().size(), 2u);
    ASSERT_EQ(customOpNode->attributes.get_outputs().size(), 0u);
}

} // namespace
