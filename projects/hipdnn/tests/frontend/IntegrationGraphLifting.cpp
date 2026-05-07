// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants;
using hipdnn_tests::buildConvFpropGraph;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;

namespace
{
// Builds a conv fprop graph via the frontend, lowers it through the backend C-API
// via build_operation_graph(), then lifts it back with fromBackendDescriptor()
// and verifies the reconstructed graph matches the original.
class IntegrationGraphLifting : public IntegrationTestFixture
{
};

// Builds a conv fprop graph, lowers via build_operation_graph(handle), extracts the
// raw descriptor, creates a new graph with fromBackendDescriptor(), and verifies
// tensor dimensions, data types, convolution parameters, and graph-level data types.
TEST_F(IntegrationGraphLifting, ConvFpropRoundTripViaCApi)
{
    auto originalGraph = buildConvFpropGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph name survives the C-API round-trip
    EXPECT_EQ(liftedGraph->get_name(), "ConvFpropTestGraph");

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u)
        << "Expected 3 tensors (X, W, Y) in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_X_UID), 0u);
    auto liftedX = tensorMap[K_FPROP_TENSOR_X_UID];
    EXPECT_EQ(liftedX->get_dim(), toVec(K_FPROP_TENSOR_X_DIMS));
    EXPECT_EQ(liftedX->get_stride(), toVec(K_FPROP_TENSOR_X_STRIDES));
    EXPECT_EQ(liftedX->get_data_type(), DataType::FLOAT);

    // Verify W tensor
    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_W_UID), 0u);
    auto liftedW = tensorMap[K_FPROP_TENSOR_W_UID];
    EXPECT_EQ(liftedW->get_dim(), toVec(K_FPROP_TENSOR_W_DIMS));
    EXPECT_EQ(liftedW->get_stride(), toVec(K_FPROP_TENSOR_W_STRIDES));
    EXPECT_EQ(liftedW->get_data_type(), DataType::FLOAT);

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_Y_UID), 0u);
    auto liftedY = tensorMap[K_FPROP_TENSOR_Y_UID];
    EXPECT_EQ(liftedY->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedY->get_dim(), toVec(K_FPROP_TENSOR_Y_DIMS));
    EXPECT_EQ(liftedY->get_stride(), toVec(K_FPROP_TENSOR_Y_STRIDES));

    // Verify the lifted graph has the correct number of sub-nodes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u)
        << "Expected 1 operation node in lifted graph"; // NOLINT(readability-implicit-bool-conversion)

    // Access the conv fprop node and verify convolution parameters
    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr)
        << "Expected a ConvolutionFpropNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_FPROP_CONV_DILATION));
    EXPECT_EQ(convNode->attributes.get_convolution_mode(), ConvolutionMode::CROSS_CORRELATION);
    EXPECT_EQ(convNode->attributes.get_name(), "conv_fprop_op");
}

// Verifies that tensors are accessible by UID on the reconstructed graph,
// confirming tensor identity is preserved through the round-trip.
TEST_F(IntegrationGraphLifting, ConvFpropTensorSharingPreserved)
{
    auto originalGraph = buildConvFpropGraph();

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // All tensors should be accessible by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    EXPECT_NE(tensorMap.count(K_FPROP_TENSOR_X_UID), 0u) << "X tensor not found by UID";
    EXPECT_NE(tensorMap.count(K_FPROP_TENSOR_W_UID), 0u) << "W tensor not found by UID";
    EXPECT_NE(tensorMap.count(K_FPROP_TENSOR_Y_UID), 0u) << "Y tensor not found by UID";

    // Verify the node references the same tensor objects via UID
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr);

    EXPECT_EQ(convNode->attributes.get_x()->get_uid(), K_FPROP_TENSOR_X_UID);
    EXPECT_EQ(convNode->attributes.get_w()->get_uid(), K_FPROP_TENSOR_W_UID);
    EXPECT_EQ(convNode->attributes.get_y()->get_uid(), K_FPROP_TENSOR_Y_UID);

    // Verify tensor objects are shared between the tensorMap and conv node attributes
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_X_UID].get(), convNode->attributes.get_x().get());
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_W_UID].get(), convNode->attributes.get_w().get());
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_Y_UID].get(), convNode->attributes.get_y().get());
}

// Builds a graph with set_preferred_engine_id_ext(42), lowers, lifts, and verifies
// that get_preferred_engine_id_ext() returns 42 on the reconstructed graph.
TEST_F(IntegrationGraphLifting, PreferredEngineIdPreservedThroughCApi)
{
    auto originalGraph = buildConvFpropGraph();
    constexpr int64_t K_PREFERRED_ENGINE_ID = 42;
    originalGraph->set_preferred_engine_id_ext(K_PREFERRED_ENGINE_ID);

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto liftedEngineId = liftedGraph->get_preferred_engine_id_ext();
    ASSERT_TRUE(liftedEngineId.has_value())
        << "Preferred engine ID should be set after lifting"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_EQ(liftedEngineId.value(), K_PREFERRED_ENGINE_ID);
}

// Verifies that fromBackendDescriptor(nullptr) returns an error.
TEST_F(IntegrationGraphLifting, NullDescriptorReturnsError)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    auto result = graph->fromBackendDescriptor(nullptr);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "fromBackendDescriptor(nullptr) should return INVALID_VALUE"; // NOLINT(readability-implicit-bool-conversion)
}

// Builds a graph with FLOAT compute, HALF intermediate, and BFLOAT16 io data types,
// lowers through the C-API, lifts, and verifies all three are preserved.
TEST_F(IntegrationGraphLifting, DataTypesPreservedThroughCApi)
{
    auto originalGraph = buildConvFpropGraph(
        "ConvFpropTestGraph", DataType::FLOAT, DataType::HALF, DataType::BFLOAT16);

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::HALF);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::BFLOAT16);
}

// Builds a conv fprop graph, serializes to binary, creates a backend descriptor
// from bytes (no handle, no finalize), calls fromBackendDescriptor(), and verifies
// the reconstructed graph matches the original.
TEST_F(IntegrationGraphLifting, ConvFpropLiftWithoutFinalization)
{
    auto originalGraph = buildConvFpropGraph();

    auto liftedGraph = liftGraphWithoutFinalization(*originalGraph);
    ASSERT_NE(liftedGraph, nullptr);

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
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_FPROP_CONV_DILATION));
    EXPECT_EQ(convNode->attributes.get_convolution_mode(), ConvolutionMode::CROSS_CORRELATION);

    // Verify tensor dims and strides
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_X_UID]->get_dim(), toVec(K_FPROP_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_X_UID]->get_stride(), toVec(K_FPROP_TENSOR_X_STRIDES));
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_W_UID]->get_dim(), toVec(K_FPROP_TENSOR_W_DIMS));
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_W_UID]->get_stride(), toVec(K_FPROP_TENSOR_W_STRIDES));
}

// Exercises the deserialize() path with a handle (full finalization).
// Builds a conv fprop graph, serializes to binary, then uses deserialize()
// with a handle and verifies the reconstructed graph.
TEST_F(IntegrationGraphLifting, DeserializeViaBackendWithHandle)
{
    auto originalGraph = buildConvFpropGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto [data, serErr] = originalGraph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    // Create a new graph and use deserialize with handle
    auto liftedGraph = std::make_shared<TestableGraphLifting>();
    result = liftedGraph->deserialize(_handle, data);
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
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_FPROP_CONV_DILATION));

    // Verify tensor dims
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_X_UID]->get_dim(), toVec(K_FPROP_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_W_UID]->get_dim(), toVec(K_FPROP_TENSOR_W_DIMS));
}

// Exercises the deserialize() path without a handle (no finalization).
// Verifies that the graph can be reconstructed from binary data without
// requiring a hipdnnHandle_t.
TEST_F(IntegrationGraphLifting, DeserializeViaBackendWithoutHandle)
{
    auto originalGraph = buildConvFpropGraph();

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto [data, serErr] = originalGraph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    // Create a new graph and use deserialize without handle
    auto liftedGraph = std::make_shared<TestableGraphLifting>();
    result = liftedGraph->deserialize(nullptr, data);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors are present
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);
    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_X_UID), 0u);
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_X_UID]->get_dim(), toVec(K_FPROP_TENSOR_X_DIMS));
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_X_UID]->get_stride(), toVec(K_FPROP_TENSOR_X_STRIDES));

    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_W_UID), 0u);
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_W_UID]->get_dim(), toVec(K_FPROP_TENSOR_W_DIMS));
    EXPECT_EQ(tensorMap[K_FPROP_TENSOR_W_UID]->get_stride(), toVec(K_FPROP_TENSOR_W_STRIDES));

    ASSERT_NE(tensorMap.count(K_FPROP_TENSOR_Y_UID), 0u);

    // Verify conv node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);
    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr);
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_FPROP_CONV_STRIDE));
}

// Verifies that fromBackendDescriptor returns an error (not a crash) when given
// a graph descriptor with no operations set.
TEST_F(IntegrationGraphLifting, EmptyGraphDescriptorReturnsError)
{
    // Create a backend graph descriptor with no operations
    hipdnnBackendDescriptor_t desc = nullptr;
    ASSERT_EQ(hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &desc),
              HIPDNN_STATUS_SUCCESS);

    // Attempt to lift — should return an error since no operations are set
    auto graph = std::make_shared<TestableGraphLifting>();
    auto result = graph->fromBackendDescriptor(desc);
    EXPECT_NE(result.code, ErrorCode::OK)
        << "fromBackendDescriptor should fail on a descriptor "
           "with no operations"; // NOLINT(readability-implicit-bool-conversion)

    hipdnnBackendDestroyDescriptor(desc);
}

// Verifies that deserialize returns an error (not a crash) when
// given corrupt (garbage) bytes.
TEST_F(IntegrationGraphLifting, DeserializeViaBackendCorruptDataReturnsError)
{
    const std::vector<uint8_t> garbage = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03};

    auto graph = std::make_shared<TestableGraphLifting>();
    auto result = graph->deserialize(_handle, garbage);
    EXPECT_NE(result.code, ErrorCode::OK)
        << "deserialize should fail on corrupt data"; // NOLINT(readability-implicit-bool-conversion)
}

// Verifies that deserialize returns an error (not a crash) when
// given an empty data vector.
TEST_F(IntegrationGraphLifting, DeserializeViaBackendEmptyDataReturnsError)
{
    const std::vector<uint8_t> empty;

    auto graph = std::make_shared<TestableGraphLifting>();
    auto result = graph->deserialize(_handle, empty);
    EXPECT_NE(result.code, ErrorCode::OK)
        << "deserialize should fail on empty data"; // NOLINT(readability-implicit-bool-conversion)
}

// Verifies that the graph name survives the C-API round-trip (lower -> lift).
TEST_F(IntegrationGraphLifting, GraphNamePreservedThroughCApi)
{
    auto originalGraph = buildConvFpropGraph("LiftingTestGraph");

    auto liftedGraph = liftGraph(*originalGraph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    EXPECT_EQ(liftedGraph->get_name(), "LiftingTestGraph");
}

// Exercises the deserialize() path and verifies the graph name
// is preserved through the deserialization path.
TEST_F(IntegrationGraphLifting, GraphNamePreservedThroughDeserializeViaBackend)
{
    auto originalGraph = buildConvFpropGraph("LiftingTestGraph");

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto [data, serErr] = originalGraph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    // Create a new graph and use deserialize without handle
    auto liftedGraph = std::make_shared<TestableGraphLifting>();
    result = liftedGraph->deserialize(nullptr, data);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_EQ(liftedGraph->get_name(), "LiftingTestGraph");
}

// Exercises the deserialize() path and verifies the preferred engine ID
// is preserved through the deserialization path.
TEST_F(IntegrationGraphLifting, PreferredEngineIdPreservedThroughDeserialize)
{
    auto originalGraph = buildConvFpropGraph();
    originalGraph->set_preferred_engine_id_ext(42);

    auto result = originalGraph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto [data, serErr] = originalGraph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    // Create a new graph and use deserialize without handle
    auto liftedGraph = std::make_shared<TestableGraphLifting>();
    result = liftedGraph->deserialize(nullptr, data);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    EXPECT_EQ(liftedGraph->get_preferred_engine_id_ext(), 42);
}

// Exercises the JSON serialize/deserialize path with a handle (full finalization)
// for a conv fprop graph.
TEST_F(IntegrationGraphLifting, JsonRoundTripWithHandle)
{
    auto originalGraph = buildConvFpropGraph();

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
    EXPECT_EQ(liftedGraph->get_name(), "ConvFpropTestGraph");
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u) << "Expected 3 tensors (X, W, Y)";

    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_FPROP_TENSOR_X_UID,
                                      "X",
                                      toVec(K_FPROP_TENSOR_X_DIMS),
                                      toVec(K_FPROP_TENSOR_X_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_FPROP_TENSOR_W_UID,
                                      "W",
                                      toVec(K_FPROP_TENSOR_W_DIMS),
                                      toVec(K_FPROP_TENSOR_W_STRIDES),
                                      DataType::FLOAT);
    hipdnn_tests::verifyTensorInGraph(tensorMap,
                                      K_FPROP_TENSOR_Y_UID,
                                      "Y",
                                      toVec(K_FPROP_TENSOR_Y_DIMS),
                                      toVec(K_FPROP_TENSOR_Y_STRIDES),
                                      DataType::FLOAT);

    // Verify sub-node count and type
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u) << "Expected 1 operation node in lifted graph";

    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr) << "Expected a ConvolutionFpropNode";

    // Verify convolution parameters
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_FPROP_CONV_DILATION));
    EXPECT_EQ(convNode->attributes.get_convolution_mode(), ConvolutionMode::CROSS_CORRELATION);
    EXPECT_EQ(convNode->attributes.get_name(), "conv_fprop_op");
}

} // namespace
