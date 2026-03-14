// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <set>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// Helper function to create a tensor with computed contiguous strides
std::shared_ptr<TensorAttributes> createTensor(const std::string& name,
                                               const std::vector<int64_t>& dims,
                                               DataType dtype,
                                               int64_t uid)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_name(name)
        .set_dim(dims)
        .set_stride(hipdnn_data_sdk::utilities::generateStrides(dims))
        .set_data_type(dtype)
        .set_uid(uid);
    return tensor;
}

// Helper function to create a 1D tensor (for scale/bias)
std::shared_ptr<TensorAttributes>
    createTensor1D(const std::string& name, int64_t size, DataType dtype, int64_t uid)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_name(name).set_dim({size}).set_stride({1}).set_data_type(dtype).set_uid(uid);
    return tensor;
}

//==============================================================================
// Parametrized Test Framework
//==============================================================================

// Serialization formats supported for round-trip testing
enum class SerializationFormat
{
    JSON,
    BINARY,
    FLATBUFFER_DETACHED,
    FLATBUFFER_OBJECT
};

//==============================================================================
// Graph Comparison Helpers
//==============================================================================

// Compare two graphs using FlatBuffer-generated == operators on unpacked types.
// We cannot use GraphT::operator== directly because it compares tensor vectors
// positionally (using std::equal), but serialization may reorder tensors.
// Instead, we match tensors by UID and use TensorAttributesT::operator== for each pair.
void expectGraphsEqual(Graph& expected, Graph& actual)
{
    using namespace hipdnn_data_sdk::data_objects;

    auto expectedBuffer = expected.toFlatBuffer();
    auto actualBuffer = actual.toFlatBuffer();

    auto expectedUnpacked = UnPackGraph(expectedBuffer.data());
    auto actualUnpacked = UnPackGraph(actualBuffer.data());

    // Manually compare graph-level fields (can't use GraphT::operator== due to tensor ordering)
    EXPECT_EQ(expectedUnpacked->name, actualUnpacked->name) << "Graph name mismatch";
    EXPECT_EQ(expectedUnpacked->compute_data_type, actualUnpacked->compute_data_type)
        << "Graph compute_data_type mismatch";
    EXPECT_EQ(expectedUnpacked->io_data_type, actualUnpacked->io_data_type)
        << "Graph io_data_type mismatch";
    EXPECT_EQ(expectedUnpacked->intermediate_data_type, actualUnpacked->intermediate_data_type)
        << "Graph intermediate_data_type mismatch";
    EXPECT_EQ(expectedUnpacked->preferred_engine_id, actualUnpacked->preferred_engine_id)
        << "Graph preferred_engine_id mismatch";

    // Compare tensors using TensorAttributesT::operator== (match by UID due to potential reordering)
    ASSERT_EQ(expectedUnpacked->tensors.size(), actualUnpacked->tensors.size())
        << "Tensor count mismatch";

    std::unordered_map<int64_t, const TensorAttributesT*> actualTensorMap;
    for(const auto& tensor : actualUnpacked->tensors)
    {
        actualTensorMap[tensor->uid] = tensor.get();
    }

    EXPECT_EQ(actualTensorMap.size(), actualUnpacked->tensors.size())
        << "Duplicate tensor UIDs detected";

    for(const auto& expTensor : expectedUnpacked->tensors)
    {
        auto it = actualTensorMap.find(expTensor->uid);
        ASSERT_NE(it, actualTensorMap.end())
            << "Tensor with uid " << expTensor->uid << " not found";

        EXPECT_EQ(*expTensor, *(it->second))
            << "TensorAttributesT::operator== failed for uid " << expTensor->uid;
    }

    // Compare nodes using NodeT::operator==
    ASSERT_EQ(expectedUnpacked->nodes.size(), actualUnpacked->nodes.size())
        << "Node count mismatch";

    for(size_t i = 0; i < expectedUnpacked->nodes.size(); ++i)
    {
        EXPECT_EQ(*expectedUnpacked->nodes[i], *actualUnpacked->nodes[i])
            << "NodeT::operator== failed for node[" << i << "]";
    }
}

// Parametrized test fixture for serialization round-trip tests
class TestGraphSerializationRoundTrip : public ::testing::TestWithParam<SerializationFormat>
{
protected:
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    void roundTripAndCompare(Graph& graph)
    {
        Graph restored;

        switch(GetParam())
        {
        case SerializationFormat::JSON:
        {
            auto json = graph.toJson();
            restored.deserialize(json);
            break;
        }
        case SerializationFormat::BINARY:
        {
            auto binary = graph.toBinary();
            const std::vector<uint8_t>& binaryCopy = binary;
            restored.deserialize(nullptr, binaryCopy);
            break;
        }
        case SerializationFormat::FLATBUFFER_DETACHED:
        {
            auto fb = graph.toFlatBuffer();
            restored.deserialize(fb);
            break;
        }
        case SerializationFormat::FLATBUFFER_OBJECT:
        {
            auto fb = graph.toFlatBuffer();
            auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(fb.data());
            restored.deserialize(fbGraph);
            break;
        }
        default:
            FAIL() << "Unknown serialization format";
        }

        ASSERT_NO_FATAL_FAILURE(expectGraphsEqual(graph, restored));
    }
};

// Custom name generator for better test names
std::string serializationFormatToString(const ::testing::TestParamInfo<SerializationFormat>& info)
{
    switch(info.param)
    {
    case SerializationFormat::JSON:
        return "Json";
    case SerializationFormat::BINARY:
        return "Binary";
    case SerializationFormat::FLATBUFFER_DETACHED:
        return "FlatBufferDetached";
    case SerializationFormat::FLATBUFFER_OBJECT:
        return "FlatBufferObject";
    default:
        return "Unknown";
    }
}

//==============================================================================
// Parametrized Round-Trip Tests
//==============================================================================

TEST_P(TestGraphSerializationRoundTrip, SimpleConvolution)
{
    Graph graph;
    graph.set_name("test_serialization_graph");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x")
        .set_dim({1, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1})
        .set_data_type(DataType::FLOAT)
        .set_uid(1);

    auto w = std::make_shared<TensorAttributes>();
    w->set_name("w")
        .set_dim({64, 64, 3, 3})
        .set_stride({576, 9, 3, 1})
        .set_data_type(DataType::FLOAT)
        .set_uid(2);

    ConvFpropAttributes convAttrs;
    convAttrs.set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});

    graph.conv_fprop(x, w, convAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, GraphAttributes)
{
    Graph graph;
    graph.set_name("attribute_test_graph");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::HALF);
    graph.set_intermediate_data_type(DataType::BFLOAT16);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, PreferredEngineId)
{
    Graph graph;
    graph.set_preferred_engine_id_ext(42);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, TensorAttributes)
{
    Graph graph;
    graph.set_name("tensor_attr_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("input_tensor")
        .set_dim({2, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1})
        .set_data_type(DataType::FLOAT)
        .set_uid(100);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, VirtualTensors)
{
    Graph graph;
    graph.set_name("virtual_tensor_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto w = createTensor("w", {64, 64, 3, 3}, DataType::FLOAT, 2);

    ConvFpropAttributes convAttrs;
    convAttrs.set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});

    // Conv output is virtual (intermediate)
    auto convOut = graph.conv_fprop(x, w, convAttrs);

    // Apply ReLU to make a multi-node graph
    PointwiseAttributes reluAttrs;
    reluAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto reluOut = graph.pointwise(convOut, reluAttrs);
    reluOut->set_output(true); // Mark as output so only convOut is virtual

    auto json = graph.toJson();

    // Should have 4 tensors: x, w, conv_out (virtual), relu_out
    EXPECT_EQ(json["tensors"].size(), 4);

    // Find the virtual tensor (conv output) and verify it's marked correctly
    bool foundVirtualTensor = false;
    int64_t virtualTensorUid = -1;
    for(const auto& tensor : json["tensors"])
    {
        if(tensor.contains("virtual") && tensor["virtual"].get<bool>())
        {
            foundVirtualTensor = true;

            // Virtual tensor must have a UID
            EXPECT_TRUE(tensor.contains("uid"));
            virtualTensorUid = tensor["uid"];

            // Must be marked as virtual
            EXPECT_TRUE(tensor["virtual"].get<bool>());

            // The remaining properties are handled by the engine, and therefore not stored.

            break;
        }
    }
    EXPECT_TRUE(foundVirtualTensor);
    EXPECT_NE(virtualTensorUid, -1);

    // Verify virtual tensor is correctly marked in FlatBuffer
    auto buffer = graph.toFlatBuffer();
    auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());

    bool foundVirtualInFB = false;
    for(size_t i = 0; i < fbGraph->tensors()->size(); ++i)
    {
        auto fbTensor = fbGraph->tensors()->Get(static_cast<flatbuffers::uoffset_t>(i));
        if(fbTensor->uid() == virtualTensorUid)
        {
            foundVirtualInFB = true;

            // Verify it's marked as virtual
            EXPECT_TRUE(fbTensor->virtual_());

            // Note: Virtual tensors are placeholders and don't store full tensor attributes.
            // The graph execution engine infers their properties from the operations.

            break;
        }
    }
    EXPECT_TRUE(foundVirtualInFB);

    // Verify full round-trip correctness
    Graph restored;
    restored.deserialize(json);
    expectGraphsEqual(graph, restored);
}

TEST_P(TestGraphSerializationRoundTrip, TernaryPointwise)
{
    Graph graph;
    graph.set_name("ternary_pointwise_test");
    graph.set_compute_data_type(DataType::FLOAT);

    auto condition = createTensor("condition", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 2);
    auto y = createTensor("y", {1, 64, 32, 32}, DataType::FLOAT, 3);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::BINARY_SELECT);
    graph.pointwise(condition, x, y, pwAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, ConvReluFusion)
{
    Graph graph;
    graph.set_name("conv_relu_fusion_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto w = createTensor("w", {128, 64, 3, 3}, DataType::FLOAT, 2);

    ConvFpropAttributes convAttrs;
    convAttrs.set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});

    auto convOut = graph.conv_fprop(x, w, convAttrs);

    PointwiseAttributes reluAttrs;
    reluAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(convOut, reluAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, ConvBiasReluFusion)
{
    Graph graph;
    graph.set_name("conv_bias_relu_fusion_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto w = createTensor("w", {128, 64, 3, 3}, DataType::FLOAT, 2);
    auto bias = createTensor1D("bias", 128, DataType::FLOAT, 3);

    ConvFpropAttributes convAttrs;
    convAttrs.set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});

    auto convOut = graph.conv_fprop(x, w, convAttrs);

    // Add bias using pointwise ADD
    PointwiseAttributes addAttrs;
    addAttrs.set_mode(PointwiseMode::ADD);
    auto biasOut = graph.pointwise(convOut, bias, addAttrs);

    // Apply ReLU
    PointwiseAttributes reluAttrs;
    reluAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(biasOut, reluAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, ResidualBlock)
{
    Graph graph;
    graph.set_name("residual_block_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto w1 = createTensor("w1", {64, 64, 3, 3}, DataType::FLOAT, 2);
    auto w2 = createTensor("w2", {64, 64, 3, 3}, DataType::FLOAT, 3);

    // First conv
    ConvFpropAttributes conv1Attrs;
    conv1Attrs.set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto conv1Out = graph.conv_fprop(x, w1, conv1Attrs);

    // ReLU
    PointwiseAttributes relu1Attrs;
    relu1Attrs.set_mode(PointwiseMode::RELU_FWD);
    auto relu1Out = graph.pointwise(conv1Out, relu1Attrs);

    // Second conv
    ConvFpropAttributes conv2Attrs;
    conv2Attrs.set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    auto conv2Out = graph.conv_fprop(relu1Out, w2, conv2Attrs);

    // Residual add
    PointwiseAttributes addAttrs;
    addAttrs.set_mode(PointwiseMode::ADD);
    auto addOut = graph.pointwise(conv2Out, x, addAttrs);

    // Final ReLU
    PointwiseAttributes relu2Attrs;
    relu2Attrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(addOut, relu2Attrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, BnTrainingActivFusion)
{
    Graph graph;
    graph.set_name("bn_training_activ_fusion_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 2);
    auto bias = createTensor1D("bias", 64, DataType::FLOAT, 3);
    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(4);

    BatchnormAttributes bnAttrs;
    bnAttrs.set_epsilon(epsilon);

    auto [y, savedMean, savedInvVariance, nextRunningMean, nextRunningVariance]
        = graph.batchnorm(x, scale, bias, bnAttrs);

    // Apply activation
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(y, pwAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, BnInfDReluBnBwdFusion)
{
    Graph graph;
    graph.set_name("bn_inf_drelu_bn_bwd_fusion_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto savedMean = createTensor1D("saved_mean", 64, DataType::FLOAT, 2);
    auto savedInvVariance = createTensor1D("saved_inv_variance", 64, DataType::FLOAT, 3);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 4);
    auto bias = createTensor1D("bias", 64, DataType::FLOAT, 5);
    auto dy = createTensor("dy", {1, 64, 32, 32}, DataType::FLOAT, 6);

    // Batchnorm inference
    BatchnormInferenceAttributes bnInfAttrs;
    auto bnY = graph.batchnorm_inference(x, savedMean, savedInvVariance, scale, bias, bnInfAttrs);

    // DReLU (ReLU backward)
    PointwiseAttributes activBwdAttrs;
    activBwdAttrs.set_mode(PointwiseMode::RELU_BWD);
    auto dxDrelu = graph.pointwise(bnY, dy, activBwdAttrs);

    // Batchnorm backward
    BatchnormBackwardAttributes bnBwdAttrs;
    bnBwdAttrs.set_saved_mean_and_inv_variance(savedMean, savedInvVariance);
    graph.batchnorm_backward(dxDrelu, x, scale, bnBwdAttrs);

    roundTripAndCompare(graph);
}

//==============================================================================
// Binary Serialization Tests
//==============================================================================

TEST(TestGraphSerialization, BinarySerializationRoundTrip)
{
    Graph graph;
    graph.set_name("binary_roundtrip_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto w = createTensor("w", {128, 64, 3, 3}, DataType::FLOAT, 2);

    ConvFpropAttributes convAttrs;
    convAttrs.set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});

    auto convOut = graph.conv_fprop(x, w, convAttrs);

    PointwiseAttributes reluAttrs;
    reluAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(convOut, reluAttrs);

    // Serialize to binary
    auto binaryData = graph.toBinary();
    EXPECT_FALSE(binaryData.empty());

    // Deserialize
    Graph newGraph;
    hipdnnHandle_t nullHandle = nullptr;
    auto err = newGraph.deserialize(nullHandle, binaryData);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    // Verify full round-trip correctness
    expectGraphsEqual(graph, newGraph);
}

TEST(TestGraphSerialization, BinaryVsJsonConsistency)
{
    Graph graph;
    graph.set_name("binary_json_consistency_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    // Serialize to JSON
    auto json = graph.toJson();

    // Serialize to binary
    auto binaryData = graph.toBinary();

    // Deserialize both
    Graph jsonGraph;
    jsonGraph.deserialize(json);

    Graph binaryGraph;
    hipdnnHandle_t nullHandle = nullptr;
    binaryGraph.deserialize(nullHandle, binaryData);

    // Verify both deserialized graphs are fully identical
    ASSERT_NO_FATAL_FAILURE(expectGraphsEqual(jsonGraph, binaryGraph));

    // Also verify they match the original
    ASSERT_NO_FATAL_FAILURE(expectGraphsEqual(graph, jsonGraph));
    ASSERT_NO_FATAL_FAILURE(expectGraphsEqual(graph, binaryGraph));
}

//==============================================================================
// FlatBuffer Object Serialization Tests
//==============================================================================

TEST(TestGraphSerialization, ToFlatBufferReturnsValidBuffer)
{
    Graph graph;
    graph.set_name("flatbuffer_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, pwAttrs);
    y->set_output(true); // Mark as output to test non-virtual tensor

    // Get flatbuffer (assigns UIDs if needed)
    auto buffer = graph.toFlatBuffer();

    // Verify buffer is valid
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_GT(buffer.size(), 0u);

    // Verify we can read the flatbuffer
    auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());
    EXPECT_NE(fbGraph, nullptr);
    EXPECT_STREQ(fbGraph->name()->c_str(), "flatbuffer_test");

    // Verify round-trip produces identical graph
    Graph restored;
    restored.fromFlatBuffer(buffer);
    expectGraphsEqual(graph, restored);
}

TEST(TestGraphSerialization, FromFlatBufferRestoresGraph)
{
    // Create original graph
    Graph graph;
    graph.set_name("from_flatbuffer_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::HALF);
    graph.set_intermediate_data_type(DataType::BFLOAT16);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto w = createTensor("w", {128, 64, 3, 3}, DataType::FLOAT, 2);

    ConvFpropAttributes convAttrs;
    convAttrs.set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    graph.conv_fprop(x, w, convAttrs);

    // Convert to flatbuffer (assigns UIDs if needed)
    auto buffer = graph.toFlatBuffer();
    auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());

    // Restore to new graph using fromFlatBuffer
    Graph newGraph;
    auto err = newGraph.fromFlatBuffer(fbGraph);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    // Verify full round-trip correctness
    expectGraphsEqual(graph, newGraph);
}

TEST(TestGraphSerialization, FlatBufferPreservesPreferredEngineId)
{
    Graph graph;
    graph.set_name("preferred_engine_flatbuffer_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);
    graph.set_preferred_engine_id_ext(42);

    auto x = createTensor("x", {1, 16}, DataType::FLOAT, 1);
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::IDENTITY);
    graph.pointwise(x, pwAttrs);

    // Round-trip through flatbuffer (assigns UIDs if needed)
    auto buffer = graph.toFlatBuffer();
    auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());

    Graph newGraph;
    auto err = newGraph.fromFlatBuffer(fbGraph);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    // Verify by re-serializing to JSON and checking the value
    auto json = newGraph.toJson();
    EXPECT_TRUE(json.contains("preferred_engine_id"));
    EXPECT_EQ(json["preferred_engine_id"], 42);

    // Full verification - ensure complete round-trip correctness
    expectGraphsEqual(graph, newGraph);
}

TEST(TestGraphSerialization, BinaryUsesPackedFlatBuffer)
{
    Graph graph;
    graph.set_name("binary_flatbuffer_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 16}, DataType::FLOAT, 1);
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    // Get binary serialization (assigns UIDs if needed)
    auto binaryData = graph.toBinary();

    // Verify it's a valid flatbuffer (not UBJSON)
    // FlatBuffers can be directly parsed with GetGraph
    auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(binaryData.data());
    EXPECT_NE(fbGraph, nullptr);
    EXPECT_STREQ(fbGraph->name()->c_str(), "binary_flatbuffer_test");

    // Compare with toFlatBuffer output - should be identical
    auto directBuffer = graph.toFlatBuffer();
    EXPECT_EQ(binaryData.size(), directBuffer.size());
    EXPECT_EQ(std::memcmp(binaryData.data(), directBuffer.data(), binaryData.size()), 0);
}

TEST(TestGraphSerialization, SerializeOverloadReturnsDetachedBuffer)
{
    Graph graph;
    graph.set_name("serialize_overload_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 16}, DataType::FLOAT, 1);
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, pwAttrs);
    y->set_output(true); // Mark as output to test non-virtual tensor

    // Use toFlatBuffer() that returns DetachedBuffer
    auto buffer = graph.toFlatBuffer();
    EXPECT_GT(buffer.size(), 0u);

    // Verify it's equivalent to calling toFlatBuffer() again
    auto directBuffer = graph.toFlatBuffer();
    EXPECT_EQ(buffer.size(), directBuffer.size());
    EXPECT_EQ(std::memcmp(buffer.data(), directBuffer.data(), buffer.size()), 0);

    // Verify round-trip produces identical graph
    Graph restored;
    restored.fromFlatBuffer(buffer);
    expectGraphsEqual(graph, restored);
}

TEST(TestGraphSerialization, DeserializeFromFlatBufferGraphObject)
{
    Graph graph;
    graph.set_name("deserialize_graph_object_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 16}, DataType::FLOAT, 1);
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, pwAttrs);
    y->set_output(true); // Mark as output to test non-virtual tensor

    // Serialize to buffer
    auto buffer = graph.toFlatBuffer();
    auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());

    // Use deserialize(const Graph*) overload
    Graph newGraph;
    auto err = newGraph.deserialize(fbGraph);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    // Verify full round-trip correctness
    expectGraphsEqual(graph, newGraph);
}

TEST(TestGraphSerialization, DeserializeFromDetachedBuffer)
{
    Graph graph;
    graph.set_name("deserialize_detached_buffer_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 16}, DataType::FLOAT, 1);
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    // Serialize to DetachedBuffer (assigns UIDs if needed)
    auto buffer = graph.toFlatBuffer();

    // Use deserialize(const DetachedBuffer&) overload
    Graph newGraph;
    auto err = newGraph.deserialize(buffer);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    // Verify restoration
    auto json = newGraph.toJson();
    EXPECT_EQ(json["name"], "deserialize_detached_buffer_test");
    EXPECT_EQ(json["nodes"].size(), 1u);
}

TEST(TestGraphSerialization, FromFlatBufferDetachedBufferOverload)
{
    Graph graph;
    graph.set_name("from_flatbuffer_detached_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 16}, DataType::FLOAT, 1);
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    // Serialize to DetachedBuffer (assigns UIDs if needed)
    auto buffer = graph.toFlatBuffer();

    // Use fromFlatBuffer(const DetachedBuffer&) overload
    Graph newGraph;
    auto err = newGraph.fromFlatBuffer(buffer);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    // Verify restoration
    auto json = newGraph.toJson();
    EXPECT_EQ(json["name"], "from_flatbuffer_detached_test");
    EXPECT_EQ(json["nodes"].size(), 1u);
}

TEST(TestGraphSerialization, ConstSerializeReturnsErrorWithoutUids)
{
    Graph graph;
    graph.set_name("const_serialize_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    // Create tensor without UID
    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x");
    x->set_dim({1, 16});
    x->set_stride({16, 1});
    x->set_data_type(DataType::FLOAT);
    // Note: NOT calling set_uid()

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    // Const serialize should return error because UIDs are not set
    const Graph& constGraph = graph;
    flatbuffers::DetachedBuffer buffer;
    auto err = constGraph.serialize(buffer);
    EXPECT_EQ(err.get_code(), ErrorCode::ATTRIBUTE_NOT_SET);

    // Non-const serialize should succeed by assigning UIDs
    auto nonConstBuffer = graph.toFlatBuffer();
    EXPECT_GT(nonConstBuffer.size(), 0u);

    // Verify UIDs were actually assigned
    EXPECT_TRUE(x->has_uid()) << "Tensor should have UID after non-const serialize";

    // Verify round-trip works after UID assignment
    Graph restored;
    restored.fromFlatBuffer(nonConstBuffer);
    expectGraphsEqual(graph, restored);
}

TEST(TestGraphSerialization, NonConstSerializeAssignsUids)
{
    Graph graph;
    graph.set_name("nonconst_serialize_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    // Create input tensor without UID
    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x");
    x->set_dim({1, 16});
    x->set_stride({16, 1});
    x->set_data_type(DataType::FLOAT);
    // Note: NOT calling set_uid()

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, pwAttrs);

    // Verify UIDs are not set before serialization
    EXPECT_FALSE(x->has_uid());
    EXPECT_FALSE(y->has_uid());

    // Non-const serialize should assign UIDs
    auto buffer = graph.toFlatBuffer();
    EXPECT_GT(buffer.size(), 0u);

    // After non-const serialize, both tensors should have UIDs assigned
    EXPECT_TRUE(x->has_uid()) << "Input tensor should have UID after serialize";
    EXPECT_TRUE(y->has_uid()) << "Output tensor should have UID after serialize";
    EXPECT_NE(x->get_uid(), y->get_uid()) << "UIDs should be unique";

    // Verify the assigned UIDs are actually in the serialized buffer
    auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());
    ASSERT_EQ(fbGraph->tensors()->size(), 2u);

    std::set<int64_t> serializedUids;
    for(size_t i = 0; i < fbGraph->tensors()->size(); ++i)
    {
        serializedUids.insert(
            fbGraph->tensors()->Get(static_cast<flatbuffers::uoffset_t>(i))->uid());
    }
    EXPECT_TRUE(serializedUids.count(x->get_uid()))
        << "Input tensor UID " << x->get_uid() << " not found in serialized buffer";
    EXPECT_TRUE(serializedUids.count(y->get_uid()))
        << "Output tensor UID " << y->get_uid() << " not found in serialized buffer";

    // Verify round-trip preserves the assigned UIDs
    Graph restored;
    restored.fromFlatBuffer(buffer);
    expectGraphsEqual(graph, restored);
}

TEST(TestGraphSerialization, ConstJsonSerializeReturnsErrorWithoutUids)
{
    Graph graph;
    graph.set_name("const_json_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    // Create tensor without UID
    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x");
    x->set_dim({1, 16});
    x->set_stride({16, 1});
    x->set_data_type(DataType::FLOAT);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    // Const JSON serialize should return error
    const Graph& constGraph = graph;
    nlohmann::json json;
    auto err = constGraph.serialize(json);
    EXPECT_EQ(err.get_code(), ErrorCode::ATTRIBUTE_NOT_SET);

    // Non-const toJson() should succeed
    auto nonConstJson = graph.toJson();
    EXPECT_EQ(nonConstJson["name"], "const_json_test");

    // Verify UIDs were assigned
    EXPECT_TRUE(x->has_uid()) << "Tensor should have UID after non-const serialize";

    // Verify round-trip works
    Graph restored;
    restored.deserialize(nonConstJson);
    expectGraphsEqual(graph, restored);
}

TEST(TestGraphSerialization, ConstBinarySerializeReturnsErrorWithoutUids)
{
    Graph graph;
    graph.set_name("const_binary_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    // Create tensor without UID
    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x");
    x->set_dim({1, 16});
    x->set_stride({16, 1});
    x->set_data_type(DataType::FLOAT);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    // Const binary serialize should return error
    const Graph& constGraph = graph;
    std::vector<uint8_t> data;
    auto err = constGraph.serialize(data);
    EXPECT_EQ(err.get_code(), ErrorCode::ATTRIBUTE_NOT_SET);

    // Non-const toBinary() should succeed
    auto nonConstData = graph.toBinary();
    EXPECT_GT(nonConstData.size(), 0u);

    // Verify UIDs were assigned
    EXPECT_TRUE(x->has_uid()) << "Tensor should have UID after non-const serialize";

    // Verify round-trip works
    Graph restored;
    hipdnnHandle_t nullHandle = nullptr;
    restored.deserialize(nullHandle, nonConstData);
    expectGraphsEqual(graph, restored);
}

TEST(TestGraphSerialization, ConstJsonSerializeSucceedsWithUids)
{
    Graph graph;
    graph.set_name("const_json_success_test");

    // Create tensor with UID already set
    auto x = createTensor("x", {1, 16}, DataType::FLOAT, 1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, pwAttrs);
    y->set_uid(2); // Set UID on output tensor

    // Const JSON serialize should succeed when UIDs are set
    const Graph& constGraph = graph;
    nlohmann::json json;
    auto err = constGraph.serialize(json);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    // Deserialize and verify round-trip correctness
    Graph restored;
    restored.deserialize(json);
    expectGraphsEqual(graph, restored);
}

TEST(TestGraphSerialization, ConstFlatBufferSerializeSucceedsWithUids)
{
    Graph graph;
    graph.set_name("const_flatbuffer_success_test");

    // Create tensor with UID already set
    auto x = createTensor("x", {1, 16}, DataType::FLOAT, 1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph.pointwise(x, pwAttrs);
    y->set_uid(2); // Set UID on output tensor

    // Const FlatBuffer serialize should succeed when UIDs are set
    const Graph& constGraph = graph;
    flatbuffers::DetachedBuffer buffer;
    auto err = constGraph.serialize(buffer);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    // Deserialize and verify round-trip correctness
    Graph restored;
    restored.fromFlatBuffer(buffer);
    expectGraphsEqual(graph, restored);
}

TEST_P(TestGraphSerializationRoundTrip, LargeDimensions)
{
    Graph graph;
    graph.set_name("large_dims_test");
    graph.set_compute_data_type(DataType::FLOAT);

    auto x = createTensor("x", {16, 1024, 128, 128}, DataType::FLOAT, 1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, SingleElementTensor)
{
    Graph graph;
    graph.set_name("single_element_test");
    graph.set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("scalar").set_dim({1}).set_stride({1}).set_data_type(DataType::FLOAT).set_uid(1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::ABS);
    graph.pointwise(x, pwAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, MultipleIndependentBranches)
{
    // Test a graph with multiple independent operations (DAG structure)
    Graph graph;
    graph.set_name("multi_branch_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);

    // Branch 1: ReLU
    PointwiseAttributes relu1Attrs;
    relu1Attrs.set_mode(PointwiseMode::RELU_FWD);
    auto branch1 = graph.pointwise(x, relu1Attrs);

    // Branch 2: Sigmoid (from same input)
    PointwiseAttributes sigmoidAttrs;
    sigmoidAttrs.set_mode(PointwiseMode::SIGMOID_FWD);
    auto branch2 = graph.pointwise(x, sigmoidAttrs);

    // Merge branches with ADD
    PointwiseAttributes addAttrs;
    addAttrs.set_mode(PointwiseMode::ADD);
    graph.pointwise(branch1, branch2, addAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, DeepChain)
{
    // Test a deeply chained graph (many sequential operations)
    Graph graph;
    graph.set_name("deep_chain_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto current = createTensor("input", {1, 64, 32, 32}, DataType::FLOAT, 1);

    // Chain of 10 ReLU operations
    for(int i = 0; i < 10; ++i)
    {
        PointwiseAttributes reluAttrs;
        reluAttrs.set_mode(PointwiseMode::RELU_FWD);
        current = graph.pointwise(current, reluAttrs);
    }

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, UniqueUidsPreserved)
{
    Graph graph;
    graph.set_name("unique_uids_test");
    graph.set_compute_data_type(DataType::FLOAT);

    // Create tensors with specific UIDs
    auto x = createTensor("x", {1, 16}, DataType::FLOAT, 42);
    auto y = createTensor("y", {1, 16}, DataType::FLOAT, 99);

    PointwiseAttributes addAttrs;
    addAttrs.set_mode(PointwiseMode::ADD);
    graph.pointwise(x, y, addAttrs);

    // Only verify UIDs for JSON format (other formats will also preserve them)
    if(GetParam() == SerializationFormat::JSON)
    {
        auto json = graph.toJson();

        // Verify UIDs are in the JSON
        std::set<int64_t> foundUids;
        for(const auto& tensor : json["tensors"])
        {
            if(tensor.contains("uid"))
            {
                foundUids.insert(tensor["uid"].get<int64_t>());
            }
        }
        EXPECT_TRUE(foundUids.count(42) > 0);
        EXPECT_TRUE(foundUids.count(99) > 0);
    }

    // Verify full round-trip equality
    roundTripAndCompare(graph);
}

//==============================================================================
// Error Handling Tests
//==============================================================================

TEST(TestGraphSerialization, DeserializeInvalidJsonGracefully)
{
    Graph graph;

    // Empty JSON object
    nlohmann::json emptyJson = nlohmann::json::object();
    auto err = graph.deserialize(emptyJson);
    // Should not crash, behavior depends on implementation
    // At minimum, should return without exception
}

TEST(TestGraphSerialization, DeserializeMalformedDataTypesGracefully)
{
    // Create a valid JSON first, then modify it
    Graph originalGraph;
    originalGraph.set_name("malformed_test");
    originalGraph.set_compute_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 16, 8, 8}, DataType::FLOAT, 1);
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    originalGraph.pointwise(x, pwAttrs);

    auto json = originalGraph.toJson();

    // Now deserialize the valid JSON - should work
    Graph newGraph;
    auto err = newGraph.deserialize(json);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);
}

TEST(TestGraphSerialization, DeserializeMissingTensorUidReturnsError)
{
    // Create a JSON with a node that references a tensor UID that doesn't exist
    nlohmann::json json;
    json["name"] = "missing_tensor_test";
    json["compute_data_type"] = "float";
    json["intermediate_data_type"] = "float";
    json["io_data_type"] = "float";

    // Add tensors with UIDs 1 and 2 (input and output)
    json["tensors"] = nlohmann::json::array();
    nlohmann::json inputTensor;
    inputTensor["uid"] = 1;
    inputTensor["name"] = "x";
    inputTensor["dims"] = {1, 16, 8, 8};
    inputTensor["strides"] = {1024, 64, 8, 1};
    inputTensor["data_type"] = "float";
    inputTensor["virtual"] = false;
    json["tensors"].push_back(inputTensor);

    nlohmann::json outputTensor;
    outputTensor["uid"] = 2;
    outputTensor["name"] = "y";
    outputTensor["dims"] = {1, 16, 8, 8};
    outputTensor["strides"] = {1024, 64, 8, 1};
    outputTensor["data_type"] = "float";
    outputTensor["virtual"] = false;
    json["tensors"].push_back(outputTensor);

    // Add a properly formed node that references UID 999 which doesn't exist
    json["nodes"] = nlohmann::json::array();
    nlohmann::json node;
    node["type"] = "PointwiseAttributes";
    node["name"] = "test_node";
    node["compute_data_type"] = "float";
    node["inputs"] = nlohmann::json::object();
    node["inputs"]["operation"] = "relu_fwd";
    node["inputs"]["relu_lower_clip"] = nullptr;
    node["inputs"]["relu_upper_clip"] = nullptr;
    node["inputs"]["relu_lower_clip_slope"] = nullptr;
    node["inputs"]["swish_beta"] = nullptr;
    node["inputs"]["elu_alpha"] = nullptr;
    node["inputs"]["softplus_beta"] = nullptr;
    node["inputs"]["axis_tensor_uid"] = nullptr;
    node["inputs"]["in_0_tensor_uid"] = 999; // This UID doesn't exist!
    node["inputs"]["in_1_tensor_uid"] = nullptr;
    node["inputs"]["in_2_tensor_uid"] = nullptr;
    node["outputs"] = nlohmann::json::object();
    node["outputs"]["out_0_tensor_uid"] = 2;
    json["nodes"].push_back(node);

    Graph graph;
    auto err = graph.deserialize(json);

    // Should return an error about missing tensor
    EXPECT_EQ(err.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(err.get_message().find("missing tensor") != std::string::npos
                || err.get_message().find("invalid reference") != std::string::npos);
}

TEST(TestGraphSerialization, DeserializeMalformedJsonReturnsError)
{
    // Test that malformed JSON (missing required fields) returns an error
    nlohmann::json json;
    json["name"] = "malformed_test";
    json["compute_data_type"] = "float";
    json["tensors"] = nlohmann::json::array();
    json["nodes"] = nlohmann::json::array();

    // Node missing required "inputs" object
    nlohmann::json badNode;
    badNode["type"] = "PointwiseAttributes";
    // Missing "inputs" and "outputs" - should throw json::out_of_range
    json["nodes"].push_back(badNode);

    Graph graph;
    auto err = graph.deserialize(json);

    // Should return an error about malformed JSON
    EXPECT_EQ(err.get_code(), ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(err.get_message().find("malformed JSON") != std::string::npos
                || err.get_message().find("Deserialization failed") != std::string::npos);
}

TEST_P(TestGraphSerializationRoundTrip, PassByValueTensor)
{
    Graph graph;
    graph.set_name("pass_by_value_test");
    graph.set_compute_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);

    // Create a scalar pass-by-value tensor
    auto alpha = std::make_shared<TensorAttributes>(2.0f);
    alpha->set_uid(2);

    // Scale x by alpha using MUL
    PointwiseAttributes mulAttrs;
    mulAttrs.set_mode(PointwiseMode::MUL);
    graph.pointwise(x, alpha, mulAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, TensorLike)
{
    Graph graph;
    graph.set_name("tensor_like_test");
    graph.set_compute_data_type(DataType::FLOAT);

    auto original = createTensor("original", {1, 64, 32, 32}, DataType::FLOAT, 1);

    // Create a tensor_like copy
    auto copy = Graph::tensor_like(original, "copy_tensor");

    // Use both in a binary operation
    PointwiseAttributes addAttrs;
    addAttrs.set_mode(PointwiseMode::ADD);
    graph.pointwise(original, copy, addAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, MatmulNode)
{
    Graph graph;
    graph.set_name("matmul_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    // Create 2D tensors for matmul: A[M,K] x B[K,N] = C[M,N]
    auto a = createTensor("a", {32, 64}, DataType::FLOAT, 1);
    auto b = createTensor("b", {64, 128}, DataType::FLOAT, 2);

    MatmulAttributes matmulAttrs;
    auto c = graph.matmul(a, b, matmulAttrs);
    c->set_output(true); // Mark as output to test non-virtual tensor

    // Only verify counts for JSON format
    if(GetParam() == SerializationFormat::JSON)
    {
        auto json = graph.toJson();
        EXPECT_EQ(json["nodes"].size(), 1);
        EXPECT_EQ(json["tensors"].size(), 3); // a, b, c
    }

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, BatchnormInferenceNodeVarianceExt)
{
    Graph graph;
    graph.set_name("batchnorm_inference_variance_ext_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto mean = createTensor1D("mean", 64, DataType::FLOAT, 2);
    auto variance = createTensor1D("variance", 64, DataType::FLOAT, 3);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 4);
    auto bias = createTensor1D("bias", 64, DataType::FLOAT, 5);
    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);

    BatchnormInferenceAttributesVarianceExt bnInfVarAttrs;

    auto y = graph.batchnorm_inference_variance_ext(
        x, mean, variance, scale, bias, epsilon, bnInfVarAttrs);

    // Only verify counts for JSON format
    if(GetParam() == SerializationFormat::JSON)
    {
        auto json = graph.toJson();
        EXPECT_EQ(json["nodes"].size(), 1);
        EXPECT_EQ(json["tensors"].size(), 7); // x, mean, variance, scale, bias, epsilon, y
    }

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, ConvFprop)
{
    Graph graph;
    graph.set_name("conv_fprop_deep_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto w = createTensor("w", {128, 64, 3, 3}, DataType::FLOAT, 2);

    ConvFpropAttributes convAttrs;
    convAttrs.set_pre_padding({2, 3})
        .set_post_padding({2, 3})
        .set_stride({2, 2})
        .set_dilation({1, 1})
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    auto y = graph.conv_fprop(x, w, convAttrs);
    y->set_output(true); // Mark as output to test non-virtual tensor

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, PointwiseWithParams)
{
    Graph graph;
    graph.set_name("pointwise_params_deep_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);

    // Test ELU with alpha parameter
    PointwiseAttributes eluAttrs;
    eluAttrs.set_mode(PointwiseMode::ELU_FWD).set_elu_alpha(0.5f);

    auto y = graph.pointwise(x, eluAttrs);
    y->set_output(true); // Mark as output to test non-virtual tensor

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, BatchnormBackward)
{
    Graph graph;
    graph.set_name("batchnorm_backward_deep_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto dy = createTensor("dy", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 2);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 3);
    auto mean = createTensor1D("mean", 64, DataType::FLOAT, 4);
    auto invVariance = createTensor1D("inv_variance", 64, DataType::FLOAT, 5);

    BatchnormBackwardAttributes bnBwdAttrs;
    bnBwdAttrs.set_mean(mean).set_inv_variance(invVariance);

    graph.batchnorm_backward(dy, x, scale, bnBwdAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, ConvDgrad)
{
    Graph graph;
    graph.set_name("conv_dgrad_deep_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto dy = createTensor("dy", {1, 128, 16, 16}, DataType::FLOAT, 1);
    auto w = createTensor("w", {128, 64, 3, 3}, DataType::FLOAT, 2);

    ConvDgradAttributes dgradAttrs;
    dgradAttrs.set_pre_padding({1, 2})
        .set_post_padding({1, 2})
        .set_stride({2, 2})
        .set_dilation({1, 1})
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    auto dx = graph.conv_dgrad(dy, w, dgradAttrs);
    dx->set_output(true); // Mark as output to test non-virtual tensor

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, ConvWgrad)
{
    Graph graph;
    graph.set_name("conv_wgrad_deep_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto dy = createTensor("dy", {1, 128, 16, 16}, DataType::FLOAT, 1);
    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 2);

    ConvWgradAttributes wgradAttrs;
    wgradAttrs.set_pre_padding({1, 1})
        .set_post_padding({1, 1})
        .set_stride({2, 2})
        .set_dilation({1, 1})
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    auto dw = graph.conv_wgrad(dy, x, wgradAttrs);
    dw->set_output(true); // Mark as output to test non-virtual tensor

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, Batchnorm)
{
    Graph graph;
    graph.set_name("batchnorm_deep_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 2);
    auto bias = createTensor1D("bias", 64, DataType::FLOAT, 3);
    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(4);

    BatchnormAttributes bnAttrs;
    bnAttrs.set_epsilon(epsilon);

    graph.batchnorm(x, scale, bias, bnAttrs);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, BatchnormInference)
{
    Graph graph;
    graph.set_name("batchnorm_inference_deep_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto mean = createTensor1D("mean", 64, DataType::FLOAT, 2);
    auto invVariance = createTensor1D("inv_variance", 64, DataType::FLOAT, 3);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 4);
    auto bias = createTensor1D("bias", 64, DataType::FLOAT, 5);

    BatchnormInferenceAttributes bnInfAttrs;

    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, bnInfAttrs);
    y->set_output(true); // Mark as output to test non-virtual tensor

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, LayernormNodeInference)
{
    Graph graph;
    graph.set_name("layernorm_inference_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {2, 64, 32, 32}, DataType::FLOAT, 1);
    auto scale = createTensor("scale", {1, 64, 32, 32}, DataType::FLOAT, 2);
    auto bias = createTensor("bias", {1, 64, 32, 32}, DataType::FLOAT, 3);
    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(4);

    LayernormAttributes lnAttrs;
    lnAttrs.set_epsilon(epsilon);
    lnAttrs.set_forward_phase(NormFwdPhase::INFERENCE);

    auto [y, mean, invVariance] = graph.layernorm(x, scale, bias, lnAttrs);

    // In inference mode, mean and inv_variance should be nullptr
    EXPECT_EQ(mean, nullptr);
    EXPECT_EQ(invVariance, nullptr);

    // Only verify counts for JSON format
    if(GetParam() == SerializationFormat::JSON)
    {
        auto json = graph.toJson();
        EXPECT_EQ(json["nodes"].size(), 1);
        EXPECT_EQ(json["tensors"].size(), 5); // x, scale, bias, epsilon, y
    }

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, LayernormNodeTraining)
{
    Graph graph;
    graph.set_name("layernorm_training_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {2, 64, 32, 32}, DataType::FLOAT, 1);
    auto scale = createTensor("scale", {1, 64, 32, 32}, DataType::FLOAT, 2);
    auto bias = createTensor("bias", {1, 64, 32, 32}, DataType::FLOAT, 3);
    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(4);

    LayernormAttributes lnAttrs;
    lnAttrs.set_epsilon(epsilon);
    lnAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto [y, mean, invVariance] = graph.layernorm(x, scale, bias, lnAttrs);

    // In training mode, mean and inv_variance should be set
    ASSERT_NE(mean, nullptr);
    ASSERT_NE(invVariance, nullptr);

    // Only verify counts for JSON format
    if(GetParam() == SerializationFormat::JSON)
    {
        auto json = graph.toJson();
        EXPECT_EQ(json["nodes"].size(), 1);
        EXPECT_EQ(json["tensors"].size(), 7); // x, scale, bias, epsilon, y, mean, inv_variance
    }

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, RMSNormNode)
{
    Graph graph;
    graph.set_name("rmsnorm_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 2);
    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(3);

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto [y, invRms] = graph.rmsnorm(x, scale, rmsnormAttrs);
    y->set_output(true);

    if(GetParam() == SerializationFormat::JSON)
    {
        auto json = graph.toJson();
        EXPECT_EQ(json["nodes"].size(), 1);
        EXPECT_EQ(json["tensors"].size(), 5); // x, scale, epsilon, y, inv_rms
    }

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, RMSNormNodeWithBias)
{
    Graph graph;
    graph.set_name("rmsnorm_bias_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 2);
    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(3);
    auto bias = createTensor1D("bias", 64, DataType::FLOAT, 4);

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_bias(bias);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto [y, invRms] = graph.rmsnorm(x, scale, rmsnormAttrs);
    y->set_output(true);

    roundTripAndCompare(graph);
}

TEST_P(TestGraphSerializationRoundTrip, CustomOpNode)
{
    Graph graph;
    graph.set_name("custom_op_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto inputA = createTensor("input_a", {2, 3}, DataType::FLOAT, 1);
    auto inputB = createTensor("input_b", {2, 3}, DataType::FLOAT, 2);

    std::vector<uint8_t> opaquePayload = {0xDE, 0xAD, 0xBE, 0xEF};

    CustomOpAttributes customAttrs;
    customAttrs.set_name("my_custom_op").set_custom_op_id("example.my_add").set_data(opaquePayload);

    auto outputs = graph.custom_op({inputA, inputB}, 1, customAttrs);
    ASSERT_EQ(outputs.size(), 1u);
    outputs[0]->set_output(true).set_dim({2, 3}).set_stride({3, 1}).set_data_type(DataType::FLOAT);

    roundTripAndCompare(graph);
}

//==============================================================================
// Test Suite Instantiation
//==============================================================================

INSTANTIATE_TEST_SUITE_P(AllFormats,
                         TestGraphSerializationRoundTrip,
                         ::testing::Values(SerializationFormat::JSON,
                                           SerializationFormat::BINARY,
                                           SerializationFormat::FLATBUFFER_DETACHED,
                                           SerializationFormat::FLATBUFFER_OBJECT),
                         serializationFormatToString);

#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
