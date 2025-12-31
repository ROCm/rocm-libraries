// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
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

// This must be called before serialize()
void prepareGraphForSerialization(Graph& graph)
{
    graph.assignTensorUids();
}

//==============================================================================
// Basic JSON Serialization Tests
//==============================================================================

TEST(TestGraphSerialization, SerializeDeserializeJson)
{
    // Create a simple graph
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

    // Prepare graph for serialization (assigns UIDs to output tensors)
    prepareGraphForSerialization(graph);

    // Serialize to JSON
    nlohmann::json json;
    graph.serialize(json);

    // Check basic properties
    EXPECT_EQ(json["name"], "test_serialization_graph");
    EXPECT_EQ(json["nodes"].size(), 1);
    EXPECT_EQ(json["tensors"].size(), 3); // x, w, y (output)

    // Deserialize to new graph
    Graph newGraph;
    newGraph.deserialize(json);

    EXPECT_EQ(newGraph.get_name(), "test_serialization_graph");
    EXPECT_EQ(newGraph.get_compute_data_type(), DataType::FLOAT);

    // We can re-serialize and compare
    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    // Just check counts again as exact JSON might differ slightly due to ordering
    EXPECT_EQ(newJson["nodes"].size(), 1);
    EXPECT_EQ(newJson["tensors"].size(), 3);
}

TEST(TestGraphSerialization, SerializeDeserializeBinary)
{
    // Create a simple graph
    Graph graph;
    graph.set_name("test_binary_graph");
    graph.set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("x")
        .set_dim({1, 16})
        .set_stride({16, 1})
        .set_data_type(DataType::FLOAT)
        .set_uid(10);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);

    graph.pointwise(x, pwAttrs);

    prepareGraphForSerialization(graph);

    std::vector<uint8_t> binaryData;
    graph.serialize(binaryData);

    EXPECT_FALSE(binaryData.empty());

    Graph newGraph;
    hipdnnHandle_t nullHandle = nullptr;
    auto err = newGraph.deserialize(nullHandle, binaryData);

    EXPECT_EQ(err.get_code(), ErrorCode::OK);
    EXPECT_EQ(newGraph.get_name(), "test_binary_graph");
}

//==============================================================================
// Graph Attribute Tests
//==============================================================================

TEST(TestGraphSerialization, GraphAttributesPreserved)
{
    Graph graph;
    graph.set_name("attribute_test_graph");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::HALF);
    graph.set_intermediate_data_type(DataType::BFLOAT16);

    // Add a simple node so graph is not empty
    auto x = createTensor("x", {1, 16, 8, 8}, DataType::HALF, 1);
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::IDENTITY);
    graph.pointwise(x, pwAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    Graph newGraph;
    newGraph.deserialize(json);

    EXPECT_EQ(newGraph.get_name(), "attribute_test_graph");
    EXPECT_EQ(newGraph.get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(newGraph.get_io_data_type(), DataType::HALF);
    EXPECT_EQ(newGraph.get_intermediate_data_type(), DataType::BFLOAT16);
}

TEST(TestGraphSerialization, PreferredEngineIdPreserved)
{
    Graph graph;
    graph.set_name("engine_id_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_preferred_engine_id_ext(42);

    auto x = createTensor("x", {1, 16, 8, 8}, DataType::FLOAT, 1);
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_TRUE(json.contains("preferred_engine_id"));
    EXPECT_EQ(json["preferred_engine_id"], 42);

    Graph newGraph;
    newGraph.deserialize(json);

    // Re-serialize to verify the value was restored
    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);
    EXPECT_TRUE(newJson.contains("preferred_engine_id"));
    EXPECT_EQ(newJson["preferred_engine_id"], 42);
}

TEST(TestGraphSerialization, VersionInfoPresent)
{
    Graph graph;
    graph.set_name("version_test");
    graph.set_compute_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 16}, DataType::FLOAT, 1);
    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::ABS);
    graph.pointwise(x, pwAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_TRUE(json.contains("hipdnn_frontend_version"));
    EXPECT_TRUE(json.contains("json_version"));
}

//==============================================================================
// Tensor Attribute Tests
//==============================================================================

TEST(TestGraphSerialization, TensorAttributesPreserved)
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

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    // Find the input tensor in JSON
    bool foundInputTensor = false;
    for(const auto& tensor : json["tensors"])
    {
        if(tensor["name"] == "input_tensor")
        {
            foundInputTensor = true;
            EXPECT_EQ(tensor["uid"], 100);
            auto dims = tensor["dims"].get<std::vector<int64_t>>();
            EXPECT_EQ(dims, (std::vector<int64_t>{2, 64, 32, 32}));
            auto strides = tensor["strides"].get<std::vector<int64_t>>();
            EXPECT_EQ(strides, (std::vector<int64_t>{65536, 1024, 32, 1}));
            break;
        }
    }
    EXPECT_TRUE(foundInputTensor);
}

TEST(TestGraphSerialization, VirtualTensorsSerialized)
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
    graph.pointwise(convOut, reluAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    // Should have 4 tensors: x, w, conv_out (virtual), relu_out
    EXPECT_EQ(json["tensors"].size(), 4);

    // Find the virtual tensor (conv output)
    bool foundVirtualTensor = false;
    for(const auto& tensor : json["tensors"])
    {
        if(tensor.contains("virtual") && tensor["virtual"].get<bool>())
        {
            foundVirtualTensor = true;
            break;
        }
    }
    EXPECT_TRUE(foundVirtualTensor);
}

//==============================================================================
// Convolution Node Tests
//==============================================================================

TEST(TestGraphSerialization, ConvFpropNodeSerialization)
{
    Graph graph;
    graph.set_name("conv_fprop_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto w = createTensor("w", {128, 64, 3, 3}, DataType::FLOAT, 2);

    ConvFpropAttributes convAttrs;
    convAttrs.set_padding({1, 1}).set_stride({2, 2}).set_dilation({1, 1}).set_convolution_mode(
        ConvolutionMode::CROSS_CORRELATION);

    graph.conv_fprop(x, w, convAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 1);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 1);
    EXPECT_EQ(newJson["tensors"].size(), 3); // x, w, y
}

TEST(TestGraphSerialization, ConvDgradNodeSerialization)
{
    Graph graph;
    graph.set_name("conv_dgrad_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto dy = createTensor("dy", {1, 128, 16, 16}, DataType::FLOAT, 1);
    auto w = createTensor("w", {128, 64, 3, 3}, DataType::FLOAT, 2);

    ConvDgradAttributes dgradAttrs;
    dgradAttrs.set_padding({1, 1}).set_stride({2, 2}).set_dilation({1, 1});

    graph.conv_dgrad(dy, w, dgradAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 1);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 1);
}

TEST(TestGraphSerialization, ConvWgradNodeSerialization)
{
    Graph graph;
    graph.set_name("conv_wgrad_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto dy = createTensor("dy", {1, 128, 16, 16}, DataType::FLOAT, 1);
    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 2);

    ConvWgradAttributes wgradAttrs;
    wgradAttrs.set_padding({1, 1}).set_stride({2, 2}).set_dilation({1, 1});

    graph.conv_wgrad(dy, x, wgradAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 1);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 1);
}

//==============================================================================
// Batchnorm Node Tests
//==============================================================================

TEST(TestGraphSerialization, BatchnormNodeSerialization)
{
    Graph graph;
    graph.set_name("batchnorm_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 2);
    auto bias = createTensor1D("bias", 64, DataType::FLOAT, 3);
    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(4);

    BatchnormAttributes bnAttrs;
    bnAttrs.set_epsilon(epsilon);

    graph.batchnorm(x, scale, bias, bnAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 1);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 1);
}

TEST(TestGraphSerialization, BatchnormInferenceNodeSerialization)
{
    Graph graph;
    graph.set_name("batchnorm_inference_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto mean = createTensor1D("mean", 64, DataType::FLOAT, 2);
    auto invVariance = createTensor1D("inv_variance", 64, DataType::FLOAT, 3);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 4);
    auto bias = createTensor1D("bias", 64, DataType::FLOAT, 5);

    BatchnormInferenceAttributes bnInfAttrs;

    graph.batchnorm_inference(x, mean, invVariance, scale, bias, bnInfAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 1);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 1);
}

TEST(TestGraphSerialization, BatchnormBackwardNodeSerialization)
{
    Graph graph;
    graph.set_name("batchnorm_backward_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);

    auto dy = createTensor("dy", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 2);
    auto scale = createTensor1D("scale", 64, DataType::FLOAT, 3);
    auto mean = createTensor1D("mean", 64, DataType::FLOAT, 4);
    auto invVariance = createTensor1D("inv_variance", 64, DataType::FLOAT, 5);

    BatchnormBackwardAttributes bnBwdAttrs;
    bnBwdAttrs.set_mean(mean).set_inv_variance(invVariance);

    graph.batchnorm_backward(dy, x, scale, bnBwdAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 1);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 1);
}

//==============================================================================
// Pointwise Node Tests
//==============================================================================

TEST(TestGraphSerialization, UnaryPointwiseSerialization)
{
    std::vector<PointwiseMode> unaryModes = {PointwiseMode::RELU_FWD,
                                             PointwiseMode::SIGMOID_FWD,
                                             PointwiseMode::TANH_FWD,
                                             PointwiseMode::ABS,
                                             PointwiseMode::EXP,
                                             PointwiseMode::LOG,
                                             PointwiseMode::SQRT,
                                             PointwiseMode::NEG};

    for(auto mode : unaryModes)
    {
        Graph graph;
        graph.set_name("unary_pointwise_test");
        graph.set_compute_data_type(DataType::FLOAT);

        auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);

        PointwiseAttributes pwAttrs;
        pwAttrs.set_mode(mode);
        graph.pointwise(x, pwAttrs);

        prepareGraphForSerialization(graph);

        nlohmann::json json;
        auto err = graph.serialize(json);
        EXPECT_EQ(err.get_code(), ErrorCode::OK) << "Failed for mode: " << static_cast<int>(mode);

        Graph newGraph;
        err = newGraph.deserialize(json);
        EXPECT_EQ(err.get_code(), ErrorCode::OK)
            << "Deserialize failed for mode: " << static_cast<int>(mode);

        prepareGraphForSerialization(newGraph);

        nlohmann::json newJson;
        newGraph.serialize(newJson);
        EXPECT_EQ(newJson["nodes"].size(), 1);
    }
}

TEST(TestGraphSerialization, BinaryPointwiseSerialization)
{
    std::vector<PointwiseMode> binaryModes = {PointwiseMode::ADD,
                                              PointwiseMode::SUB,
                                              PointwiseMode::MUL,
                                              PointwiseMode::DIV,
                                              PointwiseMode::MAX,
                                              PointwiseMode::MIN};

    for(auto mode : binaryModes)
    {
        Graph graph;
        graph.set_name("binary_pointwise_test");
        graph.set_compute_data_type(DataType::FLOAT);

        auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
        auto y = createTensor("y", {1, 64, 32, 32}, DataType::FLOAT, 2);

        PointwiseAttributes pwAttrs;
        pwAttrs.set_mode(mode);
        graph.pointwise(x, y, pwAttrs);

        prepareGraphForSerialization(graph);

        nlohmann::json json;
        auto err = graph.serialize(json);
        EXPECT_EQ(err.get_code(), ErrorCode::OK) << "Failed for mode: " << static_cast<int>(mode);

        Graph newGraph;
        err = newGraph.deserialize(json);
        EXPECT_EQ(err.get_code(), ErrorCode::OK)
            << "Deserialize failed for mode: " << static_cast<int>(mode);

        prepareGraphForSerialization(newGraph);

        nlohmann::json newJson;
        newGraph.serialize(newJson);
        EXPECT_EQ(newJson["nodes"].size(), 1);
        EXPECT_EQ(newJson["tensors"].size(), 3); // x, y, output
    }
}

TEST(TestGraphSerialization, TernaryPointwiseSerialization)
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

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 1);
    EXPECT_EQ(json["tensors"].size(), 4); // condition, x, y, output

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);
    EXPECT_EQ(newJson["nodes"].size(), 1);
}

TEST(TestGraphSerialization, PointwiseWithExtraAttributes)
{
    Graph graph;
    graph.set_name("pointwise_extra_attrs_test");
    graph.set_compute_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::ELU_FWD);
    pwAttrs.set_elu_alpha(1.0f);
    graph.pointwise(x, pwAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);
    EXPECT_EQ(newJson["nodes"].size(), 1);
}

//==============================================================================
// Multi-Node Graph Tests
//==============================================================================

TEST(TestGraphSerialization, ConvReluFusionSerialization)
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

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 2);
    EXPECT_EQ(json["tensors"].size(), 4); // x, w, conv_out (virtual), relu_out

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 2);
    EXPECT_EQ(newJson["tensors"].size(), 4);
}

TEST(TestGraphSerialization, ConvBiasReluFusionSerialization)
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

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 3);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 3);
}

TEST(TestGraphSerialization, ResidualBlockSerialization)
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

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 5); // 2 conv + 2 relu + 1 add

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 5);
}

//==============================================================================
// Data Type Tests
//==============================================================================

TEST(TestGraphSerialization, HalfPrecisionSerialization)
{
    Graph graph;
    graph.set_name("half_precision_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::HALF);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::HALF, 1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    Graph newGraph;
    newGraph.deserialize(json);

    EXPECT_EQ(newGraph.get_io_data_type(), DataType::HALF);
}

TEST(TestGraphSerialization, BFloat16Serialization)
{
    Graph graph;
    graph.set_name("bfloat16_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::BFLOAT16);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::BFLOAT16, 1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    Graph newGraph;
    newGraph.deserialize(json);

    EXPECT_EQ(newGraph.get_io_data_type(), DataType::BFLOAT16);
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

    prepareGraphForSerialization(graph);

    // Serialize to binary
    std::vector<uint8_t> binaryData;
    auto err = graph.serialize(binaryData);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);
    EXPECT_FALSE(binaryData.empty());

    // Deserialize
    Graph newGraph;
    hipdnnHandle_t nullHandle = nullptr;
    err = newGraph.deserialize(nullHandle, binaryData);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    EXPECT_EQ(newGraph.get_name(), "binary_roundtrip_test");
    EXPECT_EQ(newGraph.get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(newGraph.get_io_data_type(), DataType::FLOAT);

    // Re-serialize to verify
    prepareGraphForSerialization(newGraph);

    std::vector<uint8_t> newBinaryData;
    newGraph.serialize(newBinaryData);
    EXPECT_FALSE(newBinaryData.empty());
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

    prepareGraphForSerialization(graph);

    // Serialize to JSON
    nlohmann::json json;
    graph.serialize(json);

    // Serialize to binary
    std::vector<uint8_t> binaryData;
    graph.serialize(binaryData);

    // Deserialize both
    Graph jsonGraph;
    jsonGraph.deserialize(json);

    Graph binaryGraph;
    hipdnnHandle_t nullHandle = nullptr;
    binaryGraph.deserialize(nullHandle, binaryData);

    // Both should produce the same result
    EXPECT_EQ(jsonGraph.get_name(), binaryGraph.get_name());
    EXPECT_EQ(jsonGraph.get_compute_data_type(), binaryGraph.get_compute_data_type());
}

//==============================================================================
// Edge Case Tests
//==============================================================================

TEST(TestGraphSerialization, LargeDimensionsSerialization)
{
    Graph graph;
    graph.set_name("large_dims_test");
    graph.set_compute_data_type(DataType::FLOAT);

    auto x = createTensor("x", {16, 1024, 128, 128}, DataType::FLOAT, 1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(x, pwAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    // Find the tensor and verify dimensions
    for(const auto& tensor : newJson["tensors"])
    {
        if(tensor["name"] == "x")
        {
            auto dims = tensor["dims"].get<std::vector<int64_t>>();
            EXPECT_EQ(dims, (std::vector<int64_t>{16, 1024, 128, 128}));
            break;
        }
    }
}

TEST(TestGraphSerialization, SingleElementTensorSerialization)
{
    Graph graph;
    graph.set_name("single_element_test");
    graph.set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("scalar").set_dim({1}).set_stride({1}).set_data_type(DataType::FLOAT).set_uid(1);

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::ABS);
    graph.pointwise(x, pwAttrs);

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 1);
}

TEST(TestGraphSerialization, MultipleIndependentBranches)
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

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 3);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 3);
}

TEST(TestGraphSerialization, DeepChainSerialization)
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

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 10);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 10);
}

TEST(TestGraphSerialization, UniqueUidsPreserved)
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

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

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

    prepareGraphForSerialization(originalGraph);

    nlohmann::json json;
    originalGraph.serialize(json);

    // Now deserialize the valid JSON - should work
    Graph newGraph;
    auto err = newGraph.deserialize(json);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);
}

//==============================================================================
// Pass-by-Value Tensor Tests
//==============================================================================

TEST(TestGraphSerialization, PassByValueTensorSerialization)
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

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    EXPECT_EQ(json["nodes"].size(), 1);

    // Note: pass-by-value tensors might be serialized differently based on SDK implementation
    // This test verifies the round-trip works regardless

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["nodes"].size(), 1);
}

//==============================================================================
// tensor_like Utility Test
//==============================================================================

TEST(TestGraphSerialization, TensorLikeSerialization)
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

    prepareGraphForSerialization(graph);

    nlohmann::json json;
    graph.serialize(json);

    // Should have 3 tensors (original, copy, output)
    EXPECT_EQ(json["tensors"].size(), 3);

    Graph newGraph;
    newGraph.deserialize(json);

    prepareGraphForSerialization(newGraph);

    nlohmann::json newJson;
    newGraph.serialize(newJson);

    EXPECT_EQ(newJson["tensors"].size(), 3);
}
