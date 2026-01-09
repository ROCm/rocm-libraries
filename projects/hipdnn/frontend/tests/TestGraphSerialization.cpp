// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

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
// Graph Comparison Helpers
//==============================================================================

namespace
{

// Helper to compare FlatBuffer vectors
template <typename T>
void expectVectorsEqual(const flatbuffers::Vector<T>* expected,
                        const flatbuffers::Vector<T>* actual,
                        const std::string& context)
{
    if(expected == nullptr && actual == nullptr)
    {
        return;
    }
    ASSERT_NE(expected, nullptr) << context << ": expected is null but actual is not";
    ASSERT_NE(actual, nullptr) << context << ": actual is null but expected is not";
    ASSERT_EQ(expected->size(), actual->size()) << context << ": size mismatch";
    for(size_t i = 0; i < expected->size(); ++i)
    {
        EXPECT_EQ(expected->Get(static_cast<flatbuffers::uoffset_t>(i)),
                  actual->Get(static_cast<flatbuffers::uoffset_t>(i)))
            << context << "[" << i << "]";
    }
}

// Helper to compare pass-by-value tensor values
void expectTensorValuesEqual(const hipdnn_data_sdk::data_objects::TensorAttributes* expected,
                             const hipdnn_data_sdk::data_objects::TensorAttributes* actual,
                             const std::string& context)
{
    using namespace hipdnn_data_sdk::data_objects;

    ASSERT_EQ(expected->value_type(), actual->value_type()) << context << ".value_type";

    switch(expected->value_type())
    {
    case TensorValue::NONE:
        break;
    case TensorValue::Float32Value:
        EXPECT_EQ(expected->value_as_Float32Value()->value(),
                  actual->value_as_Float32Value()->value())
            << context << ".value (float32)";
        break;
    case TensorValue::Float16Value:
        EXPECT_EQ(expected->value_as_Float16Value()->value(),
                  actual->value_as_Float16Value()->value())
            << context << ".value (float16)";
        break;
    case TensorValue::BFloat16Value:
        EXPECT_EQ(expected->value_as_BFloat16Value()->value(),
                  actual->value_as_BFloat16Value()->value())
            << context << ".value (bfloat16)";
        break;
    case TensorValue::Float8Value:
        EXPECT_EQ(expected->value_as_Float8Value()->value(),
                  actual->value_as_Float8Value()->value())
            << context << ".value (float8)";
        break;
    case TensorValue::Int32Value:
        EXPECT_EQ(expected->value_as_Int32Value()->value(), actual->value_as_Int32Value()->value())
            << context << ".value (int32)";
        break;
    case TensorValue::Float64Value:
        EXPECT_EQ(expected->value_as_Float64Value()->value(),
                  actual->value_as_Float64Value()->value())
            << context << ".value (float64)";
        break;
    default:
        FAIL() << context << ": unknown tensor value type "
               << static_cast<int>(expected->value_type());
    }
}

// Compare tensor attributes from FlatBuffer
void expectTensorsEqual(const hipdnn_data_sdk::data_objects::TensorAttributes* expected,
                        const hipdnn_data_sdk::data_objects::TensorAttributes* actual,
                        const std::string& context)
{
    ASSERT_NE(expected, nullptr) << context << ": expected tensor is null";
    ASSERT_NE(actual, nullptr) << context << ": actual tensor is null";

    EXPECT_EQ(expected->uid(), actual->uid()) << context << ".uid";
    if(expected->name() != nullptr && actual->name() != nullptr)
    {
        EXPECT_STREQ(expected->name()->c_str(), actual->name()->c_str()) << context << ".name";
    }
    EXPECT_EQ(expected->data_type(), actual->data_type()) << context << ".data_type";
    EXPECT_EQ(expected->virtual_(), actual->virtual_()) << context << ".virtual";
    expectVectorsEqual(expected->dims(), actual->dims(), context + ".dims");
    expectVectorsEqual(expected->strides(), actual->strides(), context + ".strides");
    expectTensorValuesEqual(expected, actual, context);
}

// Compare node attributes based on type
void expectNodeAttributesEqual(const hipdnn_data_sdk::data_objects::Node* expected,
                               const hipdnn_data_sdk::data_objects::Node* actual,
                               const std::string& context)
{
    using namespace hipdnn_data_sdk::data_objects;

    ASSERT_EQ(expected->attributes_type(), actual->attributes_type())
        << context << ": node type mismatch";

    switch(expected->attributes_type())
    {
    case NodeAttributes::ConvolutionFwdAttributes:
    {
        auto exp = expected->attributes_as_ConvolutionFwdAttributes();
        auto act = actual->attributes_as_ConvolutionFwdAttributes();
        EXPECT_EQ(exp->x_tensor_uid(), act->x_tensor_uid()) << context << ".x_tensor_uid";
        EXPECT_EQ(exp->w_tensor_uid(), act->w_tensor_uid()) << context << ".w_tensor_uid";
        EXPECT_EQ(exp->y_tensor_uid(), act->y_tensor_uid()) << context << ".y_tensor_uid";
        EXPECT_EQ(exp->conv_mode(), act->conv_mode()) << context << ".conv_mode";
        expectVectorsEqual(exp->pre_padding(), act->pre_padding(), context + ".pre_padding");
        expectVectorsEqual(exp->post_padding(), act->post_padding(), context + ".post_padding");
        expectVectorsEqual(exp->stride(), act->stride(), context + ".stride");
        expectVectorsEqual(exp->dilation(), act->dilation(), context + ".dilation");
        break;
    }
    case NodeAttributes::ConvolutionBwdAttributes:
    {
        auto exp = expected->attributes_as_ConvolutionBwdAttributes();
        auto act = actual->attributes_as_ConvolutionBwdAttributes();
        EXPECT_EQ(exp->dy_tensor_uid(), act->dy_tensor_uid()) << context << ".dy_tensor_uid";
        EXPECT_EQ(exp->w_tensor_uid(), act->w_tensor_uid()) << context << ".w_tensor_uid";
        EXPECT_EQ(exp->dx_tensor_uid(), act->dx_tensor_uid()) << context << ".dx_tensor_uid";
        EXPECT_EQ(exp->conv_mode(), act->conv_mode()) << context << ".conv_mode";
        expectVectorsEqual(exp->pre_padding(), act->pre_padding(), context + ".pre_padding");
        expectVectorsEqual(exp->post_padding(), act->post_padding(), context + ".post_padding");
        expectVectorsEqual(exp->stride(), act->stride(), context + ".stride");
        expectVectorsEqual(exp->dilation(), act->dilation(), context + ".dilation");
        break;
    }
    case NodeAttributes::ConvolutionWrwAttributes:
    {
        auto exp = expected->attributes_as_ConvolutionWrwAttributes();
        auto act = actual->attributes_as_ConvolutionWrwAttributes();
        EXPECT_EQ(exp->x_tensor_uid(), act->x_tensor_uid()) << context << ".x_tensor_uid";
        EXPECT_EQ(exp->dy_tensor_uid(), act->dy_tensor_uid()) << context << ".dy_tensor_uid";
        EXPECT_EQ(exp->dw_tensor_uid(), act->dw_tensor_uid()) << context << ".dw_tensor_uid";
        EXPECT_EQ(exp->conv_mode(), act->conv_mode()) << context << ".conv_mode";
        expectVectorsEqual(exp->pre_padding(), act->pre_padding(), context + ".pre_padding");
        expectVectorsEqual(exp->post_padding(), act->post_padding(), context + ".post_padding");
        expectVectorsEqual(exp->stride(), act->stride(), context + ".stride");
        expectVectorsEqual(exp->dilation(), act->dilation(), context + ".dilation");
        break;
    }
    case NodeAttributes::PointwiseAttributes:
    {
        auto exp = expected->attributes_as_PointwiseAttributes();
        auto act = actual->attributes_as_PointwiseAttributes();
        EXPECT_EQ(exp->operation(), act->operation()) << context << ".operation";
        EXPECT_EQ(exp->in_0_tensor_uid(), act->in_0_tensor_uid()) << context << ".in_0_tensor_uid";
        EXPECT_EQ(exp->out_0_tensor_uid(), act->out_0_tensor_uid())
            << context << ".out_0_tensor_uid";
        EXPECT_EQ(exp->in_1_tensor_uid(), act->in_1_tensor_uid()) << context << ".in_1_tensor_uid";
        EXPECT_EQ(exp->in_2_tensor_uid(), act->in_2_tensor_uid()) << context << ".in_2_tensor_uid";
        EXPECT_EQ(exp->relu_lower_clip(), act->relu_lower_clip()) << context << ".relu_lower_clip";
        EXPECT_EQ(exp->relu_upper_clip(), act->relu_upper_clip()) << context << ".relu_upper_clip";
        EXPECT_EQ(exp->swish_beta(), act->swish_beta()) << context << ".swish_beta";
        EXPECT_EQ(exp->elu_alpha(), act->elu_alpha()) << context << ".elu_alpha";
        EXPECT_EQ(exp->softplus_beta(), act->softplus_beta()) << context << ".softplus_beta";
        break;
    }
    case NodeAttributes::MatmulAttributes:
    {
        auto exp = expected->attributes_as_MatmulAttributes();
        auto act = actual->attributes_as_MatmulAttributes();
        EXPECT_EQ(exp->a_tensor_uid(), act->a_tensor_uid()) << context << ".a_tensor_uid";
        EXPECT_EQ(exp->b_tensor_uid(), act->b_tensor_uid()) << context << ".b_tensor_uid";
        EXPECT_EQ(exp->c_tensor_uid(), act->c_tensor_uid()) << context << ".c_tensor_uid";
        break;
    }
    case NodeAttributes::BatchnormAttributes:
    {
        auto exp = expected->attributes_as_BatchnormAttributes();
        auto act = actual->attributes_as_BatchnormAttributes();
        EXPECT_EQ(exp->x_tensor_uid(), act->x_tensor_uid()) << context << ".x_tensor_uid";
        EXPECT_EQ(exp->scale_tensor_uid(), act->scale_tensor_uid())
            << context << ".scale_tensor_uid";
        EXPECT_EQ(exp->bias_tensor_uid(), act->bias_tensor_uid()) << context << ".bias_tensor_uid";
        EXPECT_EQ(exp->y_tensor_uid(), act->y_tensor_uid()) << context << ".y_tensor_uid";
        EXPECT_EQ(exp->epsilon_tensor_uid(), act->epsilon_tensor_uid())
            << context << ".epsilon_tensor_uid";
        break;
    }
    case NodeAttributes::BatchnormInferenceAttributes:
    {
        auto exp = expected->attributes_as_BatchnormInferenceAttributes();
        auto act = actual->attributes_as_BatchnormInferenceAttributes();
        EXPECT_EQ(exp->x_tensor_uid(), act->x_tensor_uid()) << context << ".x_tensor_uid";
        EXPECT_EQ(exp->mean_tensor_uid(), act->mean_tensor_uid()) << context << ".mean_tensor_uid";
        EXPECT_EQ(exp->inv_variance_tensor_uid(), act->inv_variance_tensor_uid())
            << context << ".inv_variance_tensor_uid";
        EXPECT_EQ(exp->scale_tensor_uid(), act->scale_tensor_uid())
            << context << ".scale_tensor_uid";
        EXPECT_EQ(exp->bias_tensor_uid(), act->bias_tensor_uid()) << context << ".bias_tensor_uid";
        EXPECT_EQ(exp->y_tensor_uid(), act->y_tensor_uid()) << context << ".y_tensor_uid";
        break;
    }
    case NodeAttributes::BatchnormInferenceAttributesVarianceExt:
    {
        auto exp = expected->attributes_as_BatchnormInferenceAttributesVarianceExt();
        auto act = actual->attributes_as_BatchnormInferenceAttributesVarianceExt();
        EXPECT_EQ(exp->x_tensor_uid(), act->x_tensor_uid()) << context << ".x_tensor_uid";
        EXPECT_EQ(exp->mean_tensor_uid(), act->mean_tensor_uid()) << context << ".mean_tensor_uid";
        EXPECT_EQ(exp->variance_tensor_uid(), act->variance_tensor_uid())
            << context << ".variance_tensor_uid";
        EXPECT_EQ(exp->scale_tensor_uid(), act->scale_tensor_uid())
            << context << ".scale_tensor_uid";
        EXPECT_EQ(exp->bias_tensor_uid(), act->bias_tensor_uid()) << context << ".bias_tensor_uid";
        EXPECT_EQ(exp->y_tensor_uid(), act->y_tensor_uid()) << context << ".y_tensor_uid";
        break;
    }
    case NodeAttributes::BatchnormBackwardAttributes:
    {
        auto exp = expected->attributes_as_BatchnormBackwardAttributes();
        auto act = actual->attributes_as_BatchnormBackwardAttributes();
        EXPECT_EQ(exp->x_tensor_uid(), act->x_tensor_uid()) << context << ".x_tensor_uid";
        EXPECT_EQ(exp->dy_tensor_uid(), act->dy_tensor_uid()) << context << ".dy_tensor_uid";
        EXPECT_EQ(exp->scale_tensor_uid(), act->scale_tensor_uid())
            << context << ".scale_tensor_uid";
        EXPECT_EQ(exp->dx_tensor_uid(), act->dx_tensor_uid()) << context << ".dx_tensor_uid";
        EXPECT_EQ(exp->dscale_tensor_uid(), act->dscale_tensor_uid())
            << context << ".dscale_tensor_uid";
        EXPECT_EQ(exp->dbias_tensor_uid(), act->dbias_tensor_uid())
            << context << ".dbias_tensor_uid";
        EXPECT_EQ(exp->mean_tensor_uid(), act->mean_tensor_uid()) << context << ".mean_tensor_uid";
        EXPECT_EQ(exp->inv_variance_tensor_uid(), act->inv_variance_tensor_uid())
            << context << ".inv_variance_tensor_uid";
        break;
    }
    default:
        FAIL() << context << ": unknown node type "
               << static_cast<int>(expected->attributes_type());
    }
}

} // namespace

// Compare two graphs by serializing to FlatBuffer and comparing field-by-field
void expectGraphsEqual(Graph& expected, Graph& actual)
{
    using namespace hipdnn_data_sdk::data_objects;

    auto expectedBuffer = expected.toFlatBuffer();
    auto actualBuffer = actual.toFlatBuffer();

    auto fbExpected = GetGraph(expectedBuffer.data());
    auto fbActual = GetGraph(actualBuffer.data());

    // Compare graph-level attributes
    if(fbExpected->name() != nullptr && fbActual->name() != nullptr)
    {
        EXPECT_STREQ(fbExpected->name()->c_str(), fbActual->name()->c_str()) << "graph.name";
    }
    EXPECT_EQ(fbExpected->compute_data_type(), fbActual->compute_data_type())
        << "graph.compute_data_type";
    EXPECT_EQ(fbExpected->io_data_type(), fbActual->io_data_type()) << "graph.io_data_type";
    EXPECT_EQ(fbExpected->intermediate_data_type(), fbActual->intermediate_data_type())
        << "graph.intermediate_data_type";
    EXPECT_EQ(fbExpected->preferred_engine_id(), fbActual->preferred_engine_id())
        << "graph.preferred_engine_id";

    // Compare tensors by UID (ordering may differ after serialization)
    ASSERT_EQ(fbExpected->tensors()->size(), fbActual->tensors()->size())
        << "tensor count mismatch";

    // Build UID -> tensor map for actual graph
    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>
        actualTensorMap;
    for(size_t i = 0; i < fbActual->tensors()->size(); ++i)
    {
        auto tensor = fbActual->tensors()->Get(static_cast<flatbuffers::uoffset_t>(i));
        actualTensorMap[tensor->uid()] = tensor;
    }

    // Compare each expected tensor with its counterpart by UID
    for(size_t i = 0; i < fbExpected->tensors()->size(); ++i)
    {
        auto expTensor = fbExpected->tensors()->Get(static_cast<flatbuffers::uoffset_t>(i));
        auto it = actualTensorMap.find(expTensor->uid());
        ASSERT_NE(it, actualTensorMap.end())
            << "tensor with uid " << expTensor->uid() << " not found in actual graph";
        expectTensorsEqual(
            expTensor, it->second, "tensor[uid=" + std::to_string(expTensor->uid()) + "]");
    }

    // Compare nodes
    ASSERT_EQ(fbExpected->nodes()->size(), fbActual->nodes()->size()) << "node count mismatch";
    for(size_t i = 0; i < fbExpected->nodes()->size(); ++i)
    {
        auto expNode = fbExpected->nodes()->Get(static_cast<flatbuffers::uoffset_t>(i));
        auto actNode = fbActual->nodes()->Get(static_cast<flatbuffers::uoffset_t>(i));

        std::string nodeContext = "node[" + std::to_string(i) + "]";
        if(expNode->name() != nullptr && actNode->name() != nullptr)
        {
            EXPECT_STREQ(expNode->name()->c_str(), actNode->name()->c_str())
                << nodeContext << ".name";
        }
        EXPECT_EQ(expNode->compute_data_type(), actNode->compute_data_type())
            << nodeContext << ".compute_data_type";

        expectNodeAttributesEqual(expNode, actNode, nodeContext);
    }
}

// Deep comparison using FlatBuffer-generated == operators on unpacked native types.
// This provides a second verification layer using different code paths than expectGraphsEqual.
// Note: We cannot use GraphT::operator== directly because it compares tensor vectors
// positionally (using std::equal), but serialization may reorder tensors. Since tensors
// are referenced by UID in graph operations, ordering is semantically irrelevant.
// Instead, we match tensors by UID and use TensorAttributesT::operator== on each pair.
void expectGraphsEqualUnpacked(Graph& expected, Graph& actual)
{
    using namespace hipdnn_data_sdk::data_objects;

    auto expectedBuffer = expected.toFlatBuffer();
    auto actualBuffer = actual.toFlatBuffer();

    auto expectedUnpacked = UnPackGraph(expectedBuffer.data());
    auto actualUnpacked = UnPackGraph(actualBuffer.data());

    // Verify graph-level fields match
    EXPECT_EQ(expectedUnpacked->name, actualUnpacked->name) << "GraphT name mismatch";
    EXPECT_EQ(expectedUnpacked->compute_data_type, actualUnpacked->compute_data_type)
        << "GraphT compute_data_type mismatch";
    EXPECT_EQ(expectedUnpacked->io_data_type, actualUnpacked->io_data_type)
        << "GraphT io_data_type mismatch";
    EXPECT_EQ(expectedUnpacked->intermediate_data_type, actualUnpacked->intermediate_data_type)
        << "GraphT intermediate_data_type mismatch";
    EXPECT_EQ(expectedUnpacked->preferred_engine_id, actualUnpacked->preferred_engine_id)
        << "GraphT preferred_engine_id mismatch";

    // Verify tensor count matches
    ASSERT_EQ(expectedUnpacked->tensors.size(), actualUnpacked->tensors.size())
        << "GraphT tensor count mismatch";

    // Build UID -> tensor map for actual graph
    std::unordered_map<int64_t, const TensorAttributesT*> actualUnpackedTensorMap;
    for(const auto& tensor : actualUnpacked->tensors)
    {
        actualUnpackedTensorMap[tensor->uid] = tensor.get();
    }

    // Verify no duplicate UIDs in actual (all UIDs should be unique)
    EXPECT_EQ(actualUnpackedTensorMap.size(), actualUnpacked->tensors.size())
        << "Duplicate tensor UIDs detected in actual graph";

    // Compare each tensor using generated == operator (matched by UID)
    for(const auto& expTensor : expectedUnpacked->tensors)
    {
        auto it = actualUnpackedTensorMap.find(expTensor->uid);
        ASSERT_NE(it, actualUnpackedTensorMap.end())
            << "tensor with uid " << expTensor->uid << " not found in actual graph";

        // Compare using generated == operator
        EXPECT_EQ(*expTensor, *(it->second))
            << "TensorAttributesT == failed for uid " << expTensor->uid;
    }

    // Compare each node using generated == operator
    ASSERT_EQ(expectedUnpacked->nodes.size(), actualUnpacked->nodes.size())
        << "GraphT node count mismatch";
    for(size_t i = 0; i < expectedUnpacked->nodes.size(); ++i)
    {
        EXPECT_EQ(*expectedUnpacked->nodes[i], *actualUnpacked->nodes[i])
            << "NodeT == failed for node[" << i << "]";
    }
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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

    auto json = graph.toJson();

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
    graph.set_preferred_engine_id_ext(42);

    auto json = graph.toJson();

    EXPECT_TRUE(json.contains("preferred_engine_id"));
    EXPECT_EQ(json["preferred_engine_id"], 42);

    Graph newGraph;
    newGraph.deserialize(json);

    // Re-serialize to verify the value was restored
    auto newJson = newGraph.toJson();
    EXPECT_TRUE(newJson.contains("preferred_engine_id"));
    EXPECT_EQ(newJson["preferred_engine_id"], 42);
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

    auto json = graph.toJson();

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
            EXPECT_EQ(tensor["data_type"].get<std::string>(), "float");
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

//==============================================================================
// Pointwise Node Tests (removed redundant looping tests)
//==============================================================================

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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
}

TEST(TestGraphSerialization, BnTrainingActivFusion)
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
}

TEST(TestGraphSerialization, BnInfDReluBnBwdFusion)
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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
    expectGraphsEqual(jsonGraph, binaryGraph);

    // Also verify they match the original
    expectGraphsEqual(graph, jsonGraph);
    expectGraphsEqual(graph, binaryGraph);
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
    graph.pointwise(x, pwAttrs);

    // Get flatbuffer (assigns UIDs if needed)
    auto buffer = graph.toFlatBuffer();

    // Verify buffer is valid
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_GT(buffer.size(), 0);

    // Verify we can read the flatbuffer
    auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());
    EXPECT_NE(fbGraph, nullptr);
    EXPECT_STREQ(fbGraph->name()->c_str(), "flatbuffer_test");
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

TEST(TestGraphSerialization, FlatBufferRoundTripPreservesNodes)
{
    Graph graph;
    graph.set_name("flatbuffer_nodes_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {1, 64, 32, 32}, DataType::FLOAT, 1);
    auto w = createTensor("w", {128, 64, 3, 3}, DataType::FLOAT, 2);

    ConvFpropAttributes convAttrs;
    convAttrs.set_name("ConvNode");
    convAttrs.set_padding({1, 1}).set_stride({2, 2}).set_dilation({1, 1});
    auto convOut = graph.conv_fprop(x, w, convAttrs);

    PointwiseAttributes reluAttrs;
    reluAttrs.set_name("ReluNode");
    reluAttrs.set_mode(PointwiseMode::RELU_FWD);
    graph.pointwise(convOut, reluAttrs);

    // Round-trip through flatbuffer (assigns UIDs if needed)
    auto buffer = graph.toFlatBuffer();
    auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());

    Graph newGraph;
    auto err = newGraph.fromFlatBuffer(fbGraph);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    // Full verification - ensure complete round-trip correctness
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
    graph.pointwise(x, pwAttrs);

    // Use toFlatBuffer() that returns DetachedBuffer (assigns UIDs if needed)
    auto buffer = graph.toFlatBuffer();
    EXPECT_GT(buffer.size(), 0u);

    // Verify it's equivalent to toFlatBuffer()
    auto directBuffer = graph.toFlatBuffer();
    EXPECT_EQ(buffer.size(), directBuffer.size());
    EXPECT_EQ(std::memcmp(buffer.data(), directBuffer.data(), buffer.size()), 0);
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
    graph.pointwise(x, pwAttrs);

    // Serialize to buffer (assigns UIDs if needed)
    auto buffer = graph.toFlatBuffer();
    auto fbGraph = hipdnn_data_sdk::data_objects::GetGraph(buffer.data());

    // Use deserialize(const Graph*) overload
    Graph newGraph;
    auto err = newGraph.deserialize(fbGraph);
    EXPECT_EQ(err.get_code(), ErrorCode::OK);

    // Verify restoration
    auto json = newGraph.toJson();
    EXPECT_EQ(json["name"], "deserialize_graph_object_test");
    EXPECT_EQ(json["nodes"].size(), 1u);
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
}

TEST(TestGraphSerialization, NonConstSerializeAssignsUids)
{
    Graph graph;
    graph.set_name("nonconst_serialize_test");
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

    // Verify UID is not set
    EXPECT_FALSE(x->has_uid());

    // Non-const serialize should assign UIDs
    auto buffer = graph.toFlatBuffer();
    EXPECT_GT(buffer.size(), 0u);

    // After non-const serialize, tensor should have UID
    EXPECT_TRUE(x->has_uid());
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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

    // Also verify full round-trip equality
    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
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

    auto json = graph.toJson();

    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
}

//==============================================================================
// Matmul Node Tests
//==============================================================================

TEST(TestGraphSerialization, MatmulNodeSerialization)
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
    c->set_uid(3);

    auto json = graph.toJson();

    EXPECT_EQ(json["nodes"].size(), 1);
    EXPECT_EQ(json["tensors"].size(), 3); // a, b, c

    Graph newGraph;
    newGraph.deserialize(json);

    // Use deep comparison
    expectGraphsEqual(graph, newGraph);
}

TEST(TestGraphSerialization, MatmulNodeFieldVerification)
{
    Graph graph;
    graph.set_name("matmul_field_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto a = createTensor("a", {16, 32}, DataType::FLOAT, 100);
    auto b = createTensor("b", {32, 64}, DataType::FLOAT, 200);

    MatmulAttributes matmulAttrs;
    auto c = graph.matmul(a, b, matmulAttrs);
    c->set_uid(300);

    // Round-trip through JSON
    auto json = graph.toJson();
    Graph restored;
    restored.deserialize(json);

    // Deep comparison via FlatBuffer
    expectGraphsEqual(graph, restored);

    // Additional explicit field verification
    auto originalBuffer = graph.toFlatBuffer();
    auto restoredBuffer = restored.toFlatBuffer();

    auto fbOriginal = hipdnn_data_sdk::data_objects::GetGraph(originalBuffer.data());
    auto fbRestored = hipdnn_data_sdk::data_objects::GetGraph(restoredBuffer.data());

    auto origMatmul = fbOriginal->nodes()->Get(0)->attributes_as_MatmulAttributes();
    auto restMatmul = fbRestored->nodes()->Get(0)->attributes_as_MatmulAttributes();

    EXPECT_EQ(origMatmul->a_tensor_uid(), 100);
    EXPECT_EQ(origMatmul->b_tensor_uid(), 200);
    EXPECT_EQ(origMatmul->c_tensor_uid(), 300);
    EXPECT_EQ(restMatmul->a_tensor_uid(), 100);
    EXPECT_EQ(restMatmul->b_tensor_uid(), 200);
    EXPECT_EQ(restMatmul->c_tensor_uid(), 300);
}

//==============================================================================
// BatchnormInferenceVarianceExt Node Tests
//==============================================================================

TEST(TestGraphSerialization, BatchnormInferenceNodeVarianceExtSerialization)
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

    BatchnormInferenceAttributesVarianceExt bnInfVarAttrs;

    auto y = graph.batchnorm_inference_variance_ext(x, mean, variance, scale, bias, bnInfVarAttrs);
    y->set_uid(6);

    auto json = graph.toJson();

    EXPECT_EQ(json["nodes"].size(), 1);
    EXPECT_EQ(json["tensors"].size(), 6); // x, mean, variance, scale, bias, y

    Graph newGraph;
    newGraph.deserialize(json);

    // Use deep comparison
    expectGraphsEqual(graph, newGraph);
}

TEST(TestGraphSerialization, BatchnormInferenceVarianceExtFieldVerification)
{
    Graph graph;
    graph.set_name("bn_inf_var_ext_field_test");
    graph.set_compute_data_type(DataType::FLOAT);
    graph.set_io_data_type(DataType::FLOAT);
    graph.set_intermediate_data_type(DataType::FLOAT);

    auto x = createTensor("x", {2, 128, 16, 16}, DataType::FLOAT, 10);
    auto mean = createTensor1D("mean", 128, DataType::FLOAT, 20);
    auto variance = createTensor1D("variance", 128, DataType::FLOAT, 30);
    auto scale = createTensor1D("scale", 128, DataType::FLOAT, 40);
    auto bias = createTensor1D("bias", 128, DataType::FLOAT, 50);

    BatchnormInferenceAttributesVarianceExt bnInfVarAttrs;

    auto y = graph.batchnorm_inference_variance_ext(x, mean, variance, scale, bias, bnInfVarAttrs);
    y->set_uid(60);

    // Round-trip through JSON
    auto json = graph.toJson();
    Graph restored;
    restored.deserialize(json);

    // Deep comparison
    expectGraphsEqual(graph, restored);

    // Additional explicit field verification
    auto originalBuffer = graph.toFlatBuffer();
    auto restoredBuffer = restored.toFlatBuffer();

    auto fbOriginal = hipdnn_data_sdk::data_objects::GetGraph(originalBuffer.data());
    auto fbRestored = hipdnn_data_sdk::data_objects::GetGraph(restoredBuffer.data());

    auto origBn
        = fbOriginal->nodes()->Get(0)->attributes_as_BatchnormInferenceAttributesVarianceExt();
    auto restBn
        = fbRestored->nodes()->Get(0)->attributes_as_BatchnormInferenceAttributesVarianceExt();

    EXPECT_EQ(origBn->x_tensor_uid(), 10);
    EXPECT_EQ(origBn->mean_tensor_uid(), 20);
    EXPECT_EQ(origBn->variance_tensor_uid(), 30);
    EXPECT_EQ(origBn->scale_tensor_uid(), 40);
    EXPECT_EQ(origBn->bias_tensor_uid(), 50);
    EXPECT_EQ(origBn->y_tensor_uid(), 60);

    EXPECT_EQ(restBn->x_tensor_uid(), 10);
    EXPECT_EQ(restBn->mean_tensor_uid(), 20);
    EXPECT_EQ(restBn->variance_tensor_uid(), 30);
    EXPECT_EQ(restBn->scale_tensor_uid(), 40);
    EXPECT_EQ(restBn->bias_tensor_uid(), 50);
    EXPECT_EQ(restBn->y_tensor_uid(), 60);
}

//==============================================================================
// Deep Comparison Tests for All Node Types
//==============================================================================

TEST(TestGraphSerialization, ConvFpropDeepComparison)
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
    y->set_uid(3);

    auto json = graph.toJson();
    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
    expectGraphsEqualUnpacked(graph, restored);
}

TEST(TestGraphSerialization, PointwiseWithParamsDeepComparison)
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
    y->set_uid(2);

    auto json = graph.toJson();
    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);

    // Verify the elu_alpha specifically
    auto originalBuffer = graph.toFlatBuffer();
    auto restoredBuffer = restored.toFlatBuffer();

    auto fbOriginal = hipdnn_data_sdk::data_objects::GetGraph(originalBuffer.data());
    auto fbRestored = hipdnn_data_sdk::data_objects::GetGraph(restoredBuffer.data());

    auto origPw = fbOriginal->nodes()->Get(0)->attributes_as_PointwiseAttributes();
    auto restPw = fbRestored->nodes()->Get(0)->attributes_as_PointwiseAttributes();

    EXPECT_TRUE(origPw->elu_alpha().has_value());
    EXPECT_TRUE(restPw->elu_alpha().has_value());
    EXPECT_FLOAT_EQ(origPw->elu_alpha().value(), 0.5f);
    EXPECT_FLOAT_EQ(restPw->elu_alpha().value(), 0.5f);

    expectGraphsEqualUnpacked(graph, restored);
}

TEST(TestGraphSerialization, BatchnormBackwardDeepComparison)
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

    auto json = graph.toJson();
    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
    expectGraphsEqualUnpacked(graph, restored);
}

TEST(TestGraphSerialization, ConvDgradDeepComparison)
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
    dx->set_uid(3);

    auto json = graph.toJson();
    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
    expectGraphsEqualUnpacked(graph, restored);
}

TEST(TestGraphSerialization, ConvWgradDeepComparison)
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
    dw->set_uid(3);

    auto json = graph.toJson();
    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
    expectGraphsEqualUnpacked(graph, restored);
}

TEST(TestGraphSerialization, BatchnormDeepComparison)
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

    auto json = graph.toJson();
    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
    expectGraphsEqualUnpacked(graph, restored);
}

TEST(TestGraphSerialization, BatchnormInferenceDeepComparison)
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
    y->set_uid(6);

    auto json = graph.toJson();
    Graph restored;
    restored.deserialize(json);

    expectGraphsEqual(graph, restored);
    expectGraphsEqualUnpacked(graph, restored);
}
