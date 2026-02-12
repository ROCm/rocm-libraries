// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffer_builder.h>
#include <gtest/gtest.h>

#include <hipdnn_data_sdk/data_objects/data_types_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/json/Graph.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_data_sdk::data_objects;

void toJsonAndBackTestSuite(const hipdnn_data_sdk::data_objects::Graph* graph,
                            const std::string& context)
{
    nlohmann::json graphJson = *graph;

    flatbuffers::FlatBufferBuilder builder;
    auto newGraphBuilder = hipdnn_data_sdk::json::to<Graph>(builder, graphJson);
    builder.Finish(newGraphBuilder);
    auto newGraph = hipdnn_data_sdk::data_objects::GetGraph(builder.GetBufferPointer());

    EXPECT_EQ(graph->compute_data_type(), newGraph->compute_data_type()) << context;
    EXPECT_EQ(graph->intermediate_data_type(), newGraph->intermediate_data_type()) << context;
    EXPECT_EQ(graph->io_data_type(), newGraph->io_data_type()) << context;
    EXPECT_EQ(graph->name()->str(), newGraph->name()->str()) << context;

    ASSERT_EQ(graph->tensors()->size(), newGraph->tensors()->size()) << context;
    auto t1 = graph->tensors()->begin();
    auto t2 = newGraph->tensors()->begin();
    for(; t1 != graph->tensors()->end() && t2 != newGraph->tensors()->end(); t1++, t2++)
    {
        std::unique_ptr<TensorAttributesT> t1ptr(t1->UnPack());
        std::unique_ptr<TensorAttributesT> t2ptr(t2->UnPack());
        EXPECT_EQ(*t1ptr, *t2ptr) << context;
    }

    ASSERT_EQ(graph->nodes()->size(), newGraph->nodes()->size()) << context;
    auto n1 = graph->nodes()->begin();
    auto n2 = newGraph->nodes()->begin();
    for(; n1 != graph->nodes()->end() && n2 != newGraph->nodes()->end(); n1++, n2++)
    {
        std::unique_ptr<NodeT> n1ptr(n1->UnPack());
        std::unique_ptr<NodeT> n2ptr(n2->UnPack());
        EXPECT_EQ(*n1ptr, *n2ptr) << context;
    }
}

TEST(TestJson, GraphToJsonAndBack)
{
    auto nodeAttributeValues = EnumValuesNodeAttributes();
    auto maxEnumValue = static_cast<size_t>(hipdnn_data_sdk::data_objects::NodeAttributes::MAX);
    for(size_t i = 0; i <= maxEnumValue; i++)
    {
        auto enumValue = nodeAttributeValues[i];
        flatbuffers::FlatBufferBuilder graphBuilder;
        const hipdnn_data_sdk::data_objects::Graph* graph = nullptr;
        std::string context;

        switch(enumValue)
        {
        case hipdnn_data_sdk::data_objects::NodeAttributes::NONE:
            graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();
            graph = hipdnn_data_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(empty valid graph)";
            break;
        case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
            graph = hipdnn_data_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid batchnorm inference graph)";
            break;
        case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt:
            graphBuilder
                = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
            graph = hipdnn_data_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid batchnorm with variance inference graph)";
            break;
        case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
            graph = hipdnn_data_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid batchnorm backward graph)";
            break;
        case hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
            graph = hipdnn_data_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid batchnorm forward training graph)";
            break;
        case hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createPointwiseGraph();
            graph = hipdnn_data_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid pointwise graph)";
            break;
        case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
            graph = hipdnn_data_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid convolution forward graph)";
            break;
        case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
            graph = hipdnn_data_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid convolution backward graph)";
            break;
        case hipdnn_data_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
            graph = hipdnn_data_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid convolution weight gradient graph)";
            break;
        case hipdnn_data_sdk::data_objects::NodeAttributes::MatmulAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
            graph = hipdnn_data_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid matmul graph)";
            break;
        default:
            FAIL() << "Unhandled NodeAttributes enum value";
            break;
        }

        toJsonAndBackTestSuite(graph, context);
    }
}

void vectorTestSuite(std::vector<int> const& vec, const std::string& context)
{
    nlohmann::json vecJson = vec;
    ASSERT_EQ(vec.size(), vecJson.size()) << context;
    for(size_t i = 0; i < vec.size(); i++)
    {
        EXPECT_EQ(vec[i], vecJson[i].get<int>()) << context;
    }
    EXPECT_EQ(vec, vecJson.get<std::vector<int>>()) << context;
}

TEST(TestJson, FromVector)
{
    vectorTestSuite({0, 1, 2, 3, 4}, "(filled vector)");
    vectorTestSuite({}, "(empty vector)");
}

template <class T>
void enumTestSuite(T value, const std::string& stringRep, const std::string& context)
{
    auto jsonStringRep = "\"" + stringRep + "\"";
    nlohmann::json jsonValue = value;
    EXPECT_EQ(value, jsonValue.get<T>()) << context;
    EXPECT_EQ(jsonValue.dump(), std::string{jsonStringRep}) << context;
    EXPECT_EQ(nlohmann::json(stringRep).get<T>(), value) << context;
}

TEST(TestJson, Enum)
{
    using namespace hipdnn_data_sdk::data_objects;

    enumTestSuite(DataType::FLOAT, "float", "(hipdnn_data_sdk::data_objects::DataType)");
    enumTestSuite(NodeAttributes::BatchnormInferenceAttributes,
                  "BatchnormInferenceAttributes",
                  "(for hipdnn_data_sdk::data_objects::NodeAttributes)");
}
