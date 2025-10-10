// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffer_builder.h>
#include <gtest/gtest.h>

#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/json/Graph.hpp>

using namespace hipdnn_sdk::data_objects;

void toJsonAndBackTestSuite(const hipdnn_sdk::data_objects::Graph* graph,
                            const std::string& context)
{
    nlohmann::json graphJson = *graph;

    flatbuffers::FlatBufferBuilder builder;
    auto newGraphBuilder = hipdnn_sdk::json::to<Graph>(builder, graphJson);
    builder.Finish(newGraphBuilder);
    auto newGraph = hipdnn_sdk::data_objects::GetGraph(builder.GetBufferPointer());

    EXPECT_EQ(graph->compute_type(), newGraph->compute_type()) << context;
    EXPECT_EQ(graph->io_type(), newGraph->io_type()) << context;
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
    {
        auto graphBuilder = hipdnn_sdk::test_utilities::createEmptyValidGraph();
        auto graph = hipdnn_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());

        toJsonAndBackTestSuite(graph, "(empty valid graph)");
    }
    {
        auto graphBuilder = hipdnn_sdk::test_utilities::createValidBatchnormInferenceGraph();
        auto graph = hipdnn_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());

        toJsonAndBackTestSuite(graph, "(valid batchnorm inference graph)");
    }
    {
        auto graphBuilder = hipdnn_sdk::test_utilities::createValidBatchnormBwdGraph();
        auto graph = hipdnn_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());

        toJsonAndBackTestSuite(graph, "(valid batchnorm backward graph)");
    }
    {
        auto graphBuilder = hipdnn_sdk::test_utilities::createBatchnormGraph();
        auto graph = hipdnn_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());

        toJsonAndBackTestSuite(graph, "(valid batchnorm backward graph)");
    }
    {
        auto graphBuilder = hipdnn_sdk::test_utilities::createPointwiseGraph();
        auto graph = hipdnn_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());

        toJsonAndBackTestSuite(graph, "(valid pointwise graph)");
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
    using namespace hipdnn_sdk::data_objects;

    enumTestSuite(DataType::FLOAT, "float", "(hipdnn_sdk::data_objects::DataType)");
    enumTestSuite(NodeAttributes::BatchnormInferenceAttributes,
                  "BatchnormInferenceAttributes",
                  "(for hipdnn_sdk::data_objects::NodeAttributes)");
}
