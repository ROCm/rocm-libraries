// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <flatbuffers/flatbuffer_builder.h>
#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;

namespace
{

void toJsonAndBackTestSuite(const hipdnn_flatbuffers_sdk::data_objects::Graph* graph,
                            const std::string& context)
{
    const nlohmann::json graphJson = *graph;

    flatbuffers::FlatBufferBuilder builder;
    auto newGraphBuilder = hipdnn_flatbuffers_sdk::json::to<Graph>(builder, graphJson);
    builder.Finish(newGraphBuilder);
    auto newGraph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(builder.GetBufferPointer());

    EXPECT_EQ(graph->compute_data_type(), newGraph->compute_data_type()) << context;
    EXPECT_EQ(graph->intermediate_data_type(), newGraph->intermediate_data_type()) << context;
    EXPECT_EQ(graph->io_data_type(), newGraph->io_data_type()) << context;
    EXPECT_EQ(graph->name()->str(), newGraph->name()->str()) << context;

    ASSERT_EQ(graph->tensors()->size(), newGraph->tensors()->size()) << context;
    auto t1 = graph->tensors()->begin();
    auto t2 = newGraph->tensors()->begin();
    for(; t1 != graph->tensors()->end() && t2 != newGraph->tensors()->end(); t1++, t2++)
    {
        const std::unique_ptr<TensorAttributesT> t1ptr(t1->UnPack());
        const std::unique_ptr<TensorAttributesT> t2ptr(t2->UnPack());
        EXPECT_EQ(*t1ptr, *t2ptr) << context;
    }

    ASSERT_EQ(graph->nodes()->size(), newGraph->nodes()->size()) << context;
    auto n1 = graph->nodes()->begin();
    auto n2 = newGraph->nodes()->begin();
    for(; n1 != graph->nodes()->end() && n2 != newGraph->nodes()->end(); n1++, n2++)
    {
        const std::unique_ptr<NodeT> n1ptr(n1->UnPack());
        const std::unique_ptr<NodeT> n2ptr(n2->UnPack());
        EXPECT_EQ(*n1ptr, *n2ptr) << context;
    }
}

} // namespace

TEST(TestJson, GraphToJsonAndBack)
{
    auto nodeAttributeValues = EnumValuesNodeAttributes();
    auto maxEnumValue
        = static_cast<size_t>(hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MAX);
    for(size_t i = 0; i <= maxEnumValue; i++)
    {
        auto enumValue = nodeAttributeValues[i];
        flatbuffers::FlatBufferBuilder graphBuilder;
        const hipdnn_flatbuffers_sdk::data_objects::Graph* graph = nullptr;
        std::string context;

        switch(enumValue)
        {
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::NONE:
            graphBuilder = hipdnn_test_sdk::utilities::createEmptyValidGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(empty valid graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid batchnorm inference graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
            BatchnormInferenceAttributesVarianceExt:
            graphBuilder
                = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid batchnorm with variance inference graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid batchnorm backward graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid batchnorm forward training graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createPointwiseGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid pointwise graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidConvFwdGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid convolution forward graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid convolution backward graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid convolution weight gradient graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid matmul graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidSdpaFwdGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid sdpa fprop graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidLayernormFpropGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid layernorm graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid rmsnorm graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::RMSNormBackwardAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid rmsnorm backward graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleDequantizeAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidBlockScaleDequantizeGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid block scale dequantize graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BlockScaleQuantizeAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidBlockScaleQuantizeGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid block scale quantize graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaBackwardAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidSdpaBwdGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid sdpa backward graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::CustomOpAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidCustomOpGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid custom op graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ReductionAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidReductionGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid reduction graph)";
            break;
        default:
            FAIL() << "Unhandled NodeAttributes enum value";
            break;
        }

        toJsonAndBackTestSuite(graph, context);
    }
}

namespace
{

void vectorTestSuite(const std::vector<int>& vec, const std::string& context)
{
    nlohmann::json vecJson = vec;
    ASSERT_EQ(vec.size(), vecJson.size()) << context;
    for(size_t i = 0; i < vec.size(); i++)
    {
        EXPECT_EQ(vec[i], vecJson[i].get<int>()) << context;
    }
    EXPECT_EQ(vec, vecJson.get<std::vector<int>>()) << context;
}

template <class T>
void enumTestSuite(T value, const std::string& stringRep, const std::string& context)
{
    auto jsonStringRep = "\"" + stringRep + "\"";
    const nlohmann::json jsonValue = value;
    EXPECT_EQ(value, jsonValue.get<T>()) << context;
    EXPECT_EQ(jsonValue.dump(), std::string{jsonStringRep}) << context;
    EXPECT_EQ(nlohmann::json(stringRep).get<T>(), value) << context;
}

} // namespace

TEST(TestJson, FromVector)
{
    vectorTestSuite({0, 1, 2, 3, 4}, "(filled vector)");
    vectorTestSuite({}, "(empty vector)");
}

TEST(TestJson, Enum)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    enumTestSuite(DataType::FLOAT, "float", "(hipdnn_flatbuffers_sdk::data_objects::DataType)");
    enumTestSuite(NodeAttributes::BatchnormInferenceAttributes,
                  "BatchnormInferenceAttributes",
                  "(for hipdnn_flatbuffers_sdk::data_objects::NodeAttributes)");
}

TEST(TestJson, FlatbufferStringToJson)
{
    flatbuffers::FlatBufferBuilder builder;
    auto strOffset = builder.CreateString("hello_world");
    auto svOffset = CreateStringValue(builder, strOffset);
    builder.Finish(svOffset);

    auto sv = flatbuffers::GetRoot<StringValue>(builder.GetBufferPointer());

    nlohmann::json j;
    flatbuffers::to_json(j, sv->value());
    EXPECT_TRUE(j.is_string());
    EXPECT_EQ(j.get<std::string>(), "hello_world");
}

TEST(TestJson, FlatbufferStringToJsonImplicit)
{
    flatbuffers::FlatBufferBuilder builder;
    auto strOffset = builder.CreateString("implicit_test");
    auto svOffset = CreateStringValue(builder, strOffset);
    builder.Finish(svOffset);

    auto sv = flatbuffers::GetRoot<StringValue>(builder.GetBufferPointer());

    const nlohmann::json j = sv->value();
    EXPECT_TRUE(j.is_string());
    EXPECT_EQ(j.get<std::string>(), "implicit_test");
}

TEST(TestJson, FlatbufferNullStringToJson)
{
    flatbuffers::FlatBufferBuilder builder;
    auto svOffset = CreateStringValue(builder);
    builder.Finish(svOffset);

    auto sv = flatbuffers::GetRoot<StringValue>(builder.GetBufferPointer());

    nlohmann::json j = "should_be_replaced";
    flatbuffers::to_json(j, sv->value());
    EXPECT_TRUE(j.is_string()) << "null flatbuffer string should leave json unchanged";
    EXPECT_EQ(j.get<std::string>(), "should_be_replaced");
}

TEST(TestJson, FlatbufferEmptyStringToJson)
{
    flatbuffers::FlatBufferBuilder builder;
    auto strOffset = builder.CreateString("");
    auto svOffset = CreateStringValue(builder, strOffset);
    builder.Finish(svOffset);

    auto sv = flatbuffers::GetRoot<StringValue>(builder.GetBufferPointer());

    nlohmann::json j;
    flatbuffers::to_json(j, sv->value());
    EXPECT_TRUE(j.is_string());
    EXPECT_EQ(j.get<std::string>(), "");
}

TEST(TestJson, FlatbufferStringVectorToJson)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<flatbuffers::String>> stringOffsets;
    stringOffsets.push_back(builder.CreateString("alpha"));
    stringOffsets.push_back(builder.CreateString("beta"));
    stringOffsets.push_back(builder.CreateString("gamma"));
    auto vecOffset = builder.CreateVector(stringOffsets);

    auto scOffset = CreateStringConstraint(builder, 100, vecOffset);
    builder.Finish(scOffset);

    auto sc = flatbuffers::GetRoot<StringConstraint>(builder.GetBufferPointer());

    nlohmann::json j;
    flatbuffers::to_json(j, sc->valid_values());
    ASSERT_TRUE(j.is_array());
    ASSERT_EQ(j.size(), 3u);
    EXPECT_EQ(j[0].get<std::string>(), "alpha");
    EXPECT_EQ(j[1].get<std::string>(), "beta");
    EXPECT_EQ(j[2].get<std::string>(), "gamma");
}

TEST(TestJson, FlatbufferEmptyStringVectorToJson)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<flatbuffers::Offset<flatbuffers::String>> emptyOffsets;
    auto vecOffset = builder.CreateVector(emptyOffsets);

    auto scOffset = CreateStringConstraint(builder, 0, vecOffset);
    builder.Finish(scOffset);

    auto sc = flatbuffers::GetRoot<StringConstraint>(builder.GetBufferPointer());

    nlohmann::json j;
    flatbuffers::to_json(j, sc->valid_values());
    ASSERT_TRUE(j.is_array());
    EXPECT_EQ(j.size(), 0u);
}

TEST(TestJson, FlatbufferNullStringVectorToJson)
{
    flatbuffers::FlatBufferBuilder builder;
    auto scOffset = CreateStringConstraint(builder);
    builder.Finish(scOffset);

    auto sc = flatbuffers::GetRoot<StringConstraint>(builder.GetBufferPointer());

    nlohmann::json j;
    flatbuffers::to_json(j, sc->valid_values());
    ASSERT_TRUE(j.is_array());
    EXPECT_EQ(j.size(), 0u);
}

TEST(TestJson, TensorAttributesBoolValueRoundTrip)
{
    for(const bool boolVal : {true, false})
    {
        flatbuffers::FlatBufferBuilder builder;
        const std::vector<int64_t> dims = {1};
        const std::vector<int64_t> strides = {1};
        const BoolValue value(boolVal);
        auto valueOffset = builder.CreateStruct(value).Union();
        auto attrOffset = CreateTensorAttributesDirect(builder,
                                                       /*uid*/ 7,
                                                       /*name*/ "boolean_attr",
                                                       DataType::BOOLEAN,
                                                       &strides,
                                                       &dims,
                                                       /*virtual*/ false,
                                                       TensorValue::BoolValue,
                                                       valueOffset);
        builder.Finish(attrOffset);

        auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

        const nlohmann::json attrJson = *attr;

        EXPECT_EQ(attrJson.at("value_type").get<TensorValue>(), TensorValue::BoolValue);
        EXPECT_EQ(attrJson.at("value").get<bool>(), boolVal);

        flatbuffers::FlatBufferBuilder roundTripBuilder;
        auto newAttrOffset
            = hipdnn_flatbuffers_sdk::json::to<TensorAttributes>(roundTripBuilder, attrJson);
        roundTripBuilder.Finish(newAttrOffset);

        auto* newAttr = flatbuffers::GetRoot<TensorAttributes>(roundTripBuilder.GetBufferPointer());

        ASSERT_EQ(newAttr->value_type(), TensorValue::BoolValue);
        ASSERT_NE(newAttr->value_as_BoolValue(), nullptr);
        EXPECT_EQ(newAttr->value_as_BoolValue()->value(), boolVal);
    }
}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
