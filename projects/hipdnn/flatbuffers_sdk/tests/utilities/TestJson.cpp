// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <flatbuffers/flatbuffer_builder.h>
#include <gtest/gtest.h>

#include <array>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/Graph.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/json/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <string>
#include <vector>

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

    const std::unique_ptr<GraphT> graphObject(graph->UnPack());
    const std::unique_ptr<GraphT> newGraphObject(newGraph->UnPack());
    ASSERT_NE(graphObject, nullptr) << context;
    ASSERT_NE(newGraphObject, nullptr) << context;
    EXPECT_EQ(*graphObject, *newGraphObject) << context;
}

} // namespace

TEST(TestJson, GraphUuidRoundTripsAsCanonicalString)
{
    auto builder = hipdnn_test_sdk::utilities::createEmptyValidGraph();
    auto graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(builder.GetBufferPointer());
    const auto id
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("01234567-89AB-4DEF-8123-456789ABCDEF");
    const auto uuid = hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid(id);

    auto graphT = std::unique_ptr<GraphT>(graph->UnPack());
    graphT->id = std::make_unique<Uuid>(uuid);
    flatbuffers::FlatBufferBuilder identifiedBuilder;
    identifiedBuilder.Finish(Graph::Pack(identifiedBuilder, graphT.get()));
    const auto* identifiedGraph = GetGraph(identifiedBuilder.GetBufferPointer());

    const nlohmann::json graphJson = *identifiedGraph;
    ASSERT_TRUE(graphJson.contains("id"));
    EXPECT_EQ(graphJson.at("id"), "01234567-89ab-4def-8123-456789abcdef");

    flatbuffers::FlatBufferBuilder roundTripBuilder;
    roundTripBuilder.Finish(hipdnn_flatbuffers_sdk::json::to<Graph>(roundTripBuilder, graphJson));
    const auto* roundTripped = GetGraph(roundTripBuilder.GetBufferPointer());
    ASSERT_NE(roundTripped->id(), nullptr);
    EXPECT_EQ(hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*roundTripped->id()), id);
}

TEST(TestJson, LegacyGraphOmitsUuidAndStillRoundTrips)
{
    auto builder = hipdnn_test_sdk::utilities::createEmptyValidGraph();
    const auto* graph = GetGraph(builder.GetBufferPointer());
    const nlohmann::json graphJson = *graph;
    EXPECT_FALSE(graphJson.contains("id"));

    flatbuffers::FlatBufferBuilder roundTripBuilder;
    roundTripBuilder.Finish(hipdnn_flatbuffers_sdk::json::to<Graph>(roundTripBuilder, graphJson));
    EXPECT_EQ(GetGraph(roundTripBuilder.GetBufferPointer())->id(), nullptr);
}

TEST(TestJson, RejectsInvalidGraphUuid)
{
    auto builder = hipdnn_test_sdk::utilities::createEmptyValidGraph();
    const auto* graph = GetGraph(builder.GetBufferPointer());
    auto graphJson = nlohmann::json(*graph);

    // A non-string id fails the type check; a malformed string fails parsing.
    graphJson["id"] = 42;
    flatbuffers::FlatBufferBuilder nonStringBuilder;
    EXPECT_THROW((void)hipdnn_flatbuffers_sdk::json::to<Graph>(nonStringBuilder, graphJson),
                 std::runtime_error);

    graphJson["id"] = "not-a-uuid";
    flatbuffers::FlatBufferBuilder malformedBuilder;
    EXPECT_THROW((void)hipdnn_flatbuffers_sdk::json::to<Graph>(malformedBuilder, graphJson),
                 std::invalid_argument);
}

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
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::LayernormBackwardAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidLayernormBwdGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid layernorm backward graph)";
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
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleFwdAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidResampleFwdGraph(true);
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid resample fwd graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ResampleBwdAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidResampleBwdGraph(true);
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid resample bwd graph)";
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MoeGroupedMatmulAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidMoeGroupedMatmulGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid MoE grouped matmul graph)";
            break;
        case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MoeGroupedMatmulBwdAttributes:
            graphBuilder = hipdnn_test_sdk::utilities::createValidMoeGroupedMatmulBwdGraph();
            graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(graphBuilder.GetBufferPointer());
            context = "(valid MoE grouped matmul backward graph)";
            break;
        default:
            FAIL() << "Unhandled NodeAttributes enum value";
            break;
        }

        toJsonAndBackTestSuite(graph, context);
    }
}

TEST(TestJson, GraphVersionSerializesToJson)
{
    flatbuffers::FlatBufferBuilder builder;
    const EngineApiVersion version(1, 2, 0);
    const auto graphOffset = CreateGraphDirect(builder,
                                               "graph",
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               nullptr,
                                               nullptr,
                                               flatbuffers::nullopt,
                                               false,
                                               &version);
    builder.Finish(graphOffset);

    const auto* graph = GetGraph(builder.GetBufferPointer());
    const nlohmann::json graphJson = *graph;

    ASSERT_TRUE(graphJson.contains("min_required_engine_api_version"));
    EXPECT_EQ(graphJson.at("min_required_engine_api_version"),
              nlohmann::json({{"major", 1}, {"minor", 2}, {"patch", 0}}));
}

TEST(TestJson, GraphVersionRoundTripsFromJson)
{
    const nlohmann::json graphJson
        = {{"name", "graph"},
           {"compute_data_type", DataType::FLOAT},
           {"io_data_type", DataType::FLOAT},
           {"intermediate_data_type", DataType::FLOAT},
           {"nodes", nlohmann::json::array()},
           {"tensors", nlohmann::json::array()},
           {"min_required_engine_api_version", {{"major", 1}, {"minor", 2}, {"patch", 0}}}};

    flatbuffers::FlatBufferBuilder builder;
    const auto graphOffset = hipdnn_flatbuffers_sdk::json::to<Graph>(builder, graphJson);
    builder.Finish(graphOffset);

    const auto* graph = GetGraph(builder.GetBufferPointer());
    const auto* version = graph->min_required_engine_api_version();
    ASSERT_NE(version, nullptr);
    EXPECT_EQ(version->major(), 1);
    EXPECT_EQ(version->minor(), 2);
    EXPECT_EQ(version->patch(), 0);

    const nlohmann::json roundTrippedJson = *graph;
    EXPECT_EQ(roundTrippedJson.at("min_required_engine_api_version"),
              graphJson.at("min_required_engine_api_version"));
}

TEST(TestJson, GraphWithoutVersionRoundTripsWithVersionAbsent)
{
    const nlohmann::json graphJson = {{"name", "graph"},
                                      {"compute_data_type", DataType::FLOAT},
                                      {"io_data_type", DataType::FLOAT},
                                      {"intermediate_data_type", DataType::FLOAT},
                                      {"nodes", nlohmann::json::array()},
                                      {"tensors", nlohmann::json::array()}};

    flatbuffers::FlatBufferBuilder builder;
    const auto graphOffset = hipdnn_flatbuffers_sdk::json::to<Graph>(builder, graphJson);
    builder.Finish(graphOffset);

    const auto* graph = GetGraph(builder.GetBufferPointer());
    EXPECT_EQ(graph->min_required_engine_api_version(), nullptr);

    const nlohmann::json roundTrippedJson = *graph;
    EXPECT_FALSE(roundTrippedJson.contains("min_required_engine_api_version"));
}
TEST(TestJson, MoeGroupedMatmulDefaultsRoundTrip)
{
    auto graphBuilder
        = hipdnn_test_sdk::utilities::createValidMoeGroupedMatmulGraph(MoeGroupedMatmulMode::NONE);
    const auto* graph = GetGraph(graphBuilder.GetBufferPointer());
    const nlohmann::json graphJson = *graph;
    const auto& nodeJson = graphJson.at("nodes").at(0);
    const auto& inputs = nodeJson.at("inputs");

    EXPECT_EQ(nodeJson.at("type").get<std::string>(), "MoeGroupedMatmulAttributes");
    EXPECT_EQ(nodeJson.at("mode").get<std::string>(), "none");
    EXPECT_EQ(nodeJson.at("top_k").get<int32_t>(), 0);
    EXPECT_FALSE(inputs.contains("token_index_tensor_uid"));
    EXPECT_FALSE(inputs.contains("token_ks_tensor_uid"));

    toJsonAndBackTestSuite(graph, "(MoE grouped matmul defaults)");
}

TEST(TestJson, MoeGroupedMatmulBwdRoundTrip)
{
    auto graphBuilder = hipdnn_test_sdk::utilities::createValidMoeGroupedMatmulBwdGraph();
    const auto* graph = GetGraph(graphBuilder.GetBufferPointer());
    const nlohmann::json graphJson = *graph;
    const auto& nodeJson = graphJson.at("nodes").at(0);
    const auto& inputs = nodeJson.at("inputs");
    const auto& outputs = nodeJson.at("outputs");

    EXPECT_EQ(nodeJson.at("type").get<std::string>(), "MoeGroupedMatmulBwdAttributes");
    EXPECT_EQ(inputs.at("doutput_tensor_uid").get<int64_t>(), 1);
    EXPECT_EQ(inputs.at("token_tensor_uid").get<int64_t>(), 2);
    EXPECT_EQ(inputs.at("first_token_offset_tensor_uid").get<int64_t>(), 3);
    EXPECT_EQ(outputs.at("dweight_tensor_uid").get<int64_t>(), 4);

    toJsonAndBackTestSuite(graph, "(MoE grouped matmul backward)");
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

TEST(TestJson, TensorAttributesRaggedOffsetAndAlignmentRoundTrip)
{
    const int64_t uid = 5;
    const int64_t raggedOffsetUid = 42;
    const int64_t alignment = 128;
    const std::vector<int64_t> dims = {4, 8, 1, 1};
    const std::vector<int64_t> strides = {8, 1, 1, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = CreateTensorAttributesDirect(builder,
                                                   uid,
                                                   "ragged_primary",
                                                   DataType::FLOAT,
                                                   &strides,
                                                   &dims,
                                                   /*virtual*/ false,
                                                   TensorValue::NONE,
                                                   /*value*/ 0,
                                                   false,
                                                   flatbuffers::Optional<int64_t>(raggedOffsetUid),
                                                   alignment);
    builder.Finish(attrOffset);

    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    // Verify fields before JSON round-trip
    ASSERT_TRUE(attr->ragged_offset_tensor_uid().has_value());
    EXPECT_EQ(attr->ragged_offset_tensor_uid().value(), raggedOffsetUid);
    EXPECT_EQ(attr->alignment(), alignment);

    // JSON round-trip
    const nlohmann::json attrJson = *attr;
    EXPECT_EQ(attrJson.at("ragged_offset_tensor_uid").get<int64_t>(), raggedOffsetUid);
    EXPECT_EQ(attrJson.at("alignment").get<int64_t>(), alignment);

    flatbuffers::FlatBufferBuilder roundTripBuilder;
    auto newAttrOffset
        = hipdnn_flatbuffers_sdk::json::to<TensorAttributes>(roundTripBuilder, attrJson);
    roundTripBuilder.Finish(newAttrOffset);

    auto* newAttr = flatbuffers::GetRoot<TensorAttributes>(roundTripBuilder.GetBufferPointer());
    ASSERT_TRUE(newAttr->ragged_offset_tensor_uid().has_value());
    EXPECT_EQ(newAttr->ragged_offset_tensor_uid().value(), raggedOffsetUid);
    EXPECT_EQ(newAttr->alignment(), alignment);
}

TEST(TestJson, TensorAttributesDefaultAlignmentAndNoRaggedOffset)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const std::vector<int64_t> strides = {1, 1, 1, 1};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = CreateTensorAttributesDirect(
        builder, 1, "plain", DataType::FLOAT, &strides, &dims, false);
    builder.Finish(attrOffset);

    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    EXPECT_FALSE(attr->ragged_offset_tensor_uid().has_value());
    EXPECT_EQ(attr->alignment(), 16);

    const nlohmann::json attrJson = *attr;
    EXPECT_FALSE(attrJson.contains("ragged_offset_tensor_uid"));
    EXPECT_EQ(attrJson.at("alignment").get<int64_t>(), 16);
}

TEST(TestJson, TensorAttributesLegacyJsonMissingAlignmentDefaultsTo16)
{
    // Simulate legacy JSON that lacks the alignment and ragged_offset_tensor_uid fields
    const nlohmann::json legacyJson = {{"uid", 1},
                                       {"name", "legacy"},
                                       {"data_type", DataType::FLOAT},
                                       {"dims", std::vector<int64_t>{1, 1, 1, 1}},
                                       {"strides", std::vector<int64_t>{1, 1, 1, 1}},
                                       {"virtual", false}};

    flatbuffers::FlatBufferBuilder builder;
    auto attrOffset = hipdnn_flatbuffers_sdk::json::to<TensorAttributes>(builder, legacyJson);
    builder.Finish(attrOffset);

    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    EXPECT_FALSE(attr->ragged_offset_tensor_uid().has_value());
    EXPECT_EQ(attr->alignment(), 16);
}

TEST(TestJson, TensorAttributesRuntimePassByValueRoundTrips)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1};
    const std::vector<int64_t> strides = {1};
    const auto attrOffset = CreateTensorAttributesDirect(builder,
                                                         /*uid*/ 7,
                                                         /*name*/ "runtime_scalar",
                                                         DataType::FLOAT,
                                                         &strides,
                                                         &dims,
                                                         /*virtual*/ false,
                                                         TensorValue::NONE,
                                                         0,
                                                         /*is_runtime_pass_by_value*/ true);
    builder.Finish(attrOffset);

    const auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    const nlohmann::json attrJson = *attr;
    ASSERT_TRUE(attrJson.at("is_runtime_pass_by_value").get<bool>());

    flatbuffers::FlatBufferBuilder roundTripBuilder;
    const auto newAttrOffset
        = hipdnn_flatbuffers_sdk::json::to<TensorAttributes>(roundTripBuilder, attrJson);
    roundTripBuilder.Finish(newAttrOffset);

    const auto* newAttr
        = flatbuffers::GetRoot<TensorAttributes>(roundTripBuilder.GetBufferPointer());
    EXPECT_TRUE(newAttr->is_runtime_pass_by_value());
}

TEST(TestJson, TensorAttributesWithoutRuntimePassByValueDefaultsFalse)
{
    const nlohmann::json attrJson = {{"uid", 7},
                                     {"name", "legacy_tensor"},
                                     {"data_type", DataType::FLOAT},
                                     {"strides", {1}},
                                     {"dims", {1}},
                                     {"virtual", false}};

    flatbuffers::FlatBufferBuilder builder;
    const auto attrOffset = hipdnn_flatbuffers_sdk::json::to<TensorAttributes>(builder, attrJson);
    builder.Finish(attrOffset);

    const auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    EXPECT_FALSE(attr->is_runtime_pass_by_value());
}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
