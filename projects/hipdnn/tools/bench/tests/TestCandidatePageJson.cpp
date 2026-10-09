// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCandidatePageJson.cpp
 * @brief Covers the JSON a catalog enumeration or sweep prints: the graph it names and the
 *        candidates it lists.
 */

#include <gtest/gtest.h>

#include <hipdnn_bench/CandidatePageJson.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace hipdnn_bench
{
namespace
{

hipdnn_frontend::EngineCandidatePage twoCandidatePage()
{
    hipdnn_frontend::EngineCandidatePage page;
    page.engineId = 7;
    page.engineName = "hipkernel:Example";
    page.engineDescriptorId = "4e06e981-aaf2-4fe1-836d-09c221009a0c";
    // The engine's structural hash of the graph, as GenericPlanBuilder fills it.
    page.graphId = "4872434071279366875";
    page.deviceId = "device-0";
    page.deviceArch = "gfx942:sramecc+:xnack-";
    page.totalCount = 2;
    hipdnn_frontend::EngineCandidate small;
    small.id = "kernel-64";
    small.variant.knobSettings["block_size"] = int64_t{64};
    hipdnn_frontend::EngineCandidate large;
    large.id = "kernel-256";
    large.variant.knobSettings["block_size"] = int64_t{256};
    page.candidates = {small, large};
    return page;
}

} // namespace

TEST(TestCandidatePageJson, TheGraphIsNamedByTheKeyAndIdGivenNotThePagesOwn)
{
    const auto json = pageJson(twoCandidatePage(), "1729", "6808882c-5c0a-8e85-8817-bfdd64ff94f2");
    EXPECT_EQ(json.at("graph_key"), "1729");
    EXPECT_EQ(json.at("graph_id"), "6808882c-5c0a-8e85-8817-bfdd64ff94f2");
}

namespace
{

/// One tensor in, one out, under @p name and @p id: enough content to key.
std::vector<uint8_t> tinyGraph(const std::string& name, uint8_t id, int64_t width)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    fb::GraphT graph;
    graph.name = name;
    std::array<uint8_t, 16> bytes{};
    bytes[0] = id;
    graph.id = std::make_unique<fb::Uuid>(flatbuffers::span<const uint8_t, 16>(bytes));
    auto tensor = std::make_unique<fb::TensorAttributesT>();
    tensor->uid = 7;
    tensor->dims = {1, width};
    tensor->strides = {width, 1};
    graph.tensors.push_back(std::move(tensor));
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(fb::Graph::Pack(builder, &graph));
    return {builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize()};
}

} // namespace

TEST(TestCandidatePageJson, AGraphsContentKeyIgnoresItsNameAndId)
{
    const auto left = tinyGraph("left", 1, 8);
    const auto right = tinyGraph("right", 2, 8);
    EXPECT_EQ(graphContentKey(left.data(), left.size()),
              graphContentKey(right.data(), right.size()));

    const auto wider = tinyGraph("left", 1, 16);
    EXPECT_NE(graphContentKey(left.data(), left.size()),
              graphContentKey(wider.data(), wider.size()));
}

TEST(TestCandidatePageJson, TheContentKeyIsTheIngestorsGraphContentKeyInDecimal)
{
    // The kernel ingestor names a catalog page by the same value, so one graph has one key.
    const auto graph = tinyGraph("g", 1, 8);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapped(graph.data(),
                                                                             graph.size());
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey key{wrapped};
    EXPECT_EQ(graphContentKey(graph.data(), graph.size()), std::to_string(key.hash()));
}

TEST(TestCandidatePageJson, BytesThatAreNotAGraphHaveNoContentKey)
{
    const std::vector<uint8_t> noise{1, 2, 3, 4};
    EXPECT_THROW(graphContentKey(noise.data(), noise.size()), std::exception);
}

TEST(TestCandidatePageJson, EveryCandidateKeepsItsIdAndKnobTuple)
{
    const auto json = pageJson(twoCandidatePage(), "key", "graph");
    ASSERT_EQ(json.at("candidates").size(), 2U);
    EXPECT_EQ(json.at("candidates")[0].at("id"), "kernel-64");
    EXPECT_EQ(json.at("candidates")[0].at("knob_settings"), nlohmann::json({{"block_size", 64}}));
    EXPECT_EQ(json.at("candidates")[1].at("knob_settings"), nlohmann::json({{"block_size", 256}}));
    EXPECT_EQ(json.at("engine_descriptor_id"), "4e06e981-aaf2-4fe1-836d-09c221009a0c");
    EXPECT_EQ(json.at("total_count"), 2U);
    EXPECT_TRUE(json.at("next_offset").is_null());
}

TEST(TestCandidatePageJson, AKnobTupleOfEveryValueTypeReadsBack)
{
    const std::map<hipdnn_frontend::KnobType_t, hipdnn_frontend::KnobValueVariant> knobs{
        {"block_size", int64_t{128}}, {"scale", 0.5}, {"layout", std::string("nhwc")}};
    EXPECT_EQ(knobJson(knobs),
              nlohmann::json({{"block_size", 128}, {"layout", "nhwc"}, {"scale", 0.5}}));
}

} // namespace hipdnn_bench
