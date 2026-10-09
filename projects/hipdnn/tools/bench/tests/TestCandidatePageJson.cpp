// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCandidatePageJson.cpp
 * @brief Covers the JSON a catalog enumeration or sweep prints: the graph it names and the
 *        candidates it lists.
 */

#include <gtest/gtest.h>

#include <hipdnn_bench/CandidatePageJson.hpp>

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

TEST(TestCandidatePageJson, TheGraphIsNamedByItsOwnIdNotTheEnginesHash)
{
    // A corpus manifest and an engine-immediate run name the graph by the id it carries; a
    // sweep naming it by the hash would join to no manifest row, regime or evaluation slice.
    const auto json = pageJson(twoCandidatePage(), "6808882c-5c0a-8e85-8817-bfdd64ff94f2");
    EXPECT_EQ(json.at("graph_id"), "6808882c-5c0a-8e85-8817-bfdd64ff94f2");
}

TEST(TestCandidatePageJson, EveryCandidateKeepsItsIdAndKnobTuple)
{
    const auto json = pageJson(twoCandidatePage(), "graph");
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
