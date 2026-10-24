// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCandidatePageJson.cpp
 * @brief Covers the JSON a catalog enumeration or sweep prints: the graph it names and the
 *        candidates it lists.
 */

#include <gtest/gtest.h>

#include <hipdnn_bench/CandidatePageJson.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <map>
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

TEST(TestCandidatePageJson, ASweepResultIsTheEntryUhdGenReads)
{
    // uhd_gen's test_generate reads the same staged entry through collect_graph, so a key
    // renamed on either side fails one of the two.
    CandidateOutcome outcome;
    outcome.rank = 0;
    outcome.succeeded = true;
    outcome.iterationsRun = 40;
    outcome.numericallyValid = true;
    outcome.validation = "agrees_with_catalog: 2 of 2 cross-checked candidates that wrote "
                         "their output produced it";
    outcome.minTimeMs = 2.0F;
    outcome.avgTimeMs = 2.75F;
    outcome.robustTimeMs = 2.5F;
    outcome.stddevMs = 0.125F;
    outcome.converged = true;

    const auto samplePath = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
                            / HIPDNN_TEST_DATA_RELDIR / HIPDNN_BENCH_CSV_SAMPLE_SUBDIR
                            / "hipdnn_bench_sweep_result.json";
    std::ifstream sample(samplePath);
    ASSERT_TRUE(sample.good()) << samplePath;
    EXPECT_EQ(sweepResultJson("kernel-a",
                              nlohmann::json{{"tile_m", 32}},
                              nlohmann::json{{"kernel.tile_m", 32}},
                              outcome),
              nlohmann::json::parse(sample));
}

TEST(TestCandidatePageJson, AnUntimedSweepResultHasNullTimingsAndItsReason)
{
    CandidateOutcome outcome;
    outcome.succeeded = false;
    outcome.avgTimeMs = 2.75F;
    const auto entry
        = sweepResultJson("kernel-b", nlohmann::json::object(), nlohmann::json::object(), outcome);
    EXPECT_FALSE(entry.at("is_valid").get<bool>());
    EXPECT_TRUE(entry.at("avg_time_ms").is_null());
    EXPECT_EQ(entry.at("skip_reason").get<std::string>().rfind("config_not_applicable", 0), 0U);
}

TEST(TestCandidatePageJson, OnlyASlottedSweepResultCarriesASlotReference)
{
    const CandidateOutcome outcome;
    const auto unslotted
        = sweepResultJson("kernel-c", nlohmann::json::object(), nlohmann::json::object(), outcome);
    EXPECT_FALSE(unslotted.contains("slot_reference"));
    const auto slotted = sweepResultJson(
        "prep=a|dq=b", nlohmann::json::object(), nlohmann::json::object(), outcome, "prep|dq");
    EXPECT_EQ(slotted.at("slot_reference"), "prep|dq");
    EXPECT_EQ(sweepResultJson("prep=a|dq=c",
                              nlohmann::json::object(),
                              nlohmann::json::object(),
                              outcome,
                              std::string())
                  .at("slot_reference"),
              "");
}

namespace
{

// The synthetic 3-slot engine of uhd_gen's tests/fixtures/slotted_sweep.csv: prep has two
// kernels, dkdv three, dq two. Ids are lowercase UUID text, as the runtime prints them.
const std::vector<std::string> PREP{"11111111-0000-4000-8000-000000000001",
                                    "11111111-0000-4000-8000-000000000002"};
const std::vector<std::string> DKDV{"22222222-0000-4000-8000-000000000001",
                                    "22222222-0000-4000-8000-000000000002",
                                    "22222222-0000-4000-8000-000000000003"};
const std::vector<std::string> DQ{"33333333-0000-4000-8000-000000000001",
                                  "33333333-0000-4000-8000-000000000002"};

/// The product, first slot varying slowest, as GenericPlanBuilder enumerates it.
std::vector<std::string> slottedCatalog()
{
    std::vector<std::string> ids;
    for(const auto& prep : PREP)
    {
        for(const auto& dkdv : DKDV)
        {
            for(const auto& dq : DQ)
            {
                std::string id = "prep=";
                id.append(prep).append("|dkdv=").append(dkdv).append("|dq=").append(dq);
                ids.push_back(std::move(id));
            }
        }
    }
    return ids;
}

/// The marks by candidate id, so a test does not depend on catalog order.
std::map<std::string, std::string> marksById(const std::vector<std::string>& ids)
{
    const auto references = slotReferences(ids);
    std::map<std::string, std::string> marks;
    for(size_t index = 0; index < ids.size(); ++index)
    {
        marks[ids[index]] = slotReferenceText(references.marks[index]);
    }
    return marks;
}

} // namespace

TEST(TestCandidatePageJson, ABareKernelIdHasNoSlotsAndAMalformedCompositeThrows)
{
    EXPECT_TRUE(slotKernels(PREP[0]).empty());
    EXPECT_THROW(slotKernels("prep=" + PREP[0] + "|dkdv"), std::runtime_error);
    EXPECT_THROW(slotKernels("prep=" + PREP[0] + "|"), std::runtime_error);
    EXPECT_THROW(slotKernels("=" + PREP[0]), std::runtime_error);
}

TEST(TestCandidatePageJson, TheReferenceProgramBelongsToEverySlotAndAVariedSlotToItself)
{
    const auto ids = slottedCatalog();
    const auto references = slotReferences(ids);
    ASSERT_EQ(references.slots, (std::vector<std::string>{"prep", "dkdv", "dq"}));
    ASSERT_EQ(references.marks.size(), ids.size());

    const auto at = [](size_t prep, size_t dkdv, size_t dq) {
        return "prep=" + PREP[prep] + "|dkdv=" + DKDV[dkdv] + "|dq=" + DQ[dq];
    };
    const auto marks = marksById(ids);
    EXPECT_EQ(ids[references.program], at(0, 0, 0));
    EXPECT_EQ(marks.at(at(0, 0, 0)), "prep|dkdv|dq");
    EXPECT_EQ(marks.at(at(1, 0, 0)), "prep");
    EXPECT_EQ(marks.at(at(0, 1, 0)), "dkdv");
    EXPECT_EQ(marks.at(at(0, 2, 0)), "dkdv");
    EXPECT_EQ(marks.at(at(0, 0, 1)), "dq");
    // Two slots off their picks is no slot's row.
    EXPECT_EQ(marks.at(at(1, 1, 0)), "");
    EXPECT_EQ(marks.at(at(1, 2, 1)), "");
    // 1 + (2-1) + (3-1) + (2-1) rows train a slot; the other 7 of 12 train none.
    const auto trains = [&](const std::string& slot) {
        return std::count_if(marks.begin(), marks.end(), [&](const auto& entry) {
            return ("|" + entry.second + "|").find("|" + slot + "|") != std::string::npos;
        });
    };
    EXPECT_EQ(trains("prep"), 2);
    EXPECT_EQ(trains("dkdv"), 3);
    EXPECT_EQ(trains("dq"), 2);
}

TEST(TestCandidatePageJson, TheReferencePickIsFirstInKernelIdOrderNotCatalogOrder)
{
    auto ids = slottedCatalog();
    const auto forward = marksById(ids);
    std::reverse(ids.begin(), ids.end());
    EXPECT_EQ(marksById(ids), forward);
    EXPECT_EQ(slotReferences(ids).program, ids.size() - 1);
}

TEST(TestCandidatePageJson, AnUnslottedCatalogHasNoSlotReferences)
{
    const auto references = slotReferences({PREP[0], PREP[1]});
    EXPECT_TRUE(references.slots.empty());
    EXPECT_TRUE(references.marks.empty());
    EXPECT_TRUE(slotReferences({}).slots.empty());
    EXPECT_THROW(slotReferences({PREP[0], "prep=" + PREP[0]}), std::runtime_error);
}

TEST(TestCandidatePageJson, ACatalogThatIsNotAProductOfItsSlotsHasNoReferenceProgram)
{
    auto ids = slottedCatalog();
    ids.erase(std::remove(
                  ids.begin(), ids.end(), "prep=" + PREP[0] + "|dkdv=" + DKDV[0] + "|dq=" + DQ[0]),
              ids.end());
    EXPECT_THROW(slotReferences(ids), std::runtime_error);
}

TEST(TestCandidatePageJson, ASlottedPageKeepsTheCompositeIdDottedKnobsAndSlotKernels)
{
    hipdnn_frontend::EngineCandidatePage page = twoCandidatePage();
    page.candidates.clear();
    hipdnn_frontend::EngineCandidate slotted;
    slotted.id = "prep=" + PREP[1] + "|dq=" + DQ[0];
    slotted.variant.knobSettings["prep.tile"] = int64_t{1};
    slotted.variant.knobSettings["dq.tile"] = int64_t{0};
    slotted.kernelFeatures = {{"kernel.prep.tile", 64}, {"kernel.dq.tile", 32}};
    page.candidates.push_back(slotted);
    page.totalCount = 1;

    const auto json = pageJson(page, "key", "graph");
    const auto& candidate = json.at("candidates")[0];
    EXPECT_EQ(candidate.at("id"), slotted.id);
    EXPECT_EQ(candidate.at("knob_settings"), nlohmann::json({{"dq.tile", 0}, {"prep.tile", 1}}));
    EXPECT_EQ(candidate.at("kernel_features").at("kernel.prep.tile"), 64);
    EXPECT_EQ(candidate.at("slot_kernels"),
              nlohmann::json::parse(R"([{"slot":"prep","kernel":")" + PREP[1]
                                    + R"("},{"slot":"dq","kernel":")" + DQ[0] + R"("}])"));
}

namespace
{

/// A slotted catalog page whose candidate N is keyed by knob `tile` = N, with no result yet.
hipdnn_frontend::EngineCandidatePage slottedPage()
{
    hipdnn_frontend::EngineCandidatePage page;
    const auto ids = slottedCatalog();
    for(size_t index = 0; index < ids.size(); ++index)
    {
        hipdnn_frontend::EngineCandidate candidate;
        candidate.id = ids[index];
        candidate.variant.knobSettings["tile"] = static_cast<int64_t>(index);
        page.candidates.push_back(candidate);
    }
    return page;
}

/// The result of running candidate @p index of @p page.
hipdnn_frontend::AutotuneResult resultFor(const hipdnn_frontend::EngineCandidatePage& page,
                                          size_t index,
                                          bool succeeded,
                                          int iterations)
{
    hipdnn_frontend::AutotuneResult result;
    result.knobSettings = {hipdnn_frontend::KnobSetting(
        "tile", std::get<int64_t>(page.candidates[index].variant.knobSettings.at("tile")))};
    result.succeeded = succeeded;
    result.iterationsRun = iterations;
    return result;
}

} // namespace

TEST(TestCandidatePageJson, ASweepIsUsableOnlyWhenItsReferenceProgramTimed)
{
    const auto page = slottedPage();
    const auto references = slotReferences(slottedCatalog());
    const auto other = (references.program + 1) % page.candidates.size();

    // Present, succeeded, timed.
    EXPECT_TRUE(
        referenceProgramTimed({resultFor(page, references.program, true, 5)}, page, references));
    // Absent: only another candidate ran.
    EXPECT_FALSE(referenceProgramTimed({resultFor(page, other, true, 5)}, page, references));
    // Present but failed, or succeeded without a single timed iteration.
    EXPECT_FALSE(
        referenceProgramTimed({resultFor(page, references.program, false, 5)}, page, references));
    EXPECT_FALSE(
        referenceProgramTimed({resultFor(page, references.program, true, 0)}, page, references));
}

TEST(TestCandidatePageJson, AnUnslottedCandidateCarriesNoSlotKernels)
{
    const auto json = pageJson(twoCandidatePage(), "key", "graph");
    EXPECT_FALSE(json.at("candidates")[0].contains("slot_kernels"));
}

} // namespace hipdnn_bench
