// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The v8 core's engine views, plans, notes and variant packs, called directly
// against the mock hipDNN backend: knob-choice validation, the knob config
// build, heuristics settings, plan build/load, tag and JSON, raw-config notes,
// and variant-pack checks.
#include <hipdnn_compatibility/cudnn/detail/v8/engine_views.h>
#include <hipdnn_compatibility/cudnn/detail/v8/notes.h>
#include <hipdnn_compatibility/cudnn/detail/v8/plan.h>
#include <hipdnn_compatibility/cudnn/detail/v8/variant_pack.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"
#include "fake_backend/ShimV8TestSupport.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace v8 = hipdnn_frontend::compatibility::cudnn_frontend::detail::v8;

using hipdnn_shim_test::CapturedSetAttribute;
using hipdnn_shim_test::shimErrorOf;

constexpr std::string_view CHOICES_LABEL = "knobChoices";
constexpr std::string_view JSON_LABEL = "planJson";
constexpr v8::VariantPackLabels PACK_LABELS{"dataPointers", "uniqueIds"};

class TestCudnnShimV8CorePlans : public hipdnn_shim_test::ShimV8MockBackendFixture
{
protected:
    // Ranked {7, -2} gives the dense index {-2, 7}.
    static constexpr std::int64_t ENGINE_ID = 7;
    static constexpr std::int64_t ENGINE_INDEX = 1;
    static constexpr std::int64_t OTHER_ID = -2;
    static constexpr std::int64_t WORKSPACE_SIZE = 4096;

    void SetUp() override
    {
        ShimV8MockBackendFixture::SetUp();
        installHeuristicResults({ENGINE_ID, OTHER_ID});
        installEngineKnobs(
            ENGINE_ID,
            {{"prov.tile_size", 8, 0, 64, 8, {}}, {"prov.split_k_slc", 1, -1, 8, 1, {}}});
        installPlanWorkspace(WORKSPACE_SIZE);
    }

    std::shared_ptr<const v8::GraphCore> graph()
    {
        return std::make_shared<const v8::GraphCore>(
            hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor(
                newMockDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)),
            "");
    }

    std::shared_ptr<const v8::EngineRecord> engine()
    {
        return v8::makeEngineRecord(graph(), ENGINE_INDEX, "index");
    }
};

TEST_F(TestCudnnShimV8CorePlans, EngineIndexMapsToRankedId)
{
    const std::shared_ptr<const v8::GraphCore> core = graph();
    const std::shared_ptr<const v8::EngineRecord> record
        = v8::makeEngineRecord(core, ENGINE_INDEX, "index");

    EXPECT_EQ(record->kind(), CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    EXPECT_EQ(record->id(), ENGINE_ID);
    EXPECT_EQ(record->index(), ENGINE_INDEX);
    EXPECT_FALSE(record->isLoaded());
    EXPECT_EQ(record->hipdnnEngine(), core->defaultEngine(ENGINE_ID));
    EXPECT_EQ(&record->graph(), core.get());
}

TEST_F(TestCudnnShimV8CorePlans, EngineIndexOutsideSpaceNamesLabel)
{
    const v8::ShimError error = shimErrorOf([&] { v8::makeEngineRecord(graph(), 2, "theIndex"); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_STREQ(error.what(), "theIndex = 2 is outside [0, 2)");

    const v8::ShimError negative
        = shimErrorOf([&] { v8::makeEngineRecord(graph(), -1, "theIndex"); });
    EXPECT_EQ(negative.status(), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8CorePlans, RankedRecordsKeepRankOrder)
{
    const std::vector<std::shared_ptr<const v8::EngineRecord>> ranked
        = v8::rankedEngineRecords(graph());
    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked[0]->id(), ENGINE_ID);
    EXPECT_EQ(ranked[0]->index(), ENGINE_INDEX);
    EXPECT_EQ(ranked[1]->id(), OTHER_ID);
    EXPECT_EQ(ranked[1]->index(), 0);
}

TEST_F(TestCudnnShimV8CorePlans, KnobMinusOneIsUnsetUnlessAccepted)
{
    const std::shared_ptr<const v8::EngineRecord> record = engine();
    // TILE_SIZE rejects -1, so -1 means "unset"; SPLIT_K_SLC accepts -1.
    const std::shared_ptr<const v8::ConfigRecord> config
        = v8::makeConfigRecord(record,
                               {{CUDNN_KNOB_TYPE_TILE_SIZE, -1}, {CUDNN_KNOB_TYPE_SPLIT_K_SLC, -1}},
                               CHOICES_LABEL);
    EXPECT_EQ(config->knobChoices(), (v8::KnobChoices{{CUDNN_KNOB_TYPE_SPLIT_K_SLC, -1}}));

    const std::shared_ptr<const v8::ConfigRecord> unset
        = v8::makeConfigRecord(record, {{CUDNN_KNOB_TYPE_TILE_SIZE, -1}}, CHOICES_LABEL);
    EXPECT_TRUE(unset->knobChoices().empty());
    // No settings: the graph's shared default config, no new ENGINECFG.
    const std::size_t configs = created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size();
    EXPECT_EQ(unset->hipdnnEngineConfig(), record->graph().defaultEngineConfig(ENGINE_ID));
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), configs);
}

TEST_F(TestCudnnShimV8CorePlans, KnobChosenTwiceIsBadParam)
{
    const v8::ShimError error = shimErrorOf([&] {
        v8::makeConfigRecord(engine(),
                             {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}, {CUDNN_KNOB_TYPE_TILE_SIZE, 24}},
                             CHOICES_LABEL);
    });
    EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_STREQ(error.what(), "knobChoices: CUDNN_KNOB_TYPE_TILE_SIZE is chosen more than once");
}

TEST_F(TestCudnnShimV8CorePlans, KnobNotExposedIsNotSupported)
{
    const v8::ShimError error = shimErrorOf(
        [&] { v8::makeConfigRecord(engine(), {{CUDNN_KNOB_TYPE_BLOCK_SIZE, 8}}, CHOICES_LABEL); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_STREQ(error.what(),
                 "knobChoices: CUDNN_KNOB_TYPE_BLOCK_SIZE is not exposed by engine 7");
}

TEST_F(TestCudnnShimV8CorePlans, KnobOffStrideIsBadParam)
{
    const v8::ShimError error = shimErrorOf(
        [&] { v8::makeConfigRecord(engine(), {{CUDNN_KNOB_TYPE_TILE_SIZE, 12}}, CHOICES_LABEL); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(std::string(error.what()).rfind("knobChoices: CUDNN_KNOB_TYPE_TILE_SIZE = 12: ", 0),
              0U)
        << error.what();
}

TEST_F(TestCudnnShimV8CorePlans, KnobConfigBuiltOnceOnFirstUse)
{
    const std::shared_ptr<const v8::EngineRecord> record = engine();
    const std::size_t configs = created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size();
    const std::shared_ptr<const v8::ConfigRecord> config
        = v8::makeConfigRecord(record, {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}, CHOICES_LABEL);
    // Validation alone builds nothing.
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), configs);

    hipdnnBackendDescriptor_t built = config->hipdnnEngineConfig();
    ASSERT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), configs + 1);
    EXPECT_EQ(built, created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).back());
    const std::vector<CapturedSetAttribute> engineCalls
        = callsOn(built, HIPDNN_ATTR_ENGINECFG_ENGINE);
    ASSERT_EQ(engineCalls.size(), 1U);
    EXPECT_EQ(engineCalls.front().value<hipdnnBackendDescriptor_t>(), record->hipdnnEngine());
    EXPECT_EQ(callsOn(built, HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES).size(), 1U);
    EXPECT_TRUE(setCountAtFinalize(built).has_value());

    EXPECT_EQ(config->hipdnnEngineConfig(), built);
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), configs + 1);
}

TEST_F(TestCudnnShimV8CorePlans, LoadedConfigHasNoGraph)
{
    const v8::KnobChoices requested
        = {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}, {CUDNN_KNOB_TYPE_SPLIT_K_SLC, -1}};
    const std::shared_ptr<const v8::ConfigRecord> loaded
        = v8::makeLoadedConfigRecord(-31,
                                     2,
                                     {CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION},
                                     {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}},
                                     requested,
                                     "x");
    EXPECT_TRUE(loaded->isLoaded());
    EXPECT_EQ(loaded->engineId(), -31);
    EXPECT_EQ(loaded->engineIndex(), 2);
    EXPECT_EQ(loaded->engine().name(), std::optional<std::string>("x"));
    EXPECT_TRUE(loaded->engine().knobs().empty());
    EXPECT_EQ(loaded->knobChoices(), (v8::KnobChoices{{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}));
    EXPECT_EQ(loaded->requestedChoices(), requested);
    EXPECT_EQ(shimErrorOf([&] { loaded->hipdnnEngineConfig(); }).status(),
              CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(shimErrorOf([&] { loaded->engine().graph(); }).status(), CUDNN_STATUS_NOT_SUPPORTED);

    const v8::ShimError reuse
        = shimErrorOf([&] { v8::makeConfigRecord(loaded->engineRecord(), {}, CHOICES_LABEL); });
    EXPECT_EQ(reuse.status(), CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestCudnnShimV8CorePlans, RecordAsChecksKind)
{
    const std::shared_ptr<const v8::ConfigRecord> config
        = v8::makeConfigRecord(engine(), {}, CHOICES_LABEL);
    const std::shared_ptr<const v8::Record> base = config;
    EXPECT_EQ(v8::recordAs<v8::ConfigRecord>(base), config);
    EXPECT_EQ(shimErrorOf([&] { v8::recordAs<v8::PlanRecord>(*base); }).status(),
              CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8CorePlans, PlanBuildReadsWorkspace)
{
    const std::shared_ptr<const v8::ConfigRecord> config
        = v8::makeConfigRecord(engine(), {}, CHOICES_LABEL);
    const std::shared_ptr<const v8::PlanRecord> plan = v8::buildPlan(config);

    EXPECT_EQ(plan->kind(), CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    EXPECT_EQ(plan->workspaceSize(), WORKSPACE_SIZE);
    EXPECT_EQ(plan->configRecord(), config);
    ASSERT_EQ(created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).size(), 1U);
    EXPECT_EQ(plan->hipdnnPlan(), created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).front());
    const std::vector<CapturedSetAttribute> configCalls
        = callsOn(plan->hipdnnPlan(), HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
    ASSERT_EQ(configCalls.size(), 1U);
    EXPECT_EQ(configCalls.front().value<hipdnnBackendDescriptor_t>(), config->hipdnnEngineConfig());
    EXPECT_TRUE(setCountAtFinalize(plan->hipdnnPlan()).has_value());
}

TEST_F(TestCudnnShimV8CorePlans, PlanJsonRoundTrip)
{
    const std::vector<std::uint8_t> payload = {0x00, 0xFF, 0x10, 0x80, 0x7F};
    installSerializedPlan(payload);
    installEngineName(ENGINE_ID, "mock_engine_seven");
    installBehaviorNotes(ENGINE_ID, {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    const std::shared_ptr<const v8::ConfigRecord> config
        = v8::makeConfigRecord(engine(), {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}, CHOICES_LABEL);
    const std::shared_ptr<const v8::PlanRecord> built = v8::buildPlan(config);

    const std::string json = built->json();
    // Serialized once: a later payload never reaches the built plan's JSON.
    installSerializedPlan({0x01});
    EXPECT_EQ(built->json(), json);
    installSerializedPlan(payload);
    const std::optional<v8::plan_json::Plan> parsed = v8::plan_json::read(json, "json");
    ASSERT_TRUE(parsed.has_value()) << json;
    EXPECT_EQ(parsed->id, ENGINE_ID);
    EXPECT_EQ(parsed->globalIndex, ENGINE_INDEX);
    EXPECT_EQ(parsed->name, std::optional<std::string>("mock_engine_seven"));
    EXPECT_EQ(parsed->behaviorNotes,
              std::vector<cudnnBackendBehaviorNote_t>{CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    EXPECT_EQ(parsed->knobs, (v8::KnobChoices{{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}));
    // Every requested choice was applied, so the JSON does not repeat them.
    EXPECT_FALSE(parsed->requestedKnobs.has_value());
    EXPECT_EQ(json.find("requestedKnobs"), std::string::npos) << json;
    EXPECT_EQ(parsed->payload, payload);

    const std::shared_ptr<const v8::PlanRecord> loaded = v8::loadPlan(_handle, json, JSON_LABEL);
    EXPECT_EQ(loaded->workspaceSize(), WORKSPACE_SIZE);
    EXPECT_TRUE(loaded->config().isLoaded());
    EXPECT_EQ(loaded->config().engineId(), ENGINE_ID);
    EXPECT_EQ(loaded->config().engineIndex(), ENGINE_INDEX);
    EXPECT_EQ(loaded->config().knobChoices(), config->knobChoices());
    EXPECT_EQ(loaded->config().requestedChoices(), config->requestedChoices());
    EXPECT_EQ(loaded->config().engine().name(), parsed->name);
    EXPECT_EQ(loaded->config().engine().behaviorNotes(), parsed->behaviorNotes);
    EXPECT_EQ(loaded->hipdnnPlan(), created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).back());
    // A restored plan writes the same JSON.
    EXPECT_EQ(loaded->json(), json);
}

TEST_F(TestCudnnShimV8CorePlans, PlanJsonIdMismatchIsBadParam)
{
    installSerializedPlan({0x01, 0x02, 0x03});
    const std::shared_ptr<const v8::PlanRecord> built
        = v8::buildPlan(v8::makeConfigRecord(engine(), {}, CHOICES_LABEL));
    std::optional<v8::plan_json::Plan> parsed = v8::plan_json::read(built->json(), "json");
    ASSERT_TRUE(parsed.has_value());
    parsed->id = 8;

    const v8::ShimError error = shimErrorOf(
        [&] { v8::loadPlan(_handle, v8::plan_json::write(*parsed, true), JSON_LABEL); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_STREQ(error.what(), "planJson: \"id\" 8 does not match engine 7 of the serialized plan");
}

TEST_F(TestCudnnShimV8CorePlans, PlanJsonWithoutPayloadIsNotSupported)
{
    v8::plan_json::Plan plan;
    plan.id = ENGINE_ID;
    const v8::ShimError error = shimErrorOf(
        [&] { v8::loadPlan(_handle, v8::plan_json::write(plan, false), JSON_LABEL); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_STREQ(error.what(),
                 "planJson has no \"payload\": the engine could not serialize the plan, so it "
                 "cannot be loaded");
}

TEST_F(TestCudnnShimV8CorePlans, PlanJsonReaderNamesLabel)
{
    const v8::ShimError error = shimErrorOf([&] { v8::plan_json::read("{", "where"); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_STREQ(error.what(),
                 "where: expected '\"' at offset 1 (expected the format written by the hipDNN "
                 "cuDNN shim)");
}

TEST_F(TestCudnnShimV8CorePlans, VariantPackSetsArraysAndFinalizesOnce)
{
    std::array<void*, 2> pointers
        = {reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000)};
    const std::array<std::int64_t, 2> uids = {3, 4};
    void* workspace = reinterpret_cast<void*>(0x3000);

    const std::shared_ptr<const v8::VariantPackRecord> pack
        = v8::buildVariantPack(pointers.data(), 2, uids.data(), 2, workspace, PACK_LABELS);
    EXPECT_EQ(pack->kind(), CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    ASSERT_EQ(created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).size(), 1U);
    hipdnnBackendDescriptor_t raw = pack->hipdnnVariantPack();
    EXPECT_EQ(raw, created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).front());

    const auto pointerCalls = callsOn(raw, HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
    ASSERT_EQ(pointerCalls.size(), 1U);
    EXPECT_EQ(pointerCalls.front().values<void*>(),
              std::vector<void*>(pointers.begin(), pointers.end()));
    const auto uidCalls = callsOn(raw, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
    ASSERT_EQ(uidCalls.size(), 1U);
    EXPECT_EQ(uidCalls.front().values<std::int64_t>(),
              std::vector<std::int64_t>(uids.begin(), uids.end()));
    const auto workspaceCalls = callsOn(raw, HIPDNN_ATTR_VARIANT_PACK_WORKSPACE);
    ASSERT_EQ(workspaceCalls.size(), 1U);
    EXPECT_EQ(workspaceCalls.front().value<void*>(), workspace);
    EXPECT_EQ(setCountAtFinalize(raw), setAttributes().size());
}

TEST_F(TestCudnnShimV8CorePlans, VariantPackCountMismatchIsBadParam)
{
    std::array<void*, 2> pointers = {nullptr, nullptr};
    const std::int64_t uid = 3;
    const v8::ShimError error = shimErrorOf(
        [&] { v8::buildVariantPack(pointers.data(), 2, &uid, 1, nullptr, PACK_LABELS); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_STREQ(error.what(), "dataPointers has 2 elements but uniqueIds has 1");
    EXPECT_TRUE(created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8CorePlans, VariantPackEmptyArrayIsBadParam)
{
    void* pointer = nullptr;
    const v8::ShimError noPointers
        = shimErrorOf([&] { v8::buildVariantPack(nullptr, 0, nullptr, 0, nullptr, PACK_LABELS); });
    EXPECT_EQ(noPointers.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_STREQ(noPointers.what(), "dataPointers is required (non-empty)");

    const v8::ShimError noUids
        = shimErrorOf([&] { v8::buildVariantPack(&pointer, 1, nullptr, 0, nullptr, PACK_LABELS); });
    EXPECT_STREQ(noUids.what(), "uniqueIds is required (non-empty)");
    EXPECT_TRUE(created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8CorePlans, VariantPackNegativeOrNullArrayIsBadParam)
{
    void* pointer = nullptr;
    const std::int64_t uid = 3;
    const v8::ShimError negative
        = shimErrorOf([&] { v8::buildVariantPack(&pointer, -1, &uid, 1, nullptr, PACK_LABELS); });
    EXPECT_EQ(negative.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_STREQ(negative.what(), "dataPointers: element count -1 is negative");

    const v8::ShimError null
        = shimErrorOf([&] { v8::buildVariantPack(&pointer, 1, nullptr, 1, nullptr, PACK_LABELS); });
    EXPECT_EQ(null.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_STREQ(null.what(), "uniqueIds: array is null");
    EXPECT_TRUE(created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8CorePlans, KnobConfigFailureSurfacesAtFirstUse)
{
    // Declared first: the backend action below reads it until the records go.
    bool fail = true;
    const std::shared_ptr<const v8::EngineRecord> record = engine();
    ON_CALL(*_mockBackend, backendFinalize(::testing::_))
        .WillByDefault([this, &fail](hipdnnBackendDescriptor_t descriptor) {
            return fail && typeOf(descriptor) == HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR
                       ? HIPDNN_STATUS_NOT_SUPPORTED
                       : HIPDNN_STATUS_SUCCESS;
        });

    // Validation succeeds; the build fails on first use.
    const std::shared_ptr<const v8::ConfigRecord> config
        = v8::makeConfigRecord(record, {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}, CHOICES_LABEL);
    const v8::ShimError failure = shimErrorOf([&] { config->hipdnnEngineConfig(); });
    EXPECT_EQ(failure.status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        std::string(failure.what()).rfind("finalizing the hipDNN engine config of engine 7: ", 0),
        0U)
        << failure.what();

    // A failed build is retried on the next use.
    fail = false;
    hipdnnBackendDescriptor_t built = config->hipdnnEngineConfig();
    EXPECT_EQ(built, created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).back());
}

TEST_F(TestCudnnShimV8CorePlans, KnobConfigBuiltOnceAcrossThreads)
{
    const std::shared_ptr<const v8::ConfigRecord> config
        = v8::makeConfigRecord(engine(), {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}, CHOICES_LABEL);
    const std::size_t configs = created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size();
    const std::size_t choices = created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR).size();

    constexpr std::size_t THREAD_COUNT = 8;
    std::array<hipdnnBackendDescriptor_t, THREAD_COUNT> seen{};
    runConcurrently(THREAD_COUNT,
                    [&config, &seen](std::size_t i) { seen.at(i) = config->hipdnnEngineConfig(); });

    ASSERT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), configs + 1);
    EXPECT_EQ(created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR).size(), choices + 1);
    for(hipdnnBackendDescriptor_t built : seen)
    {
        EXPECT_EQ(built, created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).back());
    }
}

TEST_F(TestCudnnShimV8CorePlans, HeuristicSettingsAreNotSupported)
{
    constexpr v8::HeuristicLabels LABELS{"smCount", "deviceProp"};
    const std::shared_ptr<const v8::GraphCore> core = graph();

    const v8::ShimError smCount = shimErrorOf(
        [&] { v8::heuristicResults(core, CUDNN_HEUR_MODE_FALLBACK, {true, true}, LABELS); });
    EXPECT_EQ(smCount.status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_STREQ(smCount.what(), "smCount is not supported by the hipDNN cuDNN shim");
    const v8::ShimError deviceProp = shimErrorOf(
        [&] { v8::heuristicResults(core, CUDNN_HEUR_MODE_FALLBACK, {false, true}, LABELS); });
    EXPECT_STREQ(deviceProp.what(), "deviceProp is not supported by the hipDNN cuDNN shim");
    // Rejected before the engine space is enumerated.
    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).empty());

    const std::vector<std::shared_ptr<const v8::EngineRecord>> results
        = v8::heuristicResults(core, CUDNN_HEUR_MODE_INSTANT, {}, LABELS);
    ASSERT_EQ(results.size(), 2U);
    EXPECT_EQ(results[0]->id(), ENGINE_ID);
    EXPECT_EQ(results[1]->id(), OTHER_ID);
}

TEST_F(TestCudnnShimV8CorePlans, PlanTagKeepsEveryRequestedChoice)
{
    installSerializedPlan({0x01, 0x02});
    const std::shared_ptr<const v8::EngineRecord> record = engine();
    EXPECT_EQ(v8::buildPlan(v8::makeConfigRecord(record, {}, CHOICES_LABEL))->tag(), "eng1");

    // TILE_SIZE -1 is unset, so it never reaches hipDNN, but the tag keeps it.
    const std::shared_ptr<const v8::PlanRecord> plan = v8::buildPlan(
        v8::makeConfigRecord(record,
                             {{CUDNN_KNOB_TYPE_TILE_SIZE, -1}, {CUDNN_KNOB_TYPE_SPLIT_K_SLC, 2}},
                             CHOICES_LABEL));
    const std::string tile = std::to_string(static_cast<int>(CUDNN_KNOB_TYPE_TILE_SIZE));
    const std::string splitK = std::to_string(static_cast<int>(CUDNN_KNOB_TYPE_SPLIT_K_SLC));
    EXPECT_EQ(plan->tag(), "eng1_k" + tile + "=-1_k" + splitK + "=2");

    // The JSON carries the requested choices, so a restored plan keeps the tag.
    const std::optional<v8::plan_json::Plan> parsed = v8::plan_json::read(plan->json(), "json");
    ASSERT_TRUE(parsed.has_value());
    EXPECT_EQ(parsed->knobs, (v8::KnobChoices{{CUDNN_KNOB_TYPE_SPLIT_K_SLC, 2}}));
    EXPECT_EQ(parsed->requestedKnobs,
              (v8::KnobChoices{{CUDNN_KNOB_TYPE_TILE_SIZE, -1}, {CUDNN_KNOB_TYPE_SPLIT_K_SLC, 2}}));
    const std::shared_ptr<const v8::PlanRecord> loaded
        = v8::loadPlan(_handle, plan->json(), JSON_LABEL);
    EXPECT_EQ(loaded->tag(), plan->tag());
    EXPECT_EQ(loaded->config().knobChoices(), parsed->knobs);
    EXPECT_EQ(loaded->json(), plan->json());
}

TEST_F(TestCudnnShimV8CorePlans, PlanNotesComeFromItsEngine)
{
    installBehaviorNotes(ENGINE_ID, {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    const std::shared_ptr<const v8::PlanRecord> plan
        = v8::buildPlan(v8::makeConfigRecord(engine(), {}, CHOICES_LABEL));
    EXPECT_EQ(plan->behaviorNotes(),
              std::vector<cudnnBackendBehaviorNote_t>{CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    EXPECT_EQ(plan->numericalNotes(),
              std::vector<cudnnBackendNumericalNote_t>{CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC});
}

TEST_F(TestCudnnShimV8CorePlans, PlanJsonOmitsPayloadWhenSerializeFails)
{
    ON_CALL(*_mockBackend,
            backendGetSerializedExecutionPlanExt(
                ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(HIPDNN_STATUS_NOT_SUPPORTED));
    const std::shared_ptr<const v8::PlanRecord> plan
        = v8::buildPlan(v8::makeConfigRecord(engine(), {}, CHOICES_LABEL));

    const std::string& json = plan->json();
    EXPECT_EQ(json.find("payload"), std::string::npos) << json;
    EXPECT_FALSE(v8::plan_json::read(json, "json").has_value()) << json;
    EXPECT_EQ(shimErrorOf([&] { v8::loadPlan(_handle, json, JSON_LABEL); }).status(),
              CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestCudnnShimV8CorePlans, PlanLoadWorkspaceFailureIsReported)
{
    installSerializedPlan({0x01, 0x02});
    const std::string json
        = v8::buildPlan(v8::makeConfigRecord(engine(), {}, CHOICES_LABEL))->json();
    ON_CALL(*_mockBackend,
            backendGetAttribute(::testing::_,
                                HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                ::testing::_,
                                ::testing::_,
                                ::testing::_,
                                ::testing::_))
        .WillByDefault(::testing::Return(HIPDNN_STATUS_BAD_PARAM));

    const v8::ShimError error = shimErrorOf([&] { v8::loadPlan(_handle, json, JSON_LABEL); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(std::string(error.what())
                  .rfind("reading the workspace size of the hipDNN execution plan: ", 0),
              0U)
        << error.what();
}

TEST_F(TestCudnnShimV8CorePlans, RawConfigNotesReadTheEngine)
{
    installBehaviorNotes(ENGINE_ID, {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    // Declared first: the destroy action below appends to it until the graph goes.
    std::vector<hipdnnBackendDescriptor_t> destroyed;
    const std::shared_ptr<const v8::GraphCore> core = graph();
    hipdnnBackendDescriptor_t config = core->defaultEngineConfig(ENGINE_ID);
    ASSERT_NE(config, nullptr);
    ON_CALL(*_mockBackend, backendDestroyDescriptor(::testing::_))
        .WillByDefault([&destroyed](hipdnnBackendDescriptor_t descriptor) {
            destroyed.push_back(descriptor);
            return HIPDNN_STATUS_SUCCESS;
        });
    const std::size_t engines = created(HIPDNN_BACKEND_ENGINE_DESCRIPTOR).size();

    EXPECT_TRUE(v8::engineConfigHasNumericalNote(config, CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC));
    EXPECT_FALSE(v8::engineConfigHasNumericalNote(config, CUDNN_NUMERICAL_NOTE_TENSOR_CORE));
    EXPECT_TRUE(v8::engineConfigHasBehaviorNote(config, CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION));
    EXPECT_FALSE(v8::engineConfigHasBehaviorNote(config, CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY));

    // Each lookup GETs a fresh ENGINE wrapper and destroys it.
    const std::vector<hipdnnBackendDescriptor_t> all = created(HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
    ASSERT_EQ(all.size(), engines + 4);
    for(std::size_t i = engines; i < all.size(); ++i)
    {
        EXPECT_NE(std::find(destroyed.begin(), destroyed.end(), all[i]), destroyed.end()) << i;
    }
    const std::vector<hipdnnBackendAttributeName_t> gets = getAttributeNames();
    EXPECT_EQ(std::find(gets.begin(), gets.end(), HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE), gets.end());
}

TEST_F(TestCudnnShimV8CorePlans, RawConfigNotesFailClosed)
{
    installHeuristicResults({hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID});
    const std::shared_ptr<const v8::GraphCore> core = graph();
    hipdnnBackendDescriptor_t claimed
        = core->defaultEngineConfig(hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID);
    ASSERT_NE(claimed, nullptr);
    EXPECT_FALSE(v8::engineConfigHasNumericalNote(claimed, CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC));

    // Not a config the backend knows: every query answers false.
    hipdnnBackendDescriptor_t foreign = newMockDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    EXPECT_FALSE(v8::engineConfigHasNumericalNote(foreign, CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC));
    EXPECT_FALSE(v8::engineConfigHasBehaviorNote(foreign, CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION));
    EXPECT_FALSE(v8::engineConfigHasNumericalNote(nullptr, CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC));
}

} // namespace
