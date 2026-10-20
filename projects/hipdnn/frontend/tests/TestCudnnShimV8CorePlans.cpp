// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The v8 core's engine views, plans and variant packs, called directly against
// the mock hipDNN backend: knob-choice validation, the lazily built knob config,
// plan build/load and the plan JSON, and variant-pack checks.
#include <hipdnn_compatibility/cudnn/detail/v8/engine_views.h>
#include <hipdnn_compatibility/cudnn/detail/v8/plan.h>
#include <hipdnn_compatibility/cudnn/detail/v8/variant_pack.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

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

constexpr std::string_view CHOICES_LABEL = "knobChoices";
constexpr v8::PlanLabels PLAN_LABELS{"planHandle", "graphHandle", "planJson"};
constexpr v8::VariantPackLabels PACK_LABELS{"dataPointers", "uniqueIds"};

// Runs `body`, which must throw ShimError; returns it.
template <typename Body>
v8::ShimError shimErrorOf(Body&& body)
{
    try
    {
        body();
    }
    catch(const v8::ShimError& error)
    {
        return error;
    }
    ADD_FAILURE() << "no ShimError thrown";
    return {CUDNN_STATUS_SUCCESS, ""};
}

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
            _handle,
            hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor(
                newMockDescriptor(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)),
            "");
    }

    std::shared_ptr<const v8::EngineRecord> engine()
    {
        return v8::makeEngineRecord(graph(), ENGINE_INDEX, "index");
    }

    std::vector<CapturedSetAttribute> callsOn(hipdnnBackendDescriptor_t descriptor,
                                              hipdnnBackendAttributeName_t name) const
    {
        std::vector<CapturedSetAttribute> calls = setAttributesOf(descriptor);
        std::vector<CapturedSetAttribute> named;
        for(CapturedSetAttribute& call : calls)
        {
            if(call.name == name)
            {
                named.push_back(std::move(call));
            }
        }
        return named;
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
    EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_STREQ(error.what(), "theIndex = 2 is outside [0, 2)");
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
    const std::shared_ptr<const v8::ConfigRecord> loaded = v8::makeLoadedConfigRecord(
        -31, 2, {CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION}, {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}, "x");
    EXPECT_TRUE(loaded->isLoaded());
    EXPECT_EQ(loaded->engineId(), -31);
    EXPECT_EQ(loaded->engineIndex(), 2);
    EXPECT_EQ(loaded->engine().name(), std::optional<std::string>("x"));
    EXPECT_TRUE(loaded->engine().knobs().empty());
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

TEST_F(TestCudnnShimV8CorePlans, PlanBuildReadsWorkspaceOnGraphHandle)
{
    const std::shared_ptr<const v8::ConfigRecord> config
        = v8::makeConfigRecord(engine(), {}, CHOICES_LABEL);
    const std::shared_ptr<const v8::PlanRecord> plan = v8::buildPlan(config, nullptr, PLAN_LABELS);

    EXPECT_EQ(plan->kind(), CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    EXPECT_EQ(plan->workspaceSize(), WORKSPACE_SIZE);
    EXPECT_EQ(plan->contextHandle(), _handle);
    EXPECT_EQ(plan->configRecord(), config);
    ASSERT_EQ(created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).size(), 1U);
    EXPECT_EQ(plan->hipdnnPlan(), created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).front());
    const std::vector<CapturedSetAttribute> configCalls
        = callsOn(plan->hipdnnPlan(), HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
    ASSERT_EQ(configCalls.size(), 1U);
    EXPECT_EQ(configCalls.front().value<hipdnnBackendDescriptor_t>(), config->hipdnnEngineConfig());
    EXPECT_TRUE(setCountAtFinalize(plan->hipdnnPlan()).has_value());

    // Naming the graph's own handle is accepted.
    EXPECT_EQ(v8::buildPlan(config, _handle, PLAN_LABELS)->contextHandle(), _handle);
}

TEST_F(TestCudnnShimV8CorePlans, PlanHandleOtherThanGraphHandleIsBadParam)
{
    const std::shared_ptr<const v8::ConfigRecord> config
        = v8::makeConfigRecord(engine(), {}, CHOICES_LABEL);
    const v8::ShimError error = shimErrorOf(
        [&] { v8::buildPlan(config, reinterpret_cast<cudnnHandle_t>(0x23456789), PLAN_LABELS); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_STREQ(error.what(),
                 "planHandle differs from the graphHandle of the engine config's graph");
    EXPECT_TRUE(created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8CorePlans, PlanJsonRoundTrip)
{
    const std::vector<std::uint8_t> payload = {0x00, 0xFF, 0x10, 0x80, 0x7F};
    installSerializedPlan(payload);
    installEngineName(ENGINE_ID, "mock_engine_seven");
    installBehaviorNotes(ENGINE_ID, {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    const std::shared_ptr<const v8::ConfigRecord> config
        = v8::makeConfigRecord(engine(), {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}, CHOICES_LABEL);
    const std::shared_ptr<const v8::PlanRecord> built = v8::buildPlan(config, nullptr, PLAN_LABELS);

    const std::string& json = built->json();
    EXPECT_EQ(&built->json(), &json);
    const std::optional<v8::plan_json::Plan> parsed = v8::plan_json::read(json, "json");
    ASSERT_TRUE(parsed.has_value()) << json;
    EXPECT_EQ(parsed->id, ENGINE_ID);
    EXPECT_EQ(parsed->globalIndex, ENGINE_INDEX);
    EXPECT_EQ(parsed->name, std::optional<std::string>("mock_engine_seven"));
    EXPECT_EQ(parsed->behaviorNotes,
              std::vector<cudnnBackendBehaviorNote_t>{CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    EXPECT_EQ(parsed->knobs, (v8::KnobChoices{{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}));
    EXPECT_EQ(parsed->payload, payload);

    const auto loadHandle = reinterpret_cast<cudnnHandle_t>(0x34567890);
    const std::shared_ptr<const v8::PlanRecord> loaded
        = v8::loadPlan(loadHandle, json, PLAN_LABELS);
    EXPECT_EQ(loaded->contextHandle(), loadHandle);
    EXPECT_EQ(loaded->workspaceSize(), WORKSPACE_SIZE);
    EXPECT_TRUE(loaded->config().isLoaded());
    EXPECT_EQ(loaded->config().engineId(), ENGINE_ID);
    EXPECT_EQ(loaded->config().engineIndex(), ENGINE_INDEX);
    EXPECT_EQ(loaded->config().knobChoices(), config->knobChoices());
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
        = v8::buildPlan(v8::makeConfigRecord(engine(), {}, CHOICES_LABEL), nullptr, PLAN_LABELS);
    std::optional<v8::plan_json::Plan> parsed = v8::plan_json::read(built->json(), "json");
    ASSERT_TRUE(parsed.has_value());
    parsed->id = 8;

    const v8::ShimError error = shimErrorOf(
        [&] { v8::loadPlan(_handle, v8::plan_json::write(*parsed, true), PLAN_LABELS); });
    EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_STREQ(error.what(), "planJson: \"id\" 8 does not match engine 7 of the serialized plan");
}

TEST_F(TestCudnnShimV8CorePlans, PlanJsonWithoutPayloadIsNotSupported)
{
    v8::plan_json::Plan plan;
    plan.id = ENGINE_ID;
    const v8::ShimError error = shimErrorOf(
        [&] { v8::loadPlan(_handle, v8::plan_json::write(plan, false), PLAN_LABELS); });
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

} // namespace
