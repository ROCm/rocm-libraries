// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x ExecutionPlanBuilder, loadFromJson, VariantPackBuilder and
// cudnnBackendExecute against the mock hipDNN backend: what each lowers to, the
// plan JSON format, and the execute path.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_ExecutionPlan.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "fake_backend/ShimV8BuilderPlanFixture.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using hipdnn_shim_test::BuildFailure;
using hipdnn_shim_test::CapturedExecute;
using hipdnn_shim_test::CapturedSetAttribute;
using hipdnn_shim_test::failureOf;
using hipdnn_shim_test::shimLastError;

// getJsonRepresentation() keeps the NUL the backend counts.
std::string withoutNul(std::string text)
{
    if(!text.empty() && text.back() == '\0')
    {
        text.pop_back();
    }
    return text;
}

class TestCudnnShimV8BuilderPlans : public hipdnn_shim_test::ShimV8BuilderPlanFixture
{
protected:
    // Ranked {7, -2} gives the dense index {-2, 7}; the tests use index 1.
    static constexpr std::int64_t ENGINE_ID = 7;
    static constexpr std::int64_t ENGINE_INDEX = 1;
    // Base64 "AP8QgH8=" (exercises the one-byte-short final group).
    static constexpr std::array<std::uint8_t, 5> PAYLOAD = {0x00, 0xFF, 0x10, 0x80, 0x7F};
    static constexpr const char* PAYLOAD_BASE64 = "AP8QgH8=";

    void SetUp() override
    {
        ShimV8BuilderPlanFixture::SetUp();
        installHeuristicResults({ENGINE_ID, -2});
        installSerializedPlan({PAYLOAD.begin(), PAYLOAD.end()});
        setDeserializedEngineId(ENGINE_ID);
    }

    void TearDown() override
    {
        hipdnn_frontend::detail::IHipdnnBackend::setInstance(_mockBackend);
        ShimV8BuilderPlanFixture::TearDown();
    }

    fe::ExecutionPlan_v8 builtPlan()
    {
        const fe::OperationGraph_v8 graph = convGraph();
        return planFor(graph, ENGINE_INDEX);
    }

    fe::ExecutionPlan_v8 loaded(const std::string& json)
    {
        return fe::ExecutionPlanBuilder_v8().setHandle(_handle).loadFromJson(json);
    }

    BuildFailure loadFailure(const std::string& json)
    {
        return failureOf([this, &json] { [[maybe_unused]] auto plan = loaded(json); });
    }

    // The hipDNN ENGINECFG of the engine's knob-free plans.
    hipdnnBackendDescriptor_t defaultHipdnnConfig() const
    {
        hipdnnBackendDescriptor_t found = nullptr;
        for(hipdnnBackendDescriptor_t config : created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR))
        {
            if(engineIdOf(config) == ENGINE_ID)
            {
                EXPECT_EQ(found, nullptr) << "more than one hipDNN ENGINECFG for the engine";
                found = config;
            }
        }
        return found;
    }

    static std::string expectedJsonPrefix()
    {
        return R"({"version":1,"engine":{"globalIndex":1,"id":"7")";
    }

    int deserializeCalls() const
    {
        return static_cast<int>(deserializations().size());
    }

    static cudnnStatus_t execute(cudnnHandle_t handle,
                                 const fe::ExecutionPlan_v8& plan,
                                 const fe::VariantPack_v8& pack)
    {
        return cudnnBackendExecute(handle, plan.get_raw_desc(), pack.get_raw_desc());
    }

    float _buffer = 0.0F;
};

TEST_F(TestCudnnShimV8BuilderPlans, PlanForwardsOnlyEngineConfig)
{
    const fe::ExecutionPlan_v8 plan = builtPlan();
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();

    const std::vector<hipdnnBackendDescriptor_t> plans
        = created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    ASSERT_EQ(plans.size(), 1U);
    const std::vector<CapturedSetAttribute> sets = setAttributesOf(plans[0]);
    ASSERT_EQ(sets.size(), 1U) << "only ENGINE_CONFIG is forwarded";
    EXPECT_EQ(sets[0].name, HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
    EXPECT_EQ(sets[0].value<hipdnnBackendDescriptor_t>(), defaultHipdnnConfig());
    const std::vector<hipdnnBackendDescriptor_t> done = finalized();
    EXPECT_NE(std::find(done.begin(), done.end(), plans[0]), done.end());
}

TEST_F(TestCudnnShimV8BuilderPlans, WorkspaceSizeReadFromHipdnnPlan)
{
    installPlanWorkspace(12345);
    EXPECT_EQ(builtPlan().getWorkspaceSize(), 12345);
}

TEST_F(TestCudnnShimV8BuilderPlans, ConfigRequired)
{
    fe::ExecutionPlanBuilder_v8 builder;
    const BuildFailure failure
        = failureOf([&builder] { [[maybe_unused]] auto plan = builder.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).empty());
}

// The untyped setter takes any descriptor; a tensor where the config belongs is rejected.
TEST_F(TestCudnnShimV8BuilderPlans, WrongKindConfigRejected)
{
    const fe::Tensor_v8 notAConfig = tensor(X_UID, {2, 4, 8, 8});
    fe::ExecutionPlanBuilder_v8 builder;
    builder.setEngineConfig(notAConfig.get_desc(), "ConvFwd_");

    const BuildFailure failure
        = failureOf([&builder] { [[maybe_unused]] auto plan = builder.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8BuilderPlans, HandleRequiredForLoadedPlan)
{
    const std::string text = builtPlan().getJsonRepresentation();

    fe::ExecutionPlanBuilder_v8 noHandle;
    const BuildFailure failure = failureOf(
        [&noHandle, &text] { [[maybe_unused]] auto plan = noHandle.loadFromJson(text); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(deserializeCalls(), 0);
}

TEST_F(TestCudnnShimV8BuilderPlans, JsonCarriesEngineAndPayload)
{
    const std::string raw = builtPlan().getJsonRepresentation();
    ASSERT_FALSE(raw.empty());
    EXPECT_EQ(raw.back(), '\0');
    const std::string text = withoutNul(raw);

    EXPECT_EQ(text.rfind(expectedJsonPrefix(), 0), 0U) << text;
    EXPECT_NE(text.find(R"("behaviorNotes":[)"), std::string::npos) << text;
    EXPECT_NE(text.find(R"(},"knobs":[],"payload":")"), std::string::npos) << text;
    const std::string tail = std::string(R"("payload":")") + PAYLOAD_BASE64 + "\"}";
    ASSERT_GE(text.size(), tail.size());
    EXPECT_EQ(text.substr(text.size() - tail.size()), tail) << text;
}

TEST_F(TestCudnnShimV8BuilderPlans, JsonWithoutPayloadWhenSerializeFails)
{
    setSerializeStatus(HIPDNN_STATUS_NOT_SUPPORTED);
    const std::string text = withoutNul(builtPlan().getJsonRepresentation());

    EXPECT_EQ(text.rfind(expectedJsonPrefix(), 0), 0U) << text;
    EXPECT_EQ(text.find("payload"), std::string::npos) << text;
    ASSERT_FALSE(text.empty());
    EXPECT_EQ(text.back(), '}');

    const BuildFailure failure = loadFailure(text);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(deserializeCalls(), 0);
}

TEST_F(TestCudnnShimV8BuilderPlans, JsonLoadRoundTrip)
{
    const fe::ExecutionPlan_v8 built = builtPlan();
    const std::string text = built.getJsonRepresentation();

    const fe::ExecutionPlan_v8 plan = loaded(text);
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    const std::vector<hipdnn_shim_test::CapturedDeserialize> calls = deserializations();
    ASSERT_EQ(calls.size(), 1U);
    EXPECT_EQ(calls.front().handle, _handle);
    EXPECT_EQ(calls.front().bytes, std::vector<std::uint8_t>(PAYLOAD.begin(), PAYLOAD.end()));
    EXPECT_EQ(plan.getWorkspaceSize(), WORKSPACE_SIZE);
    EXPECT_EQ(plan.getJsonRepresentation(), text);
    // The tag carries the dense index read back from the loaded plan.
    EXPECT_EQ(plan.getTag(), "eng" + std::to_string(ENGINE_INDEX));
}

TEST_F(TestCudnnShimV8BuilderPlans, LoadedPlanKeepsNotesKnobsAndName)
{
    const std::int64_t tileSize = 16;
    // SPLIT_K_SLC stays unset: the builder sends it as -1, which the tag keeps.
    installEngineKnobs(ENGINE_ID,
                       {{"prov.tile_size", 8, 0, 64, 8, {}}, {"prov.split_k_slc", 1, 1, 8, 1, {}}});
    installBehaviorNotes(ENGINE_ID, {HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    installEngineName(ENGINE_ID, "restored");
    const fe::OperationGraph_v8 graph = convGraph();
    fe::Engine_v8 built = engine(graph, ENGINE_INDEX);
    ASSERT_EQ(built.getSupportedKnobs().size(), 2U);
    for(auto& knob : built.getSupportedKnobs())
    {
        if(knob.getKnobType() == CUDNN_KNOB_TYPE_TILE_SIZE)
        {
            knob.setChoice(static_cast<std::uint64_t>(tileSize));
        }
    }
    const fe::EngineConfig_v8 config = fe::EngineConfigBuilder_v8().setEngine(built).build();
    const fe::ExecutionPlan_v8 plan = fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
    const std::string text = plan.getJsonRepresentation();

    const fe::ExecutionPlan_v8 restored = loaded(text);
    ASSERT_EQ(restored.get_status(), CUDNN_STATUS_SUCCESS) << restored.get_error();
    const std::string tile = std::to_string(static_cast<int>(CUDNN_KNOB_TYPE_TILE_SIZE));
    const std::string splitK = std::to_string(static_cast<int>(CUDNN_KNOB_TYPE_SPLIT_K_SLC));
    // The engine part survives the round trip; a loaded plan has no graph tag to prefix.
    const std::string engineTag = "eng1_k" + tile + "=16_k" + splitK + "=-1";
    EXPECT_EQ(plan.getTag(), "ConvFwd_" + engineTag);
    EXPECT_EQ(restored.getTag(), engineTag);
    EXPECT_EQ(restored.getAllBehaviorNotes(),
              std::vector<cudnnBackendBehaviorNote_t>{CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    EXPECT_EQ(restored.getAllNumericNotes(),
              std::vector<cudnnBackendNumericalNote_t>{CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC});
    EXPECT_NE(restored.getJsonRepresentation().find(R"("name":"restored")"), std::string::npos);
    EXPECT_EQ(restored.getJsonRepresentation(), text);
}

TEST_F(TestCudnnShimV8BuilderPlans, JsonReaderAcceptsWhitespaceAndKeyOrder)
{
    const std::string text = " {\n \"payload\" : \"AP8QgH8=\" , \"knobs\" : [ { \"value\" : 16 , "
                             "\"type\" : 3 } ] ,\t\"engine\" : { \"id\" : \"7\" , "
                             "\"behaviorNotes\" : [ 3 ] , \"name\" : \"a\\\"b\\\\c\\u0041\" , "
                             "\"globalIndex\" : 1 } , \"version\" : 1 }\r\n";
    const fe::ExecutionPlan_v8 plan = loaded(text);
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    EXPECT_EQ(withoutNul(plan.getJsonRepresentation()),
              R"({"version":1,"engine":{"globalIndex":1,"id":"7","name":"a\"b\\cA",)"
              R"("behaviorNotes":[3]},"knobs":[{"type":3,"value":16}],"payload":"AP8QgH8="})");
}

TEST_F(TestCudnnShimV8BuilderPlans, JsonMalformedRejected)
{
    const std::string engine = R"("engine":{"globalIndex":1,"id":"7"})";
    const std::string payload = R"("payload":"AP8QgH8=")";
    const std::string version = R"({"version":1,)";
    const std::vector<std::string> malformed = {
        "",
        "not json",
        R"({"version":2,)" + engine + "," + payload + "}",
        R"({"version":1.0,)" + engine + "," + payload + "}",
        version + R"("engine":{"globalIndex":1,"id":7},)" + payload + "}",
        version + R"("engine":{"globalIndex":1,"id":"99999999999999999999"},)" + payload + "}",
        version + R"("engine":{"globalIndex":-1,"id":"7"},)" + payload + "}",
        version + R"("engine":{"id":"7"},)" + payload + "}",
        version + payload + "}",
        version + engine + R"(,"payload":"AP8Q gH8="})",
        version + engine + R"(,"payload":"AP8QgH8"})",
        version + engine + R"(,"payload":""})",
        version + engine + R"(,"payload":"A==="})",
        version + engine + R"(,"payload":"===="})",
        version + engine + R"(,"payload":"AB=C"})",
        version + engine + R"(,"knobs":[{"type":999,"value":1}],)" + payload + "}",
        version + engine + R"(,"extra":1,)" + payload + "}",
        version + R"("engine":{"globalIndex":1,"id":"7x"},)" + payload + "}",
        version + R"("engine":{"globalIndex":1,"id":"7","behaviorNotes":[99]},)" + payload + "}",
        version + R"("engine":{"globalIndex":1,"id":"7","behaviorNotes":[-1]},)" + payload + "}",
        version + engine + R"(,"knobs":[{"type":3,"type":4,"value":1}],)" + payload + "}",
        version + engine + R"(,"requestedKnobs":[{"type":999,"value":1}],)" + payload + "}",
        version + engine + R"(,"requestedKnobs":[],"requestedKnobs":[],)" + payload + "}",
        version + R"("version":1,)" + engine + "," + payload + "}",
        version + engine + "," + payload + "} x",
        version + engine + "," + payload,
    };
    for(const std::string& text : malformed)
    {
        SCOPED_TRACE(text);
        const BuildFailure failure = loadFailure(text);
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    }
    EXPECT_EQ(deserializeCalls(), 0);
}

TEST_F(TestCudnnShimV8BuilderPlans, JsonEngineIdMismatchRejected)
{
    const std::string text = builtPlan().getJsonRepresentation();
    setDeserializedEngineId(8);
    const BuildFailure failure = loadFailure(text);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderPlans, ExtremeEngineIdsLoad)
{
    for(const std::int64_t id : {std::numeric_limits<std::int64_t>::min(),
                                 std::numeric_limits<std::int64_t>::max(),
                                 std::int64_t{0x637C0BB90F065BA2}})
    {
        SCOPED_TRACE(id);
        setDeserializedEngineId(id);
        const std::string idField = R"("id":")" + std::to_string(id) + "\"";
        const fe::ExecutionPlan_v8 plan = loaded(R"({"version":1,"engine":{"globalIndex":0,)"
                                                 + idField + R"(},"payload":"AP8QgH8="})");
        ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
        EXPECT_NE(plan.getJsonRepresentation().find(idField), std::string::npos);
    }
}

TEST_F(TestCudnnShimV8BuilderPlans, Base64PayloadsRoundTrip)
{
    const std::vector<std::pair<std::vector<std::uint8_t>, std::string>> cases = {
        {{0}, "AA=="},
        {{0, 1}, "AAE="},
        {{0, 1, 2}, "AAEC"},
        {{0, 1, 2, 3}, "AAECAw=="},
        {{0x00, 0xFF, 0x10, 0x80, 0x7F}, "AP8QgH8="},
    };
    for(const auto& [bytes, encoded] : cases)
    {
        SCOPED_TRACE(encoded);
        installSerializedPlan(bytes);
        const std::string text = builtPlan().getJsonRepresentation();
        EXPECT_NE(text.find(R"("payload":")" + encoded + "\""), std::string::npos) << text;

        const fe::ExecutionPlan_v8 plan = loaded(text);
        ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
        const std::vector<hipdnn_shim_test::CapturedDeserialize> calls = deserializations();
        ASSERT_FALSE(calls.empty());
        EXPECT_EQ(calls.back().bytes, bytes);
    }
}

TEST_F(TestCudnnShimV8BuilderPlans, ControlCharactersInNameRoundTrip)
{
    installEngineName(ENGINE_ID,
                      "a\nb\x01"
                      "c");
    const std::string text = builtPlan().getJsonRepresentation();
    EXPECT_NE(text.find(R"("name":"a\nb\u0001c")"), std::string::npos) << text;

    const fe::ExecutionPlan_v8 plan = loaded(text);
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    EXPECT_EQ(plan.getJsonRepresentation(), text);
}

TEST_F(TestCudnnShimV8BuilderPlans, VariantPackLowered)
{
    const fe::VariantPack_v8 pack = variantPack();
    ASSERT_EQ(pack.get_status(), CUDNN_STATUS_SUCCESS) << pack.get_error();

    const std::vector<hipdnnBackendDescriptor_t> packs
        = created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    ASSERT_EQ(packs.size(), 1U);
    const std::vector<CapturedSetAttribute> sets = setAttributesOf(packs[0]);
    const auto find = [&sets](hipdnnBackendAttributeName_t name) {
        return std::find_if(sets.begin(), sets.end(), [name](const CapturedSetAttribute& call) {
            return call.name == name;
        });
    };
    const auto pointers = find(HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
    ASSERT_NE(pointers, sets.end());
    EXPECT_EQ(pointers->values<void*>().size(), 3U);
    const auto uids = find(HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
    ASSERT_NE(uids, sets.end());
    EXPECT_EQ(uids->values<std::int64_t>(), (std::vector<std::int64_t>{X_UID, W_UID, Y_UID}));
    const auto workspace = find(HIPDNN_ATTR_VARIANT_PACK_WORKSPACE);
    ASSERT_NE(workspace, sets.end());
    EXPECT_NE(workspace->value<void*>(), nullptr);
    const std::vector<hipdnnBackendDescriptor_t> done = finalized();
    EXPECT_NE(std::find(done.begin(), done.end(), packs[0]), done.end());
}

TEST_F(TestCudnnShimV8BuilderPlans, VariantPackCountMismatchRejected)
{
    std::array<void*, 1> pointers = {&_buffer};
    const std::array<std::int64_t, 2> uids = {1, 2};
    fe::VariantPackBuilder_v8 builder;
    builder.setDataPointers(static_cast<std::int64_t>(pointers.size()), pointers.data())
        .setUids(static_cast<std::int64_t>(uids.size()), uids.data());

    const BuildFailure failure
        = failureOf([&builder] { [[maybe_unused]] auto pack = builder.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8BuilderPlans, VariantPackRequiresPointersAndIds)
{
    std::array<void*, 1> pointers = {&_buffer};
    for(const bool withPointers : {false, true})
    {
        SCOPED_TRACE(withPointers);
        fe::VariantPackBuilder_v8 builder;
        if(withPointers)
        {
            builder.setDataPointers(static_cast<std::int64_t>(pointers.size()), pointers.data());
        }
        const BuildFailure failure
            = failureOf([&builder] { [[maybe_unused]] auto pack = builder.build(); });
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    }

    // Empty vectors, whose data() may be null.
    std::vector<void*> noPointers;
    const std::vector<std::int64_t> noUids;
    fe::VariantPackBuilder_v8 empty;
    empty.setDataPointers(0, noPointers.data()).setUids(0, noUids.data());
    const BuildFailure failure
        = failureOf([&empty] { [[maybe_unused]] auto pack = empty.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);

    EXPECT_TRUE(created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8BuilderPlans, ExecuteIsOneBackendCall)
{
    const fe::ExecutionPlan_v8 plan = builtPlan();
    const fe::VariantPack_v8 pack = variantPack();
    auto* const hipdnnPlan = created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).back();
    auto* const hipdnnPack = created(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR).back();

    // Any other backend call fails the strict mock.
    auto strict = std::make_shared<::testing::StrictMock<Mock_hipdnn_backend>>();
    EXPECT_CALL(*strict, backendExecute(_handle, hipdnnPlan, hipdnnPack))
        .WillOnce(::testing::Return(HIPDNN_STATUS_SUCCESS));
    hipdnn_frontend::detail::IHipdnnBackend::setInstance(strict);
    const cudnnStatus_t status = execute(_handle, plan, pack);
    hipdnn_frontend::detail::IHipdnnBackend::setInstance(_mockBackend);

    EXPECT_EQ(status, CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimV8BuilderPlans, ExecuteFailureKeepsStatus)
{
    const fe::ExecutionPlan_v8 plan = builtPlan();
    const fe::VariantPack_v8 pack = variantPack();
    failExecuteFor(ENGINE_ID, 0);

    EXPECT_EQ(execute(_handle, plan, pack), CUDNN_STATUS_EXECUTION_FAILED) << shimLastError();
}

TEST_F(TestCudnnShimV8BuilderPlans, LoadedPlanExecutes)
{
    const std::string text = builtPlan().getJsonRepresentation();
    const fe::ExecutionPlan_v8 plan = loaded(text);
    const fe::VariantPack_v8 pack = variantPack();

    EXPECT_EQ(execute(_handle, plan, pack), CUDNN_STATUS_SUCCESS) << shimLastError();
    const std::vector<CapturedExecute> calls = executions();
    ASSERT_EQ(calls.size(), 1U);
    EXPECT_EQ(calls[0].handle, _handle);
    EXPECT_EQ(calls[0].plan, created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).back());
}

TEST_F(TestCudnnShimV8BuilderPlans, ExecuteFromTwoThreads)
{
    constexpr int RUNS = 200;
    const fe::ExecutionPlan_v8 plan = builtPlan();
    const fe::VariantPack_v8 pack = variantPack();
    std::atomic<int> failures{0};
    const auto run = [&] {
        for(int i = 0; i < RUNS; ++i)
        {
            if(execute(_handle, plan, pack) != CUDNN_STATUS_SUCCESS)
            {
                ++failures;
            }
        }
    };
    std::thread first(run);
    std::thread second(run);
    first.join();
    second.join();

    EXPECT_EQ(failures.load(), 0);
    const std::vector<CapturedExecute> calls = executions();
    ASSERT_EQ(calls.size(), static_cast<std::size_t>(2 * RUNS));
    auto* const hipdnnPlan = created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR).back();
    for(const CapturedExecute& call : calls)
    {
        EXPECT_EQ(call.plan, hipdnnPlan);
        EXPECT_EQ(call.handle, _handle);
    }
}

// Knob-free configs share the engine's hipDNN ENGINECFG, so concurrent plan
// builds hand the same hipDNN descriptor to two plan finalizes.
TEST_F(TestCudnnShimV8BuilderPlans, PlansFromTwoThreadsShareDefaultConfig)
{
    constexpr int PLANS = 20;
    fe::OperationGraph_v8 graph = convGraph();
    const fe::EngineConfigList configs = engineConfigs(graph);
    ASSERT_EQ(configs.size(), 2U);
    const fe::ManagedOpaqueDescriptor& config = configs.front();
    std::atomic<int> failures{0};
    const auto build = [&] {
        for(int i = 0; i < PLANS; ++i)
        {
            const BuildFailure failure = failureOf(
                [&config] { [[maybe_unused]] auto plan = planFor(config, "ConvFwd_"); });
            if(failure.thrown)
            {
                ++failures;
            }
        }
    };
    std::thread first(build);
    std::thread second(build);
    first.join();
    second.join();

    EXPECT_EQ(failures.load(), 0);
    auto* const shared = defaultHipdnnConfig();
    const std::vector<CapturedSetAttribute> sets
        = setAttributesOn(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    ASSERT_EQ(sets.size(), static_cast<std::size_t>(2 * PLANS));
    for(const CapturedSetAttribute& call : sets)
    {
        EXPECT_EQ(call.name, HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
        EXPECT_EQ(call.value<hipdnnBackendDescriptor_t>(), shared);
    }
}

enum class PlanSource
{
    BUILT,
    LOADED
};

// Executing a plan on a handle other than the one it was built or loaded on.
class TestCudnnShimV8BuilderPlansForeignHandle : public TestCudnnShimV8BuilderPlans,
                                                 public ::testing::WithParamInterface<PlanSource>
{
protected:
    fe::ExecutionPlan_v8 plan()
    {
        return GetParam() == PlanSource::BUILT ? builtPlan()
                                               : loaded(builtPlan().getJsonRepresentation());
    }

    cudnnHandle_t _otherHandle = reinterpret_cast<cudnnHandle_t>(0x23456789);
};

// cudnnBackendExecute passes straight through, so hipDNN runs the plan on the given handle.
TEST_P(TestCudnnShimV8BuilderPlansForeignHandle, RunsOnGivenHandle)
{
    const fe::ExecutionPlan_v8 executionPlan = plan();
    const fe::VariantPack_v8 pack = variantPack();

    EXPECT_EQ(execute(_otherHandle, executionPlan, pack), CUDNN_STATUS_SUCCESS) << shimLastError();
    EXPECT_EQ(execute(_handle, executionPlan, pack), CUDNN_STATUS_SUCCESS) << shimLastError();
    const std::vector<CapturedExecute> calls = executions();
    ASSERT_EQ(calls.size(), 2U);
    EXPECT_EQ(calls[0].handle, _otherHandle);
    EXPECT_EQ(calls[1].handle, _handle);
}

INSTANTIATE_TEST_SUITE_P(Source,
                         TestCudnnShimV8BuilderPlansForeignHandle,
                         ::testing::Values(PlanSource::BUILT, PlanSource::LOADED),
                         [](const ::testing::TestParamInfo<PlanSource>& info) {
                             return info.param == PlanSource::BUILT ? "Built" : "Loaded";
                         });

} // namespace
