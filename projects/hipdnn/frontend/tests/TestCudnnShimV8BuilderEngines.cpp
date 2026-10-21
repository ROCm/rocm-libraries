// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x heuristics, engine and engine config builders against the mock
// hipDNN backend: heuristic modes, rank order, the dense engine index, knob
// projection and validation, and the hipDNN configs plans are built on.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfig.h>

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "fake_backend/ShimV8BuilderPlanFixture.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using hipdnn_shim_test::BuilderFailure;
using hipdnn_shim_test::builderFailureOf;
using hipdnn_shim_test::CapturedSetAttribute;
using hipdnn_shim_test::MockKnob;
using KnobChoice = std::pair<std::string, std::int64_t>;

constexpr std::int64_t KNOB_ENGINE = 7;

// TILE_SIZE is the only knob the shim can show: BLOCK_SIZE has explicit valid
// values, "int_knob" has no cuDNN type, and the two STAGES knobs collide.
std::vector<MockKnob> mixedKnobs()
{
    return {
        {"prov.tile_size", 8, 0, 64, 8, {}},
        {"prov.block_size", 8, 0, 64, 1, {8, 16}},
        {"prov.int_knob", 1, 0, 10, 1, {}},
        {"first.stages", 1, 1, 4, 1, {}},
        {"second.stages", 1, 1, 4, 1, {}},
    };
}

std::size_t warningsContaining(const hipdnn_test_sdk::utilities::SharedLogRecorder& recorder,
                               const std::string& text)
{
    const auto logs = recorder.getRecordedLogs();
    return static_cast<std::size_t>(
        std::count_if(logs.begin(), logs.end(), [&text](const auto& log) {
            return log.severity == HIPDNN_SEV_WARN && log.message.find(text) != std::string::npos;
        }));
}

std::string knobSegment(cudnnBackendKnobType_t type, std::int64_t value)
{
    return "_k" + std::to_string(static_cast<int>(type)) + "=" + std::to_string(value);
}

// Appends a knob the engine did not report, as a caller can through
// getSupportedKnobs(). EngineConfigBuilder sends it only if a reported knob is set.
void appendKnob(fe::Engine_v8& built, cudnnBackendKnobType_t type, std::int64_t choice)
{
    using Knob = std::remove_reference_t<decltype(built.getSupportedKnobs().front())>;
    Knob knob(type, 64, 0, 1);
    knob.setChoice(static_cast<std::uint64_t>(choice));
    built.getSupportedKnobs().push_back(knob);
}

void choose(fe::Engine_v8& built, cudnnBackendKnobType_t type, std::int64_t choice)
{
    auto& knobs = built.getSupportedKnobs();
    const auto knob = std::find_if(knobs.begin(), knobs.end(), [type](const auto& candidate) {
        return candidate.getKnobType() == type;
    });
    ASSERT_NE(knob, knobs.end());
    knob->setChoice(static_cast<std::uint64_t>(choice));
}

class TestCudnnShimV8BuilderEngines : public hipdnn_shim_test::ShimV8BuilderPlanFixture
{
protected:
    using ShimV8BuilderPlanFixture::planFor;

    static fe::ExecutionPlan_v8 planFor(const fe::Engine_v8& built)
    {
        const fe::EngineConfig_v8 config = fe::EngineConfigBuilder_v8().setEngine(built).build();
        return fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
    }

    // The hipDNN engine ID behind the hipDNN plan built last.
    std::int64_t lastPlanEngineId() const
    {
        const std::vector<hipdnnBackendDescriptor_t> plans
            = created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        return plans.empty() ? -1 : engineIdOf(plans.back());
    }

    // The hipDNN ENGINECFG the hipDNN plan built last was set up with.
    hipdnnBackendDescriptor_t lastPlanConfig() const
    {
        const std::vector<hipdnnBackendDescriptor_t> plans
            = created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        if(plans.empty())
        {
            return nullptr;
        }
        const std::vector<CapturedSetAttribute> calls
            = callsOn(plans.back(), HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
        return calls.empty() ? nullptr : calls.front().value<hipdnnBackendDescriptor_t>();
    }

    std::vector<CapturedSetAttribute> callsOn(hipdnnBackendDescriptor_t descriptor,
                                              hipdnnBackendAttributeName_t name) const
    {
        std::vector<CapturedSetAttribute> result = setAttributesOf(descriptor);
        result.erase(
            std::remove_if(result.begin(),
                           result.end(),
                           [name](const CapturedSetAttribute& call) { return call.name != name; }),
            result.end());
        return result;
    }

    // Every hipDNN KNOB_CHOICE: the provider knob ID and value it was given.
    std::vector<KnobChoice> hipdnnKnobChoices() const
    {
        std::vector<KnobChoice> choices;
        for(hipdnnBackendDescriptor_t choice : created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR))
        {
            const auto types = callsOn(choice, HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE);
            const auto values = callsOn(choice, HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE);
            EXPECT_EQ(types.size(), 1U);
            EXPECT_EQ(values.size(), 1U);
            if(types.size() != 1 || values.size() != 1)
            {
                continue;
            }
            const std::vector<char> id = types.front().values<char>();
            choices.emplace_back(std::string(id.begin(), std::find(id.begin(), id.end(), '\0')),
                                 values.front().value<std::int64_t>());
        }
        return choices;
    }

    static fe::EngineHeuristics_v8 heuristics(fe::OperationGraph_v8& graph,
                                              cudnnBackendHeurMode_t mode
                                              = CUDNN_HEUR_MODE_FALLBACK)
    {
        return fe::EngineHeuristicsBuilder_v8().setOperationGraph(graph).setHeurMode(mode).build();
    }

    static BuilderFailure configFailure(const fe::Engine_v8& built)
    {
        return builderFailureOf([&built] {
            [[maybe_unused]] auto config = fe::EngineConfigBuilder_v8().setEngine(built).build();
        });
    }

    // The knob engine (index 0 of its graph) with the mixed knobs.
    static fe::Engine_v8 knobEngine(const fe::OperationGraph_v8& graph)
    {
        fe::Engine_v8 built = engine(graph, 0);
        EXPECT_EQ(built.getSupportedKnobs().size(), 1U);
        return built;
    }

    void installKnobEngine(std::vector<MockKnob> knobs = mixedKnobs())
    {
        installHeuristicResults({KNOB_ENGINE});
        installEngineKnobs(KNOB_ENGINE, std::move(knobs));
    }
};

TEST_F(TestCudnnShimV8BuilderEngines, EveryModeRunsFallback)
{
    installHeuristicResults({5, -2});
    fe::OperationGraph_v8 graph = convGraph();

    for(const cudnnBackendHeurMode_t mode :
        {CUDNN_HEUR_MODE_INSTANT, CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_B, CUDNN_HEUR_MODE_FALLBACK})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        fe::EngineHeuristics_v8 built = heuristics(graph, mode);
        const fe::EngineConfigList configs = built.getEngineConfig(built.getEngineConfigCount());
        ASSERT_EQ(configs.size(), 2U);
        std::vector<std::int64_t> ids;
        for(const fe::ManagedOpaqueDescriptor& config : configs)
        {
            const fe::ExecutionPlan_v8 plan = planFor(config, "");
            ids.push_back(lastPlanEngineId());
        }
        EXPECT_EQ(ids, (std::vector<std::int64_t>{5, -2}));
    }

    // hipDNN is only ever asked for FALLBACK.
    const std::vector<CapturedSetAttribute> sets
        = setAttributesOn(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    std::size_t modes = 0;
    for(const CapturedSetAttribute& call : sets)
    {
        if(call.name == HIPDNN_ATTR_ENGINEHEUR_MODE)
        {
            ++modes;
            EXPECT_EQ(call.value<hipdnnBackendHeurMode_t>(), HIPDNN_HEUR_MODE_FALLBACK);
        }
    }
    EXPECT_GE(modes, 1U);
}

TEST_F(TestCudnnShimV8BuilderEngines, UnhonoredModesWarnOncePerProcess)
{
    installHeuristicResults({1});
    fe::OperationGraph_v8 graph = convGraph();
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    fe::detail::warnedHeuristicModes().store(0);

    for(const cudnnBackendHeurMode_t mode : {CUDNN_HEUR_MODE_A,
                                             CUDNN_HEUR_MODE_A,
                                             CUDNN_HEUR_MODE_INSTANT,
                                             CUDNN_HEUR_MODE_FALLBACK,
                                             CUDNN_HEUR_MODE_INSTANT})
    {
        [[maybe_unused]] const fe::EngineHeuristics_v8 built = heuristics(graph, mode);
    }

    EXPECT_EQ(warningsContaining(recorder, "heuristic mode A "), 1U);
    EXPECT_EQ(warningsContaining(recorder, "heuristic mode INSTANT "), 1U);
    EXPECT_EQ(warningsContaining(recorder, "heuristic mode B "), 0U);
    EXPECT_EQ(warningsContaining(recorder, "heuristic mode FALLBACK"), 0U);
    EXPECT_EQ(warningsContaining(recorder, "not honored"), 2U);
}

TEST_F(TestCudnnShimV8BuilderEngines, HeuristicRequiresGraph)
{
    fe::EngineHeuristicsBuilder_v8 builder;
    builder.setHeurMode(CUDNN_HEUR_MODE_FALLBACK);
    const BuilderFailure failure
        = builderFailureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

// The untyped setters take any descriptor; a tensor where the graph belongs is rejected.
TEST_F(TestCudnnShimV8BuilderEngines, WrongKindGraphRejected)
{
    const fe::Tensor_v8 notAGraph = tensor(X_UID, {2, 4, 8, 8});

    fe::EngineHeuristicsBuilder_v8 heuristicsBuilder;
    heuristicsBuilder.setOperationGraph(notAGraph.get_desc(), "")
        .setHeurMode(CUDNN_HEUR_MODE_FALLBACK);
    const BuilderFailure heuristicsFailure = builderFailureOf(
        [&heuristicsBuilder] { [[maybe_unused]] auto built = heuristicsBuilder.build(); });
    ASSERT_TRUE(heuristicsFailure.thrown);
    EXPECT_EQ(heuristicsFailure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).empty());

    fe::EngineBuilder_v8 engineBuilder;
    engineBuilder.setOperationGraph(notAGraph.get_desc()).setGlobalEngineIdx(0);
    const BuilderFailure engineFailure = builderFailureOf(
        [&engineBuilder] { [[maybe_unused]] auto built = engineBuilder.build(); });
    ASSERT_TRUE(engineFailure.thrown);
    EXPECT_EQ(engineFailure.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderEngines, ResultsFollowRankWhileIndexFollowsId)
{
    installHeuristicResults({5, -2, 300, -40});
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineHeuristics_v8 built = heuristics(graph);

    ASSERT_EQ(built.getEngineConfigCount(), 4);
    const fe::EngineConfigList configs = built.getEngineConfig(4);
    ASSERT_EQ(configs.size(), 4U);

    // Rank order {5, -2, 300, -40}; dense index by signed ID {-40, -2, 5, 300}.
    const std::vector<std::int64_t> rankedIds = {5, -2, 300, -40};
    const std::vector<std::int64_t> indices = {2, 1, 3, 0};
    for(std::size_t i = 0; i < configs.size(); ++i)
    {
        SCOPED_TRACE(i);
        const fe::ExecutionPlan_v8 plan = planFor(configs[i], "");
        EXPECT_EQ(plan.getTag(), "eng" + std::to_string(indices[i]));
        EXPECT_EQ(lastPlanEngineId(), rankedIds[i]);
    }
    EXPECT_TRUE(created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8BuilderEngines, ResultsFillAtMostRequested)
{
    installHeuristicResults({3, 1, 2});
    fe::OperationGraph_v8 graph = convGraph();

    fe::EngineHeuristics_v8 two = heuristics(graph);
    const fe::EngineConfigList firstTwo = two.getEngineConfig(2);
    ASSERT_EQ(firstTwo.size(), 2U);
    const fe::ExecutionPlan_v8 plan = planFor(firstTwo[1], "");
    EXPECT_EQ(lastPlanEngineId(), 1);

    fe::EngineHeuristics_v8 more = heuristics(graph);
    EXPECT_EQ(more.getEngineConfig(5).size(), 3U);
}

TEST_F(TestCudnnShimV8BuilderEngines, FailingConfigDropsOneEngine)
{
    installHeuristicResults({1, 2, 3}, {2});
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineHeuristics_v8 built = heuristics(graph);

    ASSERT_EQ(built.getEngineConfigCount(), 2);
    const fe::EngineConfigList configs = built.getEngineConfig(2);
    ASSERT_EQ(configs.size(), 2U);
    EXPECT_EQ(planFor(configs[0], "").getTag(), "eng0");
    EXPECT_EQ(lastPlanEngineId(), 1);
    EXPECT_EQ(planFor(configs[1], "").getTag(), "eng1");
    EXPECT_EQ(lastPlanEngineId(), 3);
    EXPECT_EQ(graph.getEngineCount(), 2);
}

TEST_F(TestCudnnShimV8BuilderEngines, NoApplicableEngineGivesEmptyResults)
{
    installHeuristicResults({});
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineHeuristics_v8 built = heuristics(graph);

    EXPECT_EQ(built.getEngineConfigCount(), 0);
    EXPECT_TRUE(built.getEngineConfig(0).empty());
    EXPECT_EQ(graph.getEngineCount(), 0);
}

TEST_F(TestCudnnShimV8BuilderEngines, EngineIndexResolvesId)
{
    installHeuristicResults({300, -40, 5});
    const fe::OperationGraph_v8 graph = convGraph();
    EXPECT_EQ(graph.getEngineCount(), 3);

    const std::vector<std::int64_t> byIndex = {-40, 5, 300};
    for(std::size_t index = 0; index < byIndex.size(); ++index)
    {
        SCOPED_TRACE(index);
        const auto signedIndex = static_cast<std::int64_t>(index);
        const fe::ExecutionPlan_v8 plan = planFor(graph, signedIndex);
        EXPECT_EQ(plan.getTag(), "ConvFwd_eng" + std::to_string(signedIndex));
        EXPECT_EQ(lastPlanEngineId(), byIndex[index]);
    }

    for(const std::int64_t outOfRange : {std::int64_t{3}, std::int64_t{-1}})
    {
        SCOPED_TRACE(outOfRange);
        fe::EngineBuilder_v8 builder;
        builder.setOperationGraph(graph).setGlobalEngineIdx(outOfRange);
        const BuilderFailure failure
            = builderFailureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    }

    fe::EngineBuilder_v8 noGraph;
    noGraph.setGlobalEngineIdx(0);
    const BuilderFailure failure
        = builderFailureOf([&noGraph] { [[maybe_unused]] auto built = noGraph.build(); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

// A second heuristic and its results reuse the graph's engines; a knob-free
// config, from heuristics or EngineBuilder, plans on the engine's one hipDNN config.
TEST_F(TestCudnnShimV8BuilderEngines, KnobFreeConfigsShareDefaultConfig)
{
    installHeuristicResults({4, 9});
    fe::OperationGraph_v8 graph = convGraph();
    fe::EngineHeuristics_v8 first = heuristics(graph);
    const fe::EngineConfigList firstConfigs = first.getEngineConfig(first.getEngineConfigCount());
    const std::size_t heuristicsCreated = created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).size();
    const std::size_t configsCreated = created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size();

    fe::EngineHeuristics_v8 again = heuristics(graph);
    const fe::EngineConfigList configs = again.getEngineConfig(again.getEngineConfigCount());
    ASSERT_EQ(configs.size(), 2U);
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).size(), heuristicsCreated);
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), configsCreated);

    const fe::ExecutionPlan_v8 fromHeuristics = planFor(configs[1], "");
    auto* const shared = lastPlanConfig();
    ASSERT_NE(shared, nullptr);
    EXPECT_EQ(engineIdOf(shared), 9);
    const fe::ExecutionPlan_v8 fromFirst = planFor(firstConfigs[1], "");
    EXPECT_EQ(lastPlanConfig(), shared);
    const fe::ExecutionPlan_v8 fromEngine = planFor(graph, 1);
    EXPECT_EQ(lastPlanConfig(), shared);

    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), configsCreated);
    EXPECT_TRUE(created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8BuilderEngines, KnobChoiceProjectedOntoProviderKnob)
{
    installKnobEngine();
    const fe::OperationGraph_v8 graph = convGraph();
    const fe::ExecutionPlan_v8 plain = planFor(graph, 0);
    auto* const defaultConfig = lastPlanConfig();

    fe::Engine_v8 built = knobEngine(graph);
    choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 16);
    const fe::ExecutionPlan_v8 plan = planFor(built);
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    EXPECT_EQ(plan.getTag(), "ConvFwd_eng0" + knobSegment(CUDNN_KNOB_TYPE_TILE_SIZE, 16));

    auto* const knobbed = lastPlanConfig();
    ASSERT_NE(knobbed, nullptr);
    EXPECT_NE(knobbed, defaultConfig);
    EXPECT_EQ(engineIdOf(knobbed), KNOB_ENGINE);
    EXPECT_EQ(hipdnnKnobChoices(), (std::vector<KnobChoice>{{"prov.tile_size", 16}}));

    // KNOB_CHOICES before finalize.
    const std::vector<CapturedSetAttribute> all = setAttributes();
    const auto choicePosition = static_cast<std::size_t>(
        std::find_if(all.begin(),
                     all.end(),
                     [knobbed](const CapturedSetAttribute& call) {
                         return call.descriptor == knobbed
                                && call.name == HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES;
                     })
        - all.begin());
    ASSERT_LT(choicePosition, all.size());
    const std::optional<std::size_t> setsBeforeFinalize = setCountAtFinalize(knobbed);
    ASSERT_TRUE(setsBeforeFinalize.has_value());
    EXPECT_LT(choicePosition, *setsBeforeFinalize);
}

TEST_F(TestCudnnShimV8BuilderEngines, KnobConfigBuiltOnceAcrossThreads)
{
    installKnobEngine();
    const fe::OperationGraph_v8 graph = convGraph();
    fe::Engine_v8 built = knobEngine(graph);
    choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 32);
    const fe::EngineConfig_v8 config = fe::EngineConfigBuilder_v8().setEngine(built).build();

    constexpr std::size_t THREAD_COUNT = 8;
    std::atomic<bool> start{false};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(std::size_t i = 0; i < THREAD_COUNT; ++i)
    {
        threads.emplace_back([&] {
            while(!start.load())
            {
                std::this_thread::yield();
            }
            const BuilderFailure failure = builderFailureOf([&config] {
                [[maybe_unused]] auto plan
                    = fe::ExecutionPlanBuilder_v8().setEngineConfig(config).build();
            });
            if(failure.thrown)
            {
                ++failures;
            }
        });
    }
    start.store(true);
    for(std::thread& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(failures.load(), 0);
    EXPECT_EQ(created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR).size(), 1U);
    std::vector<hipdnnBackendDescriptor_t> planConfigs;
    for(hipdnnBackendDescriptor_t plan : created(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR))
    {
        for(const CapturedSetAttribute& call :
            callsOn(plan, HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG))
        {
            planConfigs.push_back(call.value<hipdnnBackendDescriptor_t>());
        }
    }
    ASSERT_EQ(planConfigs.size(), THREAD_COUNT);
    for(hipdnnBackendDescriptor_t planConfig : planConfigs)
    {
        EXPECT_EQ(planConfig, planConfigs.front());
    }
}

TEST_F(TestCudnnShimV8BuilderEngines, HiddenKnobChoiceNotSupported)
{
    installKnobEngine();
    const fe::OperationGraph_v8 graph = convGraph();

    // Valid values, a colliding type, and a type the engine does not have.
    for(const cudnnBackendKnobType_t type :
        {CUDNN_KNOB_TYPE_BLOCK_SIZE, CUDNN_KNOB_TYPE_STAGES, CUDNN_KNOB_TYPE_EDGE})
    {
        SCOPED_TRACE(static_cast<int>(type));
        fe::Engine_v8 built = knobEngine(graph);
        choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 8);
        appendKnob(built, type, 8);
        const BuilderFailure failure = configFailure(built);
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST_F(TestCudnnShimV8BuilderEngines, DuplicateKnobChoiceRejected)
{
    installKnobEngine();
    const fe::OperationGraph_v8 graph = convGraph();
    fe::Engine_v8 built = knobEngine(graph);
    choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 8);
    appendKnob(built, CUDNN_KNOB_TYPE_TILE_SIZE, 16);

    const BuilderFailure failure = configFailure(built);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderEngines, DuplicateHiddenKnobNotSupported)
{
    installKnobEngine();
    const fe::OperationGraph_v8 graph = convGraph();
    fe::Engine_v8 built = knobEngine(graph);
    choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 8);
    appendKnob(built, CUDNN_KNOB_TYPE_BLOCK_SIZE, 8);
    appendKnob(built, CUDNN_KNOB_TYPE_BLOCK_SIZE, 16);

    const BuilderFailure failure = configFailure(built);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestCudnnShimV8BuilderEngines, InvalidKnobValueRejected)
{
    installKnobEngine();
    const fe::OperationGraph_v8 graph = convGraph();

    // Above the maximum, below the minimum, and off the stride.
    for(const std::int64_t value : {std::int64_t{72}, std::int64_t{-8}, std::int64_t{12}})
    {
        SCOPED_TRACE(value);
        fe::Engine_v8 built = knobEngine(graph);
        choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, value);
        const BuilderFailure failure = configFailure(built);
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    }
}

// Upstream's builder sends -1 for every knob it was not asked to set.
TEST_F(TestCudnnShimV8BuilderEngines, UnsetKnobChoiceIgnored)
{
    std::vector<MockKnob> knobs = mixedKnobs();
    knobs.push_back({"prov.split_k_slc", 1, 1, 8, 1, {}});
    installKnobEngine(knobs);
    const fe::OperationGraph_v8 graph = convGraph();
    fe::Engine_v8 built = engine(graph, 0);
    ASSERT_EQ(built.getSupportedKnobs().size(), 2U);

    // Visible, hidden and repeated unset choices all drop out.
    choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 16);
    appendKnob(built, CUDNN_KNOB_TYPE_BLOCK_SIZE, -1);
    appendKnob(built, CUDNN_KNOB_TYPE_TILE_SIZE, -1);
    const fe::ExecutionPlan_v8 plan = planFor(built);
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    EXPECT_EQ(hipdnnKnobChoices(), (std::vector<KnobChoice>{{"prov.tile_size", 16}}));

    // The tag keeps every stored choice, -1 included.
    std::string expected = "ConvFwd_eng0";
    for(const auto& knob : built.getSupportedKnobs())
    {
        expected += knobSegment(knob.getKnobType(), knob.getChoice());
    }
    EXPECT_EQ(plan.getTag(), expected);
    EXPECT_NE(plan.getTag().find(knobSegment(CUDNN_KNOB_TYPE_SPLIT_K_SLC, -1)), std::string::npos);
}

TEST_F(TestCudnnShimV8BuilderEngines, MinusOneKeptWhenInRange)
{
    installKnobEngine({{"prov.tile_size", 8, 0, 64, 8, {}}, {"prov.split_k_slc", 1, -1, 8, 1, {}}});
    const fe::OperationGraph_v8 graph = convGraph();
    fe::Engine_v8 built = engine(graph, 0);
    ASSERT_EQ(built.getSupportedKnobs().size(), 2U);
    choose(built, CUDNN_KNOB_TYPE_TILE_SIZE, 16);

    const fe::ExecutionPlan_v8 plan = planFor(built);
    ASSERT_EQ(plan.get_status(), CUDNN_STATUS_SUCCESS) << plan.get_error();
    std::vector<KnobChoice> choices = hipdnnKnobChoices();
    std::sort(choices.begin(), choices.end());
    EXPECT_EQ(choices, (std::vector<KnobChoice>{{"prov.split_k_slc", -1}, {"prov.tile_size", 16}}));

    // A real choice, so a repeat is a duplicate.
    appendKnob(built, CUDNN_KNOB_TYPE_SPLIT_K_SLC, -1);
    const BuilderFailure failure = configFailure(built);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderEngines, BehaviorNotesMappedByName)
{
    installHeuristicResults({11, 12});
    installBehaviorNotes(11,
                         {HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION,
                          HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    installEngineName(11, "mock_engine_eleven");
    installSerializedPlan({0x01});
    const fe::OperationGraph_v8 graph = convGraph();

    const fe::ExecutionPlan_v8 mapped = planFor(graph, 0);
    EXPECT_EQ(mapped.getAllBehaviorNotes(),
              std::vector<cudnnBackendBehaviorNote_t>{CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    EXPECT_NE(mapped.getJsonRepresentation().find(R"("name":"mock_engine_eleven")"),
              std::string::npos);

    const fe::ExecutionPlan_v8 bare = planFor(graph, 1);
    EXPECT_TRUE(bare.getAllBehaviorNotes().empty());
    EXPECT_EQ(bare.getJsonRepresentation().find("\"name\""), std::string::npos);
}

} // namespace
