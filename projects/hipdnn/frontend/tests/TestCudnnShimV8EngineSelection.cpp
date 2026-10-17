// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// ENGINEHEUR, ENGINE, ENGINECFG, KNOB_INFO and KNOB_CHOICE against the mock
// hipDNN backend: heuristic modes, result order, the dense engine index, knob
// projection and validation, and when the knob-bearing hipDNN config is built.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

namespace backend = hipdnn_frontend::compatibility::cudnn_frontend::detail::backend;
namespace shim = hipdnn_frontend::compatibility::cudnn_frontend::detail;

using hipdnn_shim_test::CapturedSetAttribute;
using hipdnn_shim_test::MockKnob;
using Choices = std::vector<std::pair<cudnnBackendKnobType_t, std::int64_t>>;

constexpr std::int64_t KNOB_ENGINE = 7;

std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

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

class TestCudnnShimV8EngineSelection : public hipdnn_shim_test::ShimV8MockBackendFixture
{
protected:
    std::vector<cudnnBackendDescriptor_t> _owned;

    void SetUp() override
    {
        ShimV8MockBackendFixture::SetUp();
        lastError();
    }

    void TearDown() override
    {
        for(cudnnBackendDescriptor_t descriptor : _owned)
        {
            cudnnBackendDestroyDescriptor(descriptor);
        }
        _owned.clear();
        ShimV8MockBackendFixture::TearDown();
    }

    cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type)
    {
        cudnnBackendDescriptor_t descriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(type, &descriptor), CUDNN_STATUS_SUCCESS);
        _owned.push_back(descriptor);
        return descriptor;
    }

    template <typename T>
    static void set(cudnnBackendDescriptor_t descriptor,
                    cudnnBackendAttributeName_t name,
                    cudnnBackendAttributeType_t type,
                    const T& value)
    {
        EXPECT_EQ(
            cudnnBackendSetAttribute(descriptor, name, type, 1, static_cast<const void*>(&value)),
            CUDNN_STATUS_SUCCESS)
            << lastError();
    }

    static void finalize(cudnnBackendDescriptor_t descriptor)
    {
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS) << lastError();
    }

    cudnnBackendDescriptor_t tensor(std::int64_t uid)
    {
        const std::vector<std::int64_t> dims = {1, 4, 8, 8};
        const std::vector<std::int64_t> strides = {256, 64, 8, 1};
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        EXPECT_EQ(cudnnBackendSetAttribute(descriptor,
                                           CUDNN_ATTR_TENSOR_DIMENSIONS,
                                           CUDNN_TYPE_INT64,
                                           4,
                                           static_cast<const void*>(dims.data())),
                  CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(cudnnBackendSetAttribute(descriptor,
                                           CUDNN_ATTR_TENSOR_STRIDES,
                                           CUDNN_TYPE_INT64,
                                           4,
                                           static_cast<const void*>(strides.data())),
                  CUDNN_STATUS_SUCCESS);
        set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, uid);
        set(descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, std::int64_t{16});
        finalize(descriptor);
        return descriptor;
    }

    // relu(x:1) -> y:2, finalized.
    cudnnBackendDescriptor_t reluGraph()
    {
        cudnnBackendDescriptor_t relu = create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        set(relu, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, CUDNN_POINTWISE_RELU_FWD);
        set(relu, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        finalize(relu);
        cudnnBackendDescriptor_t op = create(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        set(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, relu);
        set(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, tensor(1));
        set(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, tensor(2));
        finalize(op);
        cudnnBackendDescriptor_t graph = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, _handle);
        set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, op);
        finalize(graph);
        return graph;
    }

    cudnnBackendDescriptor_t heuristic(cudnnBackendDescriptor_t graph,
                                       cudnnBackendHeurMode_t mode = CUDNN_HEUR_MODE_FALLBACK)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        set(descriptor,
            CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            graph);
        set(descriptor, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, mode);
        finalize(descriptor);
        return descriptor;
    }

    static std::int64_t resultCount(cudnnBackendDescriptor_t heuristicDescriptor)
    {
        std::int64_t count = -1;
        EXPECT_EQ(cudnnBackendGetAttribute(heuristicDescriptor,
                                           CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           0,
                                           &count,
                                           nullptr),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        return count;
    }

    // Creates `slots` caller configs and fills them from RESULTS in place.
    std::vector<cudnnBackendDescriptor_t> results(cudnnBackendDescriptor_t heuristicDescriptor,
                                                  std::size_t slots,
                                                  std::int64_t& returned)
    {
        std::vector<cudnnBackendDescriptor_t> configs;
        configs.reserve(slots);
        for(std::size_t i = 0; i < slots; ++i)
        {
            configs.push_back(create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
        }
        EXPECT_EQ(cudnnBackendGetAttribute(heuristicDescriptor,
                                           CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           static_cast<std::int64_t>(slots),
                                           &returned,
                                           static_cast<void*>(configs.data())),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        return configs;
    }

    // Upstream's pattern: fill a caller-created ENGINE from the config.
    cudnnBackendDescriptor_t engineOf(cudnnBackendDescriptor_t config)
    {
        cudnnBackendDescriptor_t engine = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        std::int64_t count = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(config,
                                           CUDNN_ATTR_ENGINECFG_ENGINE,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &count,
                                           static_cast<void*>(&engine)),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        return engine;
    }

    static std::int64_t globalIndex(cudnnBackendDescriptor_t engine)
    {
        std::int64_t index = -1;
        std::int64_t count = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(engine,
                                           CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &count,
                                           static_cast<void*>(&index)),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        return index;
    }

    cudnnBackendDescriptor_t unfinalizedEngine(cudnnBackendDescriptor_t graph, std::int64_t index)
    {
        cudnnBackendDescriptor_t engine = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        set(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, graph);
        set(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, index);
        return engine;
    }

    cudnnBackendDescriptor_t engine(cudnnBackendDescriptor_t graph, std::int64_t index)
    {
        cudnnBackendDescriptor_t descriptor = unfinalizedEngine(graph, index);
        finalize(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t knobChoice(cudnnBackendKnobType_t type, std::int64_t value)
    {
        cudnnBackendDescriptor_t choice = create(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
        set(choice, CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, type);
        set(choice, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, value);
        finalize(choice);
        return choice;
    }

    // Not finalized.
    cudnnBackendDescriptor_t config(cudnnBackendDescriptor_t engineDescriptor,
                                    const Choices& choices)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        set(descriptor,
            CUDNN_ATTR_ENGINECFG_ENGINE,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            engineDescriptor);
        std::vector<cudnnBackendDescriptor_t> knobs;
        knobs.reserve(choices.size());
        for(const auto& [type, value] : choices)
        {
            knobs.push_back(knobChoice(type, value));
        }
        EXPECT_EQ(cudnnBackendSetAttribute(descriptor,
                                           CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           static_cast<std::int64_t>(knobs.size()),
                                           static_cast<const void*>(knobs.data())),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        return descriptor;
    }

    // A finalized knob engine (KNOB_ENGINE, index 0 of its graph).
    cudnnBackendDescriptor_t knobEngine()
    {
        installHeuristicResults({KNOB_ENGINE});
        installEngineKnobs(KNOB_ENGINE, mixedKnobs());
        return engine(reluGraph(), 0);
    }

    template <typename Note>
    static std::vector<Note> notes(cudnnBackendDescriptor_t engineDescriptor,
                                   cudnnBackendAttributeName_t name,
                                   cudnnBackendAttributeType_t type)
    {
        std::int64_t count = -1;
        EXPECT_EQ(cudnnBackendGetAttribute(engineDescriptor, name, type, 0, &count, nullptr),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        std::vector<Note> result(static_cast<std::size_t>(std::max<std::int64_t>(count, 0)));
        if(!result.empty())
        {
            EXPECT_EQ(
                cudnnBackendGetAttribute(
                    engineDescriptor, name, type, count, &count, static_cast<void*>(result.data())),
                CUDNN_STATUS_SUCCESS)
                << lastError();
        }
        return result;
    }

    static std::int64_t int64Of(cudnnBackendDescriptor_t descriptor,
                                cudnnBackendAttributeName_t name)
    {
        std::int64_t value = -1;
        std::int64_t count = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(
                      descriptor, name, CUDNN_TYPE_INT64, 1, &count, static_cast<void*>(&value)),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        return value;
    }

    static backend::OperationGraphDescriptor& shimGraph(cudnnBackendDescriptor_t descriptor)
    {
        return backend::descriptorCast<backend::OperationGraphDescriptor>(
            *backend::liveHandle(descriptor)->object, CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    }

    static backend::EngineConfigDescriptor& shimConfig(cudnnBackendDescriptor_t descriptor)
    {
        return backend::descriptorCast<backend::EngineConfigDescriptor>(
            *backend::liveHandle(descriptor)->object, CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    }

    static backend::EngineDescriptor& shimEngine(cudnnBackendDescriptor_t descriptor)
    {
        return backend::descriptorCast<backend::EngineDescriptor>(
            *backend::liveHandle(descriptor)->object, CUDNN_BACKEND_ENGINE_DESCRIPTOR);
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
};

TEST_F(TestCudnnShimV8EngineSelection, EveryModeRunsFallback)
{
    installHeuristicResults({5, -2});
    cudnnBackendDescriptor_t graph = reluGraph();

    for(const cudnnBackendHeurMode_t mode :
        {CUDNN_HEUR_MODE_INSTANT, CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_B, CUDNN_HEUR_MODE_FALLBACK})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        cudnnBackendDescriptor_t descriptor = heuristic(graph, mode);
        std::int64_t returned = 0;
        const std::vector<cudnnBackendDescriptor_t> configs = results(descriptor, 2, returned);
        ASSERT_EQ(returned, 2);
        EXPECT_EQ(shimConfig(configs[0]).engineId(), 5);
        EXPECT_EQ(shimConfig(configs[1]).engineId(), -2);
    }

    // One hipDNN FALLBACK sweep serves every mode.
    const std::vector<hipdnnBackendDescriptor_t> heuristics
        = created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    ASSERT_EQ(heuristics.size(), 1U);
    const std::vector<CapturedSetAttribute> modes
        = callsOn(heuristics.front(), HIPDNN_ATTR_ENGINEHEUR_MODE);
    ASSERT_EQ(modes.size(), 1U);
    EXPECT_EQ(modes.front().value<hipdnnBackendHeurMode_t>(), HIPDNN_HEUR_MODE_FALLBACK);
}

TEST_F(TestCudnnShimV8EngineSelection, UnhonoredModesWarnOncePerProcess)
{
    installHeuristicResults({1});
    cudnnBackendDescriptor_t graph = reluGraph();
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    shim::warnedHeuristicModes().store(0);

    heuristic(graph, CUDNN_HEUR_MODE_A);
    heuristic(graph, CUDNN_HEUR_MODE_A);
    heuristic(graph, CUDNN_HEUR_MODE_INSTANT);
    heuristic(graph, CUDNN_HEUR_MODE_FALLBACK);
    heuristic(graph, CUDNN_HEUR_MODE_INSTANT);

    EXPECT_EQ(warningsContaining(recorder, "heuristic mode A "), 1U);
    EXPECT_EQ(warningsContaining(recorder, "heuristic mode INSTANT "), 1U);
    EXPECT_EQ(warningsContaining(recorder, "heuristic mode B "), 0U);
    EXPECT_EQ(warningsContaining(recorder, "heuristic mode FALLBACK"), 0U);
    EXPECT_EQ(warningsContaining(recorder, "not honored"), 2U);
}

TEST_F(TestCudnnShimV8EngineSelection, HeuristicTargetsNotSupported)
{
    cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    const std::int32_t smCount = 8;
    EXPECT_EQ(cudnnBackendSetAttribute(descriptor,
                                       CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET,
                                       CUDNN_TYPE_INT32,
                                       1,
                                       static_cast<const void*>(&smCount)),
              CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(lastError().find("CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET"), std::string::npos);
}

TEST_F(TestCudnnShimV8EngineSelection, HeuristicRequiresGraphAndMode)
{
    installHeuristicResults({1});
    cudnnBackendDescriptor_t noGraph = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    set(noGraph, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, CUDNN_HEUR_MODE_FALLBACK);
    EXPECT_EQ(cudnnBackendFinalize(noGraph), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH"), std::string::npos);

    cudnnBackendDescriptor_t noMode = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    set(noMode, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, reluGraph());
    EXPECT_EQ(cudnnBackendFinalize(noMode), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_ENGINEHEUR_MODE"), std::string::npos);
}

TEST_F(TestCudnnShimV8EngineSelection, ResultsFollowRankWhileIndexFollowsId)
{
    installHeuristicResults({5, -2, 300, -40});
    cudnnBackendDescriptor_t descriptor = heuristic(reluGraph());

    EXPECT_EQ(resultCount(descriptor), 4);
    std::int64_t returned = 0;
    const std::vector<cudnnBackendDescriptor_t> configs = results(descriptor, 4, returned);
    ASSERT_EQ(returned, 4);

    // Rank order {5, -2, 300, -40}; dense index by signed ID {-40, -2, 5, 300}.
    const std::vector<std::int64_t> rankedIds = {5, -2, 300, -40};
    const std::vector<std::int64_t> indices = {2, 1, 3, 0};
    for(std::size_t i = 0; i < configs.size(); ++i)
    {
        SCOPED_TRACE(i);
        const backend::EngineConfigDescriptor& config = shimConfig(configs[i]);
        EXPECT_TRUE(config.isFinalized());
        EXPECT_EQ(config.engineId(), rankedIds[i]);
        EXPECT_EQ(config.engineIndex(), indices[i]);
        EXPECT_TRUE(config.knobChoices().empty());
        EXPECT_EQ(globalIndex(engineOf(configs[i])), indices[i]);
    }
}

TEST_F(TestCudnnShimV8EngineSelection, ResultsFillAtMostRequested)
{
    installHeuristicResults({3, 1, 2});
    cudnnBackendDescriptor_t descriptor = heuristic(reluGraph());

    std::int64_t returned = 0;
    std::vector<cudnnBackendDescriptor_t> configs = results(descriptor, 2, returned);
    EXPECT_EQ(returned, 2);
    EXPECT_EQ(shimConfig(configs[1]).engineId(), 1);

    configs = results(descriptor, 5, returned);
    EXPECT_EQ(returned, 3);
    // Slots past the result count stay the caller's unfinalized configs.
    std::int64_t count = 0;
    cudnnBackendDescriptor_t engineSlot = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendGetAttribute(configs[3],
                                       CUDNN_ATTR_ENGINECFG_ENGINE,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&engineSlot)),
              CUDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestCudnnShimV8EngineSelection, ResultsMakeNoHipdnnConfigs)
{
    installHeuristicResults({4, 9});
    cudnnBackendDescriptor_t graph = reluGraph();
    heuristic(graph);
    const std::size_t createdConfigs = created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size();
    const std::size_t getCalls = getAttributeNames().size();
    const std::size_t finalizeCalls = finalized().size();

    cudnnBackendDescriptor_t again = heuristic(graph);
    std::int64_t returned = 0;
    const std::vector<cudnnBackendDescriptor_t> configs = results(again, 2, returned);
    ASSERT_EQ(returned, 2);

    // A second heuristic and its results reuse the graph's snapshot.
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), createdConfigs);
    EXPECT_EQ(getAttributeNames().size(), getCalls);
    EXPECT_EQ(finalized().size(), finalizeCalls);
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR).size(), 1U);

    // A knob-free config is the graph's cached default hipDNN config.
    const backend::OperationGraphDescriptor& shimGraphRef = shimGraph(graph);
    for(const auto& descriptor : configs)
    {
        const backend::EngineConfigDescriptor& config = shimConfig(descriptor);
        EXPECT_EQ(config.hipdnnEngineConfig(), shimGraphRef.defaultEngineConfig(config.engineId()));
        EXPECT_EQ(&config.graph(), &shimGraphRef);
    }
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), createdConfigs);
}

TEST_F(TestCudnnShimV8EngineSelection, FailingConfigDropsOneEngine)
{
    installHeuristicResults({1, 2, 3}, {2});
    cudnnBackendDescriptor_t descriptor = heuristic(reluGraph());

    EXPECT_EQ(resultCount(descriptor), 2);
    std::int64_t returned = 0;
    const std::vector<cudnnBackendDescriptor_t> configs = results(descriptor, 2, returned);
    ASSERT_EQ(returned, 2);
    EXPECT_EQ(shimConfig(configs[0]).engineId(), 1);
    EXPECT_EQ(shimConfig(configs[1]).engineId(), 3);
    EXPECT_EQ(shimConfig(configs[1]).engineIndex(), 1);
}

TEST_F(TestCudnnShimV8EngineSelection, NoApplicableEngineGivesEmptyResults)
{
    installHeuristicResults({});
    cudnnBackendDescriptor_t descriptor = heuristic(reluGraph());
    EXPECT_EQ(resultCount(descriptor), 0);
}

TEST_F(TestCudnnShimV8EngineSelection, EngineIndexResolvesId)
{
    installHeuristicResults({300, -40, 5});
    cudnnBackendDescriptor_t graph = reluGraph();

    const std::vector<std::int64_t> byIndex = {-40, 5, 300};
    for(std::size_t index = 0; index < byIndex.size(); ++index)
    {
        const auto signedIndex = static_cast<std::int64_t>(index);
        cudnnBackendDescriptor_t descriptor = engine(graph, signedIndex);
        const backend::EngineDescriptor& shimEngineRef = shimEngine(descriptor);
        EXPECT_EQ(shimEngineRef.engineId(), byIndex[index]);
        EXPECT_EQ(shimEngineRef.engineIndex(), signedIndex);
        EXPECT_EQ(shimEngineRef.hipdnnEngine(), shimGraph(graph).defaultEngine(byIndex[index]));
        EXPECT_EQ(globalIndex(descriptor), signedIndex);
    }

    for(const std::int64_t outOfRange : {std::int64_t{3}, std::int64_t{-1}})
    {
        cudnnBackendDescriptor_t descriptor = unfinalizedEngine(graph, outOfRange);
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_BAD_PARAM);
        EXPECT_NE(lastError().find("outside [0, 3)"), std::string::npos);
    }

    cudnnBackendDescriptor_t noGraph = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    set(noGraph, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, std::int64_t{0});
    EXPECT_EQ(cudnnBackendFinalize(noGraph), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_ENGINE_OPERATION_GRAPH"), std::string::npos);
}

TEST_F(TestCudnnShimV8EngineSelection, KnobInfoShowsOnlyProjectableKnobs)
{
    cudnnBackendDescriptor_t descriptor = knobEngine();

    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(descriptor,
                                       CUDNN_ATTR_ENGINE_KNOB_INFO,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &count,
                                       nullptr),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(count, 1);

    // Upstream asks for CUDNN_KNOB_TYPE_COUNTS caller-created descriptors.
    std::vector<cudnnBackendDescriptor_t> infos;
    infos.reserve(CUDNN_KNOB_TYPE_COUNTS);
    for(int i = 0; i < CUDNN_KNOB_TYPE_COUNTS; ++i)
    {
        infos.push_back(create(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR));
    }
    EXPECT_EQ(cudnnBackendGetAttribute(descriptor,
                                       CUDNN_ATTR_ENGINE_KNOB_INFO,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       CUDNN_KNOB_TYPE_COUNTS,
                                       &count,
                                       static_cast<void*>(infos.data())),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    ASSERT_EQ(count, 1);

    cudnnBackendKnobType_t type = CUDNN_KNOB_TYPE_COUNTS;
    std::int64_t written = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(infos[0],
                                       CUDNN_ATTR_KNOB_INFO_TYPE,
                                       CUDNN_TYPE_KNOB_TYPE,
                                       1,
                                       &written,
                                       static_cast<void*>(&type)),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    EXPECT_EQ(type, CUDNN_KNOB_TYPE_TILE_SIZE);
    EXPECT_EQ(int64Of(infos[0], CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE), 0);
    EXPECT_EQ(int64Of(infos[0], CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE), 64);
    EXPECT_EQ(int64Of(infos[0], CUDNN_ATTR_KNOB_INFO_STRIDE), 8);

    // The slots past the count stay unfinalized; a caller cannot finalize one.
    EXPECT_EQ(cudnnBackendFinalize(infos[1]), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8EngineSelection, HiddenKnobChoiceNotSupported)
{
    cudnnBackendDescriptor_t descriptor = knobEngine();

    // Valid values, a colliding type, and a type the engine does not have.
    for(const cudnnBackendKnobType_t type :
        {CUDNN_KNOB_TYPE_BLOCK_SIZE, CUDNN_KNOB_TYPE_STAGES, CUDNN_KNOB_TYPE_EDGE})
    {
        SCOPED_TRACE(static_cast<int>(type));
        cudnnBackendDescriptor_t hidden = config(descriptor, {{type, 8}});
        EXPECT_EQ(cudnnBackendFinalize(hidden), CUDNN_STATUS_NOT_SUPPORTED);
        EXPECT_NE(lastError().find("is not exposed by engine 7"), std::string::npos);
    }
}

TEST_F(TestCudnnShimV8EngineSelection, DuplicateKnobChoiceRejected)
{
    cudnnBackendDescriptor_t descriptor = knobEngine();
    cudnnBackendDescriptor_t duplicate
        = config(descriptor, {{CUDNN_KNOB_TYPE_TILE_SIZE, 8}, {CUDNN_KNOB_TYPE_TILE_SIZE, 16}});
    EXPECT_EQ(cudnnBackendFinalize(duplicate), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_KNOB_TYPE_TILE_SIZE is chosen more than once"),
              std::string::npos);
}

TEST_F(TestCudnnShimV8EngineSelection, DuplicateHiddenKnobNotSupported)
{
    cudnnBackendDescriptor_t descriptor = knobEngine();
    cudnnBackendDescriptor_t duplicate
        = config(descriptor, {{CUDNN_KNOB_TYPE_BLOCK_SIZE, 8}, {CUDNN_KNOB_TYPE_BLOCK_SIZE, 16}});
    EXPECT_EQ(cudnnBackendFinalize(duplicate), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(lastError().find("CUDNN_KNOB_TYPE_BLOCK_SIZE is not exposed by engine 7"),
              std::string::npos);
}

TEST_F(TestCudnnShimV8EngineSelection, InvalidKnobValueRejected)
{
    cudnnBackendDescriptor_t descriptor = knobEngine();
    // Above the maximum, below the minimum, and off the stride.
    for(const std::int64_t value : {std::int64_t{72}, std::int64_t{-8}, std::int64_t{12}})
    {
        SCOPED_TRACE(value);
        cudnnBackendDescriptor_t invalid = config(descriptor, {{CUDNN_KNOB_TYPE_TILE_SIZE, value}});
        EXPECT_EQ(cudnnBackendFinalize(invalid), CUDNN_STATUS_BAD_PARAM);
        EXPECT_NE(lastError().find("CUDNN_KNOB_TYPE_TILE_SIZE = " + std::to_string(value)),
                  std::string::npos);
    }
}

TEST_F(TestCudnnShimV8EngineSelection, KnobChoiceRequiresTypeAndValue)
{
    cudnnBackendDescriptor_t noValue = create(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
    set(noValue, CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, CUDNN_KNOB_TYPE_TILE_SIZE);
    EXPECT_EQ(cudnnBackendFinalize(noValue), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE"), std::string::npos);

    cudnnBackendDescriptor_t noType = create(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
    set(noType, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, std::int64_t{8});
    EXPECT_EQ(cudnnBackendFinalize(noType), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE"), std::string::npos);
}

TEST_F(TestCudnnShimV8EngineSelection, KnobConfigBuiltLazilyWithChoicesFirst)
{
    cudnnBackendDescriptor_t descriptor = knobEngine();
    const std::size_t defaultConfigs = created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size();

    cudnnBackendDescriptor_t knobbed = config(descriptor, {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}});
    finalize(knobbed);
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), defaultConfigs);
    EXPECT_TRUE(created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR).empty());

    const backend::EngineConfigDescriptor& shimKnobbed = shimConfig(knobbed);
    EXPECT_EQ(shimKnobbed.knobChoices(), (Choices{{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}));
    // Plan finalize is the production caller; the first use builds the config.
    hipdnnBackendDescriptor_t built = shimKnobbed.hipdnnEngineConfig();

    const std::vector<hipdnnBackendDescriptor_t> configs
        = created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    ASSERT_EQ(configs.size(), defaultConfigs + 1);
    hipdnnBackendDescriptor_t hipdnnConfig = configs.back();
    EXPECT_EQ(built, hipdnnConfig);
    EXPECT_NE(hipdnnConfig, shimKnobbed.graph().defaultEngineConfig(KNOB_ENGINE));

    // ENGINE <- the cached engine; KNOB_CHOICES before finalize.
    const auto engineCalls = callsOn(hipdnnConfig, HIPDNN_ATTR_ENGINECFG_ENGINE);
    ASSERT_EQ(engineCalls.size(), 1U);
    EXPECT_EQ(engineCalls.front().value<hipdnnBackendDescriptor_t>(),
              shimEngine(descriptor).hipdnnEngine());
    const auto choiceCalls = callsOn(hipdnnConfig, HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES);
    ASSERT_EQ(choiceCalls.size(), 1U);
    const std::vector<CapturedSetAttribute> all = setAttributes();
    const auto choicePosition = static_cast<std::size_t>(
        std::find_if(all.begin(),
                     all.end(),
                     [hipdnnConfig](const CapturedSetAttribute& call) {
                         return call.descriptor == hipdnnConfig
                                && call.name == HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES;
                     })
        - all.begin());
    const std::optional<std::size_t> setsBeforeFinalize = setCountAtFinalize(hipdnnConfig);
    ASSERT_TRUE(setsBeforeFinalize.has_value());
    EXPECT_LT(choicePosition, *setsBeforeFinalize);

    // The hipDNN knob choice names the provider's knob, not the cuDNN type.
    const std::vector<hipdnnBackendDescriptor_t> choices
        = created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
    ASSERT_EQ(choices.size(), 1U);
    EXPECT_EQ(choiceCalls.front().value<hipdnnBackendDescriptor_t>(), choices.front());
    const auto typeCalls = callsOn(choices.front(), HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE);
    ASSERT_EQ(typeCalls.size(), 1U);
    const std::vector<char> knobId = typeCalls.front().values<char>();
    EXPECT_EQ(std::string(knobId.begin(), std::find(knobId.begin(), knobId.end(), '\0')),
              "prov.tile_size");
    const auto valueCalls = callsOn(choices.front(), HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE);
    ASSERT_EQ(valueCalls.size(), 1U);
    EXPECT_EQ(valueCalls.front().value<std::int64_t>(), 16);

    // Built once.
    EXPECT_EQ(shimKnobbed.hipdnnEngineConfig(), hipdnnConfig);
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), defaultConfigs + 1);
}

TEST_F(TestCudnnShimV8EngineSelection, KnobChoicesGettableFromShim)
{
    cudnnBackendDescriptor_t descriptor = knobEngine();
    cudnnBackendDescriptor_t knobbed = config(descriptor, {{CUDNN_KNOB_TYPE_TILE_SIZE, 24}});
    finalize(knobbed);

    std::int64_t count = -1;
    EXPECT_EQ(cudnnBackendGetAttribute(knobbed,
                                       CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &count,
                                       nullptr),
              CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(count, 1);
    cudnnBackendDescriptor_t choice = create(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendGetAttribute(knobbed,
                                       CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&choice)),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(int64Of(choice, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE), 24);
}

TEST_F(TestCudnnShimV8EngineSelection, KnobConfigBuiltOnceAcrossThreads)
{
    cudnnBackendDescriptor_t descriptor = knobEngine();
    cudnnBackendDescriptor_t knobbed = config(descriptor, {{CUDNN_KNOB_TYPE_TILE_SIZE, 32}});
    finalize(knobbed);
    const std::size_t before = created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size();
    const backend::EngineConfigDescriptor& shimKnobbed = shimConfig(knobbed);

    constexpr std::size_t THREAD_COUNT = 8;
    std::atomic<bool> start{false};
    std::array<hipdnnBackendDescriptor_t, THREAD_COUNT> seen{};
    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(std::size_t i = 0; i < THREAD_COUNT; ++i)
    {
        threads.emplace_back([&, i] {
            while(!start.load())
            {
                std::this_thread::yield();
            }
            seen[i] = shimKnobbed.hipdnnEngineConfig();
        });
    }
    start.store(true);
    for(std::thread& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), before + 1);
    for(const auto& config : seen)
    {
        EXPECT_EQ(config, seen.front());
    }
}

TEST_F(TestCudnnShimV8EngineSelection, KnobFreeConfigUsesDefault)
{
    cudnnBackendDescriptor_t graph = reluGraph();
    installHeuristicResults({KNOB_ENGINE});
    cudnnBackendDescriptor_t plain = config(engine(graph, 0), {});
    finalize(plain);
    const std::size_t configs = created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size();

    const backend::EngineConfigDescriptor& shimPlain = shimConfig(plain);
    EXPECT_EQ(shimPlain.hipdnnEngineConfig(), shimGraph(graph).defaultEngineConfig(KNOB_ENGINE));
    EXPECT_EQ(shimPlain.graphRef().get(), backend::liveHandle(graph)->object.get());
    EXPECT_EQ(created(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR).size(), configs);
    // No knob lookup for a config without choices.
    const std::vector<hipdnnBackendAttributeName_t> gets = getAttributeNames();
    EXPECT_EQ(std::count(gets.begin(), gets.end(), HIPDNN_ATTR_ENGINE_KNOB_INFO), 0);
}

// Upstream's builder sends -1 for every knob it was not asked to set.
TEST_F(TestCudnnShimV8EngineSelection, UnsetKnobChoiceIgnored)
{
    installHeuristicResults({KNOB_ENGINE});
    std::vector<MockKnob> knobs = mixedKnobs();
    knobs.push_back({"prov.split_k_slc", 1, 1, 8, 1, {}});
    installEngineKnobs(KNOB_ENGINE, knobs);
    cudnnBackendDescriptor_t graph = reluGraph();
    cudnnBackendDescriptor_t descriptor = engine(graph, 0);

    // Visible, hidden and repeated unset choices all drop out.
    cudnnBackendDescriptor_t unset = config(descriptor,
                                            {{CUDNN_KNOB_TYPE_TILE_SIZE, -1},
                                             {CUDNN_KNOB_TYPE_BLOCK_SIZE, -1},
                                             {CUDNN_KNOB_TYPE_TILE_SIZE, -1}});
    finalize(unset);
    const backend::EngineConfigDescriptor& shimUnset = shimConfig(unset);
    EXPECT_TRUE(shimUnset.knobChoices().empty());
    EXPECT_EQ(shimUnset.hipdnnEngineConfig(), shimGraph(graph).defaultEngineConfig(KNOB_ENGINE));
    EXPECT_TRUE(created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR).empty());

    cudnnBackendDescriptor_t partial
        = config(descriptor, {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}, {CUDNN_KNOB_TYPE_SPLIT_K_SLC, -1}});
    finalize(partial);
    const backend::EngineConfigDescriptor& shimPartial = shimConfig(partial);
    EXPECT_EQ(shimPartial.knobChoices(), (Choices{{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}));
    shimPartial.hipdnnEngineConfig();
    const std::vector<hipdnnBackendDescriptor_t> hipdnnChoices
        = created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
    ASSERT_EQ(hipdnnChoices.size(), 1U);
    const auto typeCalls = callsOn(hipdnnChoices.front(), HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE);
    ASSERT_EQ(typeCalls.size(), 1U);
    const std::vector<char> knobId = typeCalls.front().values<char>();
    EXPECT_EQ(std::string(knobId.begin(), std::find(knobId.begin(), knobId.end(), '\0')),
              "prov.tile_size");

    // The GET still returns every stored choice, so plan tags keep the -1.
    std::int64_t count = -1;
    std::array<cudnnBackendDescriptor_t, 2> stored = {create(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR),
                                                      create(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR)};
    EXPECT_EQ(cudnnBackendGetAttribute(partial,
                                       CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       static_cast<std::int64_t>(stored.size()),
                                       &count,
                                       static_cast<void*>(stored.data())),
              CUDNN_STATUS_SUCCESS)
        << lastError();
    ASSERT_EQ(count, 2);
    EXPECT_EQ(int64Of(stored[0], CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE), 16);
    EXPECT_EQ(int64Of(stored[1], CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE), -1);
}

TEST_F(TestCudnnShimV8EngineSelection, MinusOneKeptWhenInRange)
{
    installHeuristicResults({KNOB_ENGINE});
    installEngineKnobs(KNOB_ENGINE, {{"prov.split_k_slc", 1, -1, 8, 1, {}}});
    cudnnBackendDescriptor_t descriptor = engine(reluGraph(), 0);

    cudnnBackendDescriptor_t chosen = config(descriptor, {{CUDNN_KNOB_TYPE_SPLIT_K_SLC, -1}});
    finalize(chosen);
    const backend::EngineConfigDescriptor& shimChosen = shimConfig(chosen);
    EXPECT_EQ(shimChosen.knobChoices(), (Choices{{CUDNN_KNOB_TYPE_SPLIT_K_SLC, -1}}));
    shimChosen.hipdnnEngineConfig();
    const std::vector<hipdnnBackendDescriptor_t> hipdnnChoices
        = created(HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
    ASSERT_EQ(hipdnnChoices.size(), 1U);
    const auto valueCalls = callsOn(hipdnnChoices.front(), HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE);
    ASSERT_EQ(valueCalls.size(), 1U);
    EXPECT_EQ(valueCalls.front().value<std::int64_t>(), -1);

    // A real choice, so a repeat is a duplicate.
    cudnnBackendDescriptor_t repeated = config(
        descriptor, {{CUDNN_KNOB_TYPE_SPLIT_K_SLC, -1}, {CUDNN_KNOB_TYPE_SPLIT_K_SLC, -1}});
    EXPECT_EQ(cudnnBackendFinalize(repeated), CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(lastError().find("is chosen more than once"), std::string::npos);
}

TEST_F(TestCudnnShimV8EngineSelection, NondeterministicUnlessClaimed)
{
    const std::int64_t deterministic = hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID;
    installHeuristicResults({deterministic, 9});
    cudnnBackendDescriptor_t graph = reluGraph();
    const std::int64_t deterministicIndex = *shimGraph(graph).engineIndexOf(deterministic);
    const std::int64_t otherIndex = *shimGraph(graph).engineIndexOf(9);

    EXPECT_TRUE(notes<cudnnBackendNumericalNote_t>(engine(graph, deterministicIndex),
                                                   CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                                   CUDNN_TYPE_NUMERICAL_NOTE)
                    .empty());
    EXPECT_EQ(notes<cudnnBackendNumericalNote_t>(engine(graph, otherIndex),
                                                 CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                                 CUDNN_TYPE_NUMERICAL_NOTE),
              std::vector<cudnnBackendNumericalNote_t>{CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC});
}

TEST_F(TestCudnnShimV8EngineSelection, BehaviorNotesMappedByName)
{
    installHeuristicResults({11, 12});
    installBehaviorNotes(11,
                         {HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION,
                          HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    installEngineName(11, "mock_engine_eleven");
    cudnnBackendDescriptor_t graph = reluGraph();

    cudnnBackendDescriptor_t mapped = engine(graph, 0);
    EXPECT_EQ(notes<cudnnBackendBehaviorNote_t>(
                  mapped, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE),
              std::vector<cudnnBackendBehaviorNote_t>{CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    EXPECT_EQ(shimEngine(mapped).engineName(), std::optional<std::string>("mock_engine_eleven"));

    cudnnBackendDescriptor_t bare = engine(graph, 1);
    EXPECT_TRUE(notes<cudnnBackendBehaviorNote_t>(
                    bare, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE)
                    .empty());
    EXPECT_EQ(shimEngine(bare).engineName(), std::nullopt);
}

TEST_F(TestCudnnShimV8EngineSelection, LoadedPlanConfigHasNoGraph)
{
    const std::shared_ptr<backend::EngineConfigDescriptor> loaded
        = backend::EngineConfigDescriptor::forLoadedPlan(-31,
                                                         2,
                                                         {CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION},
                                                         {{CUDNN_KNOB_TYPE_TILE_SIZE, 16}},
                                                         std::string("restored"));
    cudnnBackendDescriptor_t handle = backend::makeHandle(loaded);
    _owned.push_back(handle);

    EXPECT_TRUE(loaded->isFinalized());
    EXPECT_TRUE(loaded->isLoadedPlan());
    EXPECT_EQ(loaded->engineId(), -31);
    EXPECT_EQ(loaded->engineIndex(), 2);
    EXPECT_EQ(loaded->knobChoices(), (Choices{{CUDNN_KNOB_TYPE_TILE_SIZE, 16}}));
    EXPECT_EQ(loaded->engine().engineName(), std::optional<std::string>("restored"));

    cudnnBackendDescriptor_t restoredEngine = engineOf(handle);
    EXPECT_EQ(globalIndex(restoredEngine), 2);
    EXPECT_EQ(notes<cudnnBackendBehaviorNote_t>(
                  restoredEngine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE),
              std::vector<cudnnBackendBehaviorNote_t>{CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION});
    EXPECT_EQ(notes<cudnnBackendNumericalNote_t>(
                  restoredEngine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE),
              std::vector<cudnnBackendNumericalNote_t>{CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC});

    cudnnBackendDescriptor_t choice = create(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
    std::int64_t count = 0;
    EXPECT_EQ(cudnnBackendGetAttribute(handle,
                                       CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&choice)),
              CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(int64Of(choice, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE), 16);

    cudnnBackendDescriptor_t graphSlot = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    EXPECT_EQ(cudnnBackendGetAttribute(restoredEngine,
                                       CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &count,
                                       static_cast<void*>(&graphSlot)),
              CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(cudnnBackendGetAttribute(restoredEngine,
                                       CUDNN_ATTR_ENGINE_KNOB_INFO,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &count,
                                       nullptr),
              CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_THROW(static_cast<void>(loaded->graph()), backend::ShimError);
    EXPECT_THROW(static_cast<void>(loaded->hipdnnEngineConfig()), backend::ShimError);

    // A restored engine cannot seed a new config.
    cudnnBackendDescriptor_t reused = config(restoredEngine, {});
    EXPECT_EQ(cudnnBackendFinalize(reused), CUDNN_STATUS_NOT_SUPPORTED);
}

} // namespace
