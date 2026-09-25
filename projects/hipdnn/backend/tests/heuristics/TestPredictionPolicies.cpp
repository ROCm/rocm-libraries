// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestPredictionPolicies.cpp
 * @brief Unit tests for the SelectionHeuristic::ModeA / ModeB prediction policies.
 *
 * The policies ship as a backend built-in (RFC 0007 §5.3.5, §10.1), registered by
 * HeuristicPluginManager::registerBuiltIns through HeuristicPlugin::createBuiltIn
 * exactly like Config and StaticOrdering. These tests therefore resolve the policy
 * implementation the way production does — from a plain HeuristicPluginManager — so
 * a registration regression fails here rather than silently disabling prediction
 * ranking.
 */

#include "HipdnnException.hpp"
#include "heuristics/prediction/PredictionBuiltIn.hpp"
#include "plugin/HeuristicPlugin.hpp"
#include "plugin/HeuristicPluginManager.hpp"

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/device_properties_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace
{
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_backend::plugin;
using hipdnn_data_sdk::utilities::MODE_A_POLICY_NAME;
using hipdnn_data_sdk::utilities::MODE_B_POLICY_NAME;
using hipdnn_data_sdk::utilities::policyNameToId;
using hipdnn_data_sdk::utilities::ScopedResource;
using Key = std::pair<int64_t, hipdnnEnginePredictionKind_t>;

// Locate the registered built-in that serves the prediction policies. Resolving by
// policy ID rather than by plugin name is the contract the heuristic chain uses:
// EngineHeuristicDescriptor only ever asks for a policy ID.
std::shared_ptr<HeuristicPlugin> findPredictionPlugin(const HeuristicPluginManager& manager)
{
    const auto modeA = policyNameToId(MODE_A_POLICY_NAME);
    for(const auto& plugin : manager.getPlugins())
    {
        const auto policyIds = plugin->getAllPolicyIds();
        if(std::find(policyIds.begin(), policyIds.end(), modeA) != policyIds.end())
        {
            return plugin;
        }
    }
    return nullptr;
}

class TestPredictionPolicies : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // ABSOLUTE loading with no paths drops every externally loaded plugin. Built-ins
        // are not loaded from a path, so the prediction policies must still be there
        // (HeuristicPluginManager::actionAfterClearing re-registers them).
        const std::set<std::filesystem::path> noPaths;
        _manager.loadPlugins(noPaths, HIPDNN_PLUGIN_LOADING_ABSOLUTE);
        _plugin = findPredictionPlugin(_manager);
        ASSERT_NE(_plugin, nullptr) << "ModeA/ModeB are not registered as backend built-ins";

        _handle = ScopedResource<hipdnnHeuristicHandle_t>(
            _plugin->createHandle(), [this](auto handle) { _plugin->destroyHandle(handle); });
        DevicePropertiesT properties;
        properties.architecture_name = "gfx-test";
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(DeviceProperties::Pack(builder, &properties));
        const hipdnnPluginConstData_t bytes{builder.GetBufferPointer(), builder.GetSize()};
        _plugin->setDeviceProperties(_handle.get(), &bytes);
    }

    void selectMode(const char* mode, const std::vector<int64_t>& ids)
    {
        _descriptor = ScopedResource<hipdnnHeuristicPolicyDescriptor_t>(
            _plugin->createPolicyDescriptor(_handle.get(), policyNameToId(mode)),
            [this](auto descriptor) { _plugin->destroyPolicyDescriptor(descriptor); });
        _plugin->setEngineIds(_descriptor.get(), ids.data(), ids.size());
        GraphT graph;
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(Graph::Pack(builder, &graph));
        const hipdnnPluginConstData_t bytes{builder.GetBufferPointer(), builder.GetSize()};
        _plugin->setSerializedGraph(_descriptor.get(), &bytes);
    }

    EnginePredictionT& estimate(int64_t id, hipdnnEnginePredictionKind_t kind, double value)
    {
        auto& result = _predictions[{id, kind}];
        result.engine_id = id;
        result.kind = kind == HIPDNN_ENGINE_PREDICTION_ENGINE ? PredictionKind::ENGINE
                                                              : PredictionKind::CONFIGURATION;
        result.status = PredictionStatus::AVAILABLE;
        result.value = value;
        result.metric = _metric;
        if(kind == HIPDNN_ENGINE_PREDICTION_CONFIGURATION)
        {
            result.engine_config = std::make_unique<EngineConfigT>();
            result.engine_config->engine_id = id;
            auto knob = std::make_unique<KnobSettingT>();
            knob->knob_id = "tile";
            IntValueT tile;
            tile.value = 128;
            knob->value.Set(tile);
            result.engine_config->knobs.push_back(std::move(knob));
        }
        return result;
    }

    hipdnnHeuristicHostCallbacks_t host()
    {
        return {2,
                sizeof(hipdnnHeuristicHostCallbacks_t),
                this,
                [](void* context,
                   int64_t id,
                   hipdnnEnginePredictionKind_t kind,
                   hipdnnPluginConstData_t* output) {
                    auto& self = *static_cast<TestPredictionPolicies*>(context);
                    ++self._calls[{id, kind}];
                    *output = {};
                    if(id == self._malformedId)
                    {
                        *output = {self._malformed.data(), self._malformed.size()};
                        return HIPDNN_PLUGIN_STATUS_SUCCESS;
                    }
                    const auto found = self._predictions.find({id, kind});
                    if(found == self._predictions.end())
                    {
                        return HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE;
                    }
                    flatbuffers::FlatBufferBuilder builder;
                    builder.Finish(EnginePrediction::Pack(builder, &found->second));
                    self._buffers.push_back(builder.Release());
                    const auto& bytes = self._buffers.back();
                    *output = {bytes.data(), bytes.size()};
                    return HIPDNN_PLUGIN_STATUS_SUCCESS;
                },
                _metric.c_str()};
    }

    std::string _metric = "tflops";
    HeuristicPluginManager _manager;
    std::shared_ptr<HeuristicPlugin> _plugin;
    ScopedResource<hipdnnHeuristicHandle_t> _handle;
    ScopedResource<hipdnnHeuristicPolicyDescriptor_t> _descriptor;
    std::map<Key, EnginePredictionT> _predictions;
    std::map<Key, size_t> _calls;
    std::vector<flatbuffers::DetachedBuffer> _buffers;
    int64_t _malformedId = -1;
    std::vector<uint8_t> _malformed{0, 0, 0, 0};
};

// RFC 0007 §17: a first-party policy is delivered by built-in registration, so both
// policy IDs must be the FNV-1a hashes of their canonical names and must answer with
// those names — that is the only handle a caller (or HIPDNN_HEUR_POLICY_ORDER) has.
TEST_F(TestPredictionPolicies, RegistersBothPredictionPoliciesUnderTheirCanonicalNames)
{
    const auto policyIds = _plugin->getAllPolicyIds();
    EXPECT_NE(std::find(policyIds.begin(), policyIds.end(), policyNameToId(MODE_A_POLICY_NAME)),
              policyIds.end());
    EXPECT_NE(std::find(policyIds.begin(), policyIds.end(), policyNameToId(MODE_B_POLICY_NAME)),
              policyIds.end());
    EXPECT_EQ(_plugin->getPolicyName(policyNameToId(MODE_A_POLICY_NAME)), MODE_A_POLICY_NAME);
    EXPECT_EQ(_plugin->getPolicyName(policyNameToId(MODE_B_POLICY_NAME)), MODE_B_POLICY_NAME);

    // A freshly constructed manager registers them too: discovery never had to run.
    const HeuristicPluginManager fresh;
    EXPECT_NE(findPredictionPlugin(fresh), nullptr);
}

// RFC 0019 §11.2 quick policy: rank applicable engines by A (L1) alone. B (L2) is never
// evaluated under this policy, for any engine — not even to fill in the winner's kernel,
// which plan build chooses by the same metric (Open Question 21, option (b)).
TEST_F(TestPredictionPolicies, ModeARanksByL1AndNeverQueriesConfigurationPredictions)
{
    selectMode(MODE_A_POLICY_NAME, {1, 2, 3, 4});
    estimate(1, HIPDNN_ENGINE_PREDICTION_ENGINE, 10);
    estimate(2, HIPDNN_ENGINE_PREDICTION_ENGINE, 20);
    estimate(4, HIPDNN_ENGINE_PREDICTION_ENGINE, 20);
    // An L2 score that beats every L1 estimate: under ModeA it must never be asked for.
    estimate(1, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, 500);
    estimate(2, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, 1);
    const auto services = host();
    ASSERT_TRUE(_plugin->finalizeWithHost(_descriptor.get(), &services));
    EXPECT_EQ(_plugin->getSortedEngineIds(_descriptor.get()), (std::vector<int64_t>{2, 4, 1, 3}));

    const std::map<Key, size_t> expected{{{1, HIPDNN_ENGINE_PREDICTION_ENGINE}, 1u},
                                         {{2, HIPDNN_ENGINE_PREDICTION_ENGINE}, 1u},
                                         {{3, HIPDNN_ENGINE_PREDICTION_ENGINE}, 1u},
                                         {{4, HIPDNN_ENGINE_PREDICTION_ENGINE}, 1u}};
    EXPECT_EQ(_calls, expected);

    // Every engine, winner included, keeps a knob-less config naming the metric, so its
    // own selector picks the kernel at plan build by that metric.
    for(const auto id : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{4}})
    {
        const auto config = _plugin->getEngineConfig(_descriptor.get(), id);
        ASSERT_NE(config, nullptr);
        EXPECT_EQ(config->engine_id, id);
        EXPECT_TRUE(config->knobs.empty());
        EXPECT_EQ(config->ranking_metric, "tflops");
    }
}

// RFC 0019 §4.4/§11.2: engines are compared in the requested metric's direction. For
// `time` lower is better, and a value the metric rejects (a non-positive time) or an
// answer in another metric scores nothing rather than ranking first.
TEST_F(TestPredictionPolicies, TimeMetricRanksLowerFirstAndRejectsForeignOrInvalidValues)
{
    _metric = "time";
    for(const auto* mode : {MODE_A_POLICY_NAME, MODE_B_POLICY_NAME})
    {
        _predictions.clear();
        selectMode(mode, {1, 2, 3, 4, 5});
        estimate(1, HIPDNN_ENGINE_PREDICTION_ENGINE, 5.0);
        estimate(2, HIPDNN_ENGINE_PREDICTION_ENGINE, 2.0);
        estimate(3, HIPDNN_ENGINE_PREDICTION_ENGINE, 8.0);
        estimate(4, HIPDNN_ENGINE_PREDICTION_ENGINE, 0.0);
        estimate(5, HIPDNN_ENGINE_PREDICTION_ENGINE, 0.5).metric = "tflops";
        const auto services = host();
        ASSERT_TRUE(_plugin->finalizeWithHost(_descriptor.get(), &services)) << mode;
        EXPECT_EQ(_plugin->getSortedEngineIds(_descriptor.get()),
                  (std::vector<int64_t>{2, 1, 3, 4, 5}))
            << mode;
        EXPECT_EQ(_plugin->getEngineConfig(_descriptor.get(), 2)->ranking_metric, "time") << mode;
    }
}

// Equal values do not fall back to candidate-arrival order: RFC 0019 §11.2 breaks ties
// by the static rules.
TEST_F(TestPredictionPolicies, TiesFollowStaticOrderNotArrivalOrder)
{
    using hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID;
    using hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID;

    selectMode(MODE_A_POLICY_NAME, {HIPBLASLT_ENGINE_ID, MIOPEN_ENGINE_ID});
    estimate(HIPBLASLT_ENGINE_ID, HIPDNN_ENGINE_PREDICTION_ENGINE, 10);
    estimate(MIOPEN_ENGINE_ID, HIPDNN_ENGINE_PREDICTION_ENGINE, 10);
    const auto services = host();
    ASSERT_TRUE(_plugin->finalizeWithHost(_descriptor.get(), &services));
    EXPECT_EQ(_plugin->getSortedEngineIds(_descriptor.get()),
              (std::vector<int64_t>{MIOPEN_ENGINE_ID, HIPBLASLT_ENGINE_ID}));
}

// RFC 0019 §11.2 table row "No declared model": an engine that answers neither query
// "falls back to static ordering; contributes no score" and "is ordered by the existing
// static rules" — identically under both policies. The arrival order is neither.
TEST_F(TestPredictionPolicies, UnscoredEnginesFallBackToStaticOrdering)
{
    using hipdnn_data_sdk::utilities::ASM_SDPA_ENGINE_ID;
    using hipdnn_data_sdk::utilities::HIPBLASLT_ENGINE_ID;
    using hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID;
    using hipdnn_data_sdk::utilities::MIOPEN_ENGINE_ID;

    // Arrival order is the exact reverse of the static rules for the silent engines.
    const std::vector<int64_t> arrival{
        HIPBLASLT_ENGINE_ID, MIOPEN_ENGINE_DETERMINISTIC_ID, ASM_SDPA_ENGINE_ID, MIOPEN_ENGINE_ID};
    for(const auto* mode : {MODE_A_POLICY_NAME, MODE_B_POLICY_NAME})
    {
        selectMode(mode, arrival);
        estimate(HIPBLASLT_ENGINE_ID, HIPDNN_ENGINE_PREDICTION_ENGINE, 10);
        const auto services = host();
        ASSERT_TRUE(_plugin->finalizeWithHost(_descriptor.get(), &services));
        // The one scored engine keeps its rank ahead of the silent tail, and that tail
        // comes out MIOPEN, ASM_SDPA, MIOPEN_DETERMINISTIC however it arrived.
        EXPECT_EQ(_plugin->getSortedEngineIds(_descriptor.get()),
                  (std::vector<int64_t>{HIPBLASLT_ENGINE_ID,
                                        MIOPEN_ENGINE_ID,
                                        ASM_SDPA_ENGINE_ID,
                                        MIOPEN_ENGINE_DETERMINISTIC_ID}))
            << "policy " << mode;
    }
}

TEST_F(TestPredictionPolicies, ModeBMixesL2AndL1AndOwnsExactScoredConfig)
{
    selectMode(MODE_B_POLICY_NAME, {1, 2, 3, 4, 5});
    estimate(1, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, 50);
    estimate(1, HIPDNN_ENGINE_PREDICTION_ENGINE, 500); // L2 supersedes this L1 estimate.
    estimate(2, HIPDNN_ENGINE_PREDICTION_ENGINE, 60);
    estimate(3, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, -5);
    estimate(3, HIPDNN_ENGINE_PREDICTION_ENGINE, 10);
    estimate(5, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, 50);
    const auto services = host();
    ASSERT_TRUE(_plugin->finalizeWithHost(_descriptor.get(), &services));
    EXPECT_EQ(_plugin->getSortedEngineIds(_descriptor.get()),
              (std::vector<int64_t>{2, 1, 5, 3, 4}));
    EXPECT_EQ(_calls.count({1, HIPDNN_ENGINE_PREDICTION_ENGINE}), 0u);
    EXPECT_EQ(_calls.count({5, HIPDNN_ENGINE_PREDICTION_ENGINE}), 0u);
    EXPECT_EQ(_calls[Key(3, HIPDNN_ENGINE_PREDICTION_ENGINE)], 1u);

    // The host can release all prediction buffers at finalize return. Retrieval
    // must use the plugin's owned copy, not aliases into those buffers.
    _buffers.clear();
    _predictions.clear();
    const auto scored = _plugin->getEngineConfig(_descriptor.get(), 1);
    const auto fallback = _plugin->getEngineConfig(_descriptor.get(), 2);
    _descriptor = {};
    ASSERT_NE(scored, nullptr);
    EXPECT_EQ(scored->engine_id, 1);
    ASSERT_EQ(scored->knobs.size(), 1u);
    EXPECT_EQ(scored->knobs[0]->knob_id, "tile");
    ASSERT_NE(scored->knobs[0]->value.AsIntValue(), nullptr);
    EXPECT_EQ(scored->knobs[0]->value.AsIntValue()->value, 128);
    EXPECT_EQ(scored->ranking_metric, "tflops");
    ASSERT_NE(fallback, nullptr);
    EXPECT_TRUE(fallback->knobs.empty());
}

TEST_F(TestPredictionPolicies, InvalidConfigurationFallsBackWithoutDroppingEngine)
{
    selectMode(MODE_B_POLICY_NAME, {1, 2, 3, 4});
    estimate(1, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, 900).engine_config->engine_id = 99;
    estimate(1, HIPDNN_ENGINE_PREDICTION_ENGINE, 30);
    estimate(2, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, 1000).engine_config.reset();
    estimate(2, HIPDNN_ENGINE_PREDICTION_ENGINE, 40);
    estimate(3, HIPDNN_ENGINE_PREDICTION_ENGINE, 100).engine_id = 99;
    _malformedId = 4;
    const auto services = host();
    ASSERT_TRUE(_plugin->finalizeWithHost(_descriptor.get(), &services));
    EXPECT_EQ(_plugin->getSortedEngineIds(_descriptor.get()), (std::vector<int64_t>{2, 1, 3, 4}));
    EXPECT_TRUE(_plugin->getEngineConfig(_descriptor.get(), 1)->knobs.empty());
}

TEST_F(TestPredictionPolicies, MissingOrInvalidModelsDeclineAndInvalidatePreviousResults)
{
    for(const auto* mode : {MODE_A_POLICY_NAME, MODE_B_POLICY_NAME})
    {
        selectMode(mode, {1, 2, 3});
        estimate(1, HIPDNN_ENGINE_PREDICTION_ENGINE, 20);
        const auto services = host();
        ASSERT_TRUE(_plugin->finalizeWithHost(_descriptor.get(), &services));
        estimate(1, HIPDNN_ENGINE_PREDICTION_ENGINE, std::numeric_limits<double>::quiet_NaN());
        estimate(2, HIPDNN_ENGINE_PREDICTION_ENGINE, std::numeric_limits<double>::infinity());
        estimate(3, HIPDNN_ENGINE_PREDICTION_ENGINE, 200).status = PredictionStatus::UNAVAILABLE;
        EXPECT_FALSE(_plugin->finalizeWithHost(_descriptor.get(), &services));
        EXPECT_THROW(_plugin->getSortedEngineIds(_descriptor.get()),
                     hipdnn_backend::HipdnnException);
        EXPECT_THROW(_plugin->getEngineConfig(_descriptor.get(), 1),
                     hipdnn_backend::HipdnnException);
        EXPECT_FALSE(_plugin->finalizeWithHost(_descriptor.get(), nullptr));
    }
}

// File-scope: the C-ABI logging callback is a plain function pointer.
std::vector<std::string>* gPredictionLogLines = nullptr;

void capturePredictionLog(hipdnnSeverity_t, const char* message)
{
    if(gPredictionLogLines != nullptr && message != nullptr)
    {
        gPredictionLogLines->emplace_back(message);
    }
}

// RFC 0019 §11.2: a policy with nothing scored declines and says which metric and which
// levels nobody served, so a static-order result is never mistaken for a ranking.
TEST_F(TestPredictionPolicies, DeclineNamesTheMetricAndTheLevelsAsked)
{
    auto abi = hipdnn_backend::heuristics::prediction::populateFunctionTable();
    std::vector<std::string> lines;
    gPredictionLogLines = &lines;
    ASSERT_EQ(abi.setLoggingCallback(&capturePredictionLog), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_EQ(abi.setLogLevel(HIPDNN_SEV_INFO), HIPDNN_PLUGIN_STATUS_SUCCESS);

    _metric = "time";
    const std::vector<std::pair<const char*, std::string>> cases{
        {MODE_A_POLICY_NAME, "no engine serves 'time' at L1"},
        {MODE_B_POLICY_NAME, "no engine serves 'time' at L1 or L2"}};
    for(const auto& testCase : cases)
    {
        const char* mode = testCase.first;
        const std::string& reason = testCase.second;
        lines.clear();
        selectMode(mode, {1, 2});
        // A perfectly good TFLOPS answer is not a `time` answer.
        estimate(1, HIPDNN_ENGINE_PREDICTION_ENGINE, 100).metric = "tflops";
        const auto services = host();
        EXPECT_FALSE(_plugin->finalizeWithHost(_descriptor.get(), &services)) << mode;
        EXPECT_TRUE(std::any_of(lines.begin(), lines.end(), [&](const std::string& line) {
            return line.find(reason) != std::string::npos
                   && (mode != MODE_A_POLICY_NAME || line.find("L1 or L2") == std::string::npos);
        })) << mode;
    }

    abi.setLoggingCallback(nullptr);
    gPredictionLogLines = nullptr;
}

TEST_F(TestPredictionPolicies, RejectsIncompatibleScopedHostBeforeInvokingCallbacks)
{
    selectMode(MODE_A_POLICY_NAME, {1});
    auto services = host();
    services.version = 0;
    EXPECT_THROW(_plugin->finalizeWithHost(_descriptor.get(), &services),
                 hipdnn_backend::HipdnnException);
    services.version = 1;
    services.struct_size = offsetof(hipdnnHeuristicHostCallbacks_t, get_prediction);
    EXPECT_THROW(_plugin->finalizeWithHost(_descriptor.get(), &services),
                 hipdnn_backend::HipdnnException);
    services.version = 2;
    services.struct_size = sizeof(hipdnnHeuristicHostCallbacks_t);
    services.get_prediction = nullptr;
    EXPECT_THROW(_plugin->finalizeWithHost(_descriptor.get(), &services),
                 hipdnn_backend::HipdnnException);
    // A metric the registry does not know has no direction to rank by.
    services = host();
    services.ranking_metric = "flops";
    EXPECT_THROW(_plugin->finalizeWithHost(_descriptor.get(), &services),
                 hipdnn_backend::HipdnnException);
    services.ranking_metric = nullptr;
    EXPECT_THROW(_plugin->finalizeWithHost(_descriptor.get(), &services),
                 hipdnn_backend::HipdnnException);
    EXPECT_TRUE(_calls.empty());
}

// A version 1 table predates ranking_metric and ends before it; it means TFLOPS and the
// policy must not read past the table to find out.
TEST_F(TestPredictionPolicies, VersionOneHostRanksByTflops)
{
    selectMode(MODE_A_POLICY_NAME, {1, 2});
    estimate(1, HIPDNN_ENGINE_PREDICTION_ENGINE, 10);
    estimate(2, HIPDNN_ENGINE_PREDICTION_ENGINE, 20);
    auto services = host();
    services.version = 1;
    services.struct_size = offsetof(hipdnnHeuristicHostCallbacks_t, ranking_metric);
    services.ranking_metric = "time"; // Beyond struct_size: must be ignored.
    ASSERT_TRUE(_plugin->finalizeWithHost(_descriptor.get(), &services));
    EXPECT_EQ(_plugin->getSortedEngineIds(_descriptor.get()), (std::vector<int64_t>{2, 1}));
}

TEST_F(TestPredictionPolicies, ConfigCannotEscapeItsEngineOrFinalizationLifetime)
{
    selectMode(MODE_B_POLICY_NAME, {1});
    estimate(1, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, 30);
    const auto services = host();
    ASSERT_TRUE(_plugin->finalizeWithHost(_descriptor.get(), &services));
    EXPECT_THROW(_plugin->getEngineConfig(_descriptor.get(), 99), hipdnn_backend::HipdnnException);
    const int64_t replacement = 2;
    _plugin->setEngineIds(_descriptor.get(), &replacement, 1);
    EXPECT_THROW(_plugin->getEngineConfig(_descriptor.get(), 1), hipdnn_backend::HipdnnException);
    EXPECT_FALSE(_plugin->finalize(_descriptor.get())); // Legacy host: no prediction services.
}

TEST(TestPredictionPolicyBoundary, RejectsMalformedOrForeignConfigFromPlugin)
{
    auto functions = hipdnn_backend::heuristics::prediction::populateFunctionTable();
    functions.policyGetEngineConfig
        = [](hipdnnHeuristicPolicyDescriptor_t, int64_t, hipdnnPluginConstData_t* result) {
              static const uint8_t malformed[] = {0, 0, 0, 0};
              *result = {malformed, sizeof(malformed)};
              return HIPDNN_PLUGIN_STATUS_SUCCESS;
          };
    auto malformedPlugin = HeuristicPlugin::createBuiltIn(functions, "malformed-config-test");
    EXPECT_THROW(malformedPlugin->getEngineConfig(nullptr, 1), hipdnn_backend::HipdnnException);
    functions.policyGetEngineConfig
        = [](hipdnnHeuristicPolicyDescriptor_t, int64_t, hipdnnPluginConstData_t* result) {
              static thread_local flatbuffers::DetachedBuffer bytes;
              flatbuffers::FlatBufferBuilder builder;
              EngineConfigT config;
              config.engine_id = 999;
              builder.Finish(EngineConfig::Pack(builder, &config));
              bytes = builder.Release();
              *result = {bytes.data(), bytes.size()};
              return HIPDNN_PLUGIN_STATUS_SUCCESS;
          };
    auto foreignPlugin = HeuristicPlugin::createBuiltIn(functions, "foreign-config-test");
    EXPECT_THROW(foreignPlugin->getEngineConfig(nullptr, 1), hipdnn_backend::HipdnnException);
}
} // namespace
