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

    EnginePredictionT& estimate(int64_t id, hipdnnEnginePredictionKind_t kind, double tflops)
    {
        auto& result = _predictions[{id, kind}];
        result.engine_id = id;
        result.kind = kind == HIPDNN_ENGINE_PREDICTION_ENGINE ? PredictionKind::ENGINE
                                                              : PredictionKind::CONFIGURATION;
        result.status = PredictionStatus::AVAILABLE;
        result.tflops = tflops;
        if(kind == HIPDNN_ENGINE_PREDICTION_CONFIGURATION)
        {
            result.engine_config = std::make_unique<EngineConfigT>();
            result.engine_config->engine_id = id;
            auto knob = std::make_unique<KnobSettingT>();
            knob->knob_id = "tile";
            IntValueT value;
            value.value = 128;
            knob->value.Set(value);
            result.engine_config->knobs.push_back(std::move(knob));
        }
        return result;
    }

    hipdnnHeuristicHostCallbacks_t host()
    {
        return {1,
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
                }};
    }

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

TEST_F(TestPredictionPolicies, ModeARanksOnlyL1AndRetainsUnknownEngines)
{
    selectMode(MODE_A_POLICY_NAME, {1, 2, 3, 4});
    estimate(1, HIPDNN_ENGINE_PREDICTION_ENGINE, 10);
    estimate(2, HIPDNN_ENGINE_PREDICTION_ENGINE, 20);
    estimate(4, HIPDNN_ENGINE_PREDICTION_ENGINE, 20);
    estimate(1, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, 500);
    const auto services = host();
    ASSERT_TRUE(_plugin->finalizeWithHost(_descriptor.get(), &services));
    EXPECT_EQ(_plugin->getSortedEngineIds(_descriptor.get()), (std::vector<int64_t>{2, 4, 1, 3}));
    EXPECT_EQ(_calls.size(), 4u);
    for(const auto& [key, calls] : _calls)
    {
        EXPECT_EQ(key.second, HIPDNN_ENGINE_PREDICTION_ENGINE);
        EXPECT_EQ(calls, 1u);
        const auto config = _plugin->getEngineConfig(_descriptor.get(), key.first);
        ASSERT_NE(config, nullptr);
        EXPECT_EQ(config->engine_id, key.first);
        EXPECT_TRUE(config->knobs.empty());
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

TEST_F(TestPredictionPolicies, RejectsIncompatibleScopedHostBeforeInvokingCallbacks)
{
    selectMode(MODE_A_POLICY_NAME, {1});
    auto services = host();
    services.version = 2;
    EXPECT_THROW(_plugin->finalizeWithHost(_descriptor.get(), &services),
                 hipdnn_backend::HipdnnException);
    services.version = 1;
    services.struct_size = offsetof(hipdnnHeuristicHostCallbacks_t, get_prediction);
    EXPECT_THROW(_plugin->finalizeWithHost(_descriptor.get(), &services),
                 hipdnn_backend::HipdnnException);
    services.struct_size = sizeof(hipdnnHeuristicHostCallbacks_t);
    services.get_prediction = nullptr;
    EXPECT_THROW(_plugin->finalizeWithHost(_descriptor.get(), &services),
                 hipdnn_backend::HipdnnException);
    EXPECT_TRUE(_calls.empty());
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
