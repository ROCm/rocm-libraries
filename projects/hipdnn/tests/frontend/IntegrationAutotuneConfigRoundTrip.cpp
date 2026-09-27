// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "AutotuneIntegrationFixture.hpp"
#include "test_plugins/TestPluginEngineIdMap.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/detail/AutotuneConfigNames.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/FrontendGraphFactory.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#include <nlohmann/json.hpp>
#endif

using namespace hipdnn_frontend;
using hipdnn_test_sdk::utilities::OperationType;

namespace
{

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
namespace config_criterion = hipdnn_data_sdk::detail::autotune_config::criterion;
namespace config_json = hipdnn_data_sdk::detail::autotune_config::json;
namespace config_op = hipdnn_data_sdk::detail::autotune_config::op;
namespace config_tensor = hipdnn_data_sdk::detail::autotune_config::tensor;
namespace config_version = hipdnn_data_sdk::detail::autotune_config::version;

struct ConfigRoundTripCase
{
    OperationType op;
    const char* expectedOpName;
    std::vector<std::pair<std::string, int64_t>> expectedCriteria;
    std::vector<const char*> expectedTensorIds;
};

class IntegrationAutotuneConfigRoundTrip : public hipdnn_tests::AutotuneIntegrationFixture,
                                           public ::testing::WithParamInterface<ConfigRoundTripCase>
{
protected:
    void SetUp() override
    {
        AutotuneIntegrationFixture::SetUp();

        if(IsSkipped())
        {
            return;
        }

        _configFile = std::filesystem::temp_directory_path()
                      / ("hipdnn_autotune_config_roundtrip_" + std::to_string(sCounter.fetch_add(1))
                         + ".json");
        _policyOrderEnv.emplace("HIPDNN_HEUR_POLICY_ORDER",
                                "SelectionHeuristic::Config,SelectionHeuristic::StaticOrdering");
    }

    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove(_configFile, ec);
        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_HEUR_CONFIG_PATH");

        _policyOrderEnv.reset();
        AutotuneIntegrationFixture::TearDown();
    }

    void rewriteFirstEngineName(const std::string& newEngineName)
    {
        std::ifstream in(_configFile);
        ASSERT_TRUE(in.is_open()) << "Could not open config file for rewrite: " << _configFile;
        nlohmann::json json;
        in >> json;
        in.close();

        ASSERT_TRUE(json.contains(config_json::ENGINE_OVERRIDES));
        ASSERT_FALSE(json[config_json::ENGINE_OVERRIDES].empty());
        json[config_json::ENGINE_OVERRIDES][0][config_json::ENGINE_NAME] = newEngineName;

        std::ofstream out(_configFile, std::ios::trunc);
        ASSERT_TRUE(out.is_open()) << "Could not open config file for write: " << _configFile;
        out << json.dump(2) << '\n';
    }

    void reverseFirstEntryTensors()
    {
        std::ifstream in(_configFile);
        ASSERT_TRUE(in.is_open()) << "Could not open config file for tensor rewrite: "
                                  << _configFile;
        nlohmann::json json;
        in >> json;
        in.close();

        ASSERT_TRUE(json.contains(config_json::ENGINE_OVERRIDES));
        ASSERT_FALSE(json[config_json::ENGINE_OVERRIDES].empty());
        auto& tensors = json[config_json::ENGINE_OVERRIDES][0][config_json::TENSORS];
        ASSERT_TRUE(tensors.is_array());
        std::reverse(tensors.begin(), tensors.end());

        std::ofstream out(_configFile, std::ios::trunc);
        ASSERT_TRUE(out.is_open()) << "Could not open config file for write: " << _configFile;
        out << json.dump(2) << '\n';
    }

    void assertConfigEntryMatchesCase(const ConfigRoundTripCase& testCase)
    {
        std::ifstream in(_configFile);
        ASSERT_TRUE(in.is_open()) << "Config file was not created: " << _configFile;
        nlohmann::json json;
        in >> json;

        ASSERT_TRUE(json.contains(config_json::ENGINE_OVERRIDES));
        ASSERT_TRUE(json.contains(config_json::VERSION));
        EXPECT_EQ(json[config_json::VERSION], config_version::CURRENT);
        ASSERT_EQ(json[config_json::ENGINE_OVERRIDES].size(), 1u);
        const auto& entry = json[config_json::ENGINE_OVERRIDES][0];
        EXPECT_EQ(entry[config_json::OP], testCase.expectedOpName);
        ASSERT_TRUE(entry.contains(config_json::TENSORS));
        ASSERT_EQ(entry[config_json::TENSORS].size(), testCase.expectedTensorIds.size());
        for(size_t i = 0; i < testCase.expectedTensorIds.size(); ++i)
        {
            const auto& tensor = entry[config_json::TENSORS][i];
            ASSERT_TRUE(tensor.contains(config_json::TENSOR_ID));
            ASSERT_TRUE(tensor[config_json::TENSOR_ID].is_string());
            EXPECT_EQ(tensor[config_json::TENSOR_ID], testCase.expectedTensorIds[i]);
        }

        nlohmann::json expectedCriteria = nlohmann::json::object();
        for(const auto& [key, value] : testCase.expectedCriteria)
        {
            expectedCriteria[key] = value;
        }
        if(expectedCriteria.empty())
        {
            EXPECT_FALSE(entry.contains(config_json::CRITERIA));
        }
        else
        {
            ASSERT_TRUE(entry.contains(config_json::CRITERIA));
            EXPECT_EQ(entry[config_json::CRITERIA], expectedCriteria);
        }
    }

    std::filesystem::path _configFile;
    std::optional<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _policyOrderEnv;
    static std::atomic<uint64_t> sCounter;
};

std::atomic<uint64_t> IntegrationAutotuneConfigRoundTrip::sCounter{0};

TEST_P(IntegrationAutotuneConfigRoundTrip, EngineSelectionRoundTripsThroughConfigFile)
{
    const auto& testCase = GetParam();
    const int64_t engineAId = hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();
    const int64_t engineBId = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineB>();
    ASSERT_NE(engineAId, engineBId);

    {
        std::shared_ptr<graph::Graph> graph;
        std::optional<hipdnn_test_sdk::utilities::GraphTensorBundle> bundle;
        buildGraphAndBundle(testCase.op, graph, bundle);

        auto result = graph->add_all_engines();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        int64_t maxWorkspaceSize = 0;
        result = graph->get_estimated_max_workspace_size(maxWorkspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const hipdnn_data_sdk::utilities::Workspace workspace(
            static_cast<size_t>(maxWorkspaceSize));

        AutotuneConfig config;
        config.mode = TuneMode::STANDARD;
        config.strategy = AutotuneStrategy::FIXED_AVERAGE;
        config.timedIterations = 1;
        config.warmupIterations = 1;
        config.engineIdFilter = {engineAId};

        const AutotuneStorageConfig storageConfig{_configFile, false};
        std::vector<AutotuneResult> results;
        result = graph->autotune(_handle,
                                 bundle->variantPack(),
                                 workspace.get(),
                                 maxWorkspaceSize,
                                 config,
                                 storageConfig,
                                 &results);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        // Engine A is the only filtered-in engine, so it is the sole succeeded
        // (ranked) result; any other engine's plan surfaces as a filtered failed
        // result. The winner written to the config file must be engine A.
        int succeededEngineCount = 0;
        for(const auto& r : results)
        {
            if(r.succeeded)
            {
                EXPECT_EQ(r.engineId, engineAId) << "Only engine A should succeed";
                ++succeededEngineCount;
            }
            else
            {
                EXPECT_NE(r.engineId, engineAId) << "Engine A should not be filtered out";
            }
        }
        EXPECT_EQ(succeededEngineCount, 1);
        assertConfigEntryMatchesCase(testCase);
        reverseFirstEntryTensors();
    }

    {
        int64_t selectedId = 0;
        buildGraphAndGetSelectedEngineId(testCase.op, _configFile.string(), selectedId);
        EXPECT_EQ(selectedId, engineAId) << "Backend should select the engine written by autotune";
    }

    {
        rewriteFirstEngineName(std::to_string(engineBId));
        int64_t selectedId = 0;
        buildGraphAndGetSelectedEngineId(testCase.op, _configFile.string(), selectedId);
        EXPECT_EQ(selectedId, engineBId)
            << "Backend selection should follow modified config content";
    }
}

std::vector<ConfigRoundTripCase> configRoundTripCases()
{
    std::vector<ConfigRoundTripCase> cases{
        {OperationType::CONV_FORWARD,
         config_op::CONV_FPROP,
         {},
         {config_tensor::X, config_tensor::W}},
        {OperationType::CONV_BACKWARD_DATA,
         config_op::CONV_DGRAD,
         {},
         {config_tensor::DY, config_tensor::W}},
        {OperationType::CONV_BACKWARD_WEIGHTS,
         config_op::CONV_WGRAD,
         {},
         {config_tensor::X, config_tensor::DY}},
        {OperationType::CONV_FWD_BIAS_ACTIV,
         config_op::CONV_FPROP,
         {},
         {config_tensor::X, config_tensor::W}},
        {OperationType::MATMUL, config_op::MATMUL, {}, {config_tensor::A, config_tensor::B}},
        {OperationType::BATCHNORM_TRAINING,
         config_op::BATCHNORM_TRAINING,
         {},
         {config_tensor::X, config_tensor::SCALE, config_tensor::BIAS, config_tensor::EPSILON}},
        {OperationType::BATCHNORM_INFERENCE,
         config_op::BATCHNORM_INFERENCE,
         {},
         {config_tensor::X,
          config_tensor::MEAN,
          config_tensor::INV_VARIANCE,
          config_tensor::SCALE,
          config_tensor::BIAS}},
        {OperationType::BATCHNORM_INFERENCE_VARIANCE_EXT,
         config_op::BATCHNORM_INFERENCE_VARIANCE_EXT,
         {},
         {config_tensor::X,
          config_tensor::MEAN,
          config_tensor::VARIANCE,
          config_tensor::SCALE,
          config_tensor::BIAS,
          config_tensor::EPSILON}},
        {OperationType::BATCHNORM_BACKWARD,
         config_op::BATCHNORM_BACKWARD,
         {},
         {config_tensor::DY, config_tensor::X, config_tensor::SCALE}},
        {OperationType::LAYERNORM,
         config_op::LAYERNORM,
         {{config_criterion::NORM_FWD_PHASE, HIPDNN_NORM_FWD_INFERENCE}},
         {config_tensor::X, config_tensor::SCALE, config_tensor::BIAS, config_tensor::EPSILON}},
        {OperationType::RMSNORM,
         config_op::RMSNORM,
         {{config_criterion::NORM_FWD_PHASE, HIPDNN_NORM_FWD_INFERENCE}},
         {config_tensor::X, config_tensor::SCALE, config_tensor::EPSILON}},
        {OperationType::RMSNORM_BACKWARD,
         config_op::RMSNORM_BACKWARD,
         {},
         {config_tensor::DY, config_tensor::X, config_tensor::SCALE, config_tensor::INV_RMS}},
        {OperationType::REDUCTION,
         config_op::REDUCTION,
         {{config_criterion::REDUCTION_MODE, HIPDNN_REDUCE_TENSOR_ADD}},
         {config_tensor::INPUT}},
        {OperationType::RESAMPLE_FWD,
         config_op::RESAMPLE_FWD,
         {{config_criterion::RESAMPLE_MODE, HIPDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING},
          {config_criterion::PADDING_MODE, HIPDNN_PADDING_ZERO_PAD}},
         {config_tensor::X}},
        {OperationType::POINTWISE_UNARY,
         config_op::POINTWISE,
         {{config_criterion::POINTWISE_MODE, HIPDNN_POINTWISE_RELU_FWD}},
         {config_tensor::IN_0}},
        {OperationType::POINTWISE_BINARY,
         config_op::POINTWISE,
         {{config_criterion::POINTWISE_MODE, HIPDNN_POINTWISE_ADD}},
         {config_tensor::IN_0, config_tensor::IN_1}},
    };

#ifdef HIPDNN_ENABLE_SDPA
    cases.push_back({OperationType::SDPA_FORWARD,
                     config_op::SDPA_FWD,
                     {},
                     {config_tensor::Q, config_tensor::K, config_tensor::V}});
    cases.push_back({OperationType::SDPA_BACKWARD,
                     config_op::SDPA_BWD,
                     {},
                     {config_tensor::Q,
                      config_tensor::K,
                      config_tensor::V,
                      config_tensor::O,
                      config_tensor::DO,
                      config_tensor::STATS}});
#endif

    return cases;
}

INSTANTIATE_TEST_SUITE_P(SupportedOps,
                         IntegrationAutotuneConfigRoundTrip,
                         ::testing::ValuesIn(configRoundTripCases()),
                         [](const ::testing::TestParamInfo<ConfigRoundTripCase>& info) {
                             return hipdnn_test_sdk::utilities::operationTypeToString(
                                 info.param.op);
                         });

#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
} // namespace
