// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration test for autotune config file persistence round-trip.
// Autotunes to a JSON config file, then verifies the config can be loaded
// via HIPDNN_HEUR_CONFIG_PATH to rebuild and execute the graph.

#include <filesystem>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

#include "AutotuneIntegrationFixture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

class IntegrationAutotunePersistence : public hipdnn_tests::AutotuneIntegrationFixture
{
protected:
    void TearDown() override
    {
        // Clean up temp config file if it exists
        std::error_code ec;
        std::filesystem::remove(_configFile, ec);

        // Ensure the env var is always cleared
        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_HEUR_CONFIG_PATH");

        AutotuneIntegrationFixture::TearDown();
    }

    std::filesystem::path _configFile
        = std::filesystem::temp_directory_path() / "test_autotune_persistence_config.json";
};

// Test: autotune -> save config -> set override env -> build() -> execute() -> verify success
TEST_F(IntegrationAutotunePersistence, ConfigFileRoundTrip)
{
    // Phase 1: Autotune and write config file
    {
        ConvGraphBundle bundle;
        createBuiltConvGraph("autotune_persistence_test_conv", bundle);

        auto result = bundle.graph->add_all_engines();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        int64_t maxWs = 0;
        result = bundle.graph->get_estimated_max_workspace_size(maxWs);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const Workspace workspace(static_cast<size_t>(maxWs));

        AutotuneConfig config;
        config.mode = TuneMode::STANDARD;
        config.strategy = AutotuneStrategy::FIXED_AVERAGE;
        config.timedIterations = 1;
        config.warmupIterations = 1;

        const AutotuneStorageConfig storageConfig{_configFile, false};

        std::vector<AutotuneResult> results;
        result = bundle.graph->autotune(
            _handle, bundle.variantPack, workspace.get(), maxWs, config, storageConfig, &results);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        ASSERT_TRUE(std::filesystem::exists(_configFile))
            << "Config file was not created at '" << _configFile << "'";
        ASSERT_FALSE(results.empty());
        ASSERT_TRUE(results[0].succeeded) << "Winning engine did not succeed";
    }

    // Phase 2: Build a new graph using the override config file
    {
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_HEUR_CONFIG_PATH", _configFile.string().c_str());

        auto bundle = createConvGraph("autotune_persistence_test_conv");

        auto result = bundle.graph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = bundle.graph->build(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        bundle.buildVariantPack();

        buildWorkspaceAndExecute(bundle);

        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_HEUR_CONFIG_PATH");
    }
}

} // namespace
