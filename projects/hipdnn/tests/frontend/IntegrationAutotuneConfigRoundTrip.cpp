// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU-only end-to-end round-trip test for the autotune config file.
//
// One real conv graph is autotuned (restricted to a single tunable engine A so
// the winner is deterministic) and the winner is written to a JSON config. A
// fresh graph then loads that config via HIPDNN_HEUR_CONFIG_PATH and build()
// re-selects the engine. The discriminator is the engine-ID FLIP: with the file
// as-written the backend selects A; after rewriting the file's engine_name to a
// different candidate B, a fresh build() selects B. Selection tracking the file
// content across two engines (the only thing that changed) proves the config
// drove the choice.
//
// Engine IDs are compared (not name strings): get_plan_name() returns the
// frontend's lowercase hex form for unregistered test-plugin IDs; feeding that
// back through engineNameOrIdToId recovers the int64 ID. Frontend API only.

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "AutotuneIntegrationFixture.hpp"
#include "test_plugins/TestPluginEngineIdMap.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

class IntegrationAutotuneConfigRoundTrip : public hipdnn_tests::AutotuneIntegrationFixture
{
protected:
    void SetUp() override
    {
        AutotuneIntegrationFixture::SetUp();
        // The base SetUp() may GTEST_SKIP() when no device is present; that does
        // not unwind into this frame, so guard before touching anything else.
        if(IsSkipped())
        {
            return;
        }

        // The shared frontend-test main() pins HIPDNN_HEUR_POLICY_ORDER to just
        // the TestGoodHeuristic policy, which excludes SelectionHeuristic::Config
        // and means HIPDNN_HEUR_CONFIG_PATH is never consulted during build().
        // This round-trip test exists specifically to verify that config content
        // drives engine selection, so restore the production default order with
        // SelectionHeuristic::Config FIRST. When the config matches the conv node
        // the Config policy reorders the candidate engines (preferred first) and
        // the outer policy loop stops there; SelectionHeuristic::StaticOrdering is
        // the canonical fallback that enumerates the engine-plugin candidates when
        // Config declines. The candidate set (engines A and B) comes from the
        // engine plugin, not from any heuristic plugin, so dropping the test
        // heuristic from this order does not lose either candidate. The override
        // is restored to whatever main() set in TearDown via the scoped setter.
        _policyOrderEnv.emplace("HIPDNN_HEUR_POLICY_ORDER",
                                "SelectionHeuristic::Config,SelectionHeuristic::StaticOrdering");
    }

    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove(_configFile, ec);
        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_HEUR_CONFIG_PATH");
        // Restore HIPDNN_HEUR_POLICY_ORDER to the value the shared main() set so
        // sibling tests in this binary are not affected by this test's override.
        _policyOrderEnv.reset();
        AutotuneIntegrationFixture::TearDown();
    }

    // Rewrites engine_overrides[0]["engine_name"] in the config file to a new
    // value (read-modify-write). The value is written as a plain string so a
    // decimal literal exercises the Stage-1 decimal-parse branch.
    void rewriteFirstEngineName(const std::string& newEngineName)
    {
        std::ifstream in(_configFile);
        ASSERT_TRUE(in.is_open()) << "Could not open config file for rewrite: " << _configFile;
        nlohmann::json j;
        in >> j;
        in.close();

        ASSERT_TRUE(j.contains("engine_overrides"));
        ASSERT_FALSE(j["engine_overrides"].empty());
        j["engine_overrides"][0]["engine_name"] = newEngineName;

        std::ofstream out(_configFile, std::ios::trunc);
        ASSERT_TRUE(out.is_open()) << "Could not open config file for write: " << _configFile;
        out << j.dump(2);
        out.close();
    }

    std::filesystem::path _configFile
        = std::filesystem::temp_directory_path() / "test_autotune_config_round_trip.json";

    // Scoped override of HIPDNN_HEUR_POLICY_ORDER so SelectionHeuristic::Config
    // is active during build(); save/restore is handled by the setter's lifetime.
    std::optional<hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter> _policyOrderEnv;
};

TEST_F(IntegrationAutotuneConfigRoundTrip, EngineIdFlipsWithConfigContent)
{
    const int64_t engineAId = hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();
    const int64_t engineBId = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineB>();
    ASSERT_NE(engineAId, engineBId);

    // The kit builds both the write graph (Phase 1) and the read graphs (Phases
    // 2a/2b) from the same op, so their autotune match keys agree.
    const auto op = hipdnn_test_sdk::utilities::OperationType::CONV_FORWARD;

    // ── Phase 1: autotune (single tunable engine A) -> write config ──────────
    {
        std::shared_ptr<Graph> graph;
        std::optional<hipdnn_test_sdk::utilities::GraphTensorBundle> bundle;
        buildGraphAndBundle(op, graph, bundle);

        auto result = graph->add_all_engines();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        int64_t maxWs = 0;
        result = graph->get_estimated_max_workspace_size(maxWs);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const Workspace workspace(static_cast<size_t>(maxWs));

        AutotuneConfig config;
        config.mode = TuneMode::AUTO;
        config.strategy = AutotuneStrategy::SINGLE_SHOT;
        config.warmupIterations = 1;
        // Single tunable engine -> deterministic winner == A, so the writer
        // emits A's engine_name into the file.
        config.engineIdFilter = {engineAId};

        const AutotuneStorageConfig storageConfig{_configFile, false};

        std::vector<AutotuneResult> results;
        result = graph->autotune(_handle,
                                 bundle->variantPack(),
                                 workspace.get(),
                                 maxWs,
                                 config,
                                 storageConfig,
                                 &results);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // The "writer emits A" assumption is checked, not assumed.
        ASSERT_EQ(results.size(), 1u) << "Filter should select exactly 1 engine";
        EXPECT_EQ(results[0].engineId, engineAId);

        // Minimal content check (detailed op/tensor content is the Stage 3 job).
        ASSERT_TRUE(std::filesystem::exists(_configFile))
            << "Config file was not created at '" << _configFile << "'";
        std::ifstream in(_configFile);
        ASSERT_TRUE(in.is_open());
        nlohmann::json j;
        ASSERT_NO_THROW(in >> j);
        ASSERT_TRUE(j.contains("engine_overrides"));
        ASSERT_FALSE(j["engine_overrides"].empty());
    }

    // ── Phase 2a: assert-A-as-written (do NOT modify the file) ───────────────
    // Exercises the writer's actual hex-fallback output through the Stage-1
    // hex-parse path end-to-end.
    {
        int64_t selectedId = 0;
        buildGraphAndGetSelectedEngineId(op, _configFile.string(), selectedId);
        EXPECT_EQ(selectedId, engineAId)
            << "Backend should select engine A as written by the autotune writer";
    }

    // ── Phase 2b: flip to engine B (plain decimal literal) ───────────────────
    // Decimal literal also covers the Stage-1 decimal-parse branch.
    {
        rewriteFirstEngineName(std::to_string(engineBId));

        int64_t selectedId = 0;
        buildGraphAndGetSelectedEngineId(op, _configFile.string(), selectedId);
        EXPECT_EQ(selectedId, engineBId)
            << "Backend selection should flip to engine B after rewriting the config";
    }
}

} // namespace
