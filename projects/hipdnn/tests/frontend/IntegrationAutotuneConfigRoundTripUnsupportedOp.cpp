// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU-only test for the autotune config WRITER's unsupported-op behavior.
//
// An op that is unsupported for config round-trip yields an EMPTY match key from
// detail::getMatchKeyTensors (it has no op-aware selector branch). REDUCTION is
// the canonical such op: it has a graph-factory builder (so it is constructible
// and autotunable) but no op-aware branch, so it produces an empty match key.
//
// When the writer encounters an empty match key on the config-file write path it
// must NOT persist an entry for that op (a tensor-less entry can never be matched
// by the reader and would falsely imply the op round-trips). This test runs the
// real autotune write path for a REDUCTION graph with config-file write enabled
// and asserts that no override entry is written: the file is either absent or has
// an empty/absent engine_overrides array. A warning is also emitted via the
// frontend logger (not asserted here — the file omission is the observable
// contract this test pins).
//
// Frontend API only; GPU-gated via the fixture's SKIP_IF_NO_DEVICES().

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/FrontendGraphFactory.hpp>
#include <hipdnn_test_sdk/utilities/GraphExecuteTestKit.hpp>

#include "AutotuneIntegrationFixture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

class IntegrationAutotuneConfigRoundTripUnsupportedOp
    : public hipdnn_tests::AutotuneIntegrationFixture
{
protected:
    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove(_configFile, ec);
        AutotuneIntegrationFixture::TearDown();
    }

    std::filesystem::path _configFile = std::filesystem::temp_directory_path()
                                        / "test_autotune_config_round_trip_unsupported_op.json";
};

// Autotuning an unsupported op (REDUCTION, empty match key) with config write
// enabled must NOT write an override entry for it.
TEST_F(IntegrationAutotuneConfigRoundTripUnsupportedOp, UnsupportedOpWritesNoConfigEntry)
{
    const auto op = hipdnn_test_sdk::utilities::OperationType::REDUCTION;

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

    const AutotuneStorageConfig storageConfig{_configFile, false};

    std::vector<AutotuneResult> results;
    result = graph->autotune(
        _handle, bundle->variantPack(), workspace.get(), maxWs, config, storageConfig, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // The autotune itself runs (at least one engine succeeds): the omit is a
    // WRITE decision, not an autotune failure.
    assertAnySucceeded(results, "Autotune produced no successful result for the unsupported op");

    // The writer must have omitted the entry. The file is either not created at
    // all (nothing to write) or, if created, has no override entries.
    if(std::filesystem::exists(_configFile))
    {
        std::ifstream in(_configFile);
        ASSERT_TRUE(in.is_open());
        nlohmann::json j;
        ASSERT_NO_THROW(in >> j);
        if(j.contains("engine_overrides"))
        {
            EXPECT_TRUE(j["engine_overrides"].empty())
                << "Writer must not persist an override entry for an unsupported op";
        }
    }
}

} // namespace
