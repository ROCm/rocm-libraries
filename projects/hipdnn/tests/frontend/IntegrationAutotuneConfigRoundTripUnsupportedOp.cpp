// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU-only test for the autotune config WRITER's unsupported-op behavior.
//
// An op that is unsupported for config round-trip yields an empty match key. CUSTOM_OP is
// canonical here: it is constructible and autotunable by the fake plugin, but the config
// writer intentionally has no primary-node branch for plugin-defined custom matching semantics.
//
// When the writer encounters an empty match key on the config-file write path it must NOT
// persist an entry for that op (a tensor-less or under-specified entry can never be matched
// safely by the reader and would falsely imply the op round-trips). This test runs the real
// autotune write path for a CUSTOM_OP graph with config-file write enabled and asserts that no
// override entry is written: the file is either absent or has an empty/absent engine_overrides
// array. A warning is also emitted via the frontend logger (not asserted here — the file
// omission is the observable contract this test pins).
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

#include <hipdnn_data_sdk/detail/AutotuneConfigNames.hpp>
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
namespace config_json = hipdnn_data_sdk::detail::autotune_config::json;

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

std::shared_ptr<Graph> buildUnsupportedCustomOpGraph()
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("UnsupportedCustomOpConfigRoundTripGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto input0 = std::make_shared<TensorAttributes>();
    input0->set_uid(1).set_name("input0").set_data_type(DataType::FLOAT);
    input0->set_dim({2, 3}).set_stride({3, 1});

    auto input1 = std::make_shared<TensorAttributes>();
    input1->set_uid(2).set_name("input1").set_data_type(DataType::FLOAT);
    input1->set_dim({2, 3}).set_stride({3, 1});

    CustomOpAttributes attrs;
    attrs.set_custom_op_id("unsupported_custom_op")
        .set_name("unsupported_custom_op_for_config_round_trip");

    auto outputs = graph->custom_op({input0, input1}, 1, std::move(attrs));
    outputs[0]->set_uid(3).set_output(true).set_name("output0");
    outputs[0]->set_dim({2, 3}).set_stride({3, 1}).set_data_type(DataType::FLOAT);

    return graph;
}

void buildUnsupportedGraphAndBundle(
    hipdnnHandle_t handle,
    std::shared_ptr<Graph>& graph,
    std::optional<hipdnn_test_sdk::utilities::GraphTensorBundle>& bundle)
{
    graph = buildUnsupportedCustomOpGraph();

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.emplace(*graph);
}

// Autotuning an unsupported op (CUSTOM_OP, empty match key) with config write
// enabled must NOT write an override entry for it.
TEST_F(IntegrationAutotuneConfigRoundTripUnsupportedOp, UnsupportedOpWritesNoConfigEntry)
{
    std::shared_ptr<Graph> graph;
    std::optional<hipdnn_test_sdk::utilities::GraphTensorBundle> bundle;
    buildUnsupportedGraphAndBundle(_handle, graph, bundle);

    auto result = graph->add_all_engines();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t maxWs = 0;
    result = graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
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
        if(j.contains(config_json::ENGINE_OVERRIDES))
        {
            EXPECT_TRUE(j[config_json::ENGINE_OVERRIDES].empty())
                << "Writer must not persist an override entry for an unsupported op";
        }
    }
}

} // namespace
