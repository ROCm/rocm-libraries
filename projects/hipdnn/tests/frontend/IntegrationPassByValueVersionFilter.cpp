// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Frontend integration coverage for the RFC-0016 runtime pass-by-value engine
// version filter (EnginePluginResourceManager::getApplicableEngineIds).
//
// Contract under test: a graph carrying a runtime pass-by-value scalar raises
// the required engine-plugin API floor to K_PASS_BY_VALUE_MIN_API_VERSION
// ("1.2.0"). With only pre-1.2.0 fake plugins available, every engine is
// filtered out, so no execution plan can be created; a plugin reporting "1.2.0"
// clears the floor and serves the same graph; and with both loaded the filter
// admits exactly the 1.2.0 engine. The same graph built with an ordinary
// compile-time-constant scalar keeps the baseline 1.0.0 floor and is served by
// every plugin.
//
// GPU-less environments skip through the common test utility.

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

// How the epsilon scalar of the RMSNorm graph is declared.
enum class EpsilonKind
{
    RUNTIME_PASS_BY_VALUE, // runtime-with-default => requires plugin API >= 1.2.0
    COMPILE_TIME_CONSTANT // baked constant => baseline 1.0.0 floor
};

class IntegrationPassByValueVersionFilter : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
    }

    // Loads the given absolute plugin paths and creates the handle. Each test
    // picks the plugin set that isolates the behavior it exercises.
    void loadPlugins(const std::vector<std::string>& paths)
    {
        std::vector<const char*> cPaths;
        cPaths.reserve(paths.size());
        for(const auto& p : paths)
        {
            cPaths.push_back(p.c_str());
        }
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      cPaths.size(), cPaths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }

    // Build an RMSNorm (inference) graph whose epsilon scalar is either a
    // runtime pass-by-value parameter or a compile-time constant. All other
    // tensors are ordinary. Returns the graph plus its epsilon tensor so the
    // caller can assert on classification.
    static std::pair<std::shared_ptr<Graph>, std::shared_ptr<TensorAttributes>>
        buildRMSNormGraph(EpsilonKind epsilonKind)
    {
        const std::vector<int64_t> dims = {2, 3, 14, 14};
        std::vector<int64_t> scaleBiasDims = dims;
        scaleBiasDims[0] = 1;

        auto graph = std::make_shared<Graph>();
        graph->set_name("PassByValueVersionFilter")
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        auto x = Graph::tensor(
            makeTensorAttributes("X", DataType::FLOAT, dims, generateStrides(dims)));
        auto scale = Graph::tensor(makeTensorAttributes(
            "scale", DataType::FLOAT, scaleBiasDims, generateStrides(scaleBiasDims)));

        std::shared_ptr<TensorAttributes> epsilon;
        if(epsilonKind == EpsilonKind::RUNTIME_PASS_BY_VALUE)
        {
            // Runtime-with-default: value retained, runtime flag set. This is
            // the state that pushes the required engine floor to 1.2.0.
            epsilon = std::make_shared<TensorAttributes>(1e-5f, ScalarType::RUNTIME_PARAM);
        }
        else
        {
            // Compile-time constant: baked value, runtime flag clear => baseline 1.0.0.
            epsilon = std::make_shared<TensorAttributes>(1e-5f, ScalarType::COMPILE_TIME_CONST);
        }
        epsilon->set_name("epsilon");

        RMSNormAttributes attrs;
        attrs.set_name("rmsnorm");
        attrs.set_epsilon(epsilon);
        attrs.set_forward_phase(NormFwdPhase::INFERENCE);

        auto outputs = graph->rmsnorm(x, scale, std::move(attrs));
        const auto& y = outputs[0];
        y->set_output(true).set_data_type(DataType::FLOAT);

        return {graph, epsilon};
    }

    hipdnnHandle_t _handle = nullptr;
};

} // namespace

// Negative filter: a runtime pass-by-value scalar demands plugin API >= 1.2.0.
// With only a pre-1.2.0 plugin loaded, the version filter removes every engine,
// so plan creation must fail (no applicable engine) rather than silently
// dispatching to an engine that cannot honor the host-supplied scalar.
TEST_F(IntegrationPassByValueVersionFilter, RuntimeScalarYieldsNoApplicableEngine)
{
    loadPlugins({hipdnn_tests::plugin_constants::testGoodPluginPath()});

    auto [graph, epsilon] = buildRMSNormGraph(EpsilonKind::RUNTIME_PASS_BY_VALUE);

    // Guard: confirm we actually built the runtime pass-by-value state, else
    // the test would pass vacuously.
    ASSERT_TRUE(epsilon->get_is_runtime_pass_by_value());

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // The pre-1.2.0 plugin is filtered out for a runtime-pbv graph, so no
    // engine remains to plan against.
    std::vector<int64_t> rankedEngineIds;
    result = graph->get_ranked_engine_ids(rankedEngineIds);
    EXPECT_TRUE(result.code != ErrorCode::OK || rankedEngineIds.empty())
        << "Expected no applicable engine for a runtime pass-by-value graph, got "
        << rankedEngineIds.size() << " engine(s)";

    result = graph->create_execution_plans();
    EXPECT_NE(result.code, ErrorCode::OK)
        << "create_execution_plans must fail when the version filter leaves no engine";
}

// Positive control: the SAME graph shape with a compile-time-constant scalar
// keeps the baseline 1.0.0 floor and is still served by the pre-1.2.0 plugin.
// This proves the negative result above is due to the version floor, not an
// unrelated failure to serve RMSNorm.
TEST_F(IntegrationPassByValueVersionFilter, CompileTimeScalarIsServedByLegacyPlugin)
{
    loadPlugins({hipdnn_tests::plugin_constants::testGoodPluginPath()});

    auto [graph, epsilon] = buildRMSNormGraph(EpsilonKind::COMPILE_TIME_CONSTANT);

    // Guard: this must NOT be a runtime pass-by-value tensor.
    ASSERT_FALSE(epsilon->get_is_runtime_pass_by_value());
    ASSERT_TRUE(epsilon->get_has_compile_time_constant());

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<int64_t> rankedEngineIds;
    result = graph->get_ranked_engine_ids(rankedEngineIds);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_FALSE(rankedEngineIds.empty())
        << "Compile-time-constant graph should keep the baseline floor and stay served";

    result = graph->create_execution_plans();
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// Admit direction: a plugin reporting K_PASS_BY_VALUE_MIN_API_VERSION ("1.2.0")
// clears the runtime pass-by-value floor, so the SAME runtime-pbv graph that the
// legacy plugin is filtered out for is served here. Closes the admit half of
// the version filter hermetically (fake plugin, no GPU dispatch).
TEST_F(IntegrationPassByValueVersionFilter, RuntimeScalarServedByPassByValuePlugin)
{
    loadPlugins({hipdnn_tests::plugin_constants::testPassByValuePluginPath()});

    auto [graph, epsilon] = buildRMSNormGraph(EpsilonKind::RUNTIME_PASS_BY_VALUE);
    ASSERT_TRUE(epsilon->get_is_runtime_pass_by_value());

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    std::vector<int64_t> rankedEngineIds;
    result = graph->get_ranked_engine_ids(rankedEngineIds);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_FALSE(rankedEngineIds.empty())
        << "A 1.2.0 plugin must serve a runtime pass-by-value graph";

    result = graph->create_execution_plans();
    EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// Differential isolation: with BOTH a pre-1.2.0 (legacy) and a 1.2.0 plugin
// loaded, the version filter admits exactly the 1.2.0 engine for a runtime-pbv
// graph and drops the legacy engine — while the same-shape compile-time-constant
// graph keeps BOTH. Proves the filter keys on the runtime flag, not op support.
TEST_F(IntegrationPassByValueVersionFilter, RuntimeScalarFiltersToCompatiblePluginOnly)
{
    loadPlugins({hipdnn_tests::plugin_constants::testGoodPluginPath(),
                 hipdnn_tests::plugin_constants::testPassByValuePluginPath()});

    const int64_t legacyEngineId = hipdnn_tests::plugin_constants::engineId<GoodPlugin>();
    const int64_t passByValueEngineId
        = hipdnn_tests::plugin_constants::engineId<PassByValuePlugin>();

    // Runtime pass-by-value: only the 1.2.0 engine survives the floor.
    {
        auto [graph, epsilon] = buildRMSNormGraph(EpsilonKind::RUNTIME_PASS_BY_VALUE);
        ASSERT_TRUE(epsilon->get_is_runtime_pass_by_value());

        ASSERT_EQ(graph->validate().code, ErrorCode::OK);
        ASSERT_EQ(graph->build_operation_graph(_handle).code, ErrorCode::OK);

        std::vector<int64_t> rankedEngineIds;
        auto result = graph->get_ranked_engine_ids(rankedEngineIds);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), passByValueEngineId),
                  rankedEngineIds.end())
            << "1.2.0 engine must be present for a runtime pass-by-value graph";
        EXPECT_EQ(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), legacyEngineId),
                  rankedEngineIds.end())
            << "legacy (<1.2.0) engine must be filtered out for a runtime pass-by-value graph";
        EXPECT_EQ(graph->create_execution_plans().code, ErrorCode::OK);
    }

    // Compile-time constant: baseline floor, so BOTH engines remain.
    {
        auto [graph, epsilon] = buildRMSNormGraph(EpsilonKind::COMPILE_TIME_CONSTANT);
        ASSERT_FALSE(epsilon->get_is_runtime_pass_by_value());

        ASSERT_EQ(graph->validate().code, ErrorCode::OK);
        ASSERT_EQ(graph->build_operation_graph(_handle).code, ErrorCode::OK);

        std::vector<int64_t> rankedEngineIds;
        auto result = graph->get_ranked_engine_ids(rankedEngineIds);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        EXPECT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), passByValueEngineId),
                  rankedEngineIds.end())
            << "1.2.0 engine must serve a compile-time-constant graph";
        EXPECT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), legacyEngineId),
                  rankedEngineIds.end())
            << "legacy engine must also serve a compile-time-constant graph (baseline floor)";
    }
}
