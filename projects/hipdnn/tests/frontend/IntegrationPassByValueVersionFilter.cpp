// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Frontend integration coverage for the RFC-0016 runtime pass-by-value engine
// version filter (EnginePluginResourceManager::getApplicableEngineIds).
//
// Contract under test: a graph carrying a runtime pass-by-value scalar raises
// the required engine-plugin API floor to K_PASS_BY_VALUE_MIN_API_VERSION
// ("1.2.0"). With only pre-1.2.0 fake plugins available, every engine is
// filtered out, so no execution plan can be created. The *same* graph built
// with an ordinary compile-time-constant scalar keeps the baseline 1.0.0 floor
// and is still served by the pre-1.2.0 plugin.
//
// GPU-less environments skip through the common test utility.

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

        // Load only the pre-1.2.0 default fake plugin (reports API "1.0.0").
        // It generically serves the RMSNorm graph, so any "no applicable
        // engine" outcome is attributable to the version floor, not to op
        // support.
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
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
