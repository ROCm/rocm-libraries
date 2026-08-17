// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Shared fixture for the autotune frontend integration tests.
//
// Derives from the test_sdk IntegrationTestFixture and centralizes the graph
// setup that every autotune integration test needs:
//
// 1. getPluginPaths() — loads the test_autotune_plugin.
//
// 2. ConvGraphBundle — owns a conv-fprop Graph, its tensor attributes, the
//    backing device-side tensors, and the variant pack. buildVariantPack()
//    populates the variant pack from the current tensor UIDs and must be
//    called after build_operation_graph() assigns UIDs.
//
// 3. createConvGraph(name) — builds a small {1,4,4,4}/{4,4,3,3}/{1,4,4,4}
//    conv-fprop graph filled with deterministic values. The graph name is a
//    cosmetic label; each test passes its own.
//
// 4. assertAnySucceeded(results, message) — asserts the result vector is
//    non-empty and that at least one engine succeeded.
//
// Header-only because the helpers are small and only consumed by the handful
// of autotune integration translation units.

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/FrontendGraphFactory.hpp>
#include <hipdnn_test_sdk/utilities/GraphExecuteTestKit.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "test_plugins/TestPluginConstants.hpp"

namespace hipdnn_tests
{

/// Base fixture for autotune frontend integration tests.
///
/// Loads the test_autotune_plugin and provides the shared conv-fprop graph
/// bundle plus the common autotune-result assertion helper.
class AutotuneIntegrationFixture : public IntegrationTestFixture
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {plugin_constants::testAutotunePluginPath()};
    }

    struct ConvGraphBundle
    {
        ConvGraphBundle(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims)
            : xTensor(hipdnn_data_sdk::utilities::Tensor<float>(xDims))
            , wTensor(hipdnn_data_sdk::utilities::Tensor<float>(wDims))
            , yTensor(hipdnn_data_sdk::utilities::Tensor<float>(yDims))
        {
        }

        // Default-constructs with the canonical conv dims. Lets callers declare an
        // empty bundle that createBuiltConvGraph() then move-assigns into, which is
        // required because Tensor has no default constructor.
        ConvGraphBundle()
            : ConvGraphBundle({1, 4, 4, 4}, {4, 4, 3, 3}, {1, 4, 4, 4})
        {
        }

        std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> xAttr;
        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> wAttr;
        std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> yAttr;
        hipdnn_data_sdk::utilities::Tensor<float> xTensor;
        hipdnn_data_sdk::utilities::Tensor<float> wTensor;
        hipdnn_data_sdk::utilities::Tensor<float> yTensor;
        std::unordered_map<int64_t, void*> variantPack;

        // Populate the variant pack using the current tensor UIDs.
        // Must be called after build_operation_graph() assigns UIDs.
        void buildVariantPack()
        {
            variantPack.clear();
            variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
            variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
            variantPack[yAttr->get_uid()] = yTensor.memory().deviceData();
        }
    };

    /// Builds a small conv-fprop graph for autotune testing.
    /// Returns the graph plus tensors ready to populate a variant pack.
    static ConvGraphBundle createConvGraph(const std::string& name)
    {
        using namespace hipdnn_frontend;
        using namespace hipdnn_frontend::graph;
        using hipdnn_data_sdk::utilities::generateStrides;
        using hipdnn_data_sdk::utilities::TensorLayout;

        const std::vector<int64_t> xDims = {1, 4, 4, 4};
        const std::vector<int64_t> wDims = {4, 4, 3, 3};
        const std::vector<int64_t> yDims = {1, 4, 4, 4};

        ConvGraphBundle bundle(xDims, wDims, yDims);

        bundle.xTensor.fillWithValue(1.0f);
        bundle.wTensor.fillWithValue(1.0f);
        bundle.yTensor.fillWithValue(0.0f);

        auto graph = std::make_shared<Graph>();
        graph->set_name(name)
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        auto xAttr = std::make_shared<TensorAttributes>();
        xAttr->set_name("X")
            .set_dim(xDims)
            .set_stride(generateStrides(xDims, TensorLayout::NCHW.strideOrder))
            .set_data_type(DataType::FLOAT);

        auto wAttr = std::make_shared<TensorAttributes>();
        wAttr->set_name("W")
            .set_dim(wDims)
            .set_stride(generateStrides(wDims, TensorLayout::NCHW.strideOrder))
            .set_data_type(DataType::FLOAT);

        ConvFpropAttributes convAttrs;
        convAttrs.set_name("test_conv_fprop");
        convAttrs.set_padding({1, 1});
        convAttrs.set_stride({1, 1});
        convAttrs.set_dilation({1, 1});

        auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttrs);
        yAttr->set_output(true);

        bundle.graph = std::move(graph);
        bundle.xAttr = xAttr;
        bundle.wAttr = wAttr;
        bundle.yAttr = yAttr;

        return bundle;
    }

    /// Builds a conv-fprop graph and brings it through validate(),
    /// build_operation_graph(), and buildVariantPack(), asserting success at each
    /// step. The result is written to outBundle. Uses an out-parameter because
    /// gtest ASSERT_* macros require a void-returning function.
    void createBuiltConvGraph(const char* name, ConvGraphBundle& outBundle)
    {
        outBundle = createConvGraph(name);

        auto result = outBundle.graph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        result = outBundle.graph->build_operation_graph(_handle);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        outBundle.buildVariantPack();
    }

    /// Queries the execution workspace size, allocates it, and executes the graph,
    /// asserting success at each step.
    void buildWorkspaceAndExecute(ConvGraphBundle& bundle)
    {
        int64_t ws = 0;
        auto result = bundle.graph->get_workspace_size(ws);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        const hipdnn_data_sdk::utilities::Workspace execWorkspace(static_cast<size_t>(ws));
        result = bundle.graph->execute(_handle, bundle.variantPack, execWorkspace.get());
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
    }

    /// Asserts the result vector is non-empty and at least one engine succeeded.
    static void assertAnySucceeded(const std::vector<hipdnn_frontend::AutotuneResult>& results,
                                   const std::string& message
                                   = "No engine succeeded during autotune")
    {
        ASSERT_FALSE(results.empty());
        bool anySucceeded = false;
        for(const auto& r : results)
        {
            if(r.succeeded)
            {
                anySucceeded = true;
                break;
            }
        }
        ASSERT_TRUE(anySucceeded) << message;
    }

    // ── Shared execute-path kit (consolidation R1) ───────────────────────────
    //
    // The helpers below replace the hand-rolled ConvGraphBundle path with the
    // op-agnostic GraphExecuteTestKit. They take an OperationType so any future
    // deferred op (matmul, sdpa, norms, pointwise) reuses them with no new graph
    // or tensor code. The write graph and read graph MUST use the SAME op so
    // their autotune match keys agree.

    /// Builds a fresh graph for @p op via FrontendGraphFactory, brings it through
    /// validate() + build_operation_graph() (which assigns UIDs), then constructs
    /// an op-agnostic variant pack from the built graph. The graph and bundle are
    /// returned via out-parameters because gtest ASSERT_* needs a void return and
    /// because the bundle owns the tensors backing the variant pack.
    void
        buildGraphAndBundle(hipdnn_test_sdk::utilities::OperationType op,
                            std::shared_ptr<hipdnn_frontend::graph::Graph>& outGraph,
                            std::optional<hipdnn_test_sdk::utilities::GraphTensorBundle>& outBundle)
    {
        outGraph = hipdnn_test_sdk::utilities::buildGraphForOp(op);

        auto result = outGraph->validate();
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        result = outGraph->build_operation_graph(_handle);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        outBundle.emplace(*outGraph);
    }

    /// Builds a FRESH graph for @p op against the override config at @p configPath
    /// and reports the engine ID the backend selects. Sets HIPDNN_HEUR_CONFIG_PATH,
    /// runs the minimal selection path (build -> get_plan_name), then parses the
    /// plan name back to an int64 ID via engineNameOrIdToId so callers can compare
    /// IDs regardless of hex/decimal form.
    ///
    /// Op-agnostic: needs only the graph, no host tensors. Out-parameter because
    /// gtest ASSERT_* requires a void-returning function. Leaves the env var set;
    /// callers / TearDown unset it.
    void buildGraphAndGetSelectedEngineId(hipdnn_test_sdk::utilities::OperationType op,
                                          const std::string& configPath,
                                          int64_t& outEngineId)
    {
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_HEUR_CONFIG_PATH", configPath.c_str());

        auto graph = hipdnn_test_sdk::utilities::buildGraphForOp(op);

        auto result = graph->build(_handle);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        std::string planName;
        result = graph->get_plan_name(planName);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        outEngineId = hipdnn_data_sdk::utilities::engineNameOrIdToId(planName);
    }
};

} // namespace hipdnn_tests
