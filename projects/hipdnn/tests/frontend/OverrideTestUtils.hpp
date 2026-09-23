// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Shared frontend integration-test helpers for override-execute tests
// (RFC 0008).
//
// 1. `compileGraph(graph, handle)` — runs the standard 5-step compile
//    sequence (validate → build_operation_graph → create_execution_plans
//    → check_support → build_plans), asserting each step succeeds.
//
// 2. `resetAllOverrideFakePluginRecords()` — resets the suffixed
//    thread-local `LastCallRecord` for every override-execute fake plugin
//    that might have been loaded by a SetUp() routine. Resetters for
//    plugins not currently loaded are silently skipped.
//
// 3. `buildPointwiseReluGraph(name, dims, strides, overrideShapeEnabled)`
//    — builds a minimal pointwise (RELU) graph with two FLOAT tensors
//    (UIDs 1 and 2). When `strides` is empty, NCHW packed strides are
//    computed from `dims`. `overrideShapeEnabled` controls whether
//    `set_override_shape_enabled(true)` is applied.
//
// Header-only because the helpers are tiny and only consumed by a small
// number of test translation units.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend.hpp>
#include <test_plugins/TestPluginCommon.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hipdnn_tests::override_test_utils
{

/// Run the standard compile sequence for an override-or-non-override graph.
/// Each step is asserted; on failure the error message is included.
inline void compileGraph(std::shared_ptr<hipdnn_frontend::graph::Graph>& graph,
                         hipdnnHandle_t handle)
{
    auto result = graph->validate();
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(handle);
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

    result = graph->check_support();
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

    result = graph->build_plans();
    ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
}

/// Reset the suffixed TLS LastCallRecord across every override-execute fake
/// plugin that may be loaded by the calling fixture. Plugins not currently
/// loaded in the address space are silently skipped.
inline void resetAllOverrideFakePluginRecords()
{
    resetLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    resetLastCallRecordIfLoaded(hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath(),
                                "OverrideOmitting");
    resetLastCallRecordIfLoaded(hipdnn_tests::plugin_constants::testVersionLiarPluginPath(),
                                "VersionLiar");
    resetLastCallRecordIfLoaded(hipdnn_tests::plugin_constants::testSecondOverridePluginPath(),
                                "SecondOverride");
}

/// Build a minimal pointwise (RELU) graph with two FLOAT tensors (UIDs 1
/// and 2). `dims` describes the declared shape for both tensors; `strides`
/// describes the declared strides (when empty, NCHW packed strides are
/// computed from `dims`). `overrideShapeEnabled == true` opts the graph
/// into override-execute (RFC 0008).
inline std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildPointwiseReluGraph(const std::string& graphName,
                            const std::vector<int64_t>& dims,
                            const std::vector<int64_t>& strides,
                            bool overrideShapeEnabled)
{
    using hipdnn_frontend::DataType;
    using hipdnn_frontend::PointwiseMode;
    using hipdnn_frontend::graph::Graph;
    using hipdnn_frontend::graph::PointwiseAttributes;
    using hipdnn_frontend::graph::TensorAttributes;

    auto graph = std::make_shared<Graph>();
    graph->set_name(graphName)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

#ifdef HIPDNN_ENABLE_SDPA
    if(overrideShapeEnabled)
    {
        graph->set_override_shape_enabled(true);
    }
#else
    // The override-execute opt-in setter is `#ifdef HIPDNN_ENABLE_SDPA`-gated
    // (see Graph.hpp). Non-SDPA builds compile this helper for the
    // malformed-version-plugin test, which always passes
    // `overrideShapeEnabled=false`; silence the unused-parameter warning.
    (void)overrideShapeEnabled;
#endif

    // NCHW packed strides default; assumes 4-D `dims` when `strides` is
    // empty (mirrors the original three inline helpers).
    const std::vector<int64_t> packedStrides
        = strides.empty() ? hipdnn_data_sdk::utilities::generateStrides(dims) : strides;

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim(dims)
        .set_stride(packedStrides)
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes attrs;
    attrs.set_name("relu_node");
    attrs.set_mode(PointwiseMode::RELU_FWD);

    auto y = graph->pointwise(x, attrs);
    y->set_uid(2)
        .set_dim(dims)
        .set_stride(packedStrides)
        .set_data_type(DataType::FLOAT)
        .set_output(true);

    return graph;
}

} // namespace hipdnn_tests::override_test_utils
