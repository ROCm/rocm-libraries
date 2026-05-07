// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Frontend integration tests for overridable tensor shapes (RFC 0008,
// execute path). Covers:
//   * The four-corner dispatch matrix from RFC §9.4 (graph flag × plugin
//     capability) using the override-implementing and override-omitting
//     fakes published by Stream B.
//   * Test #11 (RFC 0008 plan §6.5) — non-override graph + override-implementing
//     plugin must invoke the existing executeOpGraph entry, NOT the override
//     entry.
//   * Test #15 (RFC 0008 plan §6.5) — empty-overrides pass-through for the
//     map overload: empty map + flag set is observably indistinguishable
//     from a non-override execute call.
//   * Map vs parallel-array equivalence (RFC §4.2 "pure sugar").
//
// Stream B's fake plugins record their last-called entry into a thread-local
// `TestPluginLastCallRecord` exposed via the suffixed
// `getLastCallRecord_<suffix>()` / `resetLastCallRecord_<suffix>()` C entry
// points (resolved via `getLastCallRecordIfLoaded` /
// `resetLastCallRecordIfLoaded` from `TestPluginCommon.hpp`, which dlsym
// the symbols out of the dlopen'd plugin `.so`). Per Risk #11
// (TLS-leak-between-tests), every test resets the TLS state in `SetUp()`.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "OverrideTestUtils.hpp"
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginCommon.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef HIPDNN_ENABLE_SDPA

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

/// Helper bundle: x and y device-resident tensors of the given declared dims.
template <typename DataType>
struct SimpleTensorBundle
{
    SimpleTensorBundle(const std::vector<int64_t>& dims)
        : xTensor(Tensor<DataType>(dims))
        , yTensor(Tensor<DataType>(dims))
    {
        xTensor.fillWithValue(static_cast<DataType>(1.0F));
        yTensor.fillWithValue(static_cast<DataType>(0.0F));
    }

    Tensor<DataType> xTensor;
    Tensor<DataType> yTensor;
};

/// Build a minimal pointwise (RELU) graph; declared dims are the upper bound
/// for any overrides supplied at execute time. Thin wrapper over the shared
/// `buildPointwiseReluGraph` helper that defaults strides to NCHW packed.
std::shared_ptr<Graph> createSimplePointwiseGraph(const std::string& graphName,
                                                  const std::vector<int64_t>& declaredDims,
                                                  bool dynamicShapeEnabled)
{
    return hipdnn_tests::override_test_utils::buildPointwiseReluGraph(
        graphName, declaredDims, /*strides=*/{}, dynamicShapeEnabled);
}

// Bring the shared `compileGraph(graph, handle)` helper into the file's
// anonymous namespace so existing call sites resolve unqualified.
using hipdnn_tests::override_test_utils::compileGraph;

/// Common fixture: load a configurable set of fake plugins and create a handle.
/// Per Risk #11, derived tests must reset the TLS LastCallRecord in their own
/// SetUp() before any plugin invocation.
class IntegrationOverrideExecuteBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
        // Reset the TLS LastCallRecord across every override-execute fake
        // plugin that may be loaded (see `OverrideTestUtils.hpp`).
        // Resetters for plugins not (yet) loaded are silently skipped
        // (Risk #11).
        hipdnn_tests::override_test_utils::resetAllOverrideFakePluginRecords();
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }

    /// Load the override-implementing and override-omitting fakes and create
    /// a handle. This is the standard "both plugins available" setup used by
    /// the four-corner matrix tests.
    void loadBothFakes()
    {
        const std::array<const char*, 2> paths
            = {hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath().c_str(),
               hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath().c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    /// Load only the override-omitting fake (no override entry exported).
    void loadOmittingOnly()
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath().c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    /// Load only the override-implementing fake (override entry available).
    void loadImplementingOnly()
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath().c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    hipdnnHandle_t _handle = nullptr;
};

} // namespace

// ----------------------------------------------------------------------------
// Four-corner matrix: graph flag × plugin capability (RFC §9.4).
//
//   Corner 1 (false × omitting):       legacy graph, legacy plugin → legacy entry
//   Corner 2 (false × implementing):   legacy graph, new plugin    → legacy entry (Test #11)
//   Corner 3 (true  × omitting):       new graph, legacy plugin    → no applicable engine
//   Corner 4 (true  × implementing):   new graph, new plugin       → override entry
// ----------------------------------------------------------------------------

class IntegrationOverrideExecuteFourCorner : public IntegrationOverrideExecuteBase
{
};

/// Corner 1: graph without `is_dynamic_shape_enabled`, override-omitting plugin
/// loaded. Dispatch must use `hipdnnEnginePluginExecuteOpGraph` (the legacy
/// entry). Verifies binary compatibility for the "no new feature anywhere"
/// case.
TEST_F(IntegrationOverrideExecuteFourCorner, LegacyGraphLegacyPluginUsesLegacyEntry)
{
    loadOmittingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);

    auto graph = createSimplePointwiseGraph(
        "FourCorner_LegacyGraphLegacyPlugin", dims, /*dynamicShapeEnabled=*/false);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    auto result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath(), "OverrideOmitting");
    ASSERT_NE(record, nullptr);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH)
        << "Corner 1 must dispatch through the legacy executeOpGraph entry.";
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(record->numOverrides, 0U) << "Legacy entry must receive no override metadata.";
}

/// Corner 2 / Test #11: graph without `is_dynamic_shape_enabled`, the
/// override-implementing plugin is loaded. The host MUST still pick the
/// legacy entry — the override entry is exclusively for graphs that opted in
/// at build time. (RFC 0008 plan Test #11.)
TEST_F(IntegrationOverrideExecuteFourCorner, LegacyGraphImplementingPluginUsesLegacyEntry)
{
    loadImplementingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);

    auto graph = createSimplePointwiseGraph(
        "FourCorner_LegacyGraphImplementingPlugin", dims, /*dynamicShapeEnabled=*/false);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    auto result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH)
        << "Test #11: even when the override entry is available, a graph "
           "that did not opt in must use the legacy executeOpGraph entry.";
}

/// Corner 3: graph with `is_dynamic_shape_enabled=true`, only the
/// override-omitting plugin loaded (reports `apiVersionWithoutTweak()` =
/// `"1.0.0"`). The version filter must exclude it from the applicability set,
/// so plan creation reports "no applicable engine" before any execute call.
TEST_F(IntegrationOverrideExecuteFourCorner, OverrideGraphOmittingPluginNoApplicableEngine)
{
    loadOmittingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createSimplePointwiseGraph(
        "FourCorner_OverrideGraphOmittingPlugin", dims, /*dynamicShapeEnabled=*/true);

    // Validation succeeds (it is structural, not engine-aware).
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Engine selection must fail downstream — the only loaded plugin reports
    // version "1.0.0" and is filtered out for override-flag graphs by the
    // applicability check (Stream B). The exact stage that surfaces the
    // failure is implementation-defined; what matters for this test is that
    // SOME stage rejects, and it does so without invoking any execute entry.
    const auto plansResult = graph->create_execution_plans();
    if(plansResult.code == ErrorCode::OK)
    {
        const auto supportResult = graph->check_support();
        // NOLINTNEXTLINE(readability-implicit-bool-conversion)
        EXPECT_NE(supportResult.code, ErrorCode::OK)
            << "Corner 3 must fail engine selection when no plugin meets the "
               "override-version floor.";
    }
    else
    {
        EXPECT_NE(plansResult.code, ErrorCode::OK);
    }

    // No execute entry should have been touched.
    const auto* record = getLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath(), "OverrideOmitting");
    ASSERT_NE(record, nullptr);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE)
        << "Corner 3 must not invoke any execute entry.";
}

/// Corner 4: graph with `is_dynamic_shape_enabled=true`, override-implementing
/// plugin loaded. Override execute entry is invoked with the supplied per-UID
/// shapes/strides. Workspace pointer is forwarded as-is.
TEST_F(IntegrationOverrideExecuteFourCorner, OverrideGraphImplementingPluginUsesOverrideEntry)
{
    loadImplementingOnly();

    const std::vector<int64_t> declaredDims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(declaredDims);

    auto graph = createSimplePointwiseGraph(
        "FourCorner_OverrideGraphImplementingPlugin", declaredDims, /*dynamicShapeEnabled=*/true);
    compileGraph(graph, _handle);

    // Override the X and Y tensors to a smaller shape, packed-strided.
    const std::vector<int64_t> overrideShape = {1, 3, 2, 2};
    const std::vector<int64_t> overrideStride = {int64_t{3} * 2 * 2, int64_t{2} * 2, 2, 1};

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const std::vector<int64_t> overrideUids = {1, 2};
    const std::vector<std::vector<int64_t>> overrideShapes = {overrideShape, overrideShape};
    const std::vector<std::vector<int64_t>> overrideStrides = {overrideStride, overrideStride};

    auto result = graph->execute(
        _handle, variantPack, nullptr, overrideUids, overrideShapes, overrideStrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES)
        << "Corner 4: override-flag graph + override-implementing plugin must "
           "dispatch through the override entry.";
    EXPECT_EQ(record->numOverrides, overrideUids.size());
}

// ----------------------------------------------------------------------------
// No-overrides short-circuit (Test #15) and map vs parallel-array equivalence.
// ----------------------------------------------------------------------------

class IntegrationOverrideExecuteShortCircuit : public IntegrationOverrideExecuteBase
{
};

/// Test #15: empty map + flag set. The map overload must lower to empty
/// parallel arrays, hit the no-overrides short-circuit, and dispatch through
/// the legacy executeOpGraph entry — observably identical to a non-override
/// execute.
TEST_F(IntegrationOverrideExecuteShortCircuit, EmptyMapDispatchesLegacyEntry)
{
    loadBothFakes();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);

    auto graph
        = createSimplePointwiseGraph("ShortCircuit_EmptyMap", dims, /*dynamicShapeEnabled=*/true);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const std::unordered_map<int64_t, OverrideEntry> emptyOverrides;
    auto result = graph->execute(_handle, variantPack, nullptr, emptyOverrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // With `dynamicShapeEnabled=true`, the omitting plugin is filtered out
    // by the host's version check, so dispatch lands on the
    // override-implementing fake even for the no-overrides short-circuit.
    const auto* record = getLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH)
        << "Empty-map override-execute must short-circuit to the legacy entry "
           "(RFC 0008 plan Test #15).";
    EXPECT_EQ(record->numOverrides, 0U);
}

/// Empty parallel arrays + flag set: same short-circuit as the map overload.
TEST_F(IntegrationOverrideExecuteShortCircuit, EmptyParallelArraysDispatchesLegacyEntry)
{
    loadBothFakes();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);

    auto graph = createSimplePointwiseGraph(
        "ShortCircuit_EmptyParallel", dims, /*dynamicShapeEnabled=*/true);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const std::vector<int64_t> emptyUids;
    const std::vector<std::vector<int64_t>> emptyShapes;
    const std::vector<std::vector<int64_t>> emptyStrides;

    auto result
        = graph->execute(_handle, variantPack, nullptr, emptyUids, emptyShapes, emptyStrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Same dispatch reasoning as `EmptyMapDispatchesLegacyEntry`: the
    // omitting plugin is filtered out under `dynamicShapeEnabled=true`.
    const auto* record = getLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH);
    EXPECT_EQ(record->numOverrides, 0U);
}

class IntegrationOverrideExecuteEquivalence : public IntegrationOverrideExecuteBase
{
};

/// Map vs parallel-array equivalence (RFC §4.2 "pure sugar"): the same
/// (uids, shapes, strides) supplied via either overload must produce the same
/// observable record on the override-implementing plugin.
TEST_F(IntegrationOverrideExecuteEquivalence, MapAndParallelArrayProduceSameDispatch)
{
    loadImplementingOnly();

    const std::vector<int64_t> declaredDims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(declaredDims);

    auto graph = createSimplePointwiseGraph(
        "Equivalence_MapVsArray", declaredDims, /*dynamicShapeEnabled=*/true);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const std::vector<int64_t> shape = {1, 3, 2, 2};
    const std::vector<int64_t> stride = {int64_t{3} * 2 * 2, int64_t{2} * 2, 2, 1};

    // First call: parallel-array form.
    {
        const std::vector<int64_t> uids = {1, 2};
        const std::vector<std::vector<int64_t>> shapes = {shape, shape};
        const std::vector<std::vector<int64_t>> strides = {stride, stride};
        auto result = graph->execute(_handle, variantPack, nullptr, uids, shapes, strides);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }
    const auto* recordAfterArray = getLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(recordAfterArray, nullptr);
    EXPECT_EQ(recordAfterArray->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES);
    const auto arrayUidCount = recordAfterArray->numOverrides;

    // Reset between calls so we observe just the second invocation's record.
    resetLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");

    // Second call: map form with identical content.
    {
        std::unordered_map<int64_t, OverrideEntry> overrides;
        overrides[1] = OverrideEntry{shape, stride};
        overrides[2] = OverrideEntry{shape, stride};
        auto result = graph->execute(_handle, variantPack, nullptr, overrides);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }
    const auto* recordAfterMap = getLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(recordAfterMap, nullptr);
    EXPECT_EQ(recordAfterMap->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(recordAfterMap->numOverrides, arrayUidCount)
        << "Map form must lower to the same (uids, shapes, strides) payload "
           "as the parallel-array form (RFC §4.2 'pure sugar').";
}

// ----------------------------------------------------------------------------
// Execution-plan validity guard for the override-execute parallel-array form.
//
// The non-override `Graph::execute()` overload guards against a missing
// compiled plan at the top of its body. The parallel-array override overload
// must perform the SAME guard so a user calling override-execute before
// `build_plans()` (or `from_compiled_plan_binary()`) gets a clean
// INVALID_VALUE diagnostic instead of dereferencing a null/invalid plan
// descriptor (RFC 0008 post-review fix #3).
// ----------------------------------------------------------------------------

class IntegrationOverrideExecutePlanGuard : public IntegrationOverrideExecuteBase
{
};

/// Override-execute (parallel-array form) called BEFORE `build_plans()` must
/// reject with INVALID_VALUE and surface the same wording as the non-override
/// guard. Verifies the override overload mirrors the existing top-of-body
/// guard at `Graph::execute(handle, variantPack, workspace)`.
TEST_F(IntegrationOverrideExecutePlanGuard, ArrayFormRejectedBeforeBuildPlan)
{
    loadImplementingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);

    // Build but DO NOT compile (no validate / build_operation_graph /
    // create_execution_plans / check_support / build_plans). The override
    // overload must reject before touching the (absent) plan descriptor.
    auto graph = createSimplePointwiseGraph(
        "PlanGuard_ArrayBeforeBuild", dims, /*dynamicShapeEnabled=*/true);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const std::vector<int64_t> overrideUids = {1, 2};
    const std::vector<std::vector<int64_t>> overrideShapes = {dims, dims};
    const std::vector<std::vector<int64_t>> overrideStrides
        = {{int64_t{3} * 4 * 4, int64_t{4} * 4, 4, 1}, {int64_t{3} * 4 * 4, int64_t{4} * 4, 4, 1}};

    auto result = graph->execute(
        _handle, variantPack, nullptr, overrideUids, overrideShapes, overrideStrides);

    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "Override-execute before build_plans must reject with INVALID_VALUE: " << result.err_msg;
    // Mirrors the wording of the non-override guard (Graph.hpp ~line 1818).
    EXPECT_NE(result.err_msg.find("no compiled execution plan"), std::string::npos)
        << "Diagnostic must surface the missing plan; got: " << result.err_msg;

    // The override-implementing fake must NOT have been touched: the guard
    // runs before any backend interaction.
    const auto* record = getLastCallRecordIfLoaded(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE)
        << "Plan-guard rejection must precede any backend dispatch.";
}

#endif // HIPDNN_ENABLE_SDPA
