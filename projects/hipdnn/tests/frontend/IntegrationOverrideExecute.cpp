// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Frontend integration tests for overridable tensor shapes (RFC 0008,
// execute path). Covers the graph-flag/plugin-capability dispatch matrix,
// empty-override fall-through, map vs parallel-array equivalence, and
// compiled-plan-only execution.
//
// Fake plugins record their last-called entry into a thread-local
// `TestPluginLastCallRecord` exposed via suffixed C entry points. The
// IfLoaded helpers resolve those symbols from already-loaded plugin `.so`
// handles without causing a plugin load.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "OverrideTestUtils.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginCommon.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
                                                  bool overrideShapeEnabled)
{
    return hipdnn_tests::override_test_utils::buildPointwiseReluGraph(
        graphName, declaredDims, /*strides=*/{}, overrideShapeEnabled);
}

// Bring the shared `compileGraph(graph, handle)` helper into the file's
// anonymous namespace so existing call sites resolve unqualified.
using hipdnn_tests::override_test_utils::compileGraph;

/// Common fixture: load a configurable set of fake plugins and create a handle.
/// Each test starts with any already-loaded fake-plugin records reset.
class IntegrationOverrideExecuteBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
        // Reset any override-execute fake plugin already loaded in this process.
        hipdnn_tests::override_test_utils::resetAllOverrideFakePluginRecords();
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
        _pluginLibraries.clear();
    }

    /// Load the override-implementing and override-omitting fakes and create
    /// a handle. This is the standard "both plugins available" setup used by
    /// the four-corner matrix tests.
    void loadBothFakes()
    {
        const auto& implementingPath
            = hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath();
        const auto& omittingPath = hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath();
        ownPluginLibraries({implementingPath, omittingPath});

        const std::array<const char*, 2> paths = {implementingPath.c_str(), omittingPath.c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    /// Load only the override-omitting fake (no override entry exported).
    void loadOmittingOnly()
    {
        const auto& omittingPath = hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath();
        ownPluginLibraries({omittingPath});

        const std::array<const char*, 1> paths = {omittingPath.c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    /// Load only the override-implementing fake (override entry available).
    void loadImplementingOnly()
    {
        const auto& implementingPath
            = hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath();
        ownPluginLibraries({implementingPath});

        const std::array<const char*, 1> paths = {implementingPath.c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void loadImplementingAndSecondOverride()
    {
        const auto& implementingPath
            = hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath();
        const auto& secondPath = hipdnn_tests::plugin_constants::testSecondOverridePluginPath();
        ownPluginLibraries({implementingPath, secondPath});

        const std::array<const char*, 2> paths = {implementingPath.c_str(), secondPath.c_str()};

        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void resetOverrideImplementingRecord() const
    {
        resetRecordForOwnedPlugin(
            hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
            "OverrideImplementing");
    }

    const TestPluginLastCallRecord* getOverrideImplementingRecord() const
    {
        return getRecordForOwnedPlugin(
            hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
            "OverrideImplementing");
    }

    const TestPluginLastCallRecord* getSecondOverrideRecord() const
    {
        return getRecordForOwnedPlugin(
            hipdnn_tests::plugin_constants::testSecondOverridePluginPath(), "SecondOverride");
    }

    const TestPluginLastCallRecord* getRecordForOwnedPlugin(const std::string& pluginPath,
                                                            const std::string& suffix) const
    {
        return getLastCallRecord(ownedPluginLibrary(pluginPath), suffix);
    }

    void resetRecordForOwnedPlugin(const std::string& pluginPath, const std::string& suffix) const
    {
        resetLastCallRecord(ownedPluginLibrary(pluginPath), suffix);
    }

    hipdnnHandle_t _handle = nullptr;

private:
    void ownPluginLibraries(const std::vector<std::string>& pluginPaths)
    {
        _pluginLibraries.clear();
        _pluginLibraries.reserve(pluginPaths.size());
        for(const auto& pluginPath : pluginPaths)
        {
            _pluginLibraries.emplace_back(pluginPath);
        }
    }

    const ScopedTestPluginLibrary& ownedPluginLibrary(const std::string& pluginPath) const
    {
        for(const auto& pluginLibrary : _pluginLibraries)
        {
            if(pluginLibrary.pluginPath() == pluginPath)
            {
                return pluginLibrary;
            }
        }
        throw std::logic_error("Test plugin is not owned by this fixture: " + pluginPath);
    }

    std::vector<ScopedTestPluginLibrary> _pluginLibraries;
};

std::vector<int64_t> packedNchwStrides(const std::vector<int64_t>& dims)
{
    return hipdnn_data_sdk::utilities::generateStrides(dims);
}

std::string describeCompiledPlanForFailure(const std::vector<uint8_t>& compiledPlan)
{
    return "compiled_plan_size=" + std::to_string(compiledPlan.size());
}

void createPlanOnlyGraph(hipdnnHandle_t handle,
                         const std::vector<int64_t>& dims,
                         std::shared_ptr<Graph>& restored)
{
    auto source
        = createSimplePointwiseGraph("PlanOnly_Source", dims, /*overrideShapeEnabled=*/true);
    compileGraph(source, handle);

    auto [compiledPlan, serializeResult] = source->to_compiled_plan_binary();
    ASSERT_EQ(serializeResult.code, ErrorCode::OK) << serializeResult.err_msg;
    ASSERT_FALSE(compiledPlan.empty());

    restored = std::make_shared<Graph>();
    auto restoreResult = restored->from_compiled_plan_binary(handle, compiledPlan);
    ASSERT_EQ(restoreResult.code, ErrorCode::OK) << restoreResult.err_msg << "\n"
                                                 << describeCompiledPlanForFailure(compiledPlan);
}

void expectCapturedOverrides(const TestPluginLastCallRecord& record,
                             const std::vector<int64_t>& overrideUids,
                             const std::vector<std::vector<int64_t>>& overrideShapes,
                             const std::vector<std::vector<int64_t>>& overrideStrides)
{
    ASSERT_EQ(record.numOverrides, overrideUids.size());
    ASSERT_LE(overrideUids.size(), K_MAX_TEST_OVERRIDES);
    for(size_t i = 0; i < overrideUids.size(); ++i)
    {
        SCOPED_TRACE("override[" + std::to_string(i) + "]");
        EXPECT_EQ(record.capturedUniqueIds[i], overrideUids[i]);
        ASSERT_EQ(overrideShapes[i].size(), overrideStrides[i].size());
        ASSERT_LE(overrideShapes[i].size(), K_MAX_TEST_OVERRIDE_RANK);
        EXPECT_EQ(record.capturedLengths[i], overrideShapes[i].size());
        for(size_t axis = 0; axis < overrideShapes[i].size(); ++axis)
        {
            EXPECT_EQ(record.capturedShapes[i][axis], overrideShapes[i][axis]);
            EXPECT_EQ(record.capturedStrides[i][axis], overrideStrides[i][axis]);
        }
    }
}

void expectCapturedOverridesByUid(
    const TestPluginLastCallRecord& record,
    const std::unordered_map<int64_t, OverrideEntry>& expectedOverrides)
{
    ASSERT_EQ(record.numOverrides, expectedOverrides.size());
    std::unordered_set<int64_t> seen;
    for(size_t i = 0; i < record.numOverrides; ++i)
    {
        const auto uid = record.capturedUniqueIds[i];
        EXPECT_TRUE(seen.insert(uid).second) << "Duplicate captured override UID " << uid;
        const auto iter = expectedOverrides.find(uid);
        ASSERT_NE(iter, expectedOverrides.end()) << "Unexpected override UID " << uid;
        ASSERT_EQ(iter->second.shape.size(), iter->second.stride.size());
        ASSERT_LE(iter->second.shape.size(), K_MAX_TEST_OVERRIDE_RANK);
        EXPECT_EQ(record.capturedLengths[i], iter->second.shape.size());
        for(size_t axis = 0; axis < iter->second.shape.size(); ++axis)
        {
            EXPECT_EQ(record.capturedShapes[i][axis], iter->second.shape[axis]);
            EXPECT_EQ(record.capturedStrides[i][axis], iter->second.stride[axis]);
        }
    }
}

} // namespace

class IntegrationOverrideExecutePluginDispatch : public IntegrationOverrideExecuteBase
{
};

TEST_F(IntegrationOverrideExecutePluginDispatch,
       HostLoadMakesLookupAvailableAndDispatchesLegacyEntry)
{
    loadImplementingOnly();
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);
    auto graph = createSimplePointwiseGraph(
        "PluginLookup_NoLoadMissThenHostLoad", dims, /*overrideShapeEnabled=*/false);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    resetOverrideImplementingRecord();
    auto result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* afterLoad = getOverrideImplementingRecord();
    ASSERT_NE(afterLoad, nullptr);
    EXPECT_EQ(afterLoad->whichEntry, TestPluginExecuteEntry::OP_GRAPH);
}

// ----------------------------------------------------------------------------
// Four-corner matrix: graph flag × plugin capability (RFC §9.4).
//
//   Corner 1 (false × omitting):       legacy graph, legacy plugin → legacy entry
//   Corner 2 (false × implementing):   legacy graph, new plugin    → legacy entry
//   Corner 3 (true  × omitting):       new graph, legacy plugin    → no applicable engine
//   Corner 4 (true  × implementing):   new graph, new plugin       → override entry
// ----------------------------------------------------------------------------

class IntegrationOverrideExecuteFourCorner : public IntegrationOverrideExecuteBase
{
};

/// Corner 1: graph without `is_override_shape_enabled`, override-omitting plugin
/// loaded. Dispatch must use `hipdnnEnginePluginExecuteOpGraph` (the legacy
/// entry). Verifies binary compatibility for the "no new feature anywhere"
/// case.
TEST_F(IntegrationOverrideExecuteFourCorner, LegacyGraphLegacyPluginUsesLegacyEntry)
{
    loadOmittingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);

    auto graph = createSimplePointwiseGraph(
        "FourCorner_LegacyGraphLegacyPlugin", dims, /*overrideShapeEnabled=*/false);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    auto result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getRecordForOwnedPlugin(
        hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath(), "OverrideOmitting");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH)
        << "Corner 1 must dispatch through the legacy executeOpGraph entry.";
    EXPECT_EQ(record->numOverrides, 0U) << "Legacy entry must receive no override metadata.";
}

/// Corner 2: graph without `is_override_shape_enabled`, the
/// override-implementing plugin is loaded. The host must still pick the
/// legacy entry because the override entry is only for graphs that opted in
/// at build time.
TEST_F(IntegrationOverrideExecuteFourCorner, LegacyGraphImplementingPluginUsesLegacyEntry)
{
    loadImplementingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);

    auto graph = createSimplePointwiseGraph(
        "FourCorner_LegacyGraphImplementingPlugin", dims, /*overrideShapeEnabled=*/false);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    auto result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getRecordForOwnedPlugin(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH)
        << "Even when the override entry is available, a graph that did not "
           "opt in must use the legacy executeOpGraph entry.";
}

/// Corner 3: graph with `is_override_shape_enabled=true`, only the
/// override-omitting plugin loaded (reports `apiVersionWithoutTweak()` =
/// `"1.0.0"`). The version filter must exclude it from the applicability set,
/// so plan creation reports "no applicable engine" before any execute call.
TEST_F(IntegrationOverrideExecuteFourCorner, OverrideGraphOmittingPluginNoApplicableEngine)
{
    loadOmittingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createSimplePointwiseGraph(
        "FourCorner_OverrideGraphOmittingPlugin", dims, /*overrideShapeEnabled=*/true);

    // Validation succeeds (it is structural, not engine-aware).
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Engine selection must fail downstream because the only loaded plugin
    // reports version "1.0.0" and is filtered out for override-flag graphs.
    // The exact stage that surfaces the failure is implementation-defined;
    // what matters is that some stage rejects without invoking an execute entry.
    const auto plansResult = graph->create_execution_plans();
    if(plansResult.code == ErrorCode::OK)
    {
        const auto supportResult = graph->check_support();
        EXPECT_NE(supportResult.code, ErrorCode::OK)
            << "Corner 3 must fail engine selection when no plugin meets the "
               "override-version floor.";
    }
    else
    {
        EXPECT_NE(plansResult.code, ErrorCode::OK);
    }

    // No execute entry should have been touched.
    const auto* record = getRecordForOwnedPlugin(
        hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath(), "OverrideOmitting");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE)
        << "Corner 3 must not invoke any execute entry.";
}

/// Corner 4: graph with `is_override_shape_enabled=true`, override-implementing
/// plugin loaded. Override execute entry is invoked with the supplied per-UID
/// shapes/strides. Workspace pointer is forwarded as-is.
TEST_F(IntegrationOverrideExecuteFourCorner, OverrideGraphImplementingPluginUsesOverrideEntry)
{
    loadImplementingOnly();

    const std::vector<int64_t> declaredDims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(declaredDims);

    auto graph = createSimplePointwiseGraph(
        "FourCorner_OverrideGraphImplementingPlugin", declaredDims, /*overrideShapeEnabled=*/true);
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

    int workspaceStorage = 0;
    void* workspace = &workspaceStorage;
    auto result = graph->execute(
        _handle, variantPack, workspace, overrideUids, overrideShapes, overrideStrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getRecordForOwnedPlugin(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES)
        << "Corner 4: override-flag graph + override-implementing plugin must "
           "dispatch through the override entry.";
    EXPECT_EQ(record->workspace, workspace);
    expectCapturedOverrides(*record, overrideUids, overrideShapes, overrideStrides);
}

TEST_F(IntegrationOverrideExecuteFourCorner,
       OverrideGraphWithMultipleOverridePluginsDispatchesOverrideEntry)
{
    loadImplementingAndSecondOverride();

    const std::vector<int64_t> declaredDims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(declaredDims);

    auto graph = createSimplePointwiseGraph(
        "FourCorner_TwoOverridePlugins", declaredDims, /*overrideShapeEnabled=*/true);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const std::vector<int64_t> overrideUids = {1};
    const std::vector<std::vector<int64_t>> overrideShapes = {{1, 3, 2, 2}};
    const std::vector<std::vector<int64_t>> overrideStrides = {{12, 4, 2, 1}};

    auto result = graph->execute(
        _handle, variantPack, nullptr, overrideUids, overrideShapes, overrideStrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* implementingRecord = getOverrideImplementingRecord();
    const auto* secondRecord = getSecondOverrideRecord();
    ASSERT_NE(implementingRecord, nullptr);
    ASSERT_NE(secondRecord, nullptr);

    const bool implementingDispatched
        = implementingRecord->whichEntry == TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES;
    const bool secondDispatched
        = secondRecord->whichEntry == TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES;
    EXPECT_NE(implementingDispatched, secondDispatched)
        << "Exactly one selected override-capable plugin should execute the graph.";

    if(implementingDispatched)
    {
        expectCapturedOverrides(*implementingRecord, overrideUids, overrideShapes, overrideStrides);
    }
    if(secondDispatched)
    {
        expectCapturedOverrides(*secondRecord, overrideUids, overrideShapes, overrideStrides);
    }
}

// ----------------------------------------------------------------------------
// No-overrides short-circuit and map vs parallel-array equivalence.
// ----------------------------------------------------------------------------

class IntegrationOverrideExecuteShortCircuit : public IntegrationOverrideExecuteBase
{
};

/// Empty map + flag set. The map overload must lower to empty parallel arrays,
/// hit the no-overrides short-circuit, and dispatch through the legacy
/// executeOpGraph entry.
TEST_F(IntegrationOverrideExecuteShortCircuit, EmptyMapDispatchesLegacyEntry)
{
    loadBothFakes();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);

    auto graph
        = createSimplePointwiseGraph("ShortCircuit_EmptyMap", dims, /*overrideShapeEnabled=*/true);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const std::unordered_map<int64_t, OverrideEntry> emptyOverrides;
    auto result = graph->execute(_handle, variantPack, nullptr, emptyOverrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // With `overrideShapeEnabled=true`, the omitting plugin is filtered out
    // by the host's version check, so dispatch lands on the
    // override-implementing fake even for the no-overrides short-circuit.
    const auto* record = getRecordForOwnedPlugin(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH)
        << "Empty-map override-execute must short-circuit to the legacy entry "
           "when no override metadata is supplied.";
    EXPECT_EQ(record->numOverrides, 0U);
}

/// Empty parallel arrays + flag set: same short-circuit as the map overload.
TEST_F(IntegrationOverrideExecuteShortCircuit, EmptyParallelArraysDispatchesLegacyEntry)
{
    loadBothFakes();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);

    auto graph = createSimplePointwiseGraph(
        "ShortCircuit_EmptyParallel", dims, /*overrideShapeEnabled=*/true);
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
    // omitting plugin is filtered out under `overrideShapeEnabled=true`.
    const auto* record = getRecordForOwnedPlugin(
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
        "Equivalence_MapVsArray", declaredDims, /*overrideShapeEnabled=*/true);
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
    const auto* recordAfterArray = getRecordForOwnedPlugin(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(recordAfterArray, nullptr);
    EXPECT_EQ(recordAfterArray->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES);
    expectCapturedOverrides(*recordAfterArray, {1, 2}, {shape, shape}, {stride, stride});

    // Reset between calls so we observe just the second invocation's record.
    resetRecordForOwnedPlugin(hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
                              "OverrideImplementing");

    // Second call: map form with identical content.
    {
        std::unordered_map<int64_t, OverrideEntry> overrides;
        overrides[1] = OverrideEntry{shape, stride};
        overrides[2] = OverrideEntry{shape, stride};
        auto result = graph->execute(_handle, variantPack, nullptr, overrides);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    }
    const auto* recordAfterMap = getRecordForOwnedPlugin(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(recordAfterMap, nullptr);
    EXPECT_EQ(recordAfterMap->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES);
    expectCapturedOverridesByUid(
        *recordAfterMap, {{1, OverrideEntry{shape, stride}}, {2, OverrideEntry{shape, stride}}});
}

TEST_F(IntegrationOverrideExecuteEquivalence, SinglePlanExecutesMultipleOverridePayloads)
{
    loadImplementingOnly();

    const std::vector<int64_t> declaredDims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(declaredDims);

    auto graph = createSimplePointwiseGraph(
        "Equivalence_ReusedPlanDifferentPayloads", declaredDims, /*overrideShapeEnabled=*/true);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    struct Payload
    {
        std::vector<int64_t> uids;
        std::vector<std::vector<int64_t>> shapes;
        std::vector<std::vector<int64_t>> strides;
    };

    const std::vector<Payload> payloads = {
        Payload{{1}, {{1, 3, 4, 4}}, {{48, 16, 4, 1}}},
        Payload{{2}, {{1, 3, 2, 2}}, {{12, 4, 2, 1}}},
        Payload{{1, 2}, {{1, 3, 1, 4}, {1, 3, 3, 2}}, {{48, 16, 4, 1}, {24, 8, 2, 1}}},
    };

    for(size_t i = 0; i < payloads.size(); ++i)
    {
        SCOPED_TRACE("payload[" + std::to_string(i) + "]");
        const auto& payload = payloads[i];
        resetOverrideImplementingRecord();

        auto result = graph->execute(
            _handle, variantPack, nullptr, payload.uids, payload.shapes, payload.strides);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const auto* record = getOverrideImplementingRecord();
        ASSERT_NE(record, nullptr);
        EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES);
        expectCapturedOverrides(*record, payload.uids, payload.shapes, payload.strides);
    }
}

// ----------------------------------------------------------------------------
// Execution-plan validity guard for the override-execute parallel-array form.
//
// The non-override `Graph::execute()` overload guards against a missing
// compiled plan at the top of its body. The parallel-array override overload
// must perform the SAME guard so a user calling override-execute before
// `build_plans()` (or `from_compiled_plan_binary()`) gets a clean
// INVALID_VALUE diagnostic instead of dereferencing a null/invalid plan
// descriptor.
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
        "PlanGuard_ArrayBeforeBuild", dims, /*overrideShapeEnabled=*/true);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const std::vector<int64_t> overrideUids = {1, 2};
    const std::vector<std::vector<int64_t>> overrideShapes = {dims, dims};
    const std::vector<std::vector<int64_t>> overrideStrides
        = {{int64_t{3} * 4 * 4, int64_t{4} * 4, 4, 1}, {int64_t{3} * 4 * 4, int64_t{4} * 4, 4, 1}};

    auto result = graph->execute(
        _handle, variantPack, nullptr, overrideUids, overrideShapes, overrideStrides);

    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "Override-execute before build_plans must reject with INVALID_VALUE: " << result.err_msg;
    // Mirrors the wording of the non-override guard (Graph.hpp ~line 1818).
    EXPECT_NE(result.err_msg.find("no compiled execution plan"), std::string::npos)
        << "Diagnostic must surface the missing plan; got: " << result.err_msg;

    // The override-implementing fake must NOT have been touched: the guard
    // runs before any backend interaction.
    const auto* record = getRecordForOwnedPlugin(
        hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
        "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE)
        << "Plan-guard rejection must precede any backend dispatch.";
}

// ----------------------------------------------------------------------------
// Compiled-plan-only execution.
// ----------------------------------------------------------------------------

class IntegrationOverrideExecutePlanOnly : public IntegrationOverrideExecuteBase
{
};

struct RestoredPlanRejectCase
{
    const char* name;
    std::vector<int64_t> overrideUids;
    std::vector<std::vector<int64_t>> overrideShapes;
    std::vector<std::vector<int64_t>> overrideStrides;
};

class IntegrationOverrideExecutePlanOnlyReject
    : public IntegrationOverrideExecuteBase,
      public ::testing::WithParamInterface<RestoredPlanRejectCase>
{
};

TEST_F(IntegrationOverrideExecutePlanOnly, RestoredCompiledPlanNormalExecuteUsesLegacyEntry)
{
    loadImplementingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);
    std::shared_ptr<Graph> graph;
    createPlanOnlyGraph(_handle, dims, graph);
    ASSERT_NE(graph, nullptr);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    resetOverrideImplementingRecord();
    auto result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getOverrideImplementingRecord();
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH);
    EXPECT_EQ(record->numOverrides, 0U);
}

TEST_F(IntegrationOverrideExecutePlanOnly, RestoredCompiledPlanOverrideExecuteForwardsPayload)
{
    loadImplementingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);
    std::shared_ptr<Graph> graph;
    createPlanOnlyGraph(_handle, dims, graph);
    ASSERT_NE(graph, nullptr);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const std::vector<int64_t> overrideUids = {1, 2};
    const std::vector<int64_t> shape = {1, 3, 2, 2};
    const std::vector<int64_t> stride = {12, 4, 2, 1};
    const std::vector<std::vector<int64_t>> overrideShapes = {shape, shape};
    const std::vector<std::vector<int64_t>> overrideStrides = {stride, stride};

    resetOverrideImplementingRecord();
    auto result = graph->execute(
        _handle, variantPack, nullptr, overrideUids, overrideShapes, overrideStrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getOverrideImplementingRecord();
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES);
    expectCapturedOverrides(*record, overrideUids, overrideShapes, overrideStrides);
}

TEST_F(IntegrationOverrideExecutePlanOnly, RestoredCompiledPlanForwardsDifferentOverrideRanks)
{
    loadImplementingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);
    std::shared_ptr<Graph> graph;
    createPlanOnlyGraph(_handle, dims, graph);
    ASSERT_NE(graph, nullptr);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();
    variantPack[99] = bundle.xTensor.memory().deviceData();

    const std::vector<int64_t> overrideUids = {1, 99};
    const std::vector<std::vector<int64_t>> overrideShapes = {{1, 3, 2, 2}, {2, 3, 4}};
    const std::vector<std::vector<int64_t>> overrideStrides = {{12, 4, 2, 1}, {12, 4, 1}};

    resetOverrideImplementingRecord();
    auto result = graph->execute(
        _handle, variantPack, nullptr, overrideUids, overrideShapes, overrideStrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getOverrideImplementingRecord();
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES);
    expectCapturedOverrides(*record, overrideUids, overrideShapes, overrideStrides);
}

TEST_F(IntegrationOverrideExecutePlanOnly, RestoredCompiledPlanUnknownGraphUidReachesProvider)
{
    loadImplementingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);
    std::shared_ptr<Graph> graph;
    createPlanOnlyGraph(_handle, dims, graph);
    ASSERT_NE(graph, nullptr);

    constexpr int64_t UNKNOWN_GRAPH_UID = 99;
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();
    variantPack[UNKNOWN_GRAPH_UID] = bundle.xTensor.memory().deviceData();

    const std::vector<int64_t> overrideUids = {UNKNOWN_GRAPH_UID};
    const std::vector<std::vector<int64_t>> overrideShapes = {{1, 1, 1, 1}};
    const std::vector<std::vector<int64_t>> overrideStrides = {{1, 1, 1, 1}};

    resetOverrideImplementingRecord();
    auto result = graph->execute(
        _handle, variantPack, nullptr, overrideUids, overrideShapes, overrideStrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getOverrideImplementingRecord();
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES);
    expectCapturedOverrides(*record, overrideUids, overrideShapes, overrideStrides);
}

TEST_P(IntegrationOverrideExecutePlanOnlyReject, RestoredCompiledPlanRejectsInvalidOverrides)
{
    loadImplementingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);
    std::shared_ptr<Graph> graph;
    createPlanOnlyGraph(_handle, dims, graph);
    ASSERT_NE(graph, nullptr);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const auto& testCase = GetParam();

    resetOverrideImplementingRecord();
    auto result = graph->execute(_handle,
                                 variantPack,
                                 nullptr,
                                 testCase.overrideUids,
                                 testCase.overrideShapes,
                                 testCase.overrideStrides);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;

    const auto* record = getOverrideImplementingRecord();
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE);
}

INSTANTIATE_TEST_SUITE_P(
    RestoredCompiledPlan,
    IntegrationOverrideExecutePlanOnlyReject,
    ::testing::Values(
        RestoredPlanRejectCase{"OuterArraySizeMismatch",
                               {1, 2},
                               {{1, 3, 4, 4}},
                               {packedNchwStrides({1, 3, 4, 4}), packedNchwStrides({1, 3, 4, 4})}},
        RestoredPlanRejectCase{
            "ShapeStrideRankMismatch", {1}, {{1, 3, 4}}, {packedNchwStrides({1, 3, 4, 4})}},
        RestoredPlanRejectCase{"DuplicateUids",
                               {1, 1},
                               {{1, 3, 4, 4}, {1, 3, 4, 4}},
                               {packedNchwStrides({1, 3, 4, 4}), packedNchwStrides({1, 3, 4, 4})}},
        RestoredPlanRejectCase{
            "NonPositiveShape", {1}, {{1, 3, 0, 4}}, {packedNchwStrides({1, 3, 4, 4})}},
        RestoredPlanRejectCase{"ZeroRankOverride", {1}, {{}}, {{}}},
        RestoredPlanRejectCase{"NonPositiveStride", {1}, {{1, 3, 4, 4}}, {{48, 16, 0, 1}}}),
    [](const auto& info) { return std::string(info.param.name); });

TEST_F(IntegrationOverrideExecutePlanOnly, RestoredCompiledPlanEmptyOverridesFallThrough)
{
    loadImplementingOnly();

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    SimpleTensorBundle<float> bundle(dims);
    std::shared_ptr<Graph> graph;
    createPlanOnlyGraph(_handle, dims, graph);
    ASSERT_NE(graph, nullptr);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = bundle.xTensor.memory().deviceData();
    variantPack[2] = bundle.yTensor.memory().deviceData();

    const std::vector<int64_t> emptyUids;
    const std::vector<std::vector<int64_t>> emptyShapes;
    const std::vector<std::vector<int64_t>> emptyStrides;

    resetOverrideImplementingRecord();
    auto result
        = graph->execute(_handle, variantPack, nullptr, emptyUids, emptyShapes, emptyStrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* recordAfterArrays = getOverrideImplementingRecord();
    ASSERT_NE(recordAfterArrays, nullptr);
    EXPECT_EQ(recordAfterArrays->whichEntry, TestPluginExecuteEntry::OP_GRAPH);
    EXPECT_EQ(recordAfterArrays->numOverrides, 0U);

    resetOverrideImplementingRecord();
    const std::unordered_map<int64_t, OverrideEntry> emptyOverrides;
    result = graph->execute(_handle, variantPack, nullptr, emptyOverrides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* recordAfterMap = getOverrideImplementingRecord();
    ASSERT_NE(recordAfterMap, nullptr);
    EXPECT_EQ(recordAfterMap->whichEntry, TestPluginExecuteEntry::OP_GRAPH);
    EXPECT_EQ(recordAfterMap->numOverrides, 0U);
}

#endif // HIPDNN_ENABLE_SDPA
