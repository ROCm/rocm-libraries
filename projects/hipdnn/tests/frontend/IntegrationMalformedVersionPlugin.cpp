// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Frontend integration test for RFC 0008 post-review fix #1
// (backend dispatch hardening). A plugin that reports an unparseable
// `apiVersion` string must be rejected at load time by the host's
// `EnginePluginManager::validateBeforeAdding` (which forces population of
// the lazy `PluginBase::parsedApiVersion()` cache). The bad plugin is
// caught by the existing `tryCatch` wrapper inside `loadPluginFromFile`,
// logged, and skipped; well-formed plugins loaded alongside it continue
// to function and serve dispatch.
//
// Without the load-time guard, the per-call `Version{plugin->apiVersion()}`
// in `EnginePluginResourceManager::getApplicableEngineIds` would throw
// `std::invalid_argument` on every graph execute — a single bad plugin
// could brick every dispatch in the host.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "OverrideTestUtils.hpp"
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

/// Build a minimal pointwise (RELU) graph for execute-path coverage. The
/// graph is non-override (legacy) so it admits every plugin reporting at
/// least the post-PR1 baseline `"1.0.0"` — the malformed plugin is the
/// only one that should be filtered out, and only at load time. Thin
/// wrapper over the shared `buildPointwiseReluGraph` helper.
std::shared_ptr<Graph> createSimpleReluGraph(const std::string& graphName,
                                             const std::vector<int64_t>& dims)
{
    return hipdnn_tests::override_test_utils::buildPointwiseReluGraph(
        graphName, dims, /*strides=*/{}, /*dynamicShapeEnabled=*/false);
}

class IntegrationMalformedVersionPlugin : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }
    }

    hipdnnHandle_t _handle = nullptr;
};

} // namespace

/// Dispatch-hardening fix #1: loading a malformed-version plugin alongside
/// a well-formed plugin must not crash hipdnnCreate, must not throw out of
/// graph execute, and the well-formed plugin must remain available to
/// serve the graph. Without the fix, `Version{...}` would be re-parsed in
/// `getApplicableEngineIds` on every execute and throw `std::invalid_argument`.
TEST_F(IntegrationMalformedVersionPlugin, BadPluginLoadedAlongsideGoodPluginDoesNotCrashDispatch)
{
    const std::array<const char*, 2> paths
        = {hipdnn_tests::plugin_constants::testMalformedVersionPluginPath().c_str(),
           hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};

    // The host must absorb the malformed plugin's load-time rejection inside
    // its `tryCatch` wrapper; `hipdnnCreate` should still succeed.
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    // Compile a non-override graph — the well-formed test_good_plugin
    // reports "1.0.0" so it satisfies the baseline requirement.
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    Tensor<float> xTensor(dims);
    Tensor<float> yTensor(dims);
    xTensor.fillWithValue(1.0F);
    yTensor.fillWithValue(0.0F);

    auto graph = createSimpleReluGraph("MalformedVersion_GoodAlongside", dims);
    hipdnn_tests::override_test_utils::compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = xTensor.memory().deviceData();
    variantPack[2] = yTensor.memory().deviceData();

    // Critical assertion: dispatch does NOT throw. With the per-call
    // `Version{plugin->apiVersion()}` parse, the malformed plugin would
    // trip an `std::invalid_argument` here even though it never produced
    // any applicable engine.
    auto result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

/// Defensive: even when the malformed plugin is the ONLY one loaded, the
/// host must still come up cleanly. Engine selection then has nothing to
/// offer, but that is a normal failure mode (no applicable plugin) — not a
/// crash.
TEST_F(IntegrationMalformedVersionPlugin, BadPluginLoadedAloneDoesNotCrashCreate)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testMalformedVersionPluginPath().c_str()};

    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
    // The malformed plugin is rejected at load time by `validateBeforeAdding`;
    // the rejection is caught by `loadPluginFromFile`'s `tryCatch` so the
    // backend handle still constructs successfully even with no usable plugin.
    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
}

/// RFC 0008 plan T-missing #2: the version-zero fake reports a
/// parseable but too-low API version ("0.0.0"). Distinct from the malformed
/// fake — `parsedApiVersion()` succeeds and caches the parsed value, but the
/// version comparison against the post-PR1 baseline rejects it. Loading it
/// alongside a well-formed plugin must not crash dispatch and must leave the
/// well-formed plugin available to serve the graph (parsed-but-too-low
/// rejection path is a quiet filter, not a fault).
TEST_F(IntegrationMalformedVersionPlugin,
       VersionZeroPluginLoadedAlongsideGoodPluginDoesNotCrashDispatch)
{
    const std::array<const char*, 2> paths
        = {hipdnn_tests::plugin_constants::testVersionZeroPluginPath().c_str(),
           hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};

    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    Tensor<float> xTensor(dims);
    Tensor<float> yTensor(dims);
    xTensor.fillWithValue(1.0F);
    yTensor.fillWithValue(0.0F);

    auto graph = createSimpleReluGraph("VersionZero_GoodAlongside", dims);
    hipdnn_tests::override_test_utils::compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = xTensor.memory().deviceData();
    variantPack[2] = yTensor.memory().deviceData();

    // Dispatch must succeed via the well-formed plugin; the version-zero
    // plugin is filtered out at version-check time and contributes no
    // applicable engines. Distinct code path from the malformed plugin
    // (which throws inside `parsedApiVersion()`); this exercises the
    // parsed-but-too-low rejection branch.
    auto result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

/// Defensive companion: when only the version-zero plugin is loaded, the
/// host still constructs cleanly. The plugin is admitted to load (parse
/// succeeds) but produces no applicable engines because the version
/// baseline filters it out.
TEST_F(IntegrationMalformedVersionPlugin, VersionZeroPluginLoadedAloneDoesNotCrashCreate)
{
    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testVersionZeroPluginPath().c_str()};

    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
}
