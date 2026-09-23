// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Frontend integration tests for overridable tensor shape validation.
// Detailed rule coverage lives in frontend unit tests for the detail helper.
// This file checks that invalid override payloads reject before backend
// dispatch and that graph flag round-trips survive backend pack/unpack paths.
//
// These integration tests use backend graph lowering and plugin-backed handles;
// GPU-less environments skip them through the common test utility.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "OverrideTestUtils.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <test_plugins/TestPluginCommon.hpp>
#include <test_plugins/TestPluginConstants.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef HIPDNN_ENABLE_SDPA

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

using ScopedHipdnnHandle
    = std::unique_ptr<std::remove_pointer_t<hipdnnHandle_t>, decltype(&hipdnnDestroy)>;

/// Build a minimal pointwise (RELU) graph with the supplied declared strides
/// and `set_override_shape_enabled(true)`. This is the standard "valid graph
/// to override against" used by every validation test below. Thin wrapper
/// over the shared `buildPointwiseReluGraph` helper.
std::shared_ptr<Graph> createOverridableGraph(const std::string& graphName,
                                              const std::vector<int64_t>& declaredDims,
                                              const std::vector<int64_t>& declaredStrides)
{
    return hipdnn_tests::override_test_utils::buildPointwiseReluGraph(
        graphName, declaredDims, declaredStrides, /*overrideShapeEnabled=*/true);
}

// Bring the shared `compileGraph(graph, handle)` helper into the file's
// anonymous namespace so existing call sites resolve unqualified.
using hipdnn_tests::override_test_utils::compileGraph;

/// Common fixture: load the override-implementing fake (so plan creation
/// succeeds for an override-flag graph) and reset TLS state per test.
class IntegrationOverrideValidationBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
        // Reset the TLS LastCallRecord across every override-execute fake
        // plugin that may be loaded. Plugins not loaded in this test process
        // are silently skipped by the no-load symbol lookup.
        hipdnn_tests::override_test_utils::resetAllOverrideFakePluginRecords();

        _implementingPlugin = std::make_unique<ScopedTestPluginLibrary>(
            hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath());

        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath().c_str()};

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
        _implementingPlugin.reset();
    }

    /// Convenience: build a 4-D tensor with NCHW packed strides and a tiny
    /// device-side allocation so `variantPack` pointer values are valid.
    static std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
    {
        return hipdnn_data_sdk::utilities::generateStrides(dims);
    }

    hipdnnHandle_t _handle = nullptr;
    std::unique_ptr<ScopedTestPluginLibrary> _implementingPlugin;
};

/// Helper: invoke the parallel-array overload and assert validation rejects
/// with `ErrorCode::INVALID_VALUE`. The override-implementing fake plugin
/// must NOT have been called (fixture loads only that plugin). When
/// `expectedRuleSubstring` is non-empty, the returned error message must
/// contain it.
void expectArrayRejected(std::shared_ptr<Graph>& graph,
                         [[maybe_unused]] hipdnnHandle_t handle,
                         const ScopedTestPluginLibrary& implementingPlugin,
                         std::unordered_map<int64_t, void*>& variantPack,
                         const std::vector<int64_t>& uids,
                         const std::vector<std::vector<int64_t>>& shapes,
                         const std::vector<std::vector<int64_t>>& strides,
                         const std::string& expectedRuleSubstring = "")
{
    resetLastCallRecord(implementingPlugin, "OverrideImplementing");
    auto result = graph->execute(handle, variantPack, nullptr, uids, shapes, strides);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "Parallel-array overload should have rejected: " << result.err_msg;
    if(!expectedRuleSubstring.empty())
    {
        EXPECT_NE(result.err_msg.find(expectedRuleSubstring), std::string::npos)
            << "Rejection message must identify the failing rule explicitly: " << result.err_msg;
    }
    const auto* record = getLastCallRecord(implementingPlugin, "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE)
        << "Validation must reject before any backend call.";
}

/// Helper: invoke the map-keyed overload with the same logical payload and
/// assert it rejects with the same `ErrorCode::INVALID_VALUE`.
void expectMapRejected(std::shared_ptr<Graph>& graph,
                       [[maybe_unused]] hipdnnHandle_t handle,
                       const ScopedTestPluginLibrary& implementingPlugin,
                       std::unordered_map<int64_t, void*>& variantPack,
                       const std::unordered_map<int64_t, OverrideEntry>& overrides)
{
    resetLastCallRecord(implementingPlugin, "OverrideImplementing");
    auto result = graph->execute(handle, variantPack, nullptr, overrides);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "Map overload should have rejected: " << result.err_msg;
    const auto* record = getLastCallRecord(implementingPlugin, "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE)
        << "Validation must reject before any backend call.";
}

} // namespace

// ============================================================================
// Override validation integration smoke tests. Detailed rule coverage lives in
// frontend unit tests for detail/GraphOverrideValidation.hpp.
// ============================================================================

class IntegrationOverrideValidation : public IntegrationOverrideValidationBase
{
};

TEST_F(IntegrationOverrideValidation, ArrayFormInvalidOverrideRejectedBeforeBackend)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Array_InvalidOverride", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    const std::vector<int64_t> uids = {999};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    const std::vector<std::vector<int64_t>> strides = {packedStrides(dims)};
    expectArrayRejected(graph, _handle, *_implementingPlugin, variantPack, uids, shapes, strides);
}

TEST_F(IntegrationOverrideValidation, MapFormInvalidOverrideRejectedBeforeBackend)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Map_InvalidOverride", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    std::unordered_map<int64_t, OverrideEntry> overrides;
    overrides[999] = OverrideEntry{dims, packedStrides(dims)};
    expectMapRejected(graph, _handle, *_implementingPlugin, variantPack, overrides);
}

TEST_F(IntegrationOverrideValidation, ValidOverrideReachesPlugin)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("ValidOverride", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    const std::vector<std::vector<int64_t>> strides = {packedStrides(dims)};

    resetLastCallRecord(*_implementingPlugin, "OverrideImplementing");
    auto result = graph->execute(_handle, variantPack, nullptr, uids, shapes, strides);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const auto* record = getLastCallRecord(*_implementingPlugin, "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::OP_GRAPH_WITH_OVERRIDES);
}

TEST_F(IntegrationOverrideValidation, OverrideUidMissingFromVariantPackRejectedBeforeDispatch)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("MissingVariantPackUid", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;

    const std::vector<int64_t> uids = {2};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    const std::vector<std::vector<int64_t>> strides = {packedStrides(dims)};

    resetLastCallRecord(*_implementingPlugin, "OverrideImplementing");
    auto result = graph->execute(_handle, variantPack, nullptr, uids, shapes, strides);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "Override UID 2 is a graph tensor, but it is absent from the variant pack.";
    EXPECT_NE(result.err_msg.find("variant pack"), std::string::npos);

    const auto* record = getLastCallRecord(*_implementingPlugin, "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE);
}

// ============================================================================
// Reject overrides when the graph did not opt in.
// ============================================================================

class IntegrationOverrideValidationFlagAbsent : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        ASSERT_EQ(hipInit(0), hipSuccess);
        int deviceId = 0;
        ASSERT_EQ(hipGetDevice(&deviceId), hipSuccess);
        hipdnn_tests::override_test_utils::resetAllOverrideFakePluginRecords();

        _implementingPlugin = std::make_unique<ScopedTestPluginLibrary>(
            hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath());

        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath().c_str()};
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
        _implementingPlugin.reset();
    }

    hipdnnHandle_t _handle = nullptr;
    std::unique_ptr<ScopedTestPluginLibrary> _implementingPlugin;
};

/// RFC §7.1: calling the override-execute overload on a graph that did not
/// `set_override_shape_enabled(true)` must reject with `INVALID_VALUE` and
/// must NOT invoke the backend.
TEST_F(IntegrationOverrideValidationFlagAbsent, ArrayFormRejectedWhenFlagAbsent)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};

    // Build a graph WITHOUT set_override_shape_enabled.
    auto graph = hipdnn_tests::override_test_utils::buildPointwiseReluGraph(
        "FlagAbsent_Array", dims, /*strides=*/{}, /*overrideShapeEnabled=*/false);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    const std::vector<std::vector<int64_t>> strides
        = {hipdnn_data_sdk::utilities::generateStrides(dims)};

    resetLastCallRecord(*_implementingPlugin, "OverrideImplementing");
    auto result = graph->execute(_handle, variantPack, nullptr, uids, shapes, strides);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "Override-execute on a graph that did not opt in must reject "
           "(RFC §7.1): "
        << result.err_msg;
    const auto* record = getLastCallRecord(*_implementingPlugin, "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE);
}

TEST_F(IntegrationOverrideValidationFlagAbsent, PostBuildToggleDoesNotEnableExistingPlan)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};

    auto graph = hipdnn_tests::override_test_utils::buildPointwiseReluGraph(
        "FlagAbsent_PostBuildToggle", dims, /*strides=*/{}, /*overrideShapeEnabled=*/false);
    compileGraph(graph, _handle);
    graph->set_override_shape_enabled(true);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    const std::vector<std::vector<int64_t>> strides
        = {hipdnn_data_sdk::utilities::generateStrides(dims)};

    resetLastCallRecord(*_implementingPlugin, "OverrideImplementing");
    auto result = graph->execute(_handle, variantPack, nullptr, uids, shapes, strides);
    EXPECT_NE(result.code, ErrorCode::OK)
        << "Toggling override shape support after build must not make the existing plan "
           "override-capable.";
    EXPECT_NE(result.err_msg.find("override"), std::string::npos);

    const auto* record = getLastCallRecord(*_implementingPlugin, "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE);
}

TEST_F(IntegrationOverrideValidationFlagAbsent, MapFormRejectedWhenFlagAbsent)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};

    auto graph = hipdnn_tests::override_test_utils::buildPointwiseReluGraph(
        "FlagAbsent_Map", dims, /*strides=*/{}, /*overrideShapeEnabled=*/false);
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    std::unordered_map<int64_t, OverrideEntry> overrides;
    overrides[1] = OverrideEntry{dims, hipdnn_data_sdk::utilities::generateStrides(dims)};

    resetLastCallRecord(*_implementingPlugin, "OverrideImplementing");
    auto result = graph->execute(_handle, variantPack, nullptr, overrides);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;
    const auto* record = getLastCallRecord(*_implementingPlugin, "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE);
}

// ============================================================================
// Frontend pack/unpack round-trip of `is_override_shape_enabled`.
// ============================================================================

/// Build a graph with `set_override_shape_enabled(true)`, serialize,
/// deserialize, and assert the flag survived.
///
/// Uses `to_binary()` which auto-lowers via the backend descriptor, mirroring
/// the existing `IntegrationGraphLifting` round-trip pattern. Deserialization
/// uses no handle so restored graph finalization is not required.
TEST(IntegrationOverrideRoundTrip, OverrideShapeEnabledFlagSurvivesSerialization)
{
    SKIP_IF_NO_DEVICES();

    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
    hipdnnHandle_t rawHandle = nullptr;
    ASSERT_EQ(hipdnnCreate(&rawHandle), HIPDNN_STATUS_SUCCESS);
    const ScopedHipdnnHandle handle(rawHandle, hipdnnDestroy);

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = std::make_shared<Graph>();
    graph->set_name("RoundTrip_OverrideShapeEnabled")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_override_shape_enabled(true);

    EXPECT_TRUE(graph->is_override_shape_enabled())
        << "Setter must round-trip via the in-memory getter immediately.";

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_dim(dims).set_stride({48, 16, 4, 1}).set_data_type(DataType::FLOAT);
    PointwiseAttributes attrs;
    attrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph->pointwise(x, attrs);
    y->set_uid(2)
        .set_dim(dims)
        .set_stride({48, 16, 4, 1})
        .set_data_type(DataType::FLOAT)
        .set_output(true);

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    // Lower to the backend operation graph so `to_binary()` exercises
    // `assembleGraphDescriptor()` — without this step the wire round-trip is
    // not actually performed.
    result = graph->build_operation_graph(handle.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto [data, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto restored = std::make_shared<Graph>();
    auto deserResult = restored->deserialize(nullptr, data);
    ASSERT_EQ(deserResult.code, ErrorCode::OK) << deserResult.err_msg;

    EXPECT_TRUE(restored->is_override_shape_enabled())
        << "is_override_shape_enabled must survive a serialize/deserialize round-trip.";
}

TEST(IntegrationOverrideRoundTrip, OverrideShapeEnabledFlagSurvivesJsonRoundTrip)
{
    SKIP_IF_NO_DEVICES();

    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
    hipdnnHandle_t rawHandle = nullptr;
    ASSERT_EQ(hipdnnCreate(&rawHandle), HIPDNN_STATUS_SUCCESS);
    const ScopedHipdnnHandle handle(rawHandle, hipdnnDestroy);

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = hipdnn_tests::override_test_utils::buildPointwiseReluGraph(
        "RoundTrip_OverrideShapeEnabledJson",
        dims,
        /*strides=*/{},
        /*overrideShapeEnabled=*/true);
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->build_operation_graph(handle.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto [json, serErr] = graph->to_json();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto restored = std::make_shared<Graph>();
    auto deserResult = restored->from_json(json);
    ASSERT_EQ(deserResult.code, ErrorCode::OK) << deserResult.err_msg;
    EXPECT_TRUE(restored->is_override_shape_enabled())
        << "is_override_shape_enabled must survive a JSON serialize/deserialize round-trip.";
}

/// End-to-end check that a graph with the default-false
/// `is_override_shape_enabled` value survives lowering through
/// `build_operation_graph()` plus a full serialize/deserialize round-trip
/// without flipping to true.
TEST(IntegrationOverrideRoundTrip, OverrideShapeEnabledDefaultFalseSurvivesBuildOperationGraph)
{
    SKIP_IF_NO_DEVICES();

    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
    hipdnnHandle_t rawHandle = nullptr;
    ASSERT_EQ(hipdnnCreate(&rawHandle), HIPDNN_STATUS_SUCCESS);
    const ScopedHipdnnHandle handle(rawHandle, hipdnnDestroy);

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = std::make_shared<Graph>();
    // Intentionally do NOT call set_override_shape_enabled; the default is false.
    graph->set_name("BuildOpGraph_DefaultFalse")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_dim(dims).set_stride({48, 16, 4, 1}).set_data_type(DataType::FLOAT);
    PointwiseAttributes attrs;
    attrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph->pointwise(x, attrs);
    y->set_uid(2)
        .set_dim(dims)
        .set_stride({48, 16, 4, 1})
        .set_data_type(DataType::FLOAT)
        .set_output(true);

    EXPECT_FALSE(graph->is_override_shape_enabled())
        << "Pre-validate: default (unset) must read as false.";

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_FALSE(graph->is_override_shape_enabled())
        << "Post-validate: default-false must not flip to true during validation.";

    // Critical step the existing default-false round-trip test omits: the
    // backend lowering. `assembleGraphDescriptor()` is the path that
    // propagates the flag into the backend graph descriptor; a regression
    // there is invisible without this call.
    result = graph->build_operation_graph(handle.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    EXPECT_FALSE(graph->is_override_shape_enabled())
        << "Post-build_operation_graph: default-false must survive backend "
           "lowering.";

    auto [data, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto restored = std::make_shared<Graph>();
    auto deserResult = restored->deserialize(nullptr, data);
    ASSERT_EQ(deserResult.code, ErrorCode::OK) << deserResult.err_msg;

    EXPECT_FALSE(restored->is_override_shape_enabled())
        << "Default-false must survive build_operation_graph + "
           "serialize/deserialize end-to-end.";
}

/// Round-trip the default (unset) case: a graph that never called
/// `set_override_shape_enabled` must deserialize to `false` via the
/// in-memory getter (matches the wire default for legacy graphs).
TEST(IntegrationOverrideRoundTrip, OverrideShapeEnabledDefaultFalseSurvivesSerialization)
{
    SKIP_IF_NO_DEVICES();

    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
    hipdnnHandle_t rawHandle = nullptr;
    ASSERT_EQ(hipdnnCreate(&rawHandle), HIPDNN_STATUS_SUCCESS);
    const ScopedHipdnnHandle handle(rawHandle, hipdnnDestroy);

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = std::make_shared<Graph>();
    graph->set_name("RoundTrip_DefaultFalse")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_dim(dims).set_stride({48, 16, 4, 1}).set_data_type(DataType::FLOAT);
    PointwiseAttributes attrs;
    attrs.set_mode(PointwiseMode::RELU_FWD);
    auto y = graph->pointwise(x, attrs);
    y->set_uid(2)
        .set_dim(dims)
        .set_stride({48, 16, 4, 1})
        .set_data_type(DataType::FLOAT)
        .set_output(true);

    EXPECT_FALSE(graph->is_override_shape_enabled())
        << "Default (unset) must read as false from the in-memory getter.";

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto [data, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto restored = std::make_shared<Graph>();
    auto deserResult = restored->deserialize(nullptr, data);
    ASSERT_EQ(deserResult.code, ErrorCode::OK) << deserResult.err_msg;

    EXPECT_FALSE(restored->is_override_shape_enabled())
        << "Default-false must survive serialize/deserialize as false.";
}

#endif // HIPDNN_ENABLE_SDPA
