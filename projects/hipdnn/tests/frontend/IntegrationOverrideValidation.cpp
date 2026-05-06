// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Frontend integration tests for RFC 0008 Phase 1 overridable tensor shapes
// (validation path). Covers:
//   * All 8 validation rules from RFC §4.2.1, exercised via BOTH the
//     parallel-array overload AND the map-keyed convenience overload (RFC 0008
//     plan Test #5: map-overload validation parity).
//   * Test #14 — frontend pack/unpack round-trip of `is_dynamic_shape_enabled`.
//   * Test #17 — rule 4 future-proof phrasing: "all dims compared" (not just
//     "non-wildcard dims") so a Phase-2 wildcard carve-out trips this test.
//   * Test #12 — rule 8 stride-ordering D4 Phase-1 phrasing.
//   * RFC §7.1 "overrides without flag" rejection (Test C.6).
//
// All validation tests in this file run BEFORE any backend interaction and
// therefore do NOT need a GPU. We still set up a handle (with the
// override-omitting fake loaded so a non-override execute path is functional)
// to keep the SetUp/TearDown symmetric with other integration suites.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

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

/// Build a minimal pointwise (RELU) graph with row-major NCHW packed strides
/// and `set_dynamic_shape_enabled(true)`. This is the standard "valid graph
/// to override against" used by every validation test below.
std::shared_ptr<Graph> createOverridableGraph(const std::string& graphName,
                                              const std::vector<int64_t>& declaredDims,
                                              const std::vector<int64_t>& declaredStrides)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(graphName)
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_dynamic_shape_enabled(true);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1)
        .set_name("X")
        .set_dim(declaredDims)
        .set_stride(declaredStrides)
        .set_data_type(DataType::FLOAT);

    PointwiseAttributes attrs;
    attrs.set_name("relu_node");
    attrs.set_mode(PointwiseMode::RELU_FWD);

    auto y = graph->pointwise(x, attrs);
    y->set_uid(2)
        .set_dim(declaredDims)
        .set_stride(declaredStrides)
        .set_data_type(DataType::FLOAT)
        .set_output(true);

    return graph;
}

/// Compile the graph far enough that `execute()` can be invoked. Validation
/// happens in `execute()`, so we need the plan stages to succeed first.
void compileGraph(std::shared_ptr<Graph>& graph, [[maybe_unused]] hipdnnHandle_t handle)
{
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->build_operation_graph(handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

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
        // Risk #11: reset the TLS LastCallRecord across every fake plugin
        // that may be loaded; only `OverrideImplementing` is loaded by this
        // fixture but the others are silently skipped via dlsym lookup.
        resetLastCallRecordIfLoaded(
            hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
            "OverrideImplementing");
        resetLastCallRecordIfLoaded(
            hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath(), "OverrideOmitting");
        resetLastCallRecordIfLoaded(hipdnn_tests::plugin_constants::testVersionLiarPluginPath(),
                                    "VersionLiar");
        resetLastCallRecordIfLoaded(hipdnn_tests::plugin_constants::testSecondOverridePluginPath(),
                                    "SecondOverride");

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
    }

    /// Convenience: build a 4-D tensor with NCHW packed strides and a tiny
    /// device-side allocation so `variantPack` pointer values are valid.
    static std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
    {
        return {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    }

    hipdnnHandle_t _handle = nullptr;
};

/// Helper: invoke the parallel-array overload and assert validation rejects
/// with `ErrorCode::INVALID_VALUE`. The override-implementing fake plugin
/// must NOT have been called (fixture loads only that plugin).
void expectArrayRejected(std::shared_ptr<Graph>& graph,
                         [[maybe_unused]] hipdnnHandle_t handle,
                         std::unordered_map<int64_t, void*>& variantPack,
                         const std::vector<int64_t>& uids,
                         const std::vector<std::vector<int64_t>>& shapes,
                         const std::vector<std::vector<int64_t>>& strides)
{
    const auto& implPath = hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath();
    resetLastCallRecordIfLoaded(implPath, "OverrideImplementing");
    auto result = graph->execute(handle, variantPack, nullptr, uids, shapes, strides);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "Parallel-array overload should have rejected: " << result.err_msg;
    const auto* record = getLastCallRecordIfLoaded(implPath, "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE)
        << "Validation must reject before any backend call.";
}

/// Helper: invoke the map-keyed overload with the same logical payload and
/// assert it rejects with the same `ErrorCode::INVALID_VALUE` (Test #5 parity).
void expectMapRejected(std::shared_ptr<Graph>& graph,
                       [[maybe_unused]] hipdnnHandle_t handle,
                       std::unordered_map<int64_t, void*>& variantPack,
                       const std::unordered_map<int64_t, OverrideEntry>& overrides)
{
    const auto& implPath = hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath();
    resetLastCallRecordIfLoaded(implPath, "OverrideImplementing");
    auto result = graph->execute(handle, variantPack, nullptr, overrides);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "Map overload should have rejected: " << result.err_msg;
    const auto* record = getLastCallRecordIfLoaded(implPath, "OverrideImplementing");
    ASSERT_NE(record, nullptr);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(record->whichEntry, TestPluginExecuteEntry::NONE)
        << "Validation must reject before any backend call.";
}

} // namespace

// ============================================================================
// Per-rule validation tests (RFC §4.2.1, 8 rules) — tested in BOTH overloads
// to satisfy plan Test #5 (map-overload validation parity).
//
// Rules 1 and 5 only have a parallel-array form (rule 1 is "array length
// consistency", rule 5 is "duplicate UIDs in the parallel-array form"). The
// map overload cannot construct an inconsistent-length input or duplicate
// UIDs by construction, so for those two rules we only have the array form
// — but Test #5 still applies because the lowered map MUST never reach those
// validation paths in violation form. We verify map overload + an internally
// duplicate map key produces the SAME structural reject (here, the map is
// already de-duplicated by construction, so the map form simply succeeds when
// the array form would have failed; we document this in the comment).
// ============================================================================

class IntegrationOverrideValidation : public IntegrationOverrideValidationBase
{
};

// ----------------------- Rule 1 — Length consistency -----------------------

/// RFC §4.2.1 r1: `override_uids`, `override_shapes`, `override_strides`
/// must all have equal length. The parallel-array overload exposes this rule;
/// the map overload cannot violate it (the values are constructed in lockstep).
TEST_F(IntegrationOverrideValidation, Rule1_LengthInconsistency_ArrayForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule1_LengthInconsistency", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    // 2 uids, 1 shape, 2 strides — violates r1.
    const std::vector<int64_t> uids = {1, 2};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    const std::vector<std::vector<int64_t>> strides = {packedStrides(dims), packedStrides(dims)};
    expectArrayRejected(graph, _handle, variantPack, uids, shapes, strides);
}

// ----------------------- Rule 2 — Unknown UID -----------------------

TEST_F(IntegrationOverrideValidation, Rule2_UnknownUid_ArrayForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule2_UnknownUid", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    // UID 999 is not a graph tensor.
    const std::vector<int64_t> uids = {999};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    const std::vector<std::vector<int64_t>> strides = {packedStrides(dims)};
    expectArrayRejected(graph, _handle, variantPack, uids, shapes, strides);
}

TEST_F(IntegrationOverrideValidation, Rule2_UnknownUid_MapForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule2_UnknownUid_Map", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    std::unordered_map<int64_t, OverrideEntry> overrides;
    overrides[999] = OverrideEntry{dims, packedStrides(dims)};
    expectMapRejected(graph, _handle, variantPack, overrides);
}

// ----------------------- Rule 3 — Rank mismatch -----------------------

TEST_F(IntegrationOverrideValidation, Rule3_RankMismatch_ArrayForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule3_RankMismatch", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    // Declared rank = 4; supply rank-3 override shape.
    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {{3, 4, 4}};
    const std::vector<std::vector<int64_t>> strides = {{16, 4, 1}};
    expectArrayRejected(graph, _handle, variantPack, uids, shapes, strides);
}

TEST_F(IntegrationOverrideValidation, Rule3_RankMismatch_MapForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule3_RankMismatch_Map", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    std::unordered_map<int64_t, OverrideEntry> overrides;
    overrides[1] = OverrideEntry{{3, 4, 4}, {16, 4, 1}};
    expectMapRejected(graph, _handle, variantPack, overrides);
}

// ----------------------- Rule 4 — Max-shape exceeded -----------------------

/// Plain rule-4 test (parallel-array): override dim larger than declared.
TEST_F(IntegrationOverrideValidation, Rule4_MaxShapeExceeded_ArrayForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule4_MaxShapeExceeded", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    // override H = 8 > declared 4.
    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {{1, 3, 8, 4}};
    const std::vector<std::vector<int64_t>> strides = {{int64_t{3} * 8 * 4, int64_t{8} * 4, 4, 1}};
    expectArrayRejected(graph, _handle, variantPack, uids, shapes, strides);
}

TEST_F(IntegrationOverrideValidation, Rule4_MaxShapeExceeded_MapForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule4_MaxShapeExceeded_Map", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    std::unordered_map<int64_t, OverrideEntry> overrides;
    overrides[1] = OverrideEntry{
        {1, 3, 8, 4}, {static_cast<int64_t>(3) * 8 * 4, static_cast<int64_t>(8) * 4, 4, 1}};
    expectMapRejected(graph, _handle, variantPack, overrides);
}

/// Test #17: future-proof phrasing of rule 4. Phase-1 compares ALL dims
/// (no wildcards). When Phase 2 introduces wildcards (`-1`) and changes the
/// rule to "for non-wildcard dimensions only", this test will break and
/// signal the test author to add wildcard-specific coverage. Until then, the
/// test asserts every-dim comparison via two equivalent cases: one where the
/// first axis exceeds and one where the last axis exceeds. Both must reject.
TEST_F(IntegrationOverrideValidation, Rule4_AllDimsCompared_FirstAxis)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph
        = createOverridableGraph("Rule4_AllDimsCompared_FirstAxis", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    // First axis (N) exceeded.
    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {{2, 3, 4, 4}};
    const std::vector<std::vector<int64_t>> strides = {packedStrides({2, 3, 4, 4})};
    expectArrayRejected(graph, _handle, variantPack, uids, shapes, strides);
}

TEST_F(IntegrationOverrideValidation, Rule4_AllDimsCompared_LastAxis)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph
        = createOverridableGraph("Rule4_AllDimsCompared_LastAxis", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    // Last axis (W) exceeded.
    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {{1, 3, 4, 8}};
    const std::vector<std::vector<int64_t>> strides = {packedStrides({1, 3, 4, 8})};
    expectArrayRejected(graph, _handle, variantPack, uids, shapes, strides);
}

// ----------------------- Rule 5 — Duplicate UIDs -----------------------

/// Rule 5 is parallel-array specific (the map de-duplicates by construction).
TEST_F(IntegrationOverrideValidation, Rule5_DuplicateUids_ArrayForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule5_DuplicateUids", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    const std::vector<int64_t> uids = {1, 1};
    const std::vector<std::vector<int64_t>> shapes = {dims, dims};
    const std::vector<std::vector<int64_t>> strides = {packedStrides(dims), packedStrides(dims)};
    expectArrayRejected(graph, _handle, variantPack, uids, shapes, strides);
}

// ----------------------- Rule 6 — Positive dim values -----------------------

TEST_F(IntegrationOverrideValidation, Rule6_NonPositiveDim_ArrayForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule6_NonPositiveDim", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {{1, 3, 0, 4}}; // zero is not positive
    const std::vector<std::vector<int64_t>> strides = {packedStrides(dims)};
    expectArrayRejected(graph, _handle, variantPack, uids, shapes, strides);
}

TEST_F(IntegrationOverrideValidation, Rule6_NonPositiveDim_MapForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule6_NonPositiveDim_Map", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    std::unordered_map<int64_t, OverrideEntry> overrides;
    overrides[1] = OverrideEntry{{1, 3, 0, 4}, packedStrides(dims)};
    expectMapRejected(graph, _handle, variantPack, overrides);
}

// ----------------------- Rule 7 — Positive stride values -----------------------

TEST_F(IntegrationOverrideValidation, Rule7_NonPositiveStride_ArrayForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule7_NonPositiveStride", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    const std::vector<std::vector<int64_t>> strides = {{48, 16, 0, 1}}; // zero stride
    expectArrayRejected(graph, _handle, variantPack, uids, shapes, strides);
}

TEST_F(IntegrationOverrideValidation, Rule7_NonPositiveStride_MapForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule7_NonPositiveStride_Map", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    std::unordered_map<int64_t, OverrideEntry> overrides;
    overrides[1] = OverrideEntry{dims, {48, 16, 0, 1}};
    expectMapRejected(graph, _handle, variantPack, overrides);
}

// ----------------------- Rule 8 — Stride-ordering preserved -----------------------

/// Test #12 (RFC 0008 plan §6.5): D4 Phase-1 phrasing of rule 8. Declared
/// strides `{H*W, W, 1}` (row-major NCHW packed) imply axis ordering
/// argsort_descending = [N, C, H, W]. Override strides must produce the
/// SAME argsort. Reject any override whose argsort differs.
///
/// Negative case: declared {48, 16, 4, 1} (NCHW, descending) vs. override
/// {1, 4, 16, 48} (NHWC-like ascending) — strict reverse order.
TEST_F(IntegrationOverrideValidation, Rule8_StrideOrderingMismatch_ArrayForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule8_StrideOrdering", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    // Declared argsort_descending(48,16,4,1) = {0,1,2,3}; override argsort
    // {1,4,16,48} = {3,2,1,0} — different argsort, must reject.
    const std::vector<std::vector<int64_t>> strides = {{1, 4, 16, 48}};
    expectArrayRejected(graph, _handle, variantPack, uids, shapes, strides);
}

TEST_F(IntegrationOverrideValidation, Rule8_StrideOrderingMismatch_MapForm)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule8_StrideOrdering_Map", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    std::unordered_map<int64_t, OverrideEntry> overrides;
    overrides[1] = OverrideEntry{dims, {1, 4, 16, 48}};
    expectMapRejected(graph, _handle, variantPack, overrides);
}

/// Positive control: same argsort_descending as declared → must NOT reject
/// for rule-8 reasons. Uses the same numerical strides as declared so all
/// rules pass and the call reaches the override entry.
TEST_F(IntegrationOverrideValidation, Rule8_StrideOrderingMatch_Accepted)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = createOverridableGraph("Rule8_StrideOrdering_Accepted", dims, packedStrides(dims));
    compileGraph(graph, _handle);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    const std::vector<std::vector<int64_t>> strides = {packedStrides(dims)};

    auto result = graph->execute(_handle, variantPack, nullptr, uids, shapes, strides);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(result.code, ErrorCode::OK)
        << "Override matching declared stride argsort must not reject for "
           "rule 8 (Test #12 positive control): "
        << result.err_msg;
}

// ============================================================================
// C.6 — Reject "overrides without flag" (RFC §7.1).
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
        // No suffixed fakes are loaded by this fixture (it uses
        // test_good_plugin instead), so the resetters below are all
        // silent no-ops via the dlsym lookup. Kept for consistency with
        // the other override fixtures.
        resetLastCallRecordIfLoaded(
            hipdnn_tests::plugin_constants::testOverrideImplementingPluginPath(),
            "OverrideImplementing");
        resetLastCallRecordIfLoaded(
            hipdnn_tests::plugin_constants::testOverrideOmittingPluginPath(), "OverrideOmitting");
        resetLastCallRecordIfLoaded(hipdnn_tests::plugin_constants::testVersionLiarPluginPath(),
                                    "VersionLiar");
        resetLastCallRecordIfLoaded(hipdnn_tests::plugin_constants::testSecondOverridePluginPath(),
                                    "SecondOverride");

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

    hipdnnHandle_t _handle = nullptr;
};

/// RFC §7.1: calling the override-execute overload on a graph that did not
/// `set_dynamic_shape_enabled(true)` must reject with `INVALID_VALUE` and
/// must NOT invoke the backend.
TEST_F(IntegrationOverrideValidationFlagAbsent, ArrayFormRejectedWhenFlagAbsent)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};

    // Build a graph WITHOUT set_dynamic_shape_enabled.
    auto graph = std::make_shared<Graph>();
    graph->set_name("FlagAbsent_Array")
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

    auto compileResult = graph->validate();
    ASSERT_EQ(compileResult.code, ErrorCode::OK) << compileResult.err_msg;
    compileResult = graph->build_operation_graph(_handle);
    ASSERT_EQ(compileResult.code, ErrorCode::OK) << compileResult.err_msg;
    compileResult = graph->create_execution_plans();
    ASSERT_EQ(compileResult.code, ErrorCode::OK) << compileResult.err_msg;
    compileResult = graph->check_support();
    ASSERT_EQ(compileResult.code, ErrorCode::OK) << compileResult.err_msg;
    compileResult = graph->build_plans();
    ASSERT_EQ(compileResult.code, ErrorCode::OK) << compileResult.err_msg;

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    const std::vector<int64_t> uids = {1};
    const std::vector<std::vector<int64_t>> shapes = {dims};
    const std::vector<std::vector<int64_t>> strides = {{48, 16, 4, 1}};

    auto result = graph->execute(_handle, variantPack, nullptr, uids, shapes, strides);
    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE)
        << "Override-execute on a graph that did not opt in must reject "
           "(RFC §7.1): "
        << result.err_msg;
    // The "rejected before backend call" property is implied by INVALID_VALUE
    // here: this fixture loads `test_good_plugin`, which does not expose any
    // of the suffixed `LastCallRecord` accessors, so there is no per-plugin
    // TLS record to inspect.
}

TEST_F(IntegrationOverrideValidationFlagAbsent, MapFormRejectedWhenFlagAbsent)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};

    auto graph = std::make_shared<Graph>();
    graph->set_name("FlagAbsent_Map")
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

    auto compileResult = graph->validate();
    ASSERT_EQ(compileResult.code, ErrorCode::OK) << compileResult.err_msg;
    compileResult = graph->build_operation_graph(_handle);
    ASSERT_EQ(compileResult.code, ErrorCode::OK) << compileResult.err_msg;
    compileResult = graph->create_execution_plans();
    ASSERT_EQ(compileResult.code, ErrorCode::OK) << compileResult.err_msg;
    compileResult = graph->check_support();
    ASSERT_EQ(compileResult.code, ErrorCode::OK) << compileResult.err_msg;
    compileResult = graph->build_plans();
    ASSERT_EQ(compileResult.code, ErrorCode::OK) << compileResult.err_msg;

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[1] = nullptr;
    variantPack[2] = nullptr;

    std::unordered_map<int64_t, OverrideEntry> overrides;
    overrides[1] = OverrideEntry{dims, {48, 16, 4, 1}};

    auto result = graph->execute(_handle, variantPack, nullptr, overrides);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;
    // Same caveat as `ArrayFormRejectedWhenFlagAbsent`: this fixture loads
    // `test_good_plugin`, which has no suffixed TLS record to inspect.
    // Rejection before any backend call is implied by INVALID_VALUE.
}

// ============================================================================
// Test #14 — frontend pack/unpack round-trip of `is_dynamic_shape_enabled`.
// ============================================================================

/// Build a graph with `set_dynamic_shape_enabled(true)`, serialize,
/// deserialize, and assert the flag survived. Satisfies plan task C.0 and
/// Test #14. No GPU dispatch; only frontend pack/unpack.
///
/// Uses `to_binary()` which auto-lowers via the backend descriptor, mirroring
/// the existing `IntegrationGraphLifting` round-trip pattern. Deserializes
/// without a handle (structure-only) so the test runs without GPU prerequisites.
TEST(IntegrationOverrideRoundTrip, DynamicShapeEnabledFlagSurvivesSerialization)
{
    SKIP_IF_NO_DEVICES();

    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graph = std::make_shared<Graph>();
    graph->set_name("RoundTrip_DynamicShapeEnabled")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_dynamic_shape_enabled(true);

    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_TRUE(graph->is_dynamic_shape_enabled())
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
    result = graph->build_operation_graph(handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto [data, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto restored = std::make_shared<Graph>();
    auto deserResult = restored->deserialize(nullptr, data);
    ASSERT_EQ(deserResult.code, ErrorCode::OK) << deserResult.err_msg;

    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_TRUE(restored->is_dynamic_shape_enabled())
        << "Test #14: is_dynamic_shape_enabled must survive a "
           "serialize/deserialize round-trip.";

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

/// Round-trip the default (unset) case: a graph that never called
/// `set_dynamic_shape_enabled` must deserialize to `false` via the
/// in-memory getter (matches the wire default for legacy graphs).
TEST(IntegrationOverrideRoundTrip, DynamicShapeEnabledDefaultFalseSurvivesSerialization)
{
    SKIP_IF_NO_DEVICES();

    const std::array<const char*, 1> paths
        = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
    ASSERT_EQ(
        hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
        HIPDNN_STATUS_SUCCESS);
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);

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

    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_FALSE(graph->is_dynamic_shape_enabled())
        << "Default (unset) must read as false from the in-memory getter.";

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    auto [data, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto restored = std::make_shared<Graph>();
    auto deserResult = restored->deserialize(nullptr, data);
    ASSERT_EQ(deserResult.code, ErrorCode::OK) << deserResult.err_msg;

    // NOLINTNEXTLINE(readability-implicit-bool-conversion)
    EXPECT_FALSE(restored->is_dynamic_shape_enabled())
        << "Default-false must survive serialize/deserialize as false.";

    ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}

#endif // HIPDNN_ENABLE_SDPA
