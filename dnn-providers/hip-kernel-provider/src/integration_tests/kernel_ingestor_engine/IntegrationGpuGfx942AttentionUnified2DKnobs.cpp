// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"
#include "ScopedPluginLogCapture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hip_kernel_provider::test_utilities;

/**
 * @file IntegrationGpuGfx942AttentionUnified2DKnobs.cpp
 * @brief hipkernel:Gfx942AttentionUnified2D end to end through the frontend on a gfx942
 *        device: each catalog kernel class is selected by its knobs, launched with its own
 *        geometry and checked against the CPU reference; the shapes the engine must leave
 *        alone are not offered.
 *
 * The knobs are the geometry (tile_size, num_warps, block_m_per_warp and the MFMA codepath
 * flags) and the codegen stack (K ring, direct-global Q, conflict-free V stores, the mask
 * limit and the invariant hoist). Each expected kernel id is resolved at run time from the
 * production descriptors (kernelIdsFor), and the coverage case holds the tables to exactly the
 * catalog's kernel classes. Each forced case pins a head pair whose GQA ratio its class ships, so a catalog
 * change that drops that ratio fails the case and its table row is moved with it. The kernels
 * read the head counts at run time and bake only their ratio, so the forced cases run head
 * pairs the manifest's cells never list. Every case runs with a workspace filled with NaN bytes
 * first: hipDNN does not zero it, and the kernel must not read anything it did not write.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

constexpr ServedEngine ENGINE{"hipkernel:Gfx942AttentionUnified2D", "gfx942"};

/// The catalog carries only the bottom-right causal kernel (mask_type 2); the other two masks
/// appear only in declines.
constexpr int64_t BOTTOM_RIGHT_MASK_TYPE = 2;

/// One forced knob value, as a (KMD field = knob id, value) pair.
using Knobs = std::vector<std::pair<std::string, int64_t>>;

/// One graph, the knobs forced on it (empty: the engine's own ranking picks) and the
/// metadata the selected kernel must carry beyond the graph's own fields.
struct KnobCase
{
    std::string name;
    IngestorSdpaGraphSpec shape;
    Knobs knobs;
    Knobs expected;
};

/// One graph the engine must not offer to serve, and why.
struct DeclineCase
{
    std::string name;
    IngestorSdpaGraphSpec shape;
};

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const KnobCase& knobCase, std::ostream* os)
{
    *os << knobCase.name;
}

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const DeclineCase& declineCase, std::ostream* os)
{
    *os << declineCase.name;
}

/// A kernel class: (tile_size at block_size 16, num_warps, block_m_per_warp, use_mfma_32x32x8,
/// use_transposed_qk_32x32) and the five codegen stack fields (use_k_sliced_ring,
/// use_q_direct_global, use_conflict_free_v_store, use_transposed_mask_limit,
/// use_transposed_invariant_hoist), named for the case.
struct Geometry
{
    const char* name;
    int64_t tileSize;
    int64_t numWarps;
    int64_t blockMPerWarp;
    int64_t useMfma32x32x8;
    int64_t useTransposedQk32x32;
    int64_t useKSlicedRing = 0;
    int64_t useQDirectGlobal = 0;
    int64_t useConflictFreeVStore = 0;
    int64_t useTransposedMaskLimit = 0;
    int64_t useTransposedInvariantHoist = 0;
};

/// The catalog's classes per head size; the first of each is every cell's default geometry.
/// D64's production stack (ring, direct-global Q, conflict-free V stores, mask limit and
/// hoist) rides the default's 4 x 32 geometry; fp16 D128's drops direct-global Q.
constexpr std::array<Geometry, 3> D64_GEOMETRIES{{
    {"Transposed4x32T64", 64, 4, 32, 1, 1},
    {"Production4x32T64", 64, 4, 32, 1, 1, 1, 1, 1, 1, 1},
    {"Narrow2x16T64", 64, 2, 16, 0, 0},
}};
constexpr std::array<Geometry, 6> D128_GEOMETRIES{{
    {"Transposed2x32T32", 32, 2, 32, 1, 1},
    {"Transposed4x32T32", 32, 4, 32, 1, 1},
    {"Narrow4x16T32", 32, 4, 16, 0, 0},
    {"Narrow2x32T32", 32, 2, 32, 0, 0},
    {"Narrow2x16T16", 16, 2, 16, 0, 0},
    {"Production4x32T64Ring", 64, 4, 32, 1, 1, 1, 0, 1, 1, 1},
}};

/// A (query heads, KV heads) pair. Several forced cases use pairs no manifest cell lists,
/// since one kernel serves every pair of its GQA ratio.
struct Heads
{
    int64_t queryHeads;
    int64_t kvHeads;
};

Knobs knobsOf(const Geometry& g)
{
    return {{"tile_size", g.tileSize},
            {"num_warps", g.numWarps},
            {"block_m_per_warp", g.blockMPerWarp},
            {"use_mfma_32x32x8", g.useMfma32x32x8},
            {"use_transposed_qk_32x32", g.useTransposedQk32x32},
            {"use_k_sliced_ring", g.useKSlicedRing},
            {"use_q_direct_global", g.useQDirectGlobal},
            {"use_conflict_free_v_store", g.useConflictFreeVStore},
            {"use_transposed_mask_limit", g.useTransposedMaskLimit},
            {"use_transposed_invariant_hoist", g.useTransposedInvariantHoist}};
}

/// One forced kernel class (head size x geometry x dtype) and the head pair it runs on. The
/// pair's GQA ratio is one the reduced catalog ships for that class.
struct ForcedClass
{
    int64_t headSize;
    const Geometry* geometry;
    DataType dataType;
    Heads heads;
};

/// Every kernel class the catalog ships, forced once on a prefill shape with B > 1 and
/// Sq != Skv; then cold cases the engine's own ranking serves: rocKE's production stack where
/// the catalog ships one for the cell, the default geometry where it does not (bf16 D128,
/// whose production stack lost to the default on every sweep graph). Variant reduction kept
/// 14 classes; the rest of the geometries have no case.
std::vector<KnobCase> knobCases()
{
    constexpr auto BF16 = DataType::BFLOAT16;
    constexpr auto FP16 = DataType::HALF;
    const std::array<ForcedClass, 14> forced{{
        {64, &D64_GEOMETRIES[0], BF16, {8, 8}},
        {64, &D64_GEOMETRIES[0], FP16, {24, 6}},
        {64, &D64_GEOMETRIES[1], BF16, {64, 8}},
        {64, &D64_GEOMETRIES[1], FP16, {64, 8}},
        {64, &D64_GEOMETRIES[2], BF16, {8, 8}},
        {64, &D64_GEOMETRIES[2], FP16, {8, 8}},
        {128, &D128_GEOMETRIES[0], BF16, {48, 3}},
        {128, &D128_GEOMETRIES[0], FP16, {64, 8}},
        {128, &D128_GEOMETRIES[1], BF16, {20, 10}},
        {128, &D128_GEOMETRIES[2], BF16, {16, 2}},
        {128, &D128_GEOMETRIES[3], BF16, {16, 4}},
        {128, &D128_GEOMETRIES[4], BF16, {64, 8}},
        {128, &D128_GEOMETRIES[4], FP16, {8, 8}},
        {128, &D128_GEOMETRIES[5], FP16, {20, 10}},
    }};
    std::vector<KnobCase> cases;
    for(const auto& f : forced)
    {
        const auto& g = *f.geometry;
        const auto& h = f.heads;
        cases.push_back({"D" + std::to_string(f.headSize) + "_" + g.name + "_"
                             + dtypeSuffix(f.dataType) + "_Hq" + std::to_string(h.queryHeads)
                             + "Hkv" + std::to_string(h.kvHeads),
                         {f.dataType,
                          f.headSize,
                          h.queryHeads,
                          h.kvHeads,
                          SdpaMask::BOUNDS_BOTTOM_RIGHT,
                          2,
                          96,
                          256},
                         knobsOf(g),
                         knobsOf(g)});
    }

    cases.push_back({"D64_Cold_SelectsTheProductionStack",
                     {DataType::BFLOAT16, 64, 32, 8, SdpaMask::BOUNDS_BOTTOM_RIGHT, 3, 512, 512},
                     {},
                     knobsOf(D64_GEOMETRIES[1])});
    cases.push_back({"D128_Fp16_Cold_SelectsTheProductionStack",
                     {DataType::HALF, 128, 64, 8, SdpaMask::BOUNDS_BOTTOM_RIGHT, 1, 512, 2048},
                     {},
                     knobsOf(D128_GEOMETRIES[5])});
    cases.push_back({"D128_Bf16_Cold_SelectsTheDefaultGeometry",
                     {DataType::BFLOAT16, 128, 64, 8, SdpaMask::BOUNDS_BOTTOM_RIGHT, 1, 512, 2048},
                     {},
                     knobsOf(D128_GEOMETRIES[0])});
    // The kernel derives its cu_q search trip count from the num_seqs kernarg.
    cases.push_back({"D128_Cold_Batch512",
                     {DataType::BFLOAT16, 128, 8, 2, SdpaMask::BOUNDS_BOTTOM_RIGHT, 512, 16, 64},
                     {},
                     knobsOf(D128_GEOMETRIES[0])});
    return cases;
}

/// Shapes the engine must refuse at graph or kernel match, so it is not offered at all.
std::vector<DeclineCase> declineCases()
{
    return {
        // The identity cache is cut into whole 16-token blocks.
        {"SeqKvNotAWholeNumberOfBlocks",
         {DataType::BFLOAT16, 128, 8, 2, SdpaMask::BOUNDS_BOTTOM_RIGHT, 1, 64, 264}},
        // Decode at a ratio the split-KV gfx942 engine ships is that engine's.
        {"LeavesDecodeToTheSplitKvEngine",
         {DataType::BFLOAT16, 128, 8, 2, SdpaMask::BOUNDS_BOTTOM_RIGHT, 2, 1, 512}},
        // The kernel computes the bottom-right causal band only.
        {"NoMask", {DataType::BFLOAT16, 128, 8, 2, SdpaMask::NO_MASK, 2, 96, 256}},
        {"TopLeftCausal", {DataType::BFLOAT16, 128, 8, 2, SdpaMask::BOUNDS_TOP_LEFT, 2, 96, 256}},
        {"TopLeftCausalAtEqualLengths",
         {DataType::BFLOAT16, 128, 8, 2, SdpaMask::BOUNDS_TOP_LEFT, 2, 256, 256}},
        // A GQA ratio the catalog does not ship: graph_match declines it by name.
        {"RatioThree", {DataType::BFLOAT16, 128, 24, 8, SdpaMask::BOUNDS_BOTTOM_RIGHT, 2, 96, 256}},
    };
}

std::vector<KnobSetting> knobSettingsFor(const Knobs& knobs)
{
    std::vector<KnobSetting> settings;
    for(const auto& [knob, value] : knobs)
    {
        settings.emplace_back(knob, value);
    }
    return settings;
}

/// The ids of the kernels the engine carries for @p knobCase's graph fields and expected
/// metadata, as the selection line spells them.
std::vector<std::string> kernelIdsFor(const KnobCase& knobCase)
{
    detail::MetadataKey key{
        {"dtype", detail::catalogDtypeName(knobCase.shape.dataType)},
        {"head_size", knobCase.shape.headSize},
        {"num_queries_per_kv", knobCase.shape.queryHeads / knobCase.shape.kvHeads},
        {"mask_type", BOTTOM_RIGHT_MASK_TYPE},
        {"sliding_window", int64_t{0}},
    };
    for(const auto& [field, value] : knobCase.expected)
    {
        key.emplace_back(field, value);
    }

    std::vector<std::string> ids;
    detail::forEachServedKernel(
        ENGINE, [&key, &ids](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
            if(detail::carriesKey(kernel, key))
            {
                ids.push_back(hipdnn_plugin_sdk::ingestor::toString(kernel.id));
            }
        });
    if(ids.empty())
    {
        ADD_FAILURE() << "no " << ENGINE.name << " kernel for " << ENGINE.arch << " carries "
                      << detail::describeKey(key) << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                      << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR") << "'";
    }
    return ids;
}

} // namespace

/// The shared served-engine harness (a skip off gfx942, the seed offset by UID), with a
/// workspace filled with NaN bytes before every launch.
template <typename TestCaseType>
class IntegrationGpuGfx942AttentionUnified2DBase : public ServedEngineHarness<TestCaseType>
{
protected:
    IntegrationGpuGfx942AttentionUnified2DBase()
        : ServedEngineHarness<TestCaseType>(ENGINE, "gfx942-unified-2d-knobs-case")
    {
    }

    /// NaN bytes in every workspace byte: the engine may not read what it did not write.
    void prepareWorkspace(void* workspace, size_t bytes) override
    {
        ASSERT_EQ(hipMemsetAsync(workspace, 0xFF, bytes, this->stream()), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(this->stream()), hipSuccess);
    }
};

using IntegrationGpuGfx942AttentionUnified2DKnobs
    = IntegrationGpuGfx942AttentionUnified2DBase<KnobCase>;
using IntegrationGpuGfx942AttentionUnified2DDeclines
    = IntegrationGpuGfx942AttentionUnified2DBase<DeclineCase>;
using IntegrationGpuGfx942AttentionUnified2DKnobCoverage
    = IntegrationGpuGfx942AttentionUnified2DBase<KnobCase>;

/// The knob tables register before main() names the descriptor root; this holds them to the
/// catalog. Every kernel class the catalog ships (head size, dtype, mask, geometry) must be
/// one a forced case runs, so a class reduction keeps cannot go unexecuted; a class the
/// catalog dropped is reported by the case that forces it.
TEST_F(IntegrationGpuGfx942AttentionUnified2DKnobCoverage, CoversEveryCatalogKernelClass)
{
    ASSERT_FALSE(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR").empty())
        << "HIPDNN_DESCRIPTOR_RUNTIME_DIR is not set";

    using KernelClass = std::tuple<int64_t, std::string, int64_t, Knobs>;
    const std::vector<std::string> classKnobs{"tile_size",
                                              "num_warps",
                                              "block_m_per_warp",
                                              "use_mfma_32x32x8",
                                              "use_transposed_qk_32x32",
                                              "use_k_sliced_ring",
                                              "use_q_direct_global",
                                              "use_conflict_free_v_store",
                                              "use_transposed_mask_limit",
                                              "use_transposed_invariant_hoist"};

    std::set<KernelClass> tableClasses;
    for(const auto& knobCase : knobCases())
    {
        if(!knobCase.knobs.empty())
        {
            tableClasses.emplace(knobCase.shape.headSize,
                                 detail::catalogDtypeName(knobCase.shape.dataType),
                                 BOTTOM_RIGHT_MASK_TYPE,
                                 knobCase.knobs);
        }
    }

    std::size_t served = 0;
    detail::forEachServedKernel(
        ENGINE, [&](const hipdnn_plugin_sdk::ingestor::KernelDescriptor& kernel) {
            ++served;
            const auto intField = [&kernel](const std::string& field) -> const int64_t* {
                const auto value = kernel.metadata.find(field);
                return value == kernel.metadata.end() ? nullptr
                                                      : std::get_if<int64_t>(&value->second);
            };
            const auto dtype = kernel.metadata.find("dtype");
            const auto* dtypeName = dtype == kernel.metadata.end()
                                        ? nullptr
                                        : std::get_if<std::string>(&dtype->second);
            const auto* mask = intField("mask_type");
            const auto* headSize = intField("head_size");
            Knobs knobs;
            for(const auto& knob : classKnobs)
            {
                const auto* value = intField(knob);
                if(value == nullptr)
                {
                    ADD_FAILURE() << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
                                  << " carries no integer " << knob;
                    return;
                }
                knobs.emplace_back(knob, *value);
            }
            ASSERT_TRUE(dtypeName != nullptr && mask != nullptr && headSize != nullptr)
                << "kernel " << hipdnn_plugin_sdk::ingestor::toString(kernel.id)
                << " carries no string dtype and integer mask_type and head_size";
            EXPECT_EQ(tableClasses.count({*headSize, *dtypeName, *mask, knobs}), 1U)
                << ENGINE.name << " ships kernel class (D" << *headSize << ", " << *dtypeName
                << ", mask_type " << *mask << ", "
                << detail::describeKey(detail::MetadataKey(knobs.begin(), knobs.end()))
                << ") that no forced case runs";
        });
    EXPECT_GT(served, 0U) << ENGINE.name << " serves no kernel on " << ENGINE.arch
                          << " under HIPDNN_DESCRIPTOR_RUNTIME_DIR='"
                          << hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR")
                          << "'";
}

/// The selected kernel is read from the plugin's own selection line, so a knob dropped
/// anywhere between the frontend and the plugin's filter reads as the cold winner; the output
/// is then checked against the CPU reference, which catches a kernel launched with any
/// geometry but its own, or with head counts other than the graph's.
TEST_P(IntegrationGpuGfx942AttentionUnified2DKnobs,
       SelectsTheExpectedKernelAndMatchesTheCpuReference)
{
    const auto& testCase = GetParam();

    const ScopedPluginLogCapture capture(this);
    const detail::SelectedKernelReport report(ENGINE, testCase.name, capture);
    const auto& recorder = capture.recorder();

    auto sdpa = detail::buildSdpaGraph(testCase.name, testCase.shape);
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*sdpa.graph));

    auto result
        = sdpa.graph->create_execution_plan_ext(engineId(), knobSettingsFor(testCase.knobs));
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = sdpa.graph->check_support();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = sdpa.graph->build_plans();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg << "\nCaptured logs:\n"
                                          << recorder.getRecordedLogsAsString();

    const auto selected = detail::selectedKernelId(ENGINE, recorder);
    ASSERT_TRUE(selected.has_value())
        << "no selection line from " << ENGINE.name << ". Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    // Resolved after the plan is built, so a lookup failure still leaves the selected
    // kernel in the KNOB_SELECTED line. A forced case names exactly one kernel; a cold case
    // names the kernels of its expected class and must select one.
    const auto expectedIds = kernelIdsFor(testCase);
    ASSERT_FALSE(expectedIds.empty());
    if(!testCase.knobs.empty())
    {
        ASSERT_EQ(expectedIds.size(), 1U)
            << "the forced knobs must name exactly one kernel of this graph's configuration";
        EXPECT_TRUE(recorder.hasLogContaining(std::string(detail::SELECTED_KERNEL_LOG)
                                              + expectedIds.front() + " at rank 0 from "
                                              + candidateCount(1)))
            << "Captured logs:\n"
            << recorder.getRecordedLogsAsString();
    }
    EXPECT_NE(std::find(expectedIds.begin(), expectedIds.end(), *selected), expectedIds.end())
        << "selected " << *selected << ". Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    GraphVerificationContext context(*sdpa.graph);
    const float tolerance = detail::sdpaForwardTolerance(testCase.shape.dataType);
    registerValidator(context, sdpa.output, tolerance, tolerance);
    verifyBuiltGraph(context, /*seed=*/0);
}

/// A shape the engine must leave alone is not in its ranked list: no plan, so no silent
/// wrong answer from a mask, block or grid the kernel cannot honour.
TEST_P(IntegrationGpuGfx942AttentionUnified2DDeclines, IsNotOffered)
{
    const auto& testCase = GetParam();
    auto sdpa = detail::buildSdpaGraph(testCase.name, testCase.shape);
    EXPECT_FALSE(offersToServe(*sdpa.graph))
        << ENGINE.name << " offered to serve " << testCase.name;
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionUnified2DKnobs,
                         ::testing::ValuesIn(knobCases()),
                         [](const ::testing::TestParamInfo<KnobCase>& info) {
                             return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx942AttentionUnified2DDeclines,
                         ::testing::ValuesIn(declineCases()),
                         [](const ::testing::TestParamInfo<DeclineCase>& info) {
                             return info.param.name;
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
