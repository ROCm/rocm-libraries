// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCacheFile.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"

/**
 * @file TestFlydslRmsNormEngine.cpp
 * @brief The flyDSL RMS-norm pack across all four native seams, plus a real launch.
 *
 * The census over what the pack SHIPS lives next door in TestFlydslRmsNormPacks.cpp,
 * which belongs to the census binary because its cases need a named shard. Everything
 * here runs in the ordinary unit binary.
 *
 * The pack is unusual among the ingestor packs in four ways, and each gets cases here:
 *
 *  - It ships TWO SELECTION TIERS over the same graph. A kernel whose `N` metadata names
 *    the graph's width is preferred (`priority: 100`); a generic kernel carrying the
 *    `N: 0` sentinel takes the width as a kernarg and serves any width (`priority: 10`).
 *    The kernel matcher admits both and `score` orders them, so a width no instance was
 *    baked for still plans. f32 ships the generic tier only.
 *
 *  - Its kernels take EPSILON AS AN ARGUMENT, so any finite, non-negative value is served,
 *    whether the graph fixes it or the caller supplies it at execute. One that is not --
 *    negative, NaN, infinite -- is declined by `graph_match` when baked, and refused by
 *    `launch()` when it arrives at execute. Both paths are covered below.
 *
 *  - Gamma may be f32 beside bf16/f16 x and y, served by objects built to read it so
 *    (`scale_f32: 1`); the kernel matcher pairs each graph with objects of its own form.
 *
 *  - Its kernel code is only ever KPACK. There is no embedded-source fallback, so every
 *    case that reaches `prepare()` needs the staged archive this binary's descriptor root
 *    carries, and the kernel definitions are built from the DESCRIPTORS THEMSELVES rather
 *    than by hand -- a hand-built `source.sha256` would describe bytes no packer produced.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

using data_objects::DataType;
using data_objects::NormFwdPhase;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool matches(const MatchContext& context)
{
    return matchesGraph(FLYDSL_RMSNORM, context).has_value();
}

/// The bindings a real plan build would hand the dispatch, from running the graph match.
BoundTokens bindingsFor(const MatchContext& context)
{
    auto bound = matchesGraph(FLYDSL_RMSNORM, context);
    if(!bound.has_value())
    {
        throw std::logic_error("test graph does not match the pack it is dispatched against");
    }
    return std::move(*bound);
}

/// A kernel carrying exactly the metadata triple this pack's KMD declares.
///
/// The shared makeKernel() cannot serve: it emits the pointwise pack's
/// `block_size`/`dtype`/`operation`, and both the field names and the dtype vocabulary
/// differ -- this pack spells the descriptor dialect (`bf16`), not the flatbuffer enum.
///
/// Source is left default-constructed. Every case using this one reads metadata only;
/// the cases that reach a loader take their kernels from the staged descriptors instead.
KernelDefinition makeFlydslKernel(const std::string& dtype,
                                  int64_t bakedN,
                                  int64_t blockThreads = 256,
                                  int64_t priority = 100,
                                  int64_t scaleF32 = 0,
                                  int64_t storeRstd = 0)
{
    KernelDefinition kernel;
    kernel.name = "rmsnorm_" + dtype + "_n" + std::to_string(bakedN);
    kernel.priority = priority;
    kernel.metadata = {{std::string(FLYDSL_DTYPE_FIELD), dtype},
                       {std::string(FLYDSL_N_FIELD), bakedN},
                       {std::string(FLYDSL_BLOCK_THREADS_FIELD), blockThreads},
                       {std::string(FLYDSL_SCALE_F32_FIELD), scaleF32},
                       {std::string(FLYDSL_STORE_RSTD_FIELD), storeRstd}};
    return kernel;
}

/// A kernel definition assembled from a descriptor the packer actually built, found in
/// this binary's own discovery root by its metadata.
///
/// Promoted from KernelDescriptor rather than hand-written because a kpack source is only
/// usable with the packer's own `library`, `toc_key`, `symbol`, `sha256` and the two
/// loader-filled path fields -- originDirectory to resolve the archive against, treeRoot
/// as the boundary it may not resolve outside of. Guessing any of them fails at load with
/// a diagnostic about the guess rather than about the kernel.
///
/// Returns false when the loaded set holds no kernel with that (dtype, N, scale_f32) for
/// this device, which is the normal answer on an arch this build packed nothing for. A
/// build for several arches loads every arch's shard, so a kernel staged for another device
/// is passed over.
bool findStagedKernel(const std::string& dtype,
                      int64_t bakedN,
                      KernelDefinition& out,
                      int64_t scaleF32 = 0,
                      int64_t storeRstd = 0)
{
    // A build for CDNA targets only stages no RMSNorm set at all.
    const auto* staged = findLoadedSet(FLYDSL_RMSNORM.engineName);
    if(staged == nullptr)
    {
        return false;
    }
    const auto& set = *staged;
    const auto device = packedArchDeviceProperties().gcnArchName;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            if(!device.empty() && !arches.empty()
               && std::find(arches.begin(), arches.end(), device) == arches.end())
            {
                continue;
            }
            const auto* name
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_DTYPE_FIELD);
            const auto* n = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_N_FIELD);
            const auto* f32Scale
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SCALE_F32_FIELD);
            const auto* rstd
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_STORE_RSTD_FIELD);
            if(name == nullptr || n == nullptr || f32Scale == nullptr || rstd == nullptr
               || *name != dtype || *n != bakedN || *f32Scale != scaleF32 || *rstd != storeRstd)
            {
                continue;
            }

            // A kernel may narrow the arch list its pack claims but never reach outside it,
            // so an empty list on the kernel means "the pack's".
            out = KernelDefinition{kernel.id,
                                   pack.id,
                                   pack.dispatchId,
                                   kernel.source,
                                   kernel.metadata,
                                   kernel.priority,
                                   kernel.arch.empty() ? pack.arch : kernel.arch,
                                   kernel.originDirectory,
                                   kernel.name,
                                   kernel.treeRoot};
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// graph_match: what this pack accepts
// ---------------------------------------------------------------------------

/// Plain function pointer rather than std::function: FlatBufferBuilder is move-only.
struct GraphCase
{
    std::string name;
    flatbuffers::FlatBufferBuilder (*buildGraph)();
};

/// The default graph with @p extras: the fields only RmsNormGraphExtras reaches.
flatbuffers::FlatBufferBuilder rmsNormGraphWith(const RmsNormGraphExtras& extras,
                                                NormFwdPhase phase = NormFwdPhase::NOT_SET)
{
    return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                   {8, 4096},
                                   FLYDSL_TEST_EPSILON,
                                   false,
                                   std::nullopt,
                                   std::nullopt,
                                   std::nullopt,
                                   false,
                                   false,
                                   phase,
                                   extras);
}

/// rmsNormGraphWith() at a chosen x and gamma shape, and optionally x's strides.
flatbuffers::FlatBufferBuilder
    rmsNormGraphWithShapes(const std::vector<int64_t>& xDims,
                           const std::vector<int64_t>& scaleDims,
                           const RmsNormGraphExtras& extras,
                           const std::optional<std::vector<int64_t>>& xStrides = std::nullopt)
{
    return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                   xDims,
                                   FLYDSL_TEST_EPSILON,
                                   false,
                                   scaleDims,
                                   std::nullopt,
                                   xStrides,
                                   false,
                                   false,
                                   NormFwdPhase::NOT_SET,
                                   extras);
}

/// A TRAINING graph that saves inv_rms, as the frontend builds one: the statistic is shaped
/// per row unless @p extras overrides it.
flatbuffers::FlatBufferBuilder trainingGraph(const std::vector<int64_t>& xDims = {8, 4096},
                                             const RmsNormGraphExtras& extras = {},
                                             const std::optional<std::vector<int64_t>>& scaleDims
                                             = std::nullopt,
                                             DataType dataType = DataType::BFLOAT16)
{
    return buildFlydslRmsNormGraph(dataType,
                                   xDims,
                                   FLYDSL_TEST_EPSILON,
                                   false,
                                   scaleDims,
                                   std::nullopt,
                                   std::nullopt,
                                   false,
                                   true,
                                   NormFwdPhase::TRAINING,
                                   extras);
}

std::string graphCaseName(const ::testing::TestParamInfo<GraphCase>& info)
{
    return info.param.name;
}

class TestFlydslRmsNormGraphMatcherAcceptance : public ::testing::TestWithParam<GraphCase>
{
};

TEST_P(TestFlydslRmsNormGraphMatcherAcceptance, Accepts)
{
    const GraphFixture fixture(GetParam().buildGraph());
    EXPECT_TRUE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslRmsNormGraphMatcherAcceptance,
    ::testing::Values(
        GraphCase{"Bfloat16", [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16); }},
        // Every value of a field the kernels handle, beside the refusals below: a compute
        // type at or below their f32 arithmetic, a rank-2 scalar epsilon. (A wider alignment
        // needs graph schema 1.3.0, above the engine's; the census pins that.)
        GraphCase{"ComputeTypeFloat",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.computeDataType = DataType::FLOAT;
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"ComputeTypeUnset",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.computeDataType = DataType::UNSET;
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"EpsilonOfRankTwo",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.epsilonDims = {1, 1};
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"Half", [] { return buildFlydslRmsNormGraph(DataType::HALF); }},
        // The generic tier only: see RMSNORM_GENERIC_ONLY_DTYPES in the generator.
        GraphCase{"Float", [] { return buildFlydslRmsNormGraph(DataType::FLOAT); }},
        // A stride on an axis of extent 1 is never stepped along, so any value describes
        // the same memory. hipdnn_torch describes an F.rms_norm weight this way, with
        // stride 0 on every unit axis, at rank 2...
        GraphCase{"ScaleWithStrideZeroOnItsUnitAxis",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.scaleStrides = {0, 1};
                      return rmsNormGraphWithShapes({8, 4096}, {1, 4096}, extras);
                  }},
        // ...and at rank 4.
        GraphCase{"Rank4ScaleWithStrideZeroOnItsUnitAxes",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.scaleStrides = {0, 0, 0, 1};
                      return rmsNormGraphWithShapes({2, 8, 4, 4096}, {1, 1, 1, 4096}, extras);
                  }},
        GraphCase{"ScaleWithUnitStridesOnItsUnitAxes",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.scaleStrides = {1, 1, 1, 1};
                      return rmsNormGraphWithShapes({2, 8, 4, 4096}, {1, 1, 1, 4096}, extras);
                  }},
        // cuDNN's layout, as hipDNN's cuDNN-shim samples build it: x {B*S, D, 1, 1} with
        // gamma {1, D, 1, 1}. The normalised axes are the trailing run where gamma equals
        // x -- D and the two unit axes -- not the innermost axis alone.
        GraphCase{"CudnnLayoutWithTrailingUnitAxes",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.scaleStrides = {4096, 1, 4096, 4096};
                      return rmsNormGraphWithShapes({32, 4096, 1, 1},
                                                    {1, 4096, 1, 1},
                                                    extras,
                                                    std::vector<int64_t>{4096, 1, 4096, 4096});
                  }},
        // A multi-axis normalised shape, F.rms_norm(x, (4, 1024)): one row of 4096.
        GraphCase{"MultiAxisNormalisedShape",
                  [] { return rmsNormGraphWithShapes({8, 4, 1024}, {1, 4, 1024}, {}); }},
        // The same exemption on x: a unit batch axis with a stride that matches nothing.
        GraphCase{"XWithAnOddStrideOnAUnitAxis",
                  [] {
                      return rmsNormGraphWithShapes(
                          {1, 8, 4096}, {4096}, {}, std::vector<int64_t>{7, 4096, 1});
                  }},
        // Rank above two is collapsed, not refused: the leading axes multiply into the row
        // count, which is sound precisely because packed row-major is also required.
        GraphCase{"Rank4CollapsesToRows",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {2, 3, 4, 4096}); }},
        // A width no instance is baked for: the generic tier is what makes this applicable,
        // and the graph matcher does not know about tiers at all.
        GraphCase{"AnUnbakedWidth",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 640}); }},
        // Epsilon is a kernel argument, so every usable value is served: vLLM's and HF's
        // 1e-6, PyTorch's eps=None (f32's machine epsilon), and zero.
        GraphCase{"AnEpsilonOfOneInAMillion",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}, 1e-6F); }},
        GraphCase{"TheF32MachineEpsilon",
                  [] {
                      return buildFlydslRmsNormGraph(
                          DataType::BFLOAT16, {8, 4096}, std::numeric_limits<float>::epsilon());
                  }},
        GraphCase{"AZeroEpsilon",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}, 0.0F); }},
        // An epsilon carried as f64 is read as one, so a value f32 holds is served...
        GraphCase{"AnF64Epsilon",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.epsilonF64 = 1e-5;
                      return rmsNormGraphWith(extras);
                  }},
        // ...up to the largest one f32 holds; the refusal suite's AnF64EpsilonPastTheF32Range
        // is the next double up.
        GraphCase{"AnF64EpsilonAtTheF32Maximum",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.epsilonF64 = static_cast<double>(std::numeric_limits<float>::max());
                      return rmsNormGraphWith(extras);
                  }},
        // The largest x and y one buffer view's 32-bit byte offsets reach: 1049601 rows of
        // 1023 bf16 is 2^31 - 2 bytes, the other side of 2^31 - 1 from the refusal suite's
        // PastThirtyTwoBitAddressing...
        GraphCase{"AtThirtyTwoBitAddressing",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {1049601, 1023}); }},
        // ...and in f32, 256999 rows of 2089 is 2^31 - 4 bytes.
        GraphCase{"FloatAtThirtyTwoBitAddressing",
                  [] { return buildFlydslRmsNormGraph(DataType::FLOAT, {256999, 2089}); }},
        // An f32 gamma beside 16-bit x and y: cuDNN's and PyTorch's usual mixed form.
        GraphCase{"Bfloat16WithAnF32Scale",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     DataType::FLOAT);
                  }},
        GraphCase{"HalfWithAnF32Scale",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::HALF,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     DataType::FLOAT);
                  }},
        // Not inspectable at match time. Deferred to launch(), which checks it there.
        GraphCase{
            "ARuntimeEpsilon",
            [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}, 0.0F, true); }},
        GraphCase{"InferencePhase",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     false,
                                                     false,
                                                     NormFwdPhase::INFERENCE);
                  }},
        // The training forward: the same graph plus inv_rms, one f32 per row, which the
        // `store_rstd` objects write.
        GraphCase{"TrainingSavingTheInverseRms", [] { return trainingGraph(); }},
        GraphCase{"TrainingAtRank3", [] { return trainingGraph({2, 3, 4096}); }},
        // cuDNN's layout: inv_rms {B*S, 1, 1, 1}, one per row of the D-wide run.
        GraphCase{"TrainingInTheCudnnLayout",
                  [] {
                      return trainingGraph(
                          {32, 4096, 1, 1}, {}, std::vector<int64_t>{1, 4096, 1, 1});
                  }},
        GraphCase{"TrainingAtFloat",
                  [] { return trainingGraph({8, 640}, {}, std::nullopt, DataType::FLOAT); }}),
    graphCaseName);

TEST(TestFlydslRmsNormBinding, BindsEveryTokenTheDispatchReads)
{
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {2, 3, 4096}));
    const auto bound = matchesGraph(FLYDSL_RMSNORM, fixture.context());
    ASSERT_TRUE(bound.has_value());

    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_RMSNORM.inputAToken), FLYDSL_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_RMSNORM.inputBToken), FLYDSL_SCALE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_EPSILON_TOKEN), FLYDSL_EPSILON_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_RMSNORM.outputToken), FLYDSL_Y_UID);

    // Shape, not uids: the two values the grid and the descriptors are built from. Rows is
    // the product of every axis before the normalised one, so 2x3 here and not 2.
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_ROWS_TOKEN), 6);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_COLUMNS_TOKEN), 4096);
}

TEST(TestFlydslRmsNormBinding, BindsInvRmsOnlyForTraining)
{
    const GraphFixture training(trainingGraph({2, 3, 4096}));
    const auto bound = matchesGraph(FLYDSL_RMSNORM, training.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_INV_RMS_TOKEN), FLYDSL_INV_RMS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_ROWS_TOKEN), 6);

    const GraphFixture inference(buildFlydslRmsNormGraph(DataType::BFLOAT16, {2, 3, 4096}));
    const auto inferenceBound = matchesGraph(FLYDSL_RMSNORM, inference.context());
    ASSERT_TRUE(inferenceBound.has_value());
    EXPECT_FALSE(tryGetBoundInt(*inferenceBound, FLYDSL_INV_RMS_TOKEN).has_value());
}

TEST(TestFlydslRmsNormBinding, CollapsesRank4UnderABroadcastShapedScale)
{
    // Rank 4 with a scale of (1,1,1,W) rather than the rank-1 {W} every other case uses.
    // The normalised axes are the trailing run where the scale matches x, here W alone, and
    // the scale is 1 on every axis before it, so leading unit axes are accepted the same
    // way a rank-1 scale is. Packed
    // strides here; the stride-0 form frameworks emit is the acceptance suite's
    // Rank4ScaleWithStrideZeroOnItsUnitAxes.
    // The three leading axes of x still collapse into the row count, 2x8x4.
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                       {2, 8, 4, 4096},
                                                       FLYDSL_TEST_EPSILON,
                                                       false,
                                                       std::vector<int64_t>{1, 1, 1, 4096},
                                                       std::nullopt,
                                                       std::nullopt,
                                                       false,
                                                       false,
                                                       NormFwdPhase::INFERENCE));

    const auto bound = matchesGraph(FLYDSL_RMSNORM, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_ROWS_TOKEN), 64);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_COLUMNS_TOKEN), 4096);
}

TEST(TestFlydslRmsNormBinding, SplitsTheCudnnLayoutAtTheNormalisedAxes)
{
    // Rows are the axes before gamma's trailing run, columns the run: 32 rows of 4096,
    // not 32*4096 rows of 1 as reading the innermost axis alone would give.
    RmsNormGraphExtras extras;
    extras.scaleStrides = {4096, 1, 4096, 4096};
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                       {32, 4096, 1, 1},
                                                       FLYDSL_TEST_EPSILON,
                                                       false,
                                                       std::vector<int64_t>{1, 4096, 1, 1},
                                                       std::nullopt,
                                                       std::vector<int64_t>{4096, 1, 4096, 4096},
                                                       false,
                                                       false,
                                                       NormFwdPhase::NOT_SET,
                                                       extras));

    const auto bound = matchesGraph(FLYDSL_RMSNORM, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_ROWS_TOKEN), 32);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_COLUMNS_TOKEN), 4096);
}

TEST(TestFlydslRmsNormBinding, JoinsAMultiAxisNormalisedShapeIntoOneRow)
{
    // F.rms_norm(x, (4, 1024)) on x [8, 4, 1024]: 8 rows of 4096, which the baked
    // N = 4096 instance then serves.
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                       {8, 4, 1024},
                                                       FLYDSL_TEST_EPSILON,
                                                       false,
                                                       std::vector<int64_t>{1, 4, 1024}));

    const auto bound = matchesGraph(FLYDSL_RMSNORM, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_ROWS_TOKEN), 8);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_COLUMNS_TOKEN), 4096);
    EXPECT_TRUE(
        matchesKernel(FLYDSL_RMSNORM, fixture.context(), makeFlydslKernel("bf16", 4096), *bound));
}

// ---------------------------------------------------------------------------
// graph_match: what this pack declines, and why each one matters
// ---------------------------------------------------------------------------

class TestFlydslRmsNormGraphMatcherRefusal : public ::testing::TestWithParam<GraphCase>
{
};

TEST_P(TestFlydslRmsNormGraphMatcherRefusal, Refuses)
{
    const GraphFixture fixture(GetParam().buildGraph());
    EXPECT_FALSE(matches(fixture.context()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslRmsNormGraphMatcherRefusal,
    ::testing::Values(
        // The kernels compute y = x * rsqrt(mean(x^2) + eps) * gamma and nothing else, so
        // an extra output or an extra addend is a different operation.
        GraphCase{"WithABias",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     true);
                  }},
        // inv_rms goes with TRAINING only: outside training it is an output cuDNN's
        // semantics do not produce.
        GraphCase{"SavingTheInverseRms",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     false,
                                                     true);
                  }},
        // Training needs the saved statistic for the backward pass even when the graph
        // does not name a tensor for it.
        GraphCase{"TrainingWithoutTheInverseRms",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     false,
                                                     false,
                                                     NormFwdPhase::TRAINING);
                  }},
        // The statistic as the kernel writes it: one f32 per row, dense.
        GraphCase{"InvRmsInHalf",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.invRmsDataType = DataType::HALF;
                      return trainingGraph({8, 4096}, extras);
                  }},
        GraphCase{"InvRmsPerColumn",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.invRmsDims = std::vector<int64_t>{1, 4096};
                      return trainingGraph({8, 4096}, extras);
                  }},
        GraphCase{"InvRmsOfLowerRank",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.invRmsDims = std::vector<int64_t>{8};
                      return trainingGraph({8, 4096}, extras);
                  }},
        GraphCase{"InvRmsStrided",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.invRmsStrides = std::vector<int64_t>{2, 1};
                      return trainingGraph({8, 4096}, extras);
                  }},
        GraphCase{"VirtualInvRms",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.virtualTensors.insert(FLYDSL_INV_RMS_UID);
                      return trainingGraph({8, 4096}, extras);
                  }},
        // The kernel stores the statistic through a pointer, densely: a pass-by-value
        // inv_rms has no buffer, and a ragged one's rows are not the dense rows written.
        GraphCase{"PassByValueInvRms",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.passByValueTensors.insert(FLYDSL_INV_RMS_UID);
                      return trainingGraph({8, 4096}, extras);
                  }},
        GraphCase{"RaggedInvRms",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.raggedTensors.insert(FLYDSL_INV_RMS_UID);
                      return trainingGraph({8, 4096}, extras);
                  }},
        GraphCase{"InvRmsAlignedTo2",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.alignment[FLYDSL_INV_RMS_UID] = 2;
                      return trainingGraph({8, 4096}, extras);
                  }},
        // x and y are each one buffer view with 32-bit byte offsets: 2^30 elements of bf16
        // (2 GiB) is one byte past it.
        GraphCase{"PastThirtyTwoBitAddressing",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {1 << 20, 1 << 10}); }},
        // Every field the graph carries, each value the kernels do not handle: a phase the
        // schema may add later, compute in double or integers, an alignment below their
        // 128-bit copies, operands that are not plain dense device tensors, an epsilon of
        // more than one element.
        GraphCase{"UnnamedPhase",
                  [] { return rmsNormGraphWith({}, static_cast<NormFwdPhase>(7)); }},
        GraphCase{"ComputeTypeDouble",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.computeDataType = DataType::DOUBLE;
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"ComputeTypeInt32",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.computeDataType = DataType::INT32;
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"XAlignedTo8",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.alignment[FLYDSL_X_UID] = 8;
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"GammaAlignedTo4",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.alignment[FLYDSL_SCALE_UID] = 4;
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"YAlignedTo2",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.alignment[FLYDSL_Y_UID] = 2;
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"VirtualX",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.virtualTensors.insert(FLYDSL_X_UID);
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"VirtualY",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.virtualTensors.insert(FLYDSL_Y_UID);
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"RaggedX",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.raggedTensors.insert(FLYDSL_X_UID);
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"RaggedGamma",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.raggedTensors.insert(FLYDSL_SCALE_UID);
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"EpsilonOfTwoElements",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.epsilonDims = {2};
                      return rmsNormGraphWith(extras);
                  }},
        // Only bf16, f16 and f32 were built, and there is no converting path.
        GraphCase{"AnUnsupportedDtype", [] { return buildFlydslRmsNormGraph(DataType::DOUBLE); }},
        // The 32-bit addressing bound counts bytes, so f32 reaches it at half the elements:
        // 2^29 of them is 2 GiB, one byte past it, where bf16 would still fit.
        GraphCase{"FloatPastThirtyTwoBitAddressing",
                  [] { return buildFlydslRmsNormGraph(DataType::FLOAT, {1 << 19, 1 << 10}); }},
        // Rows and width are read from the graph's shapes at prepare, which an
        // override-shape graph may replace at execute.
        GraphCase{"OverrideShapeEnabled",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.overrideShapeEnabled = true;
                      return rmsNormGraphWith(extras);
                  }},
        // Desc2D could carry this row stride, but the launch passes the width; until it
        // passes the graph's stride, a padded row is declined rather than misread.
        GraphCase{"PaddedRowStride",
                  [] {
                      return rmsNormGraphWithShapes(
                          {8, 4096}, {4096}, {}, std::vector<int64_t>{4160, 1});
                  }},
        // The unit-axis exemption does not reach the axes around it: the outer axis still
        // has to step over exactly one row.
        GraphCase{"NonContiguousLeadingAxes",
                  [] {
                      return rmsNormGraphWithShapes(
                          {2, 1, 4096}, {4096}, {}, std::vector<int64_t>{12288, 4096, 1});
                  }},
        // The frontend's sandwich case: gamma matches x on the innermost axis only, and is
        // not 1 on the axis before it, so it is a broadcast the kernel does not perform.
        GraphCase{"SandwichScale",
                  [] { return rmsNormGraphWithShapes({2, 2, 3, 3}, {1, 2, 1, 3}, {}); }},
        // More axes than x has: nothing to align it with.
        GraphCase{"ScaleOfHigherRank",
                  [] { return rmsNormGraphWithShapes({8, 4096}, {1, 1, 4096}, {}); }},
        // Gamma is read with unit stride along the normalised axis.
        GraphCase{"ScaleStridedAlongTheAxis",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.scaleStrides = {2};
                      return rmsNormGraphWithShapes({8, 4096}, {4096}, extras);
                  }},
        // Gamma is read at x's type or as f32, never as another 16-bit type.
        GraphCase{"GammaAtADifferentDtype",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     DataType::HALF);
                  }},
        // Rank one has no row to reduce over.
        GraphCase{"Rank1Input", [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {4096}); }},
        // Desc2D carries one row stride and an implied innermost stride of 1, so a
        // transposed or padded operand cannot be described at all.
        GraphCase{"NonPackedStrides",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::vector<int64_t>{1, 8});
                  }},
        // The same rank-4 shape CollapsesRank4UnderABroadcastShapedScale accepts, laid out
        // channels-last. This is the case that makes the layout contract load-bearing:
        // TensorAttributes::dims() carries LOGICAL order whatever the layout -- hipDNN
        // expresses channels-last purely in the strides (generateStrides() permutes only
        // those, and Utils.cpp's isChannelLastLayout() infers the layout back out of them)
        // -- so a matcher that read dims() alone would happily collapse N*C*H and normalise
        // over W while memory has C innermost. That is a silent wrong answer, not a slow
        // one. isPackedRowMajor() is what stops it: it walks from the innermost logical
        // axis expecting stride 1, and here W's stride is C. Overriding x alone is enough
        // because the guard runs per operand.
        GraphCase{"ChannelsLastStrides",
                  [] {
                      const std::vector<int64_t> dims{2, 8, 4, 4096};
                      return buildFlydslRmsNormGraph(
                          DataType::BFLOAT16,
                          dims,
                          FLYDSL_TEST_EPSILON,
                          false,
                          std::vector<int64_t>{1, 1, 1, 4096},
                          std::nullopt,
                          hipdnn_data_sdk::utilities::generateStrides(
                              dims, hipdnn_data_sdk::utilities::strideOrderNhwc(dims.size())));
                  }},
        // Gamma is indexed along the normalised axis with no broadcast, so it must be
        // exactly that wide...
        GraphCase{"GammaNarrowerThanTheAxis",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::vector<int64_t>{1});
                  }},
        // ...and one vector, not one per row.
        GraphCase{"GammaWithARowPerInputRow",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::vector<int64_t>{8, 4096});
                  }},
        // An epsilon no norm is computed with: every output would be NaN or infinite.
        GraphCase{"ANegativeEpsilon",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}, -1e-5F); }},
        GraphCase{"ANaNEpsilon",
                  [] {
                      return buildFlydslRmsNormGraph(
                          DataType::BFLOAT16, {8, 4096}, std::numeric_limits<float>::quiet_NaN());
                  }},
        GraphCase{"AnInfiniteEpsilon",
                  [] {
                      return buildFlydslRmsNormGraph(
                          DataType::BFLOAT16, {8, 4096}, std::numeric_limits<float>::infinity());
                  }},
        // A finite f64 epsilon no f32 holds: the kernel takes epsilon as an f32, which
        // would not be the value asked for (and is +inf further out). The acceptance
        // suite's AnF64EpsilonAtTheF32Maximum is the double just below this one.
        GraphCase{"AnF64EpsilonPastTheF32Range",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.epsilonF64
                          = std::nextafter(static_cast<double>(std::numeric_limits<float>::max()),
                                           std::numeric_limits<double>::infinity());
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"AnF64EpsilonFarPastTheF32Range",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.epsilonF64 = 1e300;
                      return rmsNormGraphWith(extras);
                  }},
        // The mixed form is a 16-bit x with an f32 gamma, never the reverse: there is no
        // object that reads a 16-bit gamma beside f32 x.
        GraphCase{"FloatWithABfloat16Scale",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::FLOAT,
                                                     {8, 4096},
                                                     FLYDSL_TEST_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     DataType::BFLOAT16);
                  }},
        // y is written in x's element type, over x's rows, packed: an f32 y would receive
        // bf16 bits, a smaller y would be written past its end, and a padded one as if it
        // were dense.
        GraphCase{"YOfAnotherDtype",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.yDataType = DataType::FLOAT;
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"YOfAnotherShape",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.yDims = std::vector<int64_t>{8, 2048};
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"PaddedRowsInY",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.yStrides = std::vector<int64_t>{4160, 1};
                      return rmsNormGraphWith(extras);
                  }},
        // x, gamma and y are read and written through device pointers; one carrying a host
        // value is a scalar, not an operand.
        GraphCase{"PassByValueX",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.passByValueTensors.insert(FLYDSL_X_UID);
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"PassByValueGamma",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.passByValueTensors.insert(FLYDSL_SCALE_UID);
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"PassByValueY",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.passByValueTensors.insert(FLYDSL_Y_UID);
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"VirtualGamma",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.virtualTensors.insert(FLYDSL_SCALE_UID);
                      return rmsNormGraphWith(extras);
                  }},
        GraphCase{"RaggedY",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.raggedTensors.insert(FLYDSL_Y_UID);
                      return rmsNormGraphWith(extras);
                  }},
        // One kernel serves one complete graph; a second node would be dropped.
        GraphCase{"TwoNodes",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.secondNode = true;
                      return rmsNormGraphWith(extras);
                  }},
        // At least 16 bytes is not enough: the 128-bit copies need a multiple of 16.
        GraphCase{"XAlignedTo24",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.alignment[FLYDSL_X_UID] = 24;
                      return rmsNormGraphWith(extras);
                  }},
        // A strides vector longer than the shape describes no tensor, even when its leading
        // entries happen to be the packed ones.
        GraphCase{"StridesLongerThanTheShape",
                  [] {
                      return rmsNormGraphWithShapes(
                          {8, 4096}, {4096}, {}, std::vector<int64_t>{4096, 1, 1});
                  }},
        // inv_rms is written one f32 at a time: at least 4-byte aligned is not enough, it
        // has to be a multiple of 4.
        GraphCase{"InvRmsAlignedTo6",
                  [] {
                      RmsNormGraphExtras extras;
                      extras.alignment[FLYDSL_INV_RMS_UID] = 6;
                      return trainingGraph({8, 4096}, extras);
                  }}),
    graphCaseName);

// ---------------------------------------------------------------------------
// kernel_match: the two tiers
// ---------------------------------------------------------------------------

TEST(TestFlydslRmsNormKernelMatcher, AdmitsTheKernelBakedForThisWidth)
{
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}));
    const auto bound = bindingsFor(fixture.context());

    EXPECT_TRUE(
        matchesKernel(FLYDSL_RMSNORM, fixture.context(), makeFlydslKernel("bf16", 4096), bound));
}

TEST(TestFlydslRmsNormKernelMatcher, RefusesAKernelBakedForAnotherWidth)
{
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}));
    const auto bound = bindingsFor(fixture.context());

    // Baking N fixes the loop trip count: a wider instance would read past the row, and a
    // narrower one would normalise only its first N columns and leave the rest unwritten.
    // Both directions, because every baked instance carries the same priority and so only
    // this rule keeps a narrower one out of the candidates.
    EXPECT_FALSE(
        matchesKernel(FLYDSL_RMSNORM, fixture.context(), makeFlydslKernel("bf16", 8192), bound));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_RMSNORM, fixture.context(), makeFlydslKernel("bf16", 2560), bound));
}

TEST(TestFlydslRmsNormKernelMatcher, TheGenericTierAdmitsAWidthNothingWasBakedFor)
{
    // 640 is not among the packed instances. Without the sentinel kernel this graph would
    // have no candidate at all, which is the whole reason the tier exists.
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 640}));
    const auto bound = bindingsFor(fixture.context());

    EXPECT_TRUE(matchesKernel(FLYDSL_RMSNORM,
                              fixture.context(),
                              makeFlydslKernel("bf16", FLYDSL_GENERIC_N, 256, 10),
                              bound));
}

TEST(TestFlydslRmsNormKernelMatcher, PairsAnF32ScaleWithTheObjectsThatReadOne)
{
    const GraphFixture mixed(buildFlydslRmsNormGraph(
        DataType::BFLOAT16, {8, 4096}, FLYDSL_TEST_EPSILON, false, std::nullopt, DataType::FLOAT));
    const auto mixedBound = bindingsFor(mixed.context());

    // An object reading gamma as bf16 would read the f32 vector as twice as many halves.
    EXPECT_TRUE(matchesKernel(
        FLYDSL_RMSNORM, mixed.context(), makeFlydslKernel("bf16", 4096, 256, 100, 1), mixedBound));
    EXPECT_FALSE(matchesKernel(
        FLYDSL_RMSNORM, mixed.context(), makeFlydslKernel("bf16", 4096, 256, 100, 0), mixedBound));
    EXPECT_TRUE(matchesKernel(FLYDSL_RMSNORM,
                              mixed.context(),
                              makeFlydslKernel("bf16", FLYDSL_GENERIC_N, 256, 10, 1),
                              mixedBound));

    // And the reverse: a bf16 gamma must not reach an object that reads f32.
    const GraphFixture plain(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}));
    const auto plainBound = bindingsFor(plain.context());
    EXPECT_FALSE(matchesKernel(
        FLYDSL_RMSNORM, plain.context(), makeFlydslKernel("bf16", 4096, 256, 100, 1), plainBound));
}

TEST(TestFlydslRmsNormKernelMatcher, PairsTrainingWithTheObjectsThatWriteInvRms)
{
    const GraphFixture training(trainingGraph());
    const auto trainingBound = bindingsFor(training.context());
    EXPECT_TRUE(matchesKernel(FLYDSL_RMSNORM,
                              training.context(),
                              makeFlydslKernel("bf16", 4096, 256, 100, 0, 1),
                              trainingBound));
    // An inference object would leave inv_rms unwritten.
    EXPECT_FALSE(matchesKernel(FLYDSL_RMSNORM,
                               training.context(),
                               makeFlydslKernel("bf16", 4096, 256, 100, 0, 0),
                               trainingBound));

    // And a training object on an inference graph would store through gamma's pointer.
    const GraphFixture inference(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}));
    EXPECT_FALSE(matchesKernel(FLYDSL_RMSNORM,
                               inference.context(),
                               makeFlydslKernel("bf16", 4096, 256, 100, 0, 1),
                               bindingsFor(inference.context())));
}

TEST(TestFlydslRmsNormKernelMatcher, RefusesAKernelOfTheWrongDtype)
{
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}));
    const auto bound = bindingsFor(fixture.context());

    EXPECT_FALSE(
        matchesKernel(FLYDSL_RMSNORM, fixture.context(), makeFlydslKernel("f16", 4096), bound));
    // Including the sentinel kernel: generic in N says nothing about the element type.
    EXPECT_FALSE(matchesKernel(FLYDSL_RMSNORM,
                               fixture.context(),
                               makeFlydslKernel("f16", FLYDSL_GENERIC_N, 256, 10),
                               bound));
}

TEST(TestFlydslRmsNormKernelMatcher, ReadsTheDescriptorDtypeVocabularyNotTheFlatbufferEnum)
{
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}));
    const auto bound = bindingsFor(fixture.context());

    // EnumNameDataType would spell this "BFLOAT16". The descriptors say "bf16", and the
    // pack translates deliberately rather than accepting both.
    EXPECT_FALSE(matchesKernel(
        FLYDSL_RMSNORM, fixture.context(), makeFlydslKernel("BFLOAT16", 4096), bound));
}

// ---------------------------------------------------------------------------
// score: which tier wins when both match
// ---------------------------------------------------------------------------

TEST(TestFlydslRmsNormScore, PrefersTheBakedInstanceOverTheGenericOne)
{
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}));
    const auto bound = bindingsFor(fixture.context());

    const auto baked = makeFlydslKernel("bf16", 4096, 256, 100);
    const auto generic = makeFlydslKernel("bf16", FLYDSL_GENERIC_N, 256, 10);

    // Both are applicable for 4096 -- the tie is broken here, not in the matcher.
    ASSERT_TRUE(matchesKernel(FLYDSL_RMSNORM, fixture.context(), baked, bound));
    ASSERT_TRUE(matchesKernel(FLYDSL_RMSNORM, fixture.context(), generic, bound));

    EXPECT_GT(scoreKernel(FLYDSL_RMSNORM, fixture.context(), baked, bound),
              scoreKernel(FLYDSL_RMSNORM, fixture.context(), generic, bound));
}

TEST(TestFlydslRmsNormScore, IsTheDescriptorPriorityAndNothingElse)
{
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}));
    const auto bound = bindingsFor(fixture.context());

    // Nothing measured, nothing shape-dependent: the ordering is authored, and the
    // autotuner is what refines it. A score derived from the shape instead would make the
    // winner cache's key insufficient.
    EXPECT_DOUBLE_EQ(
        scoreKernel(
            FLYDSL_RMSNORM, fixture.context(), makeFlydslKernel("bf16", 4096, 256, 7), bound),
        7.0);
}

// ---------------------------------------------------------------------------
// dispatch: the seams that need no device
// ---------------------------------------------------------------------------

TEST(TestFlydslRmsNormDispatch, ResolvesFromTheDispatchRegistry)
{
    EXPECT_NO_THROW(static_cast<void>(dispatchHandler(FLYDSL_RMSNORM)));
}

TEST(TestFlydslRmsNormDispatch, NeedsNoWorkspaceAtEitherTier)
{
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}));
    const auto& handler = dispatchHandler(FLYDSL_RMSNORM);
    const auto bound = bindingsFor(fixture.context());

    // The reduction is entirely in-block: one row per workgroup, LDS only. Answered from
    // metadata, so this reaches no loader and needs no device.
    EXPECT_EQ(handler.workspaceBytes(fixture.context(), bound, makeFlydslKernel("bf16", 4096)), 0U);
    EXPECT_EQ(handler.workspaceBytes(
                  fixture.context(), bound, makeFlydslKernel("bf16", FLYDSL_GENERIC_N, 256, 10)),
              0U);
}

TEST(TestFlydslRmsNormDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}));
    const auto& handler = dispatchHandler(FLYDSL_RMSNORM);

    // Binding lookup throws on a missing token before anything opens an archive.
    EXPECT_THROW(handler.prepare(fixture.context(), BoundTokens{}, makeFlydslKernel("bf16", 4096)),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---------------------------------------------------------------------------
// dispatch on device: a real launch out of the staged archive
// ---------------------------------------------------------------------------

/// Host and device x, gamma and y for one RMS-norm, freed on scope exit.
///
/// x is filled with exactly-representable values so the only rounding in the comparison is
/// the kernel's own: the reference is computed from the SAME bf16/f16 values the device
/// reads, which keeps input quantisation out of the error budget. Gamma is held as @p G,
/// which is T unless the case gives a 16-bit x an f32 gamma.
///
/// Each row is scaled by its own power of two, so every row has a different RMS: a kernel
/// that normalised every row by one row's statistic, or wrote inv_rms to the wrong row,
/// would still match a reference in which all rows agree.
template <typename T, typename G = T>
class RmsNormBuffers
{
public:
    /// @param training Also allocate inv_rms (one f32 per row). It and y start as NaN, so
    /// an element the kernel does not write fails the comparison.
    RmsNormBuffers(int64_t rows, int64_t columns, bool training = false)
        : _rows(rows)
        , _columns(columns)
        , _x(static_cast<size_t>(rows * columns))
        , _gamma(static_cast<size_t>(columns))
    {
        for(int64_t row = 0; row < rows; ++row)
        {
            for(int64_t column = 0; column < columns; ++column)
            {
                // Eighths times a power of two, so every value is exact in bf16's 8
                // mantissa bits.
                const float value
                    = std::ldexp(0.5F + static_cast<float>((row + column) % 8) * 0.125F,
                                 static_cast<int>(row % 4) - 1);
                _x[static_cast<size_t>(row * columns + column)] = T(value);
            }
        }
        for(int64_t column = 0; column < columns; ++column)
        {
            _gamma[static_cast<size_t>(column)] = G(0.5F + static_cast<float>(column % 4) * 0.25F);
        }

        EXPECT_EQ(hipSuccess, hipMalloc(&_xDevice, _x.size() * sizeof(T)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_gammaDevice, _gamma.size() * sizeof(G)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_yDevice, _x.size() * sizeof(T)));
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(_xDevice, _x.data(), _x.size() * sizeof(T), hipMemcpyHostToDevice));
        EXPECT_EQ(
            hipSuccess,
            hipMemcpy(
                _gammaDevice, _gamma.data(), _gamma.size() * sizeof(G), hipMemcpyHostToDevice));
        EXPECT_EQ(hipSuccess, hipMemset(_yDevice, 0xFF, _x.size() * sizeof(T)));
        if(training)
        {
            const auto bytes = static_cast<size_t>(rows) * sizeof(float);
            EXPECT_EQ(hipSuccess, hipMalloc(&_invRmsDevice, bytes));
            EXPECT_EQ(hipSuccess, hipMemset(_invRmsDevice, 0xFF, bytes));
        }
    }

    ~RmsNormBuffers()
    {
        static_cast<void>(hipFree(_xDevice));
        static_cast<void>(hipFree(_gammaDevice));
        static_cast<void>(hipFree(_yDevice));
        static_cast<void>(hipFree(_invRmsDevice));
    }

    RmsNormBuffers(const RmsNormBuffers&) = delete;
    RmsNormBuffers& operator=(const RmsNormBuffers&) = delete;

    std::vector<hipdnnPluginDeviceBuffer_t> descriptors() const
    {
        std::vector<hipdnnPluginDeviceBuffer_t> buffers{
            hipdnnPluginDeviceBuffer_t{FLYDSL_X_UID, _xDevice},
            hipdnnPluginDeviceBuffer_t{FLYDSL_SCALE_UID, _gammaDevice},
            hipdnnPluginDeviceBuffer_t{FLYDSL_Y_UID, _yDevice}};
        if(_invRmsDevice != nullptr)
        {
            buffers.push_back(hipdnnPluginDeviceBuffer_t{FLYDSL_INV_RMS_UID, _invRmsDevice});
        }
        return buffers;
    }

    std::vector<float> readInvRms() const
    {
        std::vector<float> result(static_cast<size_t>(_rows));
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(result.data(),
                            _invRmsDevice,
                            result.size() * sizeof(float),
                            hipMemcpyDeviceToHost));
        return result;
    }

    /// inv_rms = 1 / sqrt(mean(x^2) + eps) per row, in double from the quantised inputs.
    std::vector<double> referenceInvRms(double epsilon) const
    {
        std::vector<double> expected(static_cast<size_t>(_rows));
        for(int64_t row = 0; row < _rows; ++row)
        {
            double sumOfSquares = 0.0;
            for(int64_t column = 0; column < _columns; ++column)
            {
                const auto value = static_cast<double>(
                    static_cast<float>(_x[static_cast<size_t>(row * _columns + column)]));
                sumOfSquares += value * value;
            }
            expected[static_cast<size_t>(row)]
                = 1.0 / std::sqrt(sumOfSquares / static_cast<double>(_columns) + epsilon);
        }
        return expected;
    }

    std::vector<T> readOutput() const
    {
        std::vector<T> result(_x.size());
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(result.data(), _yDevice, _x.size() * sizeof(T), hipMemcpyDeviceToHost));
        return result;
    }

    /// y = x * rsqrt(mean(x^2) + eps) * gamma, in double from the quantised inputs.
    std::vector<double> reference(double epsilon) const
    {
        std::vector<double> expected(_x.size());
        for(int64_t row = 0; row < _rows; ++row)
        {
            double sumOfSquares = 0.0;
            for(int64_t column = 0; column < _columns; ++column)
            {
                const auto value = static_cast<double>(
                    static_cast<float>(_x[static_cast<size_t>(row * _columns + column)]));
                sumOfSquares += value * value;
            }
            const double scale
                = 1.0 / std::sqrt(sumOfSquares / static_cast<double>(_columns) + epsilon);
            for(int64_t column = 0; column < _columns; ++column)
            {
                const auto index = static_cast<size_t>(row * _columns + column);
                expected[index] = static_cast<double>(static_cast<float>(_x[index])) * scale
                                  * static_cast<double>(
                                      static_cast<float>(_gamma[static_cast<size_t>(column)]));
            }
        }
        return expected;
    }

private:
    int64_t _rows;
    int64_t _columns;
    std::vector<T> _x;
    std::vector<G> _gamma;
    void* _xDevice = nullptr;
    void* _gammaDevice = nullptr;
    void* _yDevice = nullptr;
    void* _invRmsDevice = nullptr;
};

/// How a launch case describes its operands and epsilon, beyond the dtype and the
/// rows x columns its buffers hold. The shapes and strides must split into those same
/// rows x columns, so the buffers are the same packed arrays either way.
struct RmsNormLaunchCase
{
    std::optional<std::vector<int64_t>> scaleDims;
    RmsNormGraphExtras extras;
    std::optional<std::vector<int64_t>> xDims;
    float epsilon = FLYDSL_TEST_EPSILON;
    /// Supply epsilon at execute rather than fixing it in the graph.
    bool runtimeEpsilon = false;
    /// A TRAINING graph that also saves inv_rms, served by a `store_rstd` object.
    bool training = false;
};

/// Runs the staged (@p dtype, @p bakedN) kernel over a @p rows x @p columns graph and
/// checks every element against the host reference, as @p launchCase describes the graph.
///
/// Fails when the staged set holds no such kernel on a device this build ships RMSNorm
/// for, and skips elsewhere: on an arch this build packed nothing for, there is no kernel
/// to be wrong about.
template <typename T, typename G = T>
void expectRmsNormMatchesReference(DataType dataType,
                                   const std::string& dtype,
                                   int64_t bakedN,
                                   int64_t rows,
                                   int64_t columns,
                                   double relativeTolerance,
                                   const RmsNormLaunchCase& launchCase = {})
{
    constexpr bool F32_SCALE = std::is_same_v<G, float> && !std::is_same_v<T, float>;
    KernelDefinition kernel;
    if(!findStagedKernel(dtype, bakedN, kernel, F32_SCALE ? 1 : 0, launchCase.training ? 1 : 0))
    {
        reportUnstaged("rmsnorm",
                       "nothing staged for dtype=" + dtype + " N=" + std::to_string(bakedN)
                           + " scale_f32=" + std::to_string(F32_SCALE ? 1 : 0)
                           + " store_rstd=" + std::to_string(launchCase.training ? 1 : 0));
        return;
    }

    const GraphFixture fixture(
        buildFlydslRmsNormGraph(dataType,
                                launchCase.xDims.value_or(std::vector<int64_t>{rows, columns}),
                                launchCase.epsilon,
                                launchCase.runtimeEpsilon,
                                launchCase.scaleDims,
                                F32_SCALE ? std::optional<DataType>(DataType::FLOAT) : std::nullopt,
                                std::nullopt,
                                false,
                                launchCase.training,
                                launchCase.training ? NormFwdPhase::TRAINING
                                                    : NormFwdPhase::NOT_SET,
                                launchCase.extras),
        packedArchDeviceProperties());
    const auto& handler = dispatchHandler(FLYDSL_RMSNORM);

    const auto prepared
        = handler.prepare(fixture.context(), bindingsFor(fixture.context()), kernel);
    ASSERT_NE(prepared, nullptr);

    const RmsNormBuffers<T, G> buffers(rows, columns, launchCase.training);
    const auto descriptors = buffers.descriptors();
    std::vector<hipdnnPluginDeviceBuffer_t> arguments(descriptors.begin(), descriptors.end());
    // A pass-by-value scalar is read from host memory, so this slot carries a pointer into
    // the stack.
    float epsilon = launchCase.epsilon;
    if(launchCase.runtimeEpsilon)
    {
        arguments.push_back(hipdnnPluginDeviceBuffer_t{FLYDSL_EPSILON_UID, &epsilon});
    }
    const Handle handle;

    handler.launch(
        handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    const auto actual = buffers.readOutput();
    const auto expected = buffers.reference(static_cast<double>(launchCase.epsilon));
    ASSERT_EQ(actual.size(), expected.size());

    const auto worst = worstRelativeError(actual, expected, 1e-3);
    EXPECT_LE(worst.error, relativeTolerance)
        << "at element " << worst.index << ": got " << static_cast<float>(actual[worst.index])
        << ", expected " << expected[worst.index];

    if(launchCase.training)
    {
        // f32 out of an f32 reduction and a hardware rsqrt: 1e-4 relative is far above
        // either's error and far below a wrong row, a wrong epsilon or an unwritten NaN.
        const auto invRms = buffers.readInvRms();
        const auto worstInvRms = worstRelativeError(
            invRms, buffers.referenceInvRms(static_cast<double>(launchCase.epsilon)), 0.0);
        EXPECT_LE(worstInvRms.error, 1e-4)
            << "inv_rms of row " << worstInvRms.index << ": got " << invRms[worstInvRms.index];
    }
}

TEST(TestGpuFlydslRmsNormDispatch, ComputesTheTrainingForward)
{
    SKIP_IF_NO_DEVICES();

    // y and inv_rms, at a baked width, through the generic tier, in f32, and with an f32
    // gamma beside bf16; the last also with an epsilon large beside the first rows' mean
    // squares, so inv_rms tells a kernel that ignored it apart.
    RmsNormLaunchCase launchCase;
    launchCase.training = true;
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", 4096, 4, 4096, 2e-2, launchCase);
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::half>(
        DataType::HALF, "f16", FLYDSL_GENERIC_N, 5, 640, 5e-3, launchCase);
    expectRmsNormMatchesReference<float>(
        DataType::FLOAT, "f32", FLYDSL_GENERIC_N, 3, 1000, 1e-4, launchCase);
    launchCase.epsilon = 0.5F;
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16, float>(
        DataType::BFLOAT16, "bf16", 8192, 4, 8192, 2e-2, launchCase);
}

TEST(TestGpuFlydslRmsNormDispatch, ComputesBf16AtABakedWidth)
{
    SKIP_IF_NO_DEVICES();

    // bf16 keeps 8 mantissa bits, so ~4e-3 relative is the type's own floor; 2e-2 leaves
    // room for the kernel's fp32 reduction order differing from the reference's.
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", 4096, 4, 4096, 2e-2);
}

TEST(TestGpuFlydslRmsNormDispatch, ComputesF16AtABakedWidth)
{
    SKIP_IF_NO_DEVICES();

    expectRmsNormMatchesReference<hipdnn_data_sdk::types::half>(
        DataType::HALF, "f16", 4096, 4, 4096, 5e-3);
    // DeepSeek-V3's hidden size, one of the widths baked by measurement.
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::half>(
        DataType::HALF, "f16", 7168, 4, 7168, 5e-3);
}

TEST(TestGpuFlydslRmsNormDispatch, ComputesF32WithTheGenericKernel)
{
    SKIP_IF_NO_DEVICES();

    // f32 in and out, so the only error is the kernel's f32 reduction against the
    // double reference: far inside 1e-4 at either width.
    expectRmsNormMatchesReference<float>(DataType::FLOAT, "f32", FLYDSL_GENERIC_N, 4, 4096, 1e-4);
    expectRmsNormMatchesReference<float>(DataType::FLOAT, "f32", FLYDSL_GENERIC_N, 4, 640, 1e-4);
}

TEST(TestGpuFlydslRmsNormDispatch, ComputesWithGammaDescribedAsHipdnnTorchDescribesIt)
{
    SKIP_IF_NO_DEVICES();

    // A (1, W) gamma with stride 0 on its unit axis, the form hipdnn_torch gives an
    // F.rms_norm weight: the same vector in memory, so the same answer.
    RmsNormLaunchCase launchCase;
    launchCase.scaleDims = std::vector<int64_t>{1, 4096};
    launchCase.extras.scaleStrides = {0, 1};
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", 4096, 4, 4096, 2e-2, launchCase);
}

TEST(TestGpuFlydslRmsNormDispatch, ComputesTheCudnnLayout)
{
    SKIP_IF_NO_DEVICES();

    // x {4, 4096, 1, 1} and gamma {1, 4096, 1, 1}, gamma with cuDNN's strides: 4 rows of
    // 4096.
    RmsNormLaunchCase launchCase;
    launchCase.scaleDims = std::vector<int64_t>{1, 4096, 1, 1};
    launchCase.extras.scaleStrides = {4096, 1, 4096, 4096};
    launchCase.xDims = std::vector<int64_t>{4, 4096, 1, 1};
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", 4096, 4, 4096, 2e-2, launchCase);
}

TEST(TestGpuFlydslRmsNormDispatch, ComputesAnF32ScaleBesideBf16AndF16)
{
    SKIP_IF_NO_DEVICES();

    // Gamma read as f32 by its own objects, at a baked width (the 128-bit fast path, with
    // gamma's own copies) and through the generic tier.
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16, float>(
        DataType::BFLOAT16, "bf16", 4096, 4, 4096, 2e-2);
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16, float>(
        DataType::BFLOAT16, "bf16", FLYDSL_GENERIC_N, 4, 640, 2e-2);
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::half, float>(
        DataType::HALF, "f16", 8192, 4, 8192, 5e-3);
}

TEST(TestGpuFlydslRmsNormDispatch, ComputesWithTheGraphsOwnEpsilon)
{
    SKIP_IF_NO_DEVICES();

    // The rows' mean squares run from about 0.24 to 15, so an epsilon of 0.5 moves the
    // first row's outputs by over 40%: a kernel that ignored it, or kept 1e-5, would fail
    // by far more than the tolerance. Zero is the other edge.
    for(const float epsilon : {0.5F, 0.0F})
    {
        SCOPED_TRACE(::testing::Message() << "epsilon " << epsilon);
        RmsNormLaunchCase launchCase;
        launchCase.epsilon = epsilon;
        expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
            DataType::BFLOAT16, "bf16", 4096, 4, 4096, 2e-2, launchCase);
        expectRmsNormMatchesReference<float>(
            DataType::FLOAT, "f32", FLYDSL_GENERIC_N, 4, 640, 1e-4, launchCase);
    }
}

TEST(TestGpuFlydslRmsNormDispatch, ComputesWithAnEpsilonSuppliedAtExecute)
{
    SKIP_IF_NO_DEVICES();

    RmsNormLaunchCase launchCase;
    launchCase.epsilon = 0.5F;
    launchCase.runtimeEpsilon = true;
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", 4096, 4, 4096, 2e-2, launchCase);
}

TEST(TestGpuFlydslRmsNormDispatch, TheGenericKernelComputesAWidthNothingWasBakedFor)
{
    SKIP_IF_NO_DEVICES();

    // The tier's reason for existing, on device: 640 is not a packed instance, and the
    // sentinel kernel takes the width as a kernarg instead.
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", FLYDSL_GENERIC_N, 4, 640, 2e-2);
}

TEST(TestGpuFlydslRmsNormDispatch, TheGenericKernelAgreesWithTheBakedOneAtABakedWidth)
{
    SKIP_IF_NO_DEVICES();

    // Same width, both tiers. The autotuner is free to pick either, so they have to be
    // interchangeable in everything except speed.
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", FLYDSL_GENERIC_N, 4, 4096, 2e-2);
}

// ---------------------------------------------------------------------------
// what the two tiers cost
// ---------------------------------------------------------------------------
//
// `score` prefers the baked instance over the generic one, and every case above asserts
// that ordering without ever checking that it buys anything. This is the missing half:
// one graph, run through both tiers, timed.
//
// It reports rather than ranks. On gfx1151 the baked tier wins at both shapes below, but
// which tier wins is a property of the device, and pinning that ordering here would make
// the suite fail on the first arch where it does not hold. What IS asserted is the sanity bound: both tiers
// complete, and neither is an order of magnitude off the other. A generic tier that had
// quietly lost its vectorized path would show up against that bound; a 40% difference
// would not, and belongs in a benchmark rather than a unit test.
//
// The two tiers differ in the baked width and in nothing else: both descriptors are
// `bt256`, so BLOCK_THREADS is held constant across the comparison.

/// Mean microseconds per launch of the staged (@p dtype, @p bakedN) kernel over a
/// @p rows x @p columns graph, or nullopt when this build packed no such kernel.
///
/// Launch-to-launch throughput on one stream, host dispatch included -- what a built plan
/// costs when it is executed repeatedly, which is how a plan is executed. It is not
/// isolated kernel time; both tiers pay the same host overhead, so the comparison between
/// them survives its inclusion even though the absolute number carries it.
template <typename T>
std::optional<double> timeRmsNormLaunches(
    DataType dataType, const std::string& dtype, int64_t bakedN, int64_t rows, int64_t columns)
{
    KernelDefinition kernel;
    if(!findStagedKernel(dtype, bakedN, kernel))
    {
        return std::nullopt;
    }

    const GraphFixture fixture(buildFlydslRmsNormGraph(dataType, {rows, columns}),
                               packedArchDeviceProperties());
    const auto& handler = dispatchHandler(FLYDSL_RMSNORM);

    const auto prepared
        = handler.prepare(fixture.context(), bindingsFor(fixture.context()), kernel);
    EXPECT_NE(prepared, nullptr);
    if(prepared == nullptr)
    {
        return std::nullopt;
    }

    const RmsNormBuffers<T> buffers(rows, columns);
    const auto descriptors = buffers.descriptors();
    const auto count = static_cast<uint32_t>(descriptors.size());
    const Handle handle;

    // Warm first: the first launch of a freshly loaded module pays code-object paging the
    // steady state does not, and attributing that to the tier would be reading a
    // one-off as a property of the kernel.
    const int warmupLaunches = 20;
    const int timedLaunches = 200;

    for(int i = 0; i < warmupLaunches; ++i)
    {
        handler.launch(handle, *prepared, descriptors.data(), count, nullptr);
    }
    EXPECT_EQ(hipSuccess, hipDeviceSynchronize());

    const auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < timedLaunches; ++i)
    {
        handler.launch(handle, *prepared, descriptors.data(), count, nullptr);
    }
    EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
    const auto elapsed = std::chrono::steady_clock::now() - start;

    return std::chrono::duration<double, std::micro>(elapsed).count() / timedLaunches;
}

/// The rows of the graph a tier case times.
class TestGpuFlydslRmsNormTiers : public ::testing::TestWithParam<int64_t>
{
};

TEST_P(TestGpuFlydslRmsNormTiers, CostTheBakedAndGenericTiersOverOneGraph)
{
    SKIP_IF_NO_DEVICES();

    const int64_t columns = 4096;
    const int64_t rows = GetParam();

    const auto baked = timeRmsNormLaunches<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", columns, rows, columns);
    const auto generic = timeRmsNormLaunches<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", FLYDSL_GENERIC_N, rows, columns);
    if(!baked.has_value() || !generic.has_value())
    {
        reportUnstaged("rmsnorm",
                       "nothing staged for a bf16 tier at N=" + std::to_string(columns)
                           + " or N=" + std::to_string(FLYDSL_GENERIC_N));
        return;
    }

    // Nanoseconds because RecordProperty carries integers, and a microsecond figure
    // rounded to int would quantise away most of the difference being reported.
    const auto shape = std::to_string(rows) + "x" + std::to_string(columns);
    RecordProperty("baked_" + shape + "_nsPerLaunch", static_cast<int>(*baked * 1000.0));
    RecordProperty("generic_" + shape + "_nsPerLaunch", static_cast<int>(*generic * 1000.0));
    GTEST_LOG_(INFO) << shape << " bf16: baked N=" << columns << " " << *baked
                     << " us/launch, generic " << *generic << " us/launch, ratio "
                     << "generic/baked " << (*generic / *baked);

    EXPECT_GT(*baked, 0.0);
    EXPECT_GT(*generic, 0.0);
    EXPECT_LT(*generic, *baked * 10.0) << "the generic tier lost more than an order of "
                                          "magnitude against the baked one; that is a "
                                          "collapsed code path, not a tuning difference";
    EXPECT_LT(*baked, *generic * 10.0) << "the baked tier lost more than an order of "
                                          "magnitude against the generic one, which is "
                                          "backwards for the tier `score` prefers";
}

// Two shapes, because either one alone would say something the other contradicts. Four
// rows is four workgroups: latency-bound, nothing to hide an instruction count behind, and
// the tier difference shows at its widest. 4096 rows moves 64 MiB per launch and runs near
// the memory ceiling, which compresses the same difference without erasing it.
//
// The `Full` tier only: the bound is on wall-clock ratios, which a loaded, shared CI device
// can distort, and the case answers a performance question no pre-merge run needs answered.
INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuFlydslRmsNormTiers,
                         ::testing::Values(int64_t{4}, int64_t{4096}),
                         [](const ::testing::TestParamInfo<int64_t>& info) {
                             return "Rows" + std::to_string(info.param);
                         });

/// The other half of the epsilon contract. An unusable baked epsilon is declined at
/// graph_match so another engine takes the graph; one the caller only supplies at execute
/// has no such escape -- the plan is already built -- so launch() must refuse it with
/// INVALID_VALUE rather than hand it to the kernel.
void expectLaunchRefusesAnEpsilonSuppliedAtExecute(float epsilon)
{
    KernelDefinition kernel;
    if(!findStagedKernel("bf16", 4096, kernel))
    {
        reportUnstaged("rmsnorm", "nothing staged for bf16 N=4096");
        return;
    }

    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {4, 4096}, 0.0F, true),
                               packedArchDeviceProperties());
    ASSERT_TRUE(matches(fixture.context())) << "a runtime epsilon must still pass graph_match";

    const auto& handler = dispatchHandler(FLYDSL_RMSNORM);
    const auto prepared
        = handler.prepare(fixture.context(), bindingsFor(fixture.context()), kernel);
    ASSERT_NE(prepared, nullptr);

    const RmsNormBuffers<hipdnn_data_sdk::types::bfloat16> buffers(4, 4096);
    const auto descriptors = buffers.descriptors();
    std::vector<hipdnnPluginDeviceBuffer_t> withEpsilon(descriptors.begin(), descriptors.end());

    // A pass-by-value scalar is read from host memory, not copied off the device, so this
    // slot legitimately carries a pointer into the stack.
    withEpsilon.push_back(hipdnnPluginDeviceBuffer_t{FLYDSL_EPSILON_UID, &epsilon});

    const Handle handle;
    try
    {
        handler.launch(handle,
                       *prepared,
                       withEpsilon.data(),
                       static_cast<uint32_t>(withEpsilon.size()),
                       nullptr);
        ADD_FAILURE() << "launch() accepted the execute-time epsilon " << epsilon;
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE) << error.what();
    }
}

TEST(TestGpuFlydslRmsNormDispatch, RejectsANegativeEpsilonSuppliedAtExecute)
{
    SKIP_IF_NO_DEVICES();
    expectLaunchRefusesAnEpsilonSuppliedAtExecute(-1e-2F);
}

TEST(TestGpuFlydslRmsNormDispatch, RejectsANaNEpsilonSuppliedAtExecute)
{
    SKIP_IF_NO_DEVICES();
    expectLaunchRefusesAnEpsilonSuppliedAtExecute(std::numeric_limits<float>::quiet_NaN());
}

TEST(TestGpuFlydslRmsNormDispatch, RejectsAnInfiniteEpsilonSuppliedAtExecute)
{
    SKIP_IF_NO_DEVICES();
    expectLaunchRefusesAnEpsilonSuppliedAtExecute(std::numeric_limits<float>::infinity());
}

TEST(TestGpuFlydslRmsNormDispatch, LoadsTheModuleOnceAcrossTwoPrepares)
{
    SKIP_IF_NO_DEVICES();

    KernelDefinition kernel;
    if(!findStagedKernel("bf16", 4096, kernel))
    {
        reportUnstaged("rmsnorm", "nothing staged for bf16 N=4096");
        return;
    }

    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {4, 4096}),
                               packedArchDeviceProperties());
    const auto& handler = dispatchHandler(FLYDSL_RMSNORM);

    // Warm first, then measure: the module for this (archive, toc_key, arch, device, sha)
    // may already be resident from an earlier case in this binary, and asserting a delta
    // of one against a cold cache would make the outcome depend on suite order. What the
    // cache promises is that a SECOND plan over the same kernel adds nothing.
    const auto warm = handler.prepare(fixture.context(), bindingsFor(fixture.context()), kernel);
    ASSERT_NE(warm, nullptr);

    const size_t resident = flydslRmsNormKpackModuleCache().size();
    ASSERT_GT(resident, 0U);

    const auto second = handler.prepare(fixture.context(), bindingsFor(fixture.context()), kernel);
    ASSERT_NE(second, nullptr);

    EXPECT_EQ(flydslRmsNormKpackModuleCache().size(), resident);
}

// ---------------------------------------------------------------------------
// the autotune winner-cache shard
// ---------------------------------------------------------------------------
//
// Everything above this point ranks kernels through `score`: a static number the
// descriptor declares, 100 for a baked tier and 10 for the generic one. Autotune is the
// other ranking path -- a benchmarked record, keyed by graph content and device, that is
// persisted per (engine, arch) and REPLACES the heuristic order for that key on every
// later run. `score` is what the engine believes; a record is what the device said.
//
// The generic mechanics of that record -- its encode/decode, its coverage and staleness
// rules -- belong to the plugin SDK and are covered exhaustively by its own
// `TestIngestorWinnerCache`. Nothing here re-tests them. What is only answerable from
// this side is whether the path is wired up for THIS engine: whether our descriptors
// reach the state manager as a real catalog, whether a record keyed on our graph
// displaces our own `score` ordering, and whether the shard it writes is scoped to our
// engine name rather than shared with every other pack in the provider.
//
// These cases do not benchmark. Measuring the two tiers is what
// `TestGpuFlydslRmsNormTiers` above does; what a recorded order must do is *win*, and a
// record whose ranking merely agreed with `score` could not tell the two apart. So the
// record written below is deliberately the INVERSE of the score order -- generic ahead
// of baked, the slower of the two wherever the tiers were measured. If the ordering that
// comes back is still baked-first, the record was ignored.

/// A state manager over the flyDSL descriptors this binary actually discovered.
///
/// Built from the loaded set rather than from hand-written descriptors for the same
/// reason `findStagedKernel()` promotes real ones: the ids in the catalog have to be the
/// ids a record is keyed on, and an invented pack id would make the coverage check fail
/// for a reason that has nothing to do with the cache.
///
/// Through makeStateManager() rather than the constructor the sibling kpack cases use, for
/// the engine name: the constructor takes it as its ninth argument, past two defaulted
/// ones, and an empty name silently disables the disk cache while leaving the in-memory
/// path working. A test that dropped it would pass the ordering case below and prove
/// nothing about persistence. The factory defaults it from the set and cannot be
/// mis-called that way.
///
/// By pointer because the manager holds the mutex guarding its winner cache and so is
/// neither copyable nor movable.
std::unique_ptr<KernelIngestorStateManager<Handle>> flydslStateManager()
{
    registerNativeIngestorSymbols();
    // Copied: the factory takes the set by value. `graphMatchNativeSymbol` is read from
    // the loaded set, which outlives every manager built from it.
    const auto& set = loadedSet(FLYDSL_RMSNORM.engineName);
    return makeStateManager<Handle>(set, set.engine.graphMatchNativeSymbol);
}

/// The winner key for @p context: graph content plus device, exactly as
/// `orderFromWinnerRecord()` composes it internally.
WinnerKey winnerKeyFor(const MatchContext& context)
{
    return WinnerKey{hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey{context.graph},
                     DeviceKey{context.deviceProperties}};
}

/// @p definitions as a benchmarked record, first entry fastest.
///
/// The times are fabricated and that is sound: `RankedEntry::timeMs` is documented as
/// diagnostic only, never compared across records. The ORDER is the payload.
WinnerRecord recordOver(const std::vector<KernelDefinition>& definitions)
{
    WinnerRecord record;
    record.reserve(definitions.size());
    double timeMs = 1.0;
    for(const auto& definition : definitions)
    {
        RankedEntry entry;
        entry.kernelId = definition.kernelId;
        entry.packId = definition.packId;
        entry.dispatchId = definition.dispatchId;
        entry.timeMs = timeMs;
        record.push_back(entry);
        timeMs += 1.0;
    }
    return record;
}

/// The baked `N` @p definition carries, or nullopt when it declares none.
std::optional<int64_t> bakedWidthOf(const KernelDefinition& definition)
{
    const auto* n = tryGetMetadataField<int64_t>(definition.metadata, FLYDSL_N_FIELD);
    return n == nullptr ? std::nullopt : std::optional<int64_t>(*n);
}

TEST(TestGpuFlydslRmsNormWinnerShard, ARecordedOrderDisplacesTheScoreOrder)
{
    SKIP_IF_NO_DEVICES();
    if(findLoadedSet(FLYDSL_RMSNORM.engineName) == nullptr)
    {
        reportUnstaged("rmsnorm", "this build staged no RMSNorm objects");
        return;
    }

    // TEST scope, not BINARY: main() already installed a binary-wide root, and every
    // case in this process would otherwise share one shard -- including the second case
    // below, which asserts on the shard's line count.
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cache(
        "flydsl-winners", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST);

    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {64, 4096}),
                               packedArchDeviceProperties());
    const auto context = fixture.context();
    const auto manager = flydslStateManager();

    // Both tiers, and only those two: the kernel matcher drops every other baked width,
    // and the dtype field drops f16. If this build packed neither, there is nothing to
    // rank and the case has no subject.
    const auto scored = manager->sortedDefinitions(context);
    if(scored.size() < 2)
    {
        reportUnstaged("rmsnorm",
                       "this build's descriptor root offers " + std::to_string(scored.size())
                           + " bf16 kernels at N=4096; the ordering cases need both tiers");
        return;
    }

    ASSERT_EQ(bakedWidthOf(scored.front()), std::optional<int64_t>(4096))
        << "`score` ranks the baked tier first; without that the inversion below proves "
           "nothing";
    ASSERT_EQ(bakedWidthOf(scored.back()), std::optional<int64_t>(FLYDSL_GENERIC_N));

    const std::vector<KernelDefinition> inverted(scored.rbegin(), scored.rend());
    manager->recordWinner(
        winnerKeyFor(context), recordOver(inverted), WinnerWriteCause::FRESH_MISS);

    const auto measured = manager->sortedDefinitions(context);
    ASSERT_EQ(measured.size(), scored.size());
    EXPECT_EQ(bakedWidthOf(measured.front()), std::optional<int64_t>(FLYDSL_GENERIC_N))
        << "a benchmarked record must outrank `score`; the catalog came back in priority "
           "order, so the record was not consulted";
    EXPECT_EQ(bakedWidthOf(measured.back()), std::optional<int64_t>(4096));

    // The memoized sort is what makes this worth asserting separately. A catalog is
    // cached once ranked, and a record written after that first ranking has to invalidate
    // it -- which the manager does by re-checking for a record even on an already-sorted
    // catalog, because a heuristic order is provisional and a measured one is final.
    // Ranking before recording, as this case does, is the order that exercises it.
    EXPECT_EQ(manager->winnerCacheSize(), 1U);
}

TEST(TestGpuFlydslRmsNormWinnerShard, TheShardIsScopedToThisEngineAndOutlivesTheManager)
{
    SKIP_IF_NO_DEVICES();
    if(findLoadedSet(FLYDSL_RMSNORM.engineName) == nullptr)
    {
        reportUnstaged("rmsnorm", "this build staged no RMSNorm objects");
        return;
    }

    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cache(
        "flydsl-winners", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST);

    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16, {64, 4096}),
                               packedArchDeviceProperties());
    const auto context = fixture.context();
    const auto& engineName = loadedSet(FLYDSL_RMSNORM.engineName).engine.name;
    const auto& arch = context.deviceProperties.gcnArchName;

    {
        const auto writer = flydslStateManager();
        const auto scored = writer->sortedDefinitions(context);
        if(scored.size() < 2)
        {
            reportUnstaged("rmsnorm",
                           "this build's descriptor root offers " + std::to_string(scored.size())
                               + " bf16 kernels at N=4096; the ordering cases need both tiers");
            return;
        }
        const std::vector<KernelDefinition> inverted(scored.rbegin(), scored.rend());
        writer->recordWinner(
            winnerKeyFor(context), recordOver(inverted), WinnerWriteCause::FRESH_MISS);
    }

    const auto shard = winnerCacheShardPath(engineName, arch);
    ASSERT_FALSE(shard.empty()) << "no usable cache root; the scoped temp dir did not take";
    EXPECT_TRUE(std::filesystem::exists(shard)) << shard.string();

    // The arch component is verbatim, not hashed -- deleting one arch's cache by hand is
    // a supported thing to do, so it has to be readable as the arch. The engine
    // component is sanitized and carries a hash suffix, so it is checked for
    // distinctness rather than for spelling.
    EXPECT_EQ(shard.parent_path().filename().string(), arch);
    EXPECT_NE(winnerCacheShardPath("hip_kernel_provider.some_other_engine", arch), shard)
        << "two engines share a shard; one engine's sweep would then be served to the "
           "other";

    // Exactly one record, not one per ranking. The shard is append-only and read
    // last-line-wins, so a write path that appended on every miss would grow a line per
    // process for a graph that never changes; FRESH_MISS adopts an existing entry for the
    // key instead. Counted by decoding rather than by counting lines, so the version
    // stamp LineStore heads the file with is not mistaken for a record.
    {
        std::ifstream file(shard);
        ASSERT_TRUE(file.is_open());
        std::string line;
        int records = 0;
        while(std::getline(file, line))
        {
            if(decodeWinnerRecordLine(line).has_value())
            {
                ++records;
            }
        }
        EXPECT_EQ(records, 1) << "one graph, one ranking, one line";
    }

    // A fresh manager: empty in-memory cache, same descriptors, same graph. Anything it
    // knows about the earlier ranking it can only have read back off disk, which is the
    // whole point of persisting -- the sweep is paid once, not once per process.
    const auto reader = flydslStateManager();
    EXPECT_EQ(reader->winnerCacheSize(), 0U) << "a new manager must start with nothing cached";

    const auto measured = reader->sortedDefinitions(context);
    ASSERT_GE(measured.size(), 2U);
    EXPECT_EQ(bakedWidthOf(measured.front()), std::optional<int64_t>(FLYDSL_GENERIC_N))
        << "the shard did not survive the manager; this reader fell back to `score`";
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
