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
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"

/**
 * @file TestFlydslRmsNormEngine.cpp
 * @brief The flyDSL RMS-norm pack across all four native seams, plus a real launch.
 *
 * The census over what the pack SHIPS lives next door in TestFlydslRmsNormPacks.cpp,
 * which belongs to the census binary because its cases need a named shard. Everything
 * here runs in the ordinary unit binary.
 *
 * The pack is unusual among the ingestor packs in three ways, and each gets cases here:
 *
 *  - It ships TWO SELECTION TIERS over the same graph. A kernel whose `N` metadata names
 *    the graph's width is preferred (`priority: 100`); a generic kernel carrying the
 *    `N: 0` sentinel takes the width as a kernarg and serves any width (`priority: 10`).
 *    The kernel matcher admits both and `score` orders them, so a width no instance was
 *    baked for still plans.
 *
 *  - Its kernels BAKE EPSILON. A graph asking for a materially different one is not one
 *    these kernels compute, so `graph_match` declines it -- quietly, so another engine
 *    takes the graph rather than the plan build failing. An epsilon the caller supplies at
 *    execute cannot be seen then, so `launch()` re-checks it and throws there instead.
 *    Both paths are covered below; they are one condition reported two ways.
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

/// The current device, with the arch stripped of its feature suffix.
///
/// hipGetDeviceProperties reports `gfx1151:xnack-`; the packer names shards with the bare
/// arch, and prepare() passes this string straight through to the archive lookup. Asking
/// in the wrong spelling fails to find a shard that is sitting right there.
hipdnn_plugin_sdk::ingestor::DeviceProperties packedArchDeviceProperties()
{
    auto properties = currentDeviceProperties();
    const auto colon = properties.gcnArchName.find(':');
    if(colon != std::string::npos)
    {
        properties.gcnArchName.resize(colon);
    }
    return properties;
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
                                  int64_t priority = 100)
{
    KernelDefinition kernel;
    kernel.name = "rmsnorm_" + dtype + "_n" + std::to_string(bakedN);
    kernel.priority = priority;
    kernel.metadata = {{std::string(FLYDSL_DTYPE_FIELD), dtype},
                       {std::string(FLYDSL_N_FIELD), bakedN},
                       {std::string(FLYDSL_BLOCK_THREADS_FIELD), blockThreads}};
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
/// Returns false when the loaded set holds no kernel with that (dtype, N) for this device,
/// which is the normal answer on an arch this build packed nothing for. A build for several
/// arches loads every arch's shard, so a kernel staged for another device is passed over.
bool findStagedKernel(const std::string& dtype, int64_t bakedN, KernelDefinition& out)
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
            if(name == nullptr || n == nullptr || *name != dtype || *n != bakedN)
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
        GraphCase{"Half", [] { return buildFlydslRmsNormGraph(DataType::HALF); }},
        // Rank above two is collapsed, not refused: the leading axes multiply into the row
        // count, which is sound precisely because packed row-major is also required.
        GraphCase{"Rank4CollapsesToRows",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {2, 3, 4, 4096}); }},
        // A width no instance is baked for: the generic tier is what makes this applicable,
        // and the graph matcher does not know about tiers at all.
        GraphCase{"AnUnbakedWidth",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 640}); }},
        // Within the relative tolerance of the baked constant, so these kernels do compute it.
        GraphCase{"AnEpsilonWithinTolerance",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}, 1.005e-5F); }},
        // Not inspectable at match time. Deferred to launch(), which re-checks it there.
        GraphCase{
            "ARuntimeEpsilon",
            [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}, 0.0F, true); }},
        GraphCase{"InferencePhase",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_BAKED_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     false,
                                                     false,
                                                     NormFwdPhase::INFERENCE);
                  }}),
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
    // the product of every axis outside the innermost, so 2x3 here and not 2.
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_ROWS_TOKEN), 6);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_COLUMNS_TOKEN), 4096);
}

TEST(TestFlydslRmsNormBinding, CollapsesRank4UnderABroadcastShapedScale)
{
    // Rank 4 with a scale of (1,1,1,W) rather than the rank-1 {W} every other case uses.
    // The matcher asks only that the scale carry ONE row and that its innermost extent be
    // the width, so leading unit axes are accepted the same way a rank-1 scale is -- the
    // shape a framework emits when gamma is broadcast against x is not a refusal here.
    // The three leading axes of x still collapse into the row count, 2x8x4.
    const GraphFixture fixture(buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                       {2, 8, 4, 4096},
                                                       FLYDSL_BAKED_EPSILON,
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
                                                     FLYDSL_BAKED_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     true);
                  }},
        // The vendored instances are built with store_rstd=False, so there is nowhere to
        // put the statistic even though the kernarg slots for it exist.
        GraphCase{"SavingTheInverseRms",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_BAKED_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     false,
                                                     true);
                  }},
        // Training needs the saved statistic for the backward pass even when the graph
        // does not name a tensor for it.
        GraphCase{"TrainingPhase",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_BAKED_EPSILON,
                                                     false,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     false,
                                                     false,
                                                     NormFwdPhase::TRAINING);
                  }},
        // Only two dtypes were baked, and there is no upconverting path.
        GraphCase{"AnUnsupportedDtype", [] { return buildFlydslRmsNormGraph(DataType::FLOAT); }},
        // One instance, one dtype for all three operands -- gamma is read at x's type.
        GraphCase{"GammaAtADifferentDtype",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_BAKED_EPSILON,
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
                                                     FLYDSL_BAKED_EPSILON,
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
                          FLYDSL_BAKED_EPSILON,
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
                                                     FLYDSL_BAKED_EPSILON,
                                                     false,
                                                     std::vector<int64_t>{1});
                  }},
        // ...and one vector, not one per row.
        GraphCase{"GammaWithARowPerInputRow",
                  [] {
                      return buildFlydslRmsNormGraph(DataType::BFLOAT16,
                                                     {8, 4096},
                                                     FLYDSL_BAKED_EPSILON,
                                                     false,
                                                     std::vector<int64_t>{8, 4096});
                  }},
        // Declined rather than approximated, and declined QUIETLY: the graph is still
        // computable, just not by these kernels, so another engine must get the chance.
        GraphCase{"ABakedEpsilonOutsideTolerance",
                  [] { return buildFlydslRmsNormGraph(DataType::BFLOAT16, {8, 4096}, 1e-6F); }}),
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

    // Baking N fixes the loop trip count, so a mismatched instance would read past the row.
    EXPECT_FALSE(
        matchesKernel(FLYDSL_RMSNORM, fixture.context(), makeFlydslKernel("bf16", 8192), bound));
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
/// reads, which keeps input quantisation out of the error budget.
template <typename T>
class RmsNormBuffers
{
public:
    RmsNormBuffers(int64_t rows, int64_t columns)
        : _rows(rows)
        , _columns(columns)
        , _x(static_cast<size_t>(rows * columns))
        , _gamma(static_cast<size_t>(columns))
    {
        for(int64_t row = 0; row < rows; ++row)
        {
            for(int64_t column = 0; column < columns; ++column)
            {
                // Eighths, so every value is exact in bf16's 8 mantissa bits.
                const float value = 0.5F + static_cast<float>((row + column) % 8) * 0.125F;
                _x[static_cast<size_t>(row * columns + column)] = T(value);
            }
        }
        for(int64_t column = 0; column < columns; ++column)
        {
            _gamma[static_cast<size_t>(column)] = T(0.5F + static_cast<float>(column % 4) * 0.25F);
        }

        EXPECT_EQ(hipSuccess, hipMalloc(&_xDevice, _x.size() * sizeof(T)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_gammaDevice, _gamma.size() * sizeof(T)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_yDevice, _x.size() * sizeof(T)));
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(_xDevice, _x.data(), _x.size() * sizeof(T), hipMemcpyHostToDevice));
        EXPECT_EQ(
            hipSuccess,
            hipMemcpy(
                _gammaDevice, _gamma.data(), _gamma.size() * sizeof(T), hipMemcpyHostToDevice));
        EXPECT_EQ(hipSuccess, hipMemset(_yDevice, 0, _x.size() * sizeof(T)));
    }

    ~RmsNormBuffers()
    {
        static_cast<void>(hipFree(_xDevice));
        static_cast<void>(hipFree(_gammaDevice));
        static_cast<void>(hipFree(_yDevice));
    }

    RmsNormBuffers(const RmsNormBuffers&) = delete;
    RmsNormBuffers& operator=(const RmsNormBuffers&) = delete;

    std::array<hipdnnPluginDeviceBuffer_t, 3> descriptors() const
    {
        return {hipdnnPluginDeviceBuffer_t{FLYDSL_X_UID, _xDevice},
                hipdnnPluginDeviceBuffer_t{FLYDSL_SCALE_UID, _gammaDevice},
                hipdnnPluginDeviceBuffer_t{FLYDSL_Y_UID, _yDevice}};
    }

    std::vector<T> readOutput() const
    {
        std::vector<T> result(_x.size());
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(result.data(), _yDevice, _x.size() * sizeof(T), hipMemcpyDeviceToHost));
        return result;
    }

    /// y = x * rsqrt(mean(x^2) + eps) * gamma, in double from the quantised inputs.
    std::vector<double> reference() const
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
            const double scale = 1.0
                                 / std::sqrt(sumOfSquares / static_cast<double>(_columns)
                                             + static_cast<double>(FLYDSL_BAKED_EPSILON));
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
    std::vector<T> _gamma;
    void* _xDevice = nullptr;
    void* _gammaDevice = nullptr;
    void* _yDevice = nullptr;
};

/// Runs the staged (@p dtype, @p bakedN) kernel over a @p rows x @p columns graph and
/// checks every element against the host reference.
///
/// Skips rather than fails when the staged set holds no such kernel: on an arch this build
/// packed nothing for, there is no kernel to be wrong about.
template <typename T>
void expectRmsNormMatchesReference(DataType dataType,
                                   const std::string& dtype,
                                   int64_t bakedN,
                                   int64_t rows,
                                   int64_t columns,
                                   double relativeTolerance)
{
    KernelDefinition kernel;
    if(!findStagedKernel(dtype, bakedN, kernel))
    {
        GTEST_SKIP() << "nothing packed for dtype=" << dtype << " N=" << bakedN
                     << " in this build's descriptor root";
    }

    const GraphFixture fixture(buildFlydslRmsNormGraph(dataType, {rows, columns}),
                               packedArchDeviceProperties());
    const auto& handler = dispatchHandler(FLYDSL_RMSNORM);

    const auto prepared
        = handler.prepare(fixture.context(), bindingsFor(fixture.context()), kernel);
    ASSERT_NE(prepared, nullptr);

    const RmsNormBuffers<T> buffers(rows, columns);
    const auto descriptors = buffers.descriptors();
    const Handle handle;

    handler.launch(
        handle, *prepared, descriptors.data(), static_cast<uint32_t>(descriptors.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    const auto actual = buffers.readOutput();
    const auto expected = buffers.reference();
    ASSERT_EQ(actual.size(), expected.size());

    double worst = 0.0;
    size_t worstIndex = 0;
    for(size_t i = 0; i < actual.size(); ++i)
    {
        const double error
            = std::abs(static_cast<double>(static_cast<float>(actual[i])) - expected[i])
              / std::max(std::abs(expected[i]), 1e-3);
        if(error > worst)
        {
            worst = error;
            worstIndex = i;
        }
    }
    EXPECT_LE(worst, relativeTolerance)
        << "at element " << worstIndex << ": got " << static_cast<float>(actual[worstIndex])
        << ", expected " << expected[worstIndex];
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

TEST(TestGpuFlydslRmsNormTiers, CostTheBakedAndGenericTiersOverOneGraph)
{
    SKIP_IF_NO_DEVICES();

    // Two shapes, because either one alone would say something the other contradicts.
    // Four rows is four workgroups: latency-bound, nothing to hide an instruction count
    // behind, and the tier difference shows at its widest. 4096 rows moves 64 MiB per
    // launch and runs near the memory ceiling, which compresses the same difference
    // without erasing it. One number alone would overstate or bury the effect.
    const int64_t columns = 4096;
    const std::array<int64_t, 2> rowCounts{4, 4096};

    for(const auto rows : rowCounts)
    {
        SCOPED_TRACE(::testing::Message() << rows << "x" << columns);

        const auto baked = timeRmsNormLaunches<hipdnn_data_sdk::types::bfloat16>(
            DataType::BFLOAT16, "bf16", columns, rows, columns);
        const auto generic = timeRmsNormLaunches<hipdnn_data_sdk::types::bfloat16>(
            DataType::BFLOAT16, "bf16", FLYDSL_GENERIC_N, rows, columns);

        if(!baked.has_value() || !generic.has_value())
        {
            GTEST_SKIP() << "this build's descriptor root lacks a bf16 tier at N=" << columns
                         << " or N=" << FLYDSL_GENERIC_N;
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
}

TEST(TestGpuFlydslRmsNormDispatch, RejectsARuntimeEpsilonTheKernelDoesNotBake)
{
    SKIP_IF_NO_DEVICES();

    KernelDefinition kernel;
    if(!findStagedKernel("bf16", 4096, kernel))
    {
        GTEST_SKIP() << "nothing packed for bf16 N=4096 in this build's descriptor root";
    }

    // The other half of the epsilon contract. A baked mismatch is declined silently at
    // graph_match so another engine takes the graph; a mismatch the caller only supplies
    // at execute has no such escape -- the plan is already built -- so it is an error.
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
    float epsilon = 1e-2F;
    withEpsilon.push_back(hipdnnPluginDeviceBuffer_t{FLYDSL_EPSILON_UID, &epsilon});

    const Handle handle;
    EXPECT_THROW(handler.launch(handle,
                                *prepared,
                                withEpsilon.data(),
                                static_cast<uint32_t>(withEpsilon.size()),
                                nullptr),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGpuFlydslRmsNormDispatch, LoadsTheModuleOnceAcrossTwoPrepares)
{
    SKIP_IF_NO_DEVICES();

    KernelDefinition kernel;
    if(!findStagedKernel("bf16", 4096, kernel))
    {
        GTEST_SKIP() << "nothing packed for bf16 N=4096 in this build's descriptor root";
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
// of baked, which §18.8 measured to be the slower of the two. If the ordering that comes
// back is still baked-first, the record was ignored.

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
        GTEST_SKIP() << "this build staged no RMSNorm objects";
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
        GTEST_SKIP() << "this build's descriptor root offers " << scored.size()
                     << " bf16 kernels at N=4096; the ordering cases need both tiers";
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
        GTEST_SKIP() << "this build staged no RMSNorm objects";
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
            GTEST_SKIP() << "this build's descriptor root offers " << scored.size()
                         << " bf16 kernels at N=4096; the ordering cases need both tiers";
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
