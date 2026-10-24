// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file TestFlydslSdpaAiwClasses.cpp
 * @brief Which shipped hipkernel:flydsl_sdpa_aiw object serves a graph, for the classes the
 *        aiw rows added after the engine landed: the padded tiers up to 384 and 512 (head
 *        dims 264-512, each plain or with either additive mask), and an attention sink with
 *        either additive mask at head_dim 64 and 128.
 *
 * The host half reads the loaded aiw shard, not hand-written metadata: for each graph, every
 * shipped gfx1201 object the engine's matchers admit, ranked by the engine's own score. A
 * graph a row was added for must find that row's object on top; one no row covers must find
 * none. The device half (gfx12; elsewhere it skips) launches the
 * top-ranked staged object through the RDNA device harness against its double-precision
 * reference, the LSE included, so the object the engine would pick is the one checked.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

/// The arch the host half matches the gfx12-generic objects for: a member this build packed
/// them for (a build packs only its GPU_TARGETS), or gfx1201 when it packed none, so the cases
/// skip as before.
std::string aiwArch()
{
    const auto arches = flydslShippedArches("sdpa_aiw");
    return arches.empty() ? std::string("gfx1201") : arches.front();
}

int64_t intField(const KernelDefinition& kernel, const char* field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return 0;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    return value == nullptr ? 0 : *value;
}

/// Every object the loaded aiw set ships for @p arch, or nullopt when the build loads none.
std::optional<std::vector<KernelDefinition>> loadedAiwObjects(const std::string& arch)
{
    const auto* set = findLoadedSet(FLYDSL_SDPA_AIW.engineName);
    if(set == nullptr)
    {
        return std::nullopt;
    }
    std::vector<KernelDefinition> kernels;
    for(const auto& pack : set->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            if(std::find(arches.begin(), arches.end(), arch) == arches.end())
            {
                continue;
            }
            kernels.push_back(KernelDefinition{kernel.id,
                                               pack.id,
                                               pack.dispatchId,
                                               kernel.source,
                                               kernel.metadata,
                                               kernel.priority,
                                               arches,
                                               kernel.originDirectory,
                                               kernel.name,
                                               kernel.treeRoot,
                                               kernel.sources});
        }
    }
    return kernels;
}

/// The objects of @p kernels @p engine admits for @p spec on @p properties, best first by
/// the engine's own score.
std::vector<KernelDefinition> rankedObjects(const PackSymbols& engine,
                                            const SdpaGraphSpec& spec,
                                            const std::vector<KernelDefinition>& kernels,
                                            const DeviceProperties& properties)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), properties);
    const auto bound = matchesGraph(engine, fixture.context());
    if(!bound.has_value())
    {
        return {};
    }
    std::vector<std::pair<double, KernelDefinition>> scored;
    for(const auto& kernel : kernels)
    {
        if(matchesKernel(engine, fixture.context(), kernel, *bound))
        {
            scored.emplace_back(scoreKernel(engine, fixture.context(), kernel, *bound), kernel);
        }
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first;
    });
    std::vector<KernelDefinition> serving;
    serving.reserve(scored.size());
    for(auto& [score, kernel] : scored)
    {
        serving.push_back(std::move(kernel));
    }
    return serving;
}

/// The aiw objects of @p kernels that serve @p spec on @p properties, best first.
std::vector<KernelDefinition> servingObjects(const SdpaGraphSpec& spec,
                                             const std::vector<KernelDefinition>& kernels,
                                             const DeviceProperties& properties)
{
    return rankedObjects(FLYDSL_SDPA_AIW, spec, kernels, properties);
}

DeviceProperties aiwDeviceProperties()
{
    DeviceProperties properties;
    properties.gcnArchName = aiwArch();
    properties.warpSize = 32;
    properties.multiProcessorCount = 64;
    return properties;
}

/// Sets @p kernels to the loaded aiw objects of aiwArch(). Without them it stays empty, with a
/// failure if this build ships aiw and a skip otherwise; the caller returns on empty.
void requireLoadedAiwObjects(std::vector<KernelDefinition>& kernels)
{
    kernels = loadedAiwObjects(aiwArch()).value_or(std::vector<KernelDefinition>{});
    if(!kernels.empty())
    {
        return;
    }
    if(!flydslShippedArches("sdpa_aiw").empty())
    {
        ADD_FAILURE() << "this build ships sdpa_aiw but loaded no " << aiwArch()
                      << " objects for it";
        return;
    }
    GTEST_SKIP() << "this build ships no sdpa_aiw objects";
}

/// The mask variants an aiw object is built with: none, one in Q's dtype, an f32 one.
enum class AiwMask
{
    NONE,
    QUERY_TYPE,
    F32
};

void setMask(SdpaGraphSpec& spec, AiwMask mask)
{
    spec.withBias = mask != AiwMask::NONE;
    spec.alignment.erase(FLYDSL_SDPA_RDNA_BIAS_UID);
    if(mask == AiwMask::F32)
    {
        spec.biasDataType = DataType::FLOAT;
        // aiw loads an f32 mask 8 x f32 at a time at that type's natural alignment.
        spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
    }
    else if(mask == AiwMask::QUERY_TYPE)
    {
        spec.biasDataType = spec.dataType;
    }
}

std::string maskName(AiwMask mask)
{
    if(mask == AiwMask::F32)
    {
        return "f32 mask";
    }
    return mask == AiwMask::QUERY_TYPE ? "query-dtype mask" : "no mask";
}

// ---------------------------------------------------------------------------
// host: the padded tiers up to 384 and 512
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaAiwClasses, ServesHeadDimsPast256OnTheNarrowestWideTier)
{
    std::vector<KernelDefinition> kernels;
    requireLoadedAiwObjects(kernels);
    if(kernels.empty())
    {
        return;
    }
    // Every multiple of 8 from 264 to 512 runs on a padded tier: up to 384 on the 384 tier
    // (the narrower outranks the wider), past it on the 512 one; both ends of each range,
    // and 384 and 512 themselves (no exact object at either), in both dtypes, with and
    // without a causal mask, plain and with either mask.
    for(const auto dataType : {DataType::BFLOAT16, DataType::HALF})
    {
        for(const auto causal : {SdpaMask::NONE, SdpaMask::BOTTOM_RIGHT_CAUSAL})
        {
            for(const auto mask : {AiwMask::NONE, AiwMask::QUERY_TYPE, AiwMask::F32})
            {
                for(const int64_t headDim : {264, 288, 320, 384, 392, 448, 512})
                {
                    SdpaGraphSpec spec;
                    spec.dataType = dataType;
                    spec.mask = causal;
                    spec.headDim = headDim;
                    setMask(spec, mask);
                    SCOPED_TRACE("head_dim " + std::to_string(headDim) + ", " + maskName(mask)
                                 + (causal == SdpaMask::NONE ? "" : ", causal")
                                 + (dataType == DataType::HALF ? ", f16" : ", bf16"));
                    const auto serving = servingObjects(spec, kernels, aiwDeviceProperties());
                    ASSERT_FALSE(serving.empty());
                    const auto& top = serving.front();
                    EXPECT_EQ(intField(top, FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD), 0) << top.name;
                    EXPECT_EQ(intField(top, FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD),
                              headDim <= 384 ? 384 : 512)
                        << top.name;
                    EXPECT_EQ(intField(top, FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD),
                              mask == AiwMask::NONE ? 0 : 1)
                        << top.name;
                    EXPECT_EQ(intField(top, FLYDSL_SDPA_AIW_BIAS_F32_FIELD),
                              mask == AiwMask::F32 ? 1 : 0)
                        << top.name;
                    // Each wide tier serves its own range only once the narrower tier has
                    // stopped: the up-to-512 object is also a candidate below 384, ranked
                    // under it.
                    if(headDim <= 384)
                    {
                        ASSERT_GE(serving.size(), 2U);
                        EXPECT_EQ(intField(serving[1], FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD), 512)
                            << serving[1].name;
                    }
                }
            }
        }
    }
}

TEST(TestFlydslSdpaAiwClasses, DeclinesHeadDimsNoTierReaches)
{
    std::vector<KernelDefinition> kernels;
    requireLoadedAiwObjects(kernels);
    if(kernels.empty())
    {
        return;
    }
    // Past the widest tile, and between tiers off the multiple-of-8 grid.
    for(const int64_t headDim : {520, 576, 1024, 388, 508})
    {
        SdpaGraphSpec spec;
        spec.headDim = headDim;
        EXPECT_TRUE(servingObjects(spec, kernels, aiwDeviceProperties()).empty()) << headDim;
        setMask(spec, AiwMask::F32);
        EXPECT_TRUE(servingObjects(spec, kernels, aiwDeviceProperties()).empty())
            << headDim << " with a mask";
    }
}

TEST(TestFlydslSdpaAiwClasses, ServesAValueHeadDimOfItsOwnOnTheTierOfTheWiderOne)
{
    std::vector<KernelDefinition> kernels;
    requireLoadedAiwObjects(kernels);
    if(kernels.empty())
    {
        return;
    }
    // H1: a padded tier takes Q/K's and V's widths at runtime, so V narrower or wider than
    // Q/K runs on the narrowest tier that holds the wider of the two. MLA's exact 192/128
    // object ranks above every tier.
    struct Pair
    {
        int64_t qk;
        int64_t v;
        int64_t tier;
    };
    for(const auto& [qk, v, tier] : {Pair{128, 64, 128},
                                     Pair{96, 128, 128},
                                     Pair{200, 128, 224},
                                     Pair{256, 64, 256},
                                     Pair{384, 256, 384},
                                     Pair{320, 128, 384},
                                     Pair{256, 384, 384},
                                     Pair{512, 384, 512}})
    {
        SdpaGraphSpec spec;
        spec.headDim = qk;
        spec.valueHeadDim = v;
        const auto serving = servingObjects(spec, kernels, aiwDeviceProperties());
        ASSERT_FALSE(serving.empty()) << qk << "/" << v;
        EXPECT_EQ(intField(serving.front(), FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD), 0)
            << serving.front().name;
        EXPECT_EQ(intField(serving.front(), FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD), tier)
            << qk << "/" << v << ": " << serving.front().name;
    }
    SdpaGraphSpec mla;
    mla.headDim = 192;
    mla.valueHeadDim = 128;
    const auto mlaServing = servingObjects(mla, kernels, aiwDeviceProperties());
    ASSERT_FALSE(mlaServing.empty());
    EXPECT_EQ(intField(mlaServing.front(), FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD), 192)
        << mlaServing.front().name;

    // Past the widest tile on either side: absorbed MLA (576/512) has no aiw row (the RDNA
    // decode engine serves it).
    for(const auto& [qk, v, tier] : {Pair{576, 512, 0}, Pair{520, 256, 0}, Pair{256, 520, 0}})
    {
        SdpaGraphSpec spec;
        spec.headDim = qk;
        spec.valueHeadDim = v;
        EXPECT_TRUE(servingObjects(spec, kernels, aiwDeviceProperties()).empty()) << qk << "/" << v;
    }
}

// ---------------------------------------------------------------------------
// host: an attention sink with an additive mask
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaAiwClasses, PairsASinkWithAMaskOnTheSinkMaskObjects)
{
    std::vector<KernelDefinition> kernels;
    requireLoadedAiwObjects(kernels);
    if(kernels.empty())
    {
        return;
    }
    // A sink and a mask at 64 and 128 run on the object built with both, in the mask's
    // dtype; a sink alone still runs on the sink object without one.
    for(const auto dataType : {DataType::BFLOAT16, DataType::HALF})
    {
        for(const auto causal : {SdpaMask::NONE, SdpaMask::TOP_LEFT_CAUSAL})
        {
            for(const int64_t headDim : {64, 128})
            {
                for(const auto mask : {AiwMask::NONE, AiwMask::QUERY_TYPE, AiwMask::F32})
                {
                    SdpaGraphSpec spec;
                    spec.dataType = dataType;
                    spec.mask = causal;
                    spec.headDim = headDim;
                    spec.withSink = true;
                    setMask(spec, mask);
                    SCOPED_TRACE("head_dim " + std::to_string(headDim) + ", sink, " + maskName(mask)
                                 + (causal == SdpaMask::NONE ? "" : ", causal")
                                 + (dataType == DataType::HALF ? ", f16" : ", bf16"));
                    const auto serving = servingObjects(spec, kernels, aiwDeviceProperties());
                    ASSERT_FALSE(serving.empty());
                    const auto& top = serving.front();
                    EXPECT_EQ(intField(top, FLYDSL_SDPA_RDNA_HEAD_DIM_FIELD), headDim) << top.name;
                    EXPECT_EQ(intField(top, FLYDSL_SDPA_RDNA_HAS_SINK_FIELD), 1) << top.name;
                    EXPECT_EQ(intField(top, FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD),
                              mask == AiwMask::NONE ? 0 : 1)
                        << top.name;
                    EXPECT_EQ(intField(top, FLYDSL_SDPA_AIW_BIAS_F32_FIELD),
                              mask == AiwMask::F32 ? 1 : 0)
                        << top.name;
                    // Exactly one object serves each such graph.
                    EXPECT_EQ(serving.size(), 1U) << top.name;
                }
            }
        }
    }
}

TEST(TestFlydslSdpaAiwClasses, DeclinesASinkWithAMaskOffTheExactSinkHeadDims)
{
    std::vector<KernelDefinition> kernels;
    requireLoadedAiwObjects(kernels);
    if(kernels.empty())
    {
        return;
    }
    // Sinks ship at the exact head dims 64 and 128 only: with a mask as without, a padded
    // head dim or 256 with a sink has no object.
    for(const int64_t headDim : {80, 96, 120, 256})
    {
        for(const auto mask : {AiwMask::NONE, AiwMask::QUERY_TYPE, AiwMask::F32})
        {
            SdpaGraphSpec spec;
            spec.headDim = headDim;
            spec.withSink = true;
            setMask(spec, mask);
            EXPECT_TRUE(servingObjects(spec, kernels, aiwDeviceProperties()).empty())
                << headDim << ", " << maskName(mask);
        }
    }
}

TEST(TestFlydslSdpaAiwClasses, DeclinesASinkOnAWideTier)
{
    std::vector<KernelDefinition> kernels;
    requireLoadedAiwObjects(kernels);
    if(kernels.empty())
    {
        return;
    }
    // Sinks ship on the exact head dims 64 and 128 only.
    SdpaGraphSpec spec;
    spec.headDim = 384;
    spec.withSink = true;
    EXPECT_TRUE(servingObjects(spec, kernels, aiwDeviceProperties()).empty());
    spec.headDim = 512;
    EXPECT_TRUE(servingObjects(spec, kernels, aiwDeviceProperties()).empty());
}

// ---------------------------------------------------------------------------
// device: the top-ranked staged object against the reference
// ---------------------------------------------------------------------------

/// Launches the object @p engine ranks first for @p spec on this device. Returns the object,
/// or nullopt after recording a skip or a failure.
std::optional<KernelDefinition> launchTopRanked(const SdpaGraphSpec& spec,
                                                const PackSymbols& engine,
                                                RdnaObjectEngine family,
                                                const char* op)
{
    const auto staged = stagedKernels(engine.engineName);
    if(staged.empty())
    {
        reportUnstaged(op, std::string("this build staged no ") + op + " objects for this device");
        return std::nullopt;
    }
    const auto ranked = rankedObjects(engine, spec, staged, packedArchDeviceProperties());
    if(ranked.empty())
    {
        ADD_FAILURE() << "no staged " << op << " object admits the graph";
        return std::nullopt;
    }
    const auto& top = ranked.front();
    SCOPED_TRACE(top.name);
    expectRdnaObjectMatchesReference(spec, family, top);
    return top;
}

std::optional<KernelDefinition> launchAiw(const SdpaGraphSpec& spec)
{
    return launchTopRanked(spec, FLYDSL_SDPA_AIW, RdnaObjectEngine::AIW, "sdpa_aiw");
}

TEST(TestGpuFlydslSdpaAiwClasses, ComputesHeadDimsPast256OnTheWideTiers)
{
    SKIP_IF_NO_DEVICES();
    struct WideCase
    {
        int64_t headDim;
        SdpaMask mask;
        DataType dataType;
        SdpaLayout layout;
        int64_t tier;
    };
    for(const auto& [headDim, mask, dataType, layout, tier] :
        {WideCase{264, SdpaMask::TOP_LEFT_CAUSAL, DataType::BFLOAT16, SdpaLayout::BSHD, 384},
         WideCase{384, SdpaMask::NONE, DataType::HALF, SdpaLayout::BHSD, 384},
         WideCase{448, SdpaMask::SLIDING_WINDOW, DataType::BFLOAT16, SdpaLayout::BSHD, 512},
         WideCase{512, SdpaMask::BOTTOM_RIGHT_CAUSAL, DataType::BFLOAT16, SdpaLayout::BSHD, 512},
         WideCase{512, SdpaMask::NONE, DataType::HALF, SdpaLayout::BSHD, 512}})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        SdpaGraphSpec spec;
        spec.dataType = dataType;
        spec.layout = layout;
        spec.headDim = headDim;
        spec.mask = mask;
        spec.heads = 4;
        spec.kvHeads = 2;
        spec.seqLenQ = 150;
        spec.seqLenKv = 200;
        spec.scale = 0.05F;
        spec.stats = SdpaStats::LSE;
        const auto top = launchAiw(spec);
        if(top.has_value())
        {
            EXPECT_EQ(intField(*top, FLYDSL_SDPA_RDNA_HEAD_DIM_MAX_FIELD), tier) << top->name;
        }
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
    }
}

TEST(TestGpuFlydslSdpaAiwClasses, AddsAMaskOnTheWideTiers)
{
    SKIP_IF_NO_DEVICES();
    // An f32 mask per head with a bottom-right causal mask at 320 (the 384 tier), one in the
    // query's dtype at 512; then COVERAGE §5.7's mask rows on each tier: rows at the mask
    // dtype's lowest finite value (O = mean(V), the LSE near the mask), then -inf on every
    // key of every third row and on the first 64 keys of the others (O = 0, LSE = -inf on
    // the rows the bound leaves no other key).
    for(const int64_t headDim : {320, 512})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        SdpaGraphSpec spec;
        spec.headDim = headDim;
        spec.heads = 4;
        spec.kvHeads = 2;
        spec.seqLenQ = 77;
        // A multiple of 8: the mask's rows must stay aligned for the object's vector loads.
        spec.seqLenKv = 104;
        spec.scale = 0.05F;
        spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
        spec.stats = SdpaStats::LSE;
        setMask(spec, headDim == 320 ? AiwMask::F32 : AiwMask::QUERY_TYPE);
        spec.biasDims = {1, spec.heads, spec.seqLenQ, spec.seqLenKv};
        static_cast<void>(launchAiw(spec));
        spec.biasDims.clear();
        spec.biasMasksRows = true;
        static_cast<void>(launchAiw(spec));
        spec.biasMasksRows = false;
        spec.biasNegativeInfinityRows = true;
        spec.biasNegativeInfinityLeadingKeys = 64;
        setMask(spec, headDim == 320 ? AiwMask::QUERY_TYPE : AiwMask::F32);
        static_cast<void>(launchAiw(spec));
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
    }
}

TEST(TestGpuFlydslSdpaAiwClasses, GivesTheRdnaDecodeObjectsAnswerAt512)
{
    SKIP_IF_NO_DEVICES();
    // Equal 512/512 with one KV head is served at a decode tile by the RDNA decode engine
    // too: both engines' top-ranked objects against the same reference, for any number of
    // query rows (autotune picks between them).
    SdpaGraphSpec spec;
    spec.headDim = 512;
    spec.heads = 8;
    spec.kvHeads = 1;
    spec.seqLenQ = 40;
    spec.seqLenKv = 600;
    spec.scale = 0.05F;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.stats = SdpaStats::LSE;
    static_cast<void>(launchAiw(spec));
    if(HasFatalFailure() || IsSkipped())
    {
        return;
    }
    static_cast<void>(launchTopRanked(
        spec, FLYDSL_SDPA_RDNA_DECODE, RdnaObjectEngine::DECODE, "sdpa_rdna_decode"));
}

/// The objects a sink-and-mask graph must land on: exact 64 or 128, a sink and a mask.
void expectSinkMaskObject(const std::optional<KernelDefinition>& top, AiwMask mask)
{
    if(!top.has_value())
    {
        return;
    }
    EXPECT_EQ(intField(*top, FLYDSL_SDPA_RDNA_HAS_SINK_FIELD), 1) << top->name;
    EXPECT_EQ(intField(*top, FLYDSL_SDPA_RDNA_HAS_BIAS_FIELD), 1) << top->name;
    EXPECT_EQ(intField(*top, FLYDSL_SDPA_AIW_BIAS_F32_FIELD), mask == AiwMask::F32 ? 1 : 0)
        << top->name;
}

TEST(TestGpuFlydslSdpaAiwClasses, AddsASinkWithAMask)
{
    SKIP_IF_NO_DEVICES();
    // A sink beside a mask per head: an f32 mask with a causal sliding window at 64 (bf16),
    // a mask in the query's dtype with a bottom-right causal mask at 128 (f16), GQA, the
    // LSE (which carries the sink's term).
    struct SinkMaskCase
    {
        int64_t headDim;
        DataType dataType;
        SdpaMask mask;
        AiwMask bias;
    };
    for(const auto& [headDim, dataType, mask, bias] :
        {SinkMaskCase{64, DataType::BFLOAT16, SdpaMask::SLIDING_WINDOW, AiwMask::F32},
         SinkMaskCase{128, DataType::HALF, SdpaMask::BOTTOM_RIGHT_CAUSAL, AiwMask::QUERY_TYPE},
         SinkMaskCase{128, DataType::BFLOAT16, SdpaMask::NONE, AiwMask::F32},
         SinkMaskCase{64, DataType::HALF, SdpaMask::TOP_LEFT_CAUSAL, AiwMask::QUERY_TYPE}})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim) + ", " + maskName(bias));
        SdpaGraphSpec spec;
        spec.dataType = dataType;
        spec.headDim = headDim;
        spec.heads = 8;
        spec.kvHeads = 2;
        spec.seqLenQ = 150;
        spec.seqLenKv = 200;
        spec.mask = mask;
        spec.withSink = true;
        spec.stats = SdpaStats::LSE;
        setMask(spec, bias);
        spec.biasDims = {1, spec.heads, spec.seqLenQ, spec.seqLenKv};
        expectSinkMaskObject(launchAiw(spec), bias);
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
    }
}

TEST(TestGpuFlydslSdpaAiwClasses, AddsASinkWithAMaskThroughTheEdgeCases)
{
    SKIP_IF_NO_DEVICES();
    // COVERAGE §5.7 with a sink and a mask together, both masks' dtypes:
    // - rows at the mask dtype's lowest finite value on every key: their scores sit about
    //   -FLT_MAX below the sink, so the sink takes the whole softmax (O = 0, LSE = the sink),
    //   as PyTorch gives -- the kernel's max must include the sink, not overflow past it;
    // - -inf on every key of every third row and the first 64 keys of the others, with a
    //   bottom-right diagonal: the rows the diagonal leaves no other key are keyless (O = 0,
    //   LSE = the sink);
    // - a sink of -inf with those rows: O = 0 and LSE = -inf, not NaN;
    // - a sink far above every score: LSE = the sink, O about 0.
    for(const auto bias : {AiwMask::F32, AiwMask::QUERY_TYPE})
    {
        for(const int64_t headDim : {64, 128})
        {
            SCOPED_TRACE("head_dim " + std::to_string(headDim) + ", " + maskName(bias));
            SdpaGraphSpec spec;
            spec.headDim = headDim;
            spec.heads = 4;
            spec.kvHeads = 2;
            spec.seqLenQ = 77;
            // A multiple of 8: the mask's rows must stay aligned for the object's loads.
            spec.seqLenKv = 104;
            spec.withSink = true;
            spec.stats = SdpaStats::LSE;
            setMask(spec, bias);
            spec.biasMasksRows = true;
            expectSinkMaskObject(launchAiw(spec), bias);
            spec.biasMasksRows = false;
            spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
            spec.biasNegativeInfinityRows = true;
            spec.biasNegativeInfinityLeadingKeys = 64;
            expectSinkMaskObject(launchAiw(spec), bias);
            spec.sinkNegativeInfinity = true;
            expectSinkMaskObject(launchAiw(spec), bias);
            spec.sinkNegativeInfinity = false;
            spec.biasNegativeInfinityRows = false;
            spec.biasNegativeInfinityLeadingKeys = 0;
            spec.sinkValue = 200.0F;
            expectSinkMaskObject(launchAiw(spec), bias);
            if(HasFatalFailure() || IsSkipped())
            {
                return;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
