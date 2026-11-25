// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/PackedPlanTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdSerialization.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file TestFlydslSdpaBwdSaveLoad.cpp
 * @brief The FlyDSL SDPA backward engines' plans saved and loaded through the plugin hooks.
 *
 * Every device case builds a plan through the engine, runs it on fixed inputs, saves it through
 * the plugin save hook, loads the bytes through the load hook on a second handle, runs the
 * loaded plan on the same inputs and compares its outputs with the fresh plan's, byte for byte.
 * No case compares with a CPU reference or within a tolerance: that comparison is valid only
 * for a form whose fresh output is bit-reproducible, which the FreshPlanOutputIsBitReproducible
 * cases show for each form before anything is compared.
 *
 * The split engine's plan is a slot program of three stages (prep, dkdv, dq), and its dq stage
 * is the `dq_reduce` handler's, with the reduce kernel as a second source (two code objects),
 * when the graph has a broadcast dBias; a graph whose V has other heads than K is served by the
 * u2 dkdv and dq handlers. The merged engine ships on gfx950 only, and its reduce kernel is the
 * second launch of the `merged_reduce` handler. The cases of an engine no device here can run
 * skip with a message that names the architecture they need.
 *
 * Refusals are driven through the load hook on a re-encoded or resealed copy of the saved
 * bytes, so that each one is the check it targets.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device;
namespace hooks = hip_kernel_provider::kernel_ingestor_engine::testing::plan_hooks;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using serialization::IngestorMultiKernelPayload;
using serialization::IngestorPlanKind;
using serialization::IngestorPlanRefusal;

using Outputs = std::vector<std::vector<uint8_t>>;

constexpr std::array<const char*, 4> OUTPUT_NAMES{"dQ", "dK", "dV", "dBias"};

// ---------------------------------------------------------------------------
// Cases and comparisons
// ---------------------------------------------------------------------------

/// Two sets of the values a caller supplies at execute. Each member differs between them.
BwdScalarValues firstValues()
{
    return {0.07F, 0x1234567, 77};
}

BwdScalarValues secondValues()
{
    return {0.11F, 0x7654321, 4242};
}

/// Fails, naming the output and the first offset, when @p actual differs from @p expected.
void expectSameOutputs(const Outputs& actual, const Outputs& expected, const std::string& what)
{
    ASSERT_EQ(actual.size(), expected.size()) << what;
    for(size_t group = 0; group < expected.size(); ++group)
    {
        ASSERT_EQ(actual[group].size(), expected[group].size())
            << what << ": " << OUTPUT_NAMES[group];
        const auto [left, right]
            = std::mismatch(actual[group].begin(), actual[group].end(), expected[group].begin());
        if(left != actual[group].end())
        {
            ADD_FAILURE() << what << ": " << OUTPUT_NAMES[group] << " first differs at byte offset "
                          << (left - actual[group].begin()) << " (0x" << std::hex
                          << static_cast<int>(*left) << " against 0x" << static_cast<int>(*right)
                          << ")";
        }
    }
}

/// Whether any output differs between @p lhs and @p rhs.
bool outputsDiffer(const Outputs& lhs, const Outputs& rhs)
{
    return lhs != rhs;
}

/// Every gradient was written: none still holds the all-ones bytes it was poisoned with.
void expectWritten(const Outputs& outputs)
{
    ASSERT_EQ(outputs.size(), OUTPUT_NAMES.size());
    for(size_t group = 0; group < 3; ++group)
    {
        const auto& bytes = outputs[group];
        ASSERT_FALSE(bytes.empty()) << OUTPUT_NAMES[group];
        EXPECT_FALSE(std::all_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b == 0xFF; }))
            << OUTPUT_NAMES[group] << " was never written";
    }
}

std::string deviceArch()
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
    {
        return "no device";
    }
    return packedArchDeviceProperties().gcnArchName;
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The split engine: the gfx11-generic content serves it on gfx11 devices, and the gfx950, gfx942
/// and gfx90a content on those devices.
class TestGpuFlydslSdpaBwdSaveLoad : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSaveLoad()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }
};

/// A case that runs on a gfx950 device only: no other architecture ships its content.
class Gfx950SaveLoad : public FlydslSdpaBwdDevice
{
protected:
    explicit Gfx950SaveLoad(const std::vector<std::string_view>& ops)
        : FlydslSdpaBwdDevice(ops)
    {
    }

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        const auto arch = deviceArch();
        if(arch.rfind("gfx950", 0) != 0)
        {
            GTEST_SKIP() << "this round trip runs on a gfx950 device; " << arch
                         << " ships no such content";
        }
        FlydslSdpaBwdDevice::SetUp();
    }
};

class TestGpuFlydslSdpaBwdSaveLoadMerged : public Gfx950SaveLoad
{
protected:
    TestGpuFlydslSdpaBwdSaveLoadMerged()
        : Gfx950SaveLoad(MERGED_OPS)
    {
    }
};

// ---------------------------------------------------------------------------
// Running a plan fresh, saving it, and running the load
// ---------------------------------------------------------------------------

/// Loads @p bytes through the load hook on a second handle and runs the loaded plan on @p rig's
/// inputs with @p values. The loaded plan reports the fresh plan's workspace.
void runLoaded(BwdRoundTripRig& rig,
               const std::vector<uint8_t>& bytes,
               const BwdScalarValues* values,
               Outputs& outputs)
{
    const auto loadHandle = hooks::makeHandle();
    hooks::LoadedPlan loaded(*loadHandle);
    ASSERT_NO_FATAL_FAILURE(loaded.load(bytes));
    EXPECT_EQ(loaded.plan().getWorkspaceSize(*loadHandle),
              rig.plan().getWorkspaceSize(rig.handle()));
    ASSERT_NO_FATAL_FAILURE(rig.run(loaded.plan(), *loadHandle, values, outputs));
}

/// The engine's plan for @p c: built, run once on @p values (so a benchmarking plan has chosen
/// what it serves), saved through the hook into @p bytes. @p fresh receives that run's output.
void buildRunAndSave(BwdRoundTripRig& rig,
                     std::string_view engine,
                     const BwdCase& c,
                     const BwdScalarValues* values,
                     std::vector<uint8_t>& bytes,
                     Outputs& fresh,
                     BwdScalarMode mode = BwdScalarMode::NONE,
                     const BwdScalarValues& baked = {})
{
    ASSERT_NO_FATAL_FAILURE(rig.build(engine, c, mode, baked));
    ASSERT_NO_FATAL_FAILURE(rig.run(rig.plan(), rig.handle(), values, fresh));
    ASSERT_NO_FATAL_FAILURE(expectWritten(fresh));
    ASSERT_NO_FATAL_FAILURE(hooks::save(rig.handle(), rig.context(), bytes));
}

/// The common round trip: build, run, save, load, run again, compare. @p bytes receives the
/// saved bytes.
void expectRoundTrip(std::string_view engine, const BwdCase& c, std::vector<uint8_t>& bytes)
{
    BwdRoundTripRig rig;
    Outputs fresh;
    ASSERT_NO_FATAL_FAILURE(buildRunAndSave(rig, engine, c, nullptr, bytes, fresh));
    for(int run = 0; run < 2; ++run)
    {
        Outputs loaded;
        ASSERT_NO_FATAL_FAILURE(runLoaded(rig, bytes, nullptr, loaded));
        ASSERT_NO_FATAL_FAILURE(
            expectSameOutputs(loaded, fresh, "loaded plan, run " + std::to_string(run)));
    }
}

/// A fresh plan run twice on identical inputs writes identical bytes.
void expectBitReproducible(std::string_view engine,
                           const BwdCase& c,
                           BwdScalarMode mode = BwdScalarMode::NONE)
{
    BwdRoundTripRig rig;
    ASSERT_NO_FATAL_FAILURE(rig.build(engine, c, mode));
    const auto values = firstValues();
    Outputs first;
    Outputs second;
    ASSERT_NO_FATAL_FAILURE(rig.run(rig.plan(), rig.handle(), &values, first));
    ASSERT_NO_FATAL_FAILURE(rig.run(rig.plan(), rig.handle(), &values, second));
    ASSERT_NO_FATAL_FAILURE(expectWritten(first));
    ASSERT_NO_FATAL_FAILURE(expectSameOutputs(second, first, "second run of one fresh plan"));
}

IngestorMultiKernelPayload kindTwoPayloadOf(const std::vector<uint8_t>& bytes)
{
    IngestorMultiKernelPayload payload;
    hooks::decodeMultiKernel(bytes, payload);
    return payload;
}

serialization::IngestorStagePayload* stageOf(IngestorMultiKernelPayload& payload,
                                             std::string_view slot)
{
    const auto found = std::find_if(payload.stages.begin(),
                                    payload.stages.end(),
                                    [&](const auto& s) { return s.slot == slot; });
    return found == payload.stages.end() ? nullptr : &*found;
}

std::vector<std::string> launchNamesOfStage(const serialization::IngestorStagePayload& stage)
{
    std::vector<std::string> names;
    names.reserve(stage.kernels.size());
    for(const auto& kernel : stage.kernels)
    {
        names.push_back(kernel.launchName);
    }
    return names;
}

/// The ragged offset uid and multiplier a launch saved for the tensor @p tensor.
std::pair<int64_t, int64_t> savedRaggedOffset(const MetadataValues& values,
                                              const std::string& tensor)
{
    return {std::get<int64_t>(values.at(tensor + ".ragged_offset_uid")),
            std::get<int64_t>(values.at(tensor + ".ragged_offset_multiplier"))};
}

/// The launch values of a stage saved exactly the ragged tensors of @p c, each with the offset
/// uid of its role and the multiplier of its sequence-major layout, and the rest dense.
void expectSavedRaggedOffsets(const MetadataValues& values, const BwdCase& c)
{
    const auto ragged = raggedTensorsOf(c);
    for(const char* tensor : {"q", "k", "v", "o", "do", "dq", "dk", "dv", "stats"})
    {
        SCOPED_TRACE(tensor);
        const auto found = std::find_if(
            ragged.begin(), ragged.end(), [&](const auto& r) { return r.name == tensor; });
        const auto expected = found == ragged.end()
                                  ? std::make_pair(int64_t{-1}, int64_t{0})
                                  : std::make_pair(found->offsetUid, found->multiplier);
        EXPECT_EQ(savedRaggedOffset(values, tensor), expected);
    }
    if(c.ragged == BwdRagged::QUERY_SIDE)
    {
        // The stats' strides are saved: packed (T, H) with no batch stride.
        EXPECT_EQ(std::get<int64_t>(values.at("stats.stride_batch")), 0);
        EXPECT_EQ(std::get<int64_t>(values.at("stats.stride_head")), 1);
        EXPECT_EQ(std::get<int64_t>(values.at("stats.stride_seq")), c.heads);
    }
}

/// What a round trip saved: the kind of the plan, the launch values of each stage, and the
/// launch names of the last stage (the dq stage of a split plan, the only stage of a merged
/// one; empty for a one-launch plan).
struct SavedPlan
{
    IngestorPlanKind kind = IngestorPlanKind::SINGLE_KERNEL_PLAN;
    std::vector<MetadataValues> stageValues;
    std::vector<std::string> lastStageLaunches;
};

SavedPlan savedPlanOf(const std::vector<uint8_t>& bytes)
{
    SavedPlan saved;
    saved.kind = hooks::headerOf(bytes).kind;
    if(saved.kind == IngestorPlanKind::MULTI_KERNEL_PLAN)
    {
        const auto payload = kindTwoPayloadOf(bytes);
        for(const auto& stage : payload.stages)
        {
            saved.stageValues.push_back(stage.launchValues);
        }
        if(!payload.stages.empty())
        {
            saved.lastStageLaunches = launchNamesOfStage(payload.stages.back());
        }
    }
    else
    {
        saved.stageValues.push_back(
            serialization::decodeIngestorPlan(bytes.data(), bytes.size()).launchValues);
    }
    return saved;
}

/// Round trips @p c through @p engine (the loaded plan's output equals the fresh plan's, byte for
/// byte), and expects the saved plan to be of kind @p kind, to end in the launches
/// @p lastStageLaunches, and to carry the ragged offsets of @p c in every stage.
void expectRaggedRoundTrip(std::string_view engine,
                           const BwdCase& c,
                           IngestorPlanKind kind,
                           const std::vector<std::string>& lastStageLaunches)
{
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(expectRoundTrip(engine, c, bytes));
    const auto saved = savedPlanOf(bytes);
    EXPECT_EQ(saved.kind, kind);
    EXPECT_EQ(saved.lastStageLaunches, lastStageLaunches);
    ASSERT_FALSE(saved.stageValues.empty());
    for(const auto& values : saved.stageValues)
    {
        expectSavedRaggedOffsets(values, c);
    }
}

/// A graph whose V has other heads than K and the launches its dq stage saves: the u2 dkdv and dq
/// handlers serve it, the dkdv stage with a dV-only launch over V's heads and the dq stage, when
/// the graph is padded, with a V-head tail-zero launch.
struct ValueHeadsRoundTrip
{
    BwdCase graph;
    std::vector<std::string> dqLaunches;
};

std::vector<ValueHeadsRoundTrip> valueHeadsRoundTripCases()
{
    return {
        {splitValueHeadsCase(DataType::HALF, 8, 4, 2, false), {""}},
        {splitValueHeadsCase(DataType::HALF, 8, 2, 4, false), {""}},
        {splitValueHeadsCase(DataType::HALF, 8, 4, 2, true), {"dq", "tailzero"}},
        {splitValueHeadsCase(DataType::HALF, 8, 2, 4, true), {"dq", "tailzero"}},
    };
}

// ---------------------------------------------------------------------------
// Bit reproducibility of each form, before any comparison relies on it
// ---------------------------------------------------------------------------

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, FreshPlanOutputIsBitReproducible)
{
    expectBitReproducible(SPLIT_ENGINE, plainCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, FreshPlanOutputIsBitReproducibleWithABroadcastDbias)
{
    // The dq stage launches its reduce kernel.
    expectBitReproducible(SPLIT_ENGINE, broadcastBiasCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, FreshPlanOutputIsBitReproducibleWithRuntimeScalars)
{
    expectBitReproducible(SPLIT_ENGINE, plainCase(), BwdScalarMode::RUNTIME);
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, FreshPlanOutputIsBitReproducibleWithAnF32Bias)
{
    expectBitReproducible(SPLIT_ENGINE, f32BroadcastBiasCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, FreshPlanOutputIsBitReproducibleWithValueHeads)
{
    for(const auto& c : valueHeadsRoundTripCases())
    {
        expectBitReproducible(SPLIT_ENGINE, c.graph);
    }
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, FreshPlanOutputIsBitReproducibleWithRaggedOffsets)
{
    expectBitReproducible(SPLIT_ENGINE, raggedQuerySideCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, FreshPlanOutputIsBitReproducibleWithRaggedOffsets)
{
    expectBitReproducible(MERGED_ENGINE, raggedQuerySideCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, FreshPlanOutputIsBitReproducibleWithEachRaggedShape)
{
    for(const auto& c : {raggedKeySideCase(), raggedGradientsCase(), paddedCase()})
    {
        expectBitReproducible(SPLIT_ENGINE, c);
    }
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, FreshPlanOutputIsBitReproducibleWithEachRaggedShape)
{
    for(const auto& c : {raggedKeySideCase(), raggedGradientsCase(), paddedCase()})
    {
        expectBitReproducible(MERGED_ENGINE, c);
    }
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, FreshPlanOutputIsBitReproducible)
{
    expectBitReproducible(MERGED_ENGINE, plainCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, FreshPlanOutputIsBitReproducibleWithItsReduceKernel)
{
    expectBitReproducible(MERGED_ENGINE, broadcastBiasCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, FreshPlanOutputIsBitReproducibleWithRuntimeScalars)
{
    expectBitReproducible(MERGED_ENGINE, plainCase(), BwdScalarMode::RUNTIME);
}

// ---------------------------------------------------------------------------
// Round trips
// ---------------------------------------------------------------------------

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsASlottedPlanWithExactOutput)
{
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(expectRoundTrip(SPLIT_ENGINE, plainCase(), bytes));

    // Three slots, one kernel each, none of them named.
    IngestorMultiKernelPayload payload;
    ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));
    ASSERT_EQ(payload.stages.size(), 3U);
    for(const auto slot : {"prep", "dkdv", "dq"})
    {
        const auto* stage = stageOf(payload, slot);
        ASSERT_NE(stage, nullptr) << slot;
        EXPECT_EQ(launchNamesOfStage(*stage), std::vector<std::string>{""}) << slot;
    }
    EXPECT_GT(payload.workspaceBytes, 0U);
}

// A graph with a ragged query side: every slot saves the offsets and the stats' strides, and the
// dq stage launches the tail-zero kernel for the dense dK and dV beside its own kernel, saved as
// a launch of its own.
TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsASlottedPlanWithRaggedOffsets)
{
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(expectRoundTrip(SPLIT_ENGINE, raggedQuerySideCase(), bytes));

    IngestorMultiKernelPayload payload;
    ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));
    ASSERT_EQ(payload.stages.size(), 3U);
    for(const auto& stage : payload.stages)
    {
        SCOPED_TRACE(stage.slot);
        expectSavedRaggedOffsets(stage.launchValues, raggedQuerySideCase());
    }
    const auto* dq = stageOf(payload, "dq");
    ASSERT_NE(dq, nullptr);
    EXPECT_EQ(launchNamesOfStage(*dq), (std::vector<std::string>{"dq", "tailzero"}));
}

// The K side ragged: K, V, dK and dV carry offsets and dQ is dense, so the dq stage still
// launches the tail-zero kernel.
TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsASlottedPlanWithARaggedKeySide)
{
    expectRaggedRoundTrip(
        SPLIT_ENGINE, raggedKeySideCase(), IngestorPlanKind::MULTI_KERNEL_PLAN, {"dq", "tailzero"});
}

// Every gradient ragged: nothing is dense, so no tail-zero launch runs.
TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsASlottedPlanWithEveryGradientRagged)
{
    expectRaggedRoundTrip(
        SPLIT_ENGINE, raggedGradientsCase(), IngestorPlanKind::MULTI_KERNEL_PLAN, {""});
}

// A padded graph with nothing ragged: every gradient is dense, so the tail-zero launch runs and
// every ragged offset is saved as absent.
TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsASlottedPlanOfAPaddedGraphWithNothingRagged)
{
    expectRaggedRoundTrip(
        SPLIT_ENGINE, paddedCase(), IngestorPlanKind::MULTI_KERNEL_PLAN, {"dq", "tailzero"});
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsASlottedPlanUsingTheSourcesReduceForm)
{
    const auto bwdCase = broadcastBiasCase();
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(expectRoundTrip(SPLIT_ENGINE, bwdCase, bytes));

    IngestorMultiKernelPayload payload;
    ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));
    const auto* dq = stageOf(payload, "dq");
    ASSERT_NE(dq, nullptr);
    EXPECT_EQ(dq->dispatchSymbol, FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1);
    EXPECT_EQ(launchNamesOfStage(*dq), (std::vector<std::string>{"dq", "dbias_reduce"}));
    ASSERT_EQ(dq->kernels.size(), 2U);
    EXPECT_NE(dq->kernels[0].symbol, dq->kernels[1].symbol);
    // The kernel variant decides whether the `dbias_reduce` entry names a second kernel of the dq
    // row's own object, which the payload stores once, or an object of its own. gfx950 always
    // shares; gfx942 and gfx90a share for a FLOAT query only, and their 16-bit rows use a
    // standalone object; gfx11 never shares.
    const auto arch = deviceArch();
    const bool sharesTheObject = arch.rfind("gfx950", 0) == 0
                                 || ((arch.rfind("gfx942", 0) == 0 || arch.rfind("gfx90a", 0) == 0)
                                     && bwdCase.dtype == DataType::FLOAT);
    EXPECT_EQ(dq->kernels[0].codeObjectIndex == dq->kernels[1].codeObjectIndex, sharesTheObject)
        << arch;
}

// An f32 bias beside a half-precision Q: the dq stage's reduce kernel and the dBias it writes are
// as wide as the bias, and the saved launch values record the bias width beside Q's.
TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsASlottedPlanWithAnF32BiasBesideAHalfPrecisionQuery)
{
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(expectRoundTrip(SPLIT_ENGINE, f32BroadcastBiasCase(), bytes));

    IngestorMultiKernelPayload payload;
    ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));
    const auto* dq = stageOf(payload, "dq");
    ASSERT_NE(dq, nullptr);
    EXPECT_EQ(dq->dispatchSymbol, FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1);
    EXPECT_EQ(launchNamesOfStage(*dq), (std::vector<std::string>{"dq", "dbias_reduce"}));
    EXPECT_EQ(std::get<int64_t>(dq->launchValues.at("element_bytes")), 2);
    EXPECT_EQ(std::get<int64_t>(dq->launchValues.at("bias_element_bytes")), 4);
}

// Each case writes the dV of V's heads, not K's: the loaded plan's dV-only launch and tail-zero
// launch take the grids over V's heads, so its output equals the fresh plan's.
TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsASlottedPlanWithValueHeadsDifferentFromKeyHeads)
{
    for(const auto& c : valueHeadsRoundTripCases())
    {
        SCOPED_TRACE("heads " + std::to_string(c.graph.heads) + ", key heads "
                     + std::to_string(c.graph.kvHeads) + ", value heads "
                     + std::to_string(c.graph.valueHeads)
                     + (c.graph.padded() ? ", padded" : ", unpadded"));
        std::vector<uint8_t> bytes;
        ASSERT_NO_FATAL_FAILURE(expectRoundTrip(SPLIT_ENGINE, c.graph, bytes));

        IngestorMultiKernelPayload payload;
        ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));
        ASSERT_EQ(payload.stages.size(), 3U);
        const auto* dkdv = stageOf(payload, "dkdv");
        const auto* dq = stageOf(payload, "dq");
        ASSERT_NE(dkdv, nullptr);
        ASSERT_NE(dq, nullptr);
        EXPECT_EQ(dkdv->dispatchSymbol, FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1);
        EXPECT_EQ(launchNamesOfStage(*dkdv), (std::vector<std::string>{"dkdv", "dvonly"}));
        EXPECT_EQ(dq->dispatchSymbol, FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1);
        EXPECT_EQ(launchNamesOfStage(*dq), c.dqLaunches);
        for(const auto* stage : {dkdv, dq})
        {
            EXPECT_EQ(std::get<int64_t>(stage->launchValues.at("num_v_heads")), c.graph.valueHeads);
        }
    }
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, RoundTripsAMergedPlanWithRaggedOffsets)
{
    std::vector<uint8_t> bytes;
    ASSERT_NO_FATAL_FAILURE(expectRoundTrip(MERGED_ENGINE, raggedQuerySideCase(), bytes));

    // The tail-zero launch zeroes the dense dK and dV past their lengths: a second launch, so a
    // kind 2 payload.
    EXPECT_EQ(hooks::headerOf(bytes).kind, IngestorPlanKind::MULTI_KERNEL_PLAN);
    IngestorMultiKernelPayload payload;
    ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));
    ASSERT_EQ(payload.stages.size(), 1U);
    EXPECT_EQ(payload.stages[0].dispatchSymbol, FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1);
    EXPECT_EQ(launchNamesOfStage(payload.stages[0]),
              (std::vector<std::string>{"merged", "tailzero"}));
    expectSavedRaggedOffsets(payload.stages[0].launchValues, raggedQuerySideCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, RoundTripsAMergedPlanWithARaggedKeySide)
{
    expectRaggedRoundTrip(MERGED_ENGINE,
                          raggedKeySideCase(),
                          IngestorPlanKind::MULTI_KERNEL_PLAN,
                          {"merged", "tailzero"});
}

// Every gradient ragged: the merged kernel is the plan's only launch, a kind 1 payload.
TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, RoundTripsAMergedPlanWithEveryGradientRagged)
{
    expectRaggedRoundTrip(
        MERGED_ENGINE, raggedGradientsCase(), IngestorPlanKind::SINGLE_KERNEL_PLAN, {});
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, RoundTripsAMergedPlanOfAPaddedGraphWithNothingRagged)
{
    expectRaggedRoundTrip(
        MERGED_ENGINE, paddedCase(), IngestorPlanKind::MULTI_KERNEL_PLAN, {"merged", "tailzero"});
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, RoundTripsAMergedPlanWithAndWithoutItsReduceKernel)
{
    {
        SCOPED_TRACE("without a dBias: one launch, a kind 1 payload");
        std::vector<uint8_t> bytes;
        ASSERT_NO_FATAL_FAILURE(expectRoundTrip(MERGED_ENGINE, plainCase(), bytes));
        const auto header = hooks::headerOf(bytes);
        EXPECT_EQ(header.kind, IngestorPlanKind::SINGLE_KERNEL_PLAN);
        EXPECT_EQ(header.formatMinor, serialization::INGESTOR_PLAN_SINGLE_KERNEL_MINOR);
    }
    {
        SCOPED_TRACE("with a broadcast dBias: the reduce kernel runs, a kind 2 payload");
        std::vector<uint8_t> bytes;
        ASSERT_NO_FATAL_FAILURE(expectRoundTrip(MERGED_ENGINE, broadcastBiasCase(), bytes));
        const auto header = hooks::headerOf(bytes);
        EXPECT_EQ(header.kind, IngestorPlanKind::MULTI_KERNEL_PLAN);
        IngestorMultiKernelPayload payload;
        ASSERT_NO_FATAL_FAILURE(hooks::decodeMultiKernel(bytes, payload));
        ASSERT_EQ(payload.stages.size(), 1U);
        EXPECT_EQ(payload.stages[0].dispatchSymbol,
                  FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1);
        EXPECT_EQ(launchNamesOfStage(payload.stages[0]),
                  (std::vector<std::string>{"merged", "dbias_reduce"}));
    }
}

// ---------------------------------------------------------------------------
// Runtime pass-by-value scalars
// ---------------------------------------------------------------------------

/// A broadcast dBias at a head dim the merged engine ships a dropout object for. The merged
/// objects with dropout and a bias that store the dBias (store_db is not 0) are, in half
/// precision, the d224 and d256 ones: the narrower dropout objects with a bias build the dQ
/// role without its dB store, and a graph that asks for a dBias does not match them.
BwdCase broadcastBiasWithDropoutObjectCase()
{
    auto c = broadcastBiasCase();
    c.headDim = 224;
    return c;
}

/// One fresh plan object serves the whole case, so what it selects can never differ between the
/// runs compared: it runs with the first values, is saved, and then the loaded plan and the
/// same fresh plan run with the second values. Their outputs are equal, and differ from the
/// first values' output, so each value is read at execute.
void expectRuntimeScalarsRoundTrip(std::string_view engine, const BwdCase& c)
{
    BwdRoundTripRig rig;
    const auto first = firstValues();
    const auto second = secondValues();
    std::vector<uint8_t> bytes;
    Outputs freshFirst;
    ASSERT_NO_FATAL_FAILURE(
        buildRunAndSave(rig, engine, c, &first, bytes, freshFirst, BwdScalarMode::RUNTIME));

    Outputs loadedSecond;
    ASSERT_NO_FATAL_FAILURE(runLoaded(rig, bytes, &second, loadedSecond));
    Outputs freshSecond;
    ASSERT_NO_FATAL_FAILURE(rig.run(rig.plan(), rig.handle(), &second, freshSecond));
    ASSERT_NO_FATAL_FAILURE(
        expectSameOutputs(loadedSecond, freshSecond, "loaded plan, second values"));
    EXPECT_TRUE(outputsDiffer(freshFirst, freshSecond))
        << "the second values give the first values' output: a value is not read at execute";

    // The loaded plan reads the first values too, and then gives the first values' output.
    Outputs loadedFirst;
    ASSERT_NO_FATAL_FAILURE(runLoaded(rig, bytes, &first, loadedFirst));
    ASSERT_NO_FATAL_FAILURE(
        expectSameOutputs(loadedFirst, freshFirst, "loaded plan, first values"));

    // Each scalar is read on its own.
    const std::array<std::pair<const char*, BwdScalarValues>, 3> single{
        {{"scale", {second.scale, first.seed, first.offset}},
         {"seed", {first.scale, second.seed, first.offset}},
         {"offset", {first.scale, first.seed, second.offset}}}};
    for(const auto& [name, values] : single)
    {
        Outputs changed;
        ASSERT_NO_FATAL_FAILURE(rig.run(rig.plan(), rig.handle(), &values, changed));
        EXPECT_TRUE(outputsDiffer(freshFirst, changed))
            << "changing only the " << name << " does not change the output";
    }

    // The payload lists the three UIDs and holds no value that was supplied. A plan one launch
    // serves is a kind 1 payload; a plan with a second launch is a kind 2 payload.
    std::vector<int64_t> listed;
    if(hooks::headerOf(bytes).kind == IngestorPlanKind::MULTI_KERNEL_PLAN)
    {
        listed = kindTwoPayloadOf(bytes).runtimePassByValueUids;
    }
    else
    {
        listed
            = serialization::decodeIngestorPlan(bytes.data(), bytes.size()).runtimePassByValueUids;
    }
    EXPECT_EQ(listed, (std::vector<int64_t>{uid::SCALE, uid::SEED, uid::OFFSET}));
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsRuntimePassByValueScaleSeedAndOffset)
{
    expectRuntimeScalarsRoundTrip(SPLIT_ENGINE, plainCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsRuntimePassByValueScalarsWithABroadcastBias)
{
    expectRuntimeScalarsRoundTrip(SPLIT_ENGINE, broadcastBiasCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, RoundTripsRuntimePassByValueScaleSeedAndOffset)
{
    expectRuntimeScalarsRoundTrip(MERGED_ENGINE, plainCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, RoundTripsRuntimePassByValueScalarsWithABroadcastBias)
{
    expectRuntimeScalarsRoundTrip(MERGED_ENGINE, broadcastBiasWithDropoutObjectCase());
}

// Runtime scalars beside ragged offsets: the scale, seed and offset are read at execute through
// the plan the launch values restored, and the loaded plan's ragged tensors are addressed as the
// fresh plan's.
TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsRuntimePassByValueScalarsWithRaggedOffsets)
{
    expectRuntimeScalarsRoundTrip(SPLIT_ENGINE, raggedQuerySideCase());
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, RoundTripsRuntimePassByValueScalarsWithRaggedOffsets)
{
    expectRuntimeScalarsRoundTrip(MERGED_ENGINE, raggedQuerySideCase());
}

void expectSavedBytesIgnoreUserValues(std::string_view engine)
{
    BwdRoundTripRig rig;
    const auto first = firstValues();
    const auto second = secondValues();
    std::vector<uint8_t> afterFirst;
    Outputs freshFirst;
    ASSERT_NO_FATAL_FAILURE(buildRunAndSave(
        rig, engine, plainCase(), &first, afterFirst, freshFirst, BwdScalarMode::RUNTIME));

    Outputs freshSecond;
    ASSERT_NO_FATAL_FAILURE(rig.run(rig.plan(), rig.handle(), &second, freshSecond));
    ASSERT_TRUE(outputsDiffer(freshFirst, freshSecond)) << "the two value sets give one output";
    std::vector<uint8_t> afterSecond;
    ASSERT_NO_FATAL_FAILURE(hooks::save(rig.handle(), rig.context(), afterSecond));

    EXPECT_EQ(afterFirst, afterSecond);
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, SavedBytesDoNotDependOnTheUserSuppliedValues)
{
    expectSavedBytesIgnoreUserValues(SPLIT_ENGINE);
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, SavedBytesDoNotDependOnTheUserSuppliedValues)
{
    expectSavedBytesIgnoreUserValues(MERGED_ENGINE);
}

/// The scalars are baked into the graph: the saved plan holds them, lists no UID, and the
/// loaded plan, given no value at execute, matches the fresh plan.
void expectBakedScalarsRoundTrip(std::string_view engine)
{
    const BwdScalarValues baked{0.09F, 31, 5};
    BwdRoundTripRig rig;
    std::vector<uint8_t> bytes;
    Outputs fresh;
    ASSERT_NO_FATAL_FAILURE(buildRunAndSave(
        rig, engine, plainCase(), nullptr, bytes, fresh, BwdScalarMode::BAKED, baked));

    Outputs loaded;
    ASSERT_NO_FATAL_FAILURE(runLoaded(rig, bytes, nullptr, loaded));
    ASSERT_NO_FATAL_FAILURE(expectSameOutputs(loaded, fresh, "loaded plan"));

    // The baked values are the saved ones: another graph with other baked values computes
    // another output, so the equality above is not one of values that did not matter.
    {
        BwdRoundTripRig other;
        Outputs otherOutputs;
        ASSERT_NO_FATAL_FAILURE(
            other.build(engine, plainCase(), BwdScalarMode::BAKED, BwdScalarValues{0.13F, 99, 6}));
        ASSERT_NO_FATAL_FAILURE(other.run(other.plan(), other.handle(), nullptr, otherOutputs));
        EXPECT_TRUE(outputsDiffer(fresh, otherOutputs));
    }

    const auto header = hooks::headerOf(bytes);
    std::vector<int64_t> listed;
    std::vector<serialization::IngestorStagePayload> stages;
    if(header.kind == IngestorPlanKind::MULTI_KERNEL_PLAN)
    {
        const auto payload = kindTwoPayloadOf(bytes);
        listed = payload.runtimePassByValueUids;
        stages = payload.stages;
    }
    else
    {
        const auto decoded = serialization::decodeIngestorPlan(bytes.data(), bytes.size());
        listed = decoded.runtimePassByValueUids;
        serialization::IngestorStagePayload stage;
        stage.launchValues = decoded.launchValues;
        stages.push_back(stage);
    }
    EXPECT_TRUE(listed.empty());
    const auto bakedOf = [&](const std::string& name) {
        for(const auto& stage : stages)
        {
            const auto found = stage.launchValues.find(name);
            if(found != stage.launchValues.end())
            {
                return found->second;
            }
        }
        return hipdnn_plugin_sdk::ingestor::MetadataValue{};
    };
    EXPECT_EQ(bakedOf("seed.baked"), hipdnn_plugin_sdk::ingestor::MetadataValue{baked.seed});
    EXPECT_EQ(bakedOf("offset.baked"), hipdnn_plugin_sdk::ingestor::MetadataValue{baked.offset});
    EXPECT_EQ(bakedOf("scale.baked"),
              hipdnn_plugin_sdk::ingestor::MetadataValue{static_cast<double>(baked.scale)});
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RoundTripsABakedScalarFromItsSavedValue)
{
    expectBakedScalarsRoundTrip(SPLIT_ENGINE);
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoadMerged, RoundTripsABakedScalarFromItsSavedValue)
{
    expectBakedScalarsRoundTrip(MERGED_ENGINE);
}

// ---------------------------------------------------------------------------
// The format at the hook
// ---------------------------------------------------------------------------

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, AMultiKernelPlanIsSavedWithAFormat1Point1Header)
{
    BwdRoundTripRig rig;
    std::vector<uint8_t> bytes;
    Outputs fresh;
    ASSERT_NO_FATAL_FAILURE(buildRunAndSave(rig, SPLIT_ENGINE, plainCase(), nullptr, bytes, fresh));

    const auto header = hooks::headerOf(bytes);
    EXPECT_EQ(header.kind, IngestorPlanKind::MULTI_KERNEL_PLAN);
    EXPECT_EQ(header.formatMajor, serialization::INGESTOR_PLAN_FORMAT_MAJOR);
    EXPECT_EQ(header.formatMinor, serialization::INGESTOR_PLAN_MULTI_KERNEL_MINOR);
    EXPECT_EQ(header.formatMinor, 1U);
}

// ---------------------------------------------------------------------------
// Refusals at the hook
// ---------------------------------------------------------------------------

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RefusesALoadWhoseListedUidsAreNotBoundByTheHandlers)
{
    BwdRoundTripRig rig;
    std::vector<uint8_t> bytes;
    Outputs fresh;
    ASSERT_NO_FATAL_FAILURE(buildRunAndSave(rig, SPLIT_ENGINE, plainCase(), nullptr, bytes, fresh));
    auto payload = kindTwoPayloadOf(bytes);
    ASSERT_TRUE(payload.runtimePassByValueUids.empty());

    // The control: the saved bytes load.
    const auto loadHandle = hooks::makeHandle();
    {
        hooks::LoadedPlan loaded(*loadHandle);
        ASSERT_NO_FATAL_FAILURE(loaded.load(bytes));
    }

    payload.runtimePassByValueUids = {uid::SEED};
    hooks::expectLoadRefusal(*loadHandle,
                             serialization::encodeIngestorPlan(payload),
                             IngestorPlanRefusal::INCOMPATIBLE,
                             "lists the runtime pass-by-value UIDs");
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RefusesAStageWhoseHandlerRefusesItsLaunchValues)
{
    BwdRoundTripRig rig;
    std::vector<uint8_t> bytes;
    Outputs fresh;
    ASSERT_NO_FATAL_FAILURE(buildRunAndSave(rig, SPLIT_ENGINE, plainCase(), nullptr, bytes, fresh));
    auto payload = kindTwoPayloadOf(bytes);

    // A batch of 0 is out of the range the contract gives the value.
    auto& stage = payload.stages.at(0);
    ASSERT_NE(stage.launchValues.find("batch"), stage.launchValues.end());
    stage.launchValues["batch"] = int64_t{0};

    const auto loadHandle = hooks::makeHandle();
    const auto message = hooks::expectLoadRefusal(*loadHandle,
                                                  serialization::encodeIngestorPlan(payload),
                                                  IngestorPlanRefusal::INCOMPATIBLE,
                                                  "'batch'");
    EXPECT_NE(message.find(stage.dispatchSymbol), std::string::npos) << message;
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RefusesADamagedCodeObjectInALaterSlot)
{
    BwdRoundTripRig rig;
    std::vector<uint8_t> bytes;
    Outputs fresh;
    ASSERT_NO_FATAL_FAILURE(buildRunAndSave(rig, SPLIT_ENGINE, plainCase(), nullptr, bytes, fresh));
    const auto payload = kindTwoPayloadOf(bytes);
    ASSERT_GE(payload.stages.size(), 2U);

    // The second slot's object, which is not the first slot's.
    const auto index = payload.stages.at(1).kernels.at(0).codeObjectIndex;
    ASSERT_NE(index, payload.stages.at(0).kernels.at(0).codeObjectIndex);
    auto damaged = bytes;
    ASSERT_NO_FATAL_FAILURE(
        hooks::damageStoredCodeObject(damaged, payload.codeObjects.at(index).codeObject));

    const auto loadHandle = hooks::makeHandle();
    hooks::expectLoadRefusal(*loadHandle,
                             damaged,
                             IngestorPlanRefusal::DAMAGED,
                             "of code object " + std::to_string(index) + " does not match");
}

TEST_F(TestGpuFlydslSdpaBwdSaveLoad, RefusesALaunchSetThatDiffersFromTheHandlersAtTheHook)
{
    const auto loadHandle = hooks::makeHandle();
    {
        SCOPED_TRACE("the reduce kernel dropped from the dq stage of a broadcast-dBias plan");
        BwdRoundTripRig rig;
        std::vector<uint8_t> bytes;
        Outputs fresh;
        ASSERT_NO_FATAL_FAILURE(
            buildRunAndSave(rig, SPLIT_ENGINE, broadcastBiasCase(), nullptr, bytes, fresh));
        auto payload = kindTwoPayloadOf(bytes);
        auto* dq = stageOf(payload, "dq");
        ASSERT_NE(dq, nullptr);
        ASSERT_EQ(dq->kernels.size(), 2U);
        dq->kernels.pop_back();

        hooks::expectLoadRefusal(*loadHandle,
                                 serialization::encodeIngestorPlan(payload),
                                 IngestorPlanRefusal::INCOMPATIBLE,
                                 "for the saved problem");
    }
    {
        SCOPED_TRACE("an extra kernel added to a no-dBias plan");
        BwdRoundTripRig rig;
        std::vector<uint8_t> bytes;
        Outputs fresh;
        ASSERT_NO_FATAL_FAILURE(
            buildRunAndSave(rig, SPLIT_ENGINE, plainCase(), nullptr, bytes, fresh));
        auto payload = kindTwoPayloadOf(bytes);
        auto* dq = stageOf(payload, "dq");
        ASSERT_NE(dq, nullptr);
        ASSERT_EQ(dq->kernels.size(), 1U);
        auto extra = dq->kernels.front();
        extra.launchName = "extra";
        dq->kernels.front().launchName = "dq";
        dq->kernels.push_back(std::move(extra));

        hooks::expectLoadRefusal(*loadHandle,
                                 serialization::encodeIngestorPlan(payload),
                                 IngestorPlanRefusal::INCOMPATIBLE,
                                 "for the saved problem");
    }
}

// ---------------------------------------------------------------------------
// The engines report the note
// ---------------------------------------------------------------------------

/// The engine reports the plan serialization note on its loaded descriptor set. Reads no
/// device and no census binary: the set is the one this build staged, and an engine this build
/// ships nothing for has none to read. @p shippedOp is the content op the engine's kernels
/// are staged under: a device that ships it must have a loaded set.
void expectLoadedSetReportsTheNote(std::string_view engineName,
                                   std::string_view shippedFor,
                                   std::string_view shippedOp)
{
    const auto* set = findLoadedSet(engineName);
    if(set == nullptr)
    {
        reportUnstaged(shippedOp,
                       std::string(engineName) + " has no content staged for this build's device ("
                           + deviceArch() + "); it ships " + std::string(shippedFor) + " content");
        return;
    }
    expectEngineReportsPlanSerialization(*set);
}

TEST(TestFlydslSdpaBwdSaveLoad, SplitEngineReportsThePlanSerializationNoteOnItsLoadedSet)
{
    expectLoadedSetReportsTheNote(
        SPLIT_ENGINE, "gfx11-generic, gfx950, gfx942 and gfx90a", "sdpa_bwd_dq");
}

TEST(TestFlydslSdpaBwdSaveLoad, MergedEngineReportsThePlanSerializationNoteOnItsLoadedSet)
{
    expectLoadedSetReportsTheNote(MERGED_ENGINE, "gfx950", "sdpa_bwd_merged");
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
