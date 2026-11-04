// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaHeadPad.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaHeadPad.cpp
 * @brief Head dims off 8 on the multiple-of-8 objects (FlydslSdpaHeadPad.hpp): which graphs are
 *        padded, the padded graph, the workspace plan, and that every other graph reaches the
 *        engine's own functions and handler untouched. The device cases are the engines'
 *        (TestFlydslSdpaRdnaEngine.cpp, TestFlydslSdpaCdnaEngine.cpp).
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;
using hipdnn_plugin_sdk::ingestor::PreparedDispatch;

// Restated rather than shared with the pack, as TestFlydslSdpaCdnaEngine.cpp does.
constexpr PackSymbols FLYDSL_SDPA_CDNA{"hipkernel:flydsl_sdpa_cdna",
                                       "hipkernel.flydsl_sdpa_cdna.graph_match",
                                       "",
                                       "hipkernel.flydsl_sdpa_cdna.kernel_match",
                                       "hipkernel.flydsl_sdpa_cdna.score",
                                       "hipkernel.flydsl_sdpa_cdna.dispatch",
                                       "flydsl_sdpa_cdna.q.uid",
                                       "flydsl_sdpa_cdna.k.uid",
                                       "flydsl_sdpa_cdna.o.uid"};

/// A compact graph at @p headDim (V @p headDimV): every row pitch the head dim itself.
SdpaGraphSpec compactSpec(int64_t headDim, std::optional<int64_t> headDimV = std::nullopt)
{
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.batch = 2;
    spec.heads = 4;
    spec.kvHeads = 2;
    spec.seqLenQ = 37;
    spec.seqLenKv = 61;
    spec.headDim = headDim;
    spec.valueHeadDim = headDimV;
    spec.stats = SdpaStats::LSE;
    return spec;
}

const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensorOf(const MatchContext& context,
                                                                       int64_t uid)
{
    const auto* tensor = flydsl_sdpa::findTensor(context, uid);
    EXPECT_NE(tensor, nullptr) << uid;
    return *tensor;
}

std::vector<int64_t> dimsOf(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    return {tensor.dims()->begin(), tensor.dims()->end()};
}

std::vector<int64_t> stridesOf(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    return {tensor.strides()->begin(), tensor.strides()->end()};
}

TEST(TestFlydslSdpaHeadPad, PadsExactlyTheGraphsWithAHeadDimOffEight)
{
    for(const int64_t headDim : {8, 16, 64, 128, 256})
    {
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(compactSpec(headDim)));
        EXPECT_FALSE(flydsl_sdpa::needsHeadPad(fixture.context())) << headDim;
    }
    for(const int64_t headDim : {1, 7, 12, 23, 241, 257})
    {
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(compactSpec(headDim)));
        EXPECT_TRUE(flydsl_sdpa::needsHeadPad(fixture.context())) << headDim;
    }
    // V's head dim alone, either way round.
    const GraphFixture valueOdd(buildFlydslSdpaRdnaGraph(compactSpec(64, 37)));
    EXPECT_TRUE(flydsl_sdpa::needsHeadPad(valueOdd.context()));
    const GraphFixture queryOdd(buildFlydslSdpaRdnaGraph(compactSpec(37, 64)));
    EXPECT_TRUE(flydsl_sdpa::needsHeadPad(queryOdd.context()));
}

// The next multiple of 8, but 225-255 to 256, where the exact 256 objects serve (gfx12's aiw
// drops only there past 224).
TEST(TestFlydslSdpaHeadPad, PadsToTheNextMultipleOfEightOrTo256)
{
    EXPECT_EQ(flydsl_sdpa::headPadWidth(1), 8);
    EXPECT_EQ(flydsl_sdpa::headPadWidth(7), 8);
    EXPECT_EQ(flydsl_sdpa::headPadWidth(113), 120);
    EXPECT_EQ(flydsl_sdpa::headPadWidth(217), 224);
    EXPECT_EQ(flydsl_sdpa::headPadWidth(225), 256);
    EXPECT_EQ(flydsl_sdpa::headPadWidth(241), 256);
    EXPECT_EQ(flydsl_sdpa::headPadWidth(255), 256);
    EXPECT_EQ(flydsl_sdpa::headPadWidth(257), 264);
}

TEST(TestFlydslSdpaHeadPad, PadsQKVAndOAndNothingElse)
{
    auto spec = compactSpec(37, 23);
    spec.withBias = true;
    spec.biasDataType = DataType::HALF;
    spec.biasDims = {spec.batch, spec.heads, spec.seqLenQ, spec.seqLenKv};
    spec.scale = 0.3F;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto padded = flydsl_sdpa::buildHeadPadGraph(fixture.context());
    ASSERT_NE(padded, nullptr);
    const auto context = padded->contextFor(fixture.context());

    // Q and K 40 wide, V and O 24; each contiguous BSHD, declaring 16-byte alignment.
    const auto expectPadded = [&](int64_t uid, int64_t heads, int64_t sequence, int64_t width) {
        const auto& tensor = tensorOf(context, uid);
        EXPECT_EQ(dimsOf(tensor), (std::vector<int64_t>{spec.batch, heads, sequence, width}))
            << uid;
        EXPECT_EQ(stridesOf(tensor),
                  (std::vector<int64_t>{sequence * heads * width, width, heads * width, 1}))
            << uid;
        EXPECT_EQ(tensor.alignment(), 16) << uid;
        EXPECT_EQ(tensor.data_type(), DataType::HALF) << uid;
    };
    expectPadded(FLYDSL_SDPA_RDNA_Q_UID, spec.heads, spec.seqLenQ, 40);
    expectPadded(FLYDSL_SDPA_RDNA_K_UID, spec.kvHeads, spec.seqLenKv, 40);
    expectPadded(FLYDSL_SDPA_RDNA_V_UID, spec.kvHeads, spec.seqLenKv, 24);
    expectPadded(FLYDSL_SDPA_RDNA_O_UID, spec.heads, spec.seqLenQ, 24);

    // The bias and the LSE as the graph has them; the scale the graph's own.
    for(const int64_t uid : {FLYDSL_SDPA_RDNA_BIAS_UID, FLYDSL_SDPA_RDNA_STATS_UID})
    {
        const auto& before = tensorOf(fixture.context(), uid);
        const auto& after = tensorOf(context, uid);
        EXPECT_EQ(dimsOf(after), dimsOf(before)) << uid;
        EXPECT_EQ(stridesOf(after), stridesOf(before)) << uid;
        EXPECT_EQ(after.alignment(), before.alignment()) << uid;
    }
    EXPECT_EQ(context.graph.getTensorMap().size(), fixture.context().graph.getTensorMap().size());
    const auto* attributes = flydsl_sdpa::sdpaNode(context);
    const auto* original = flydsl_sdpa::sdpaNode(fixture.context());
    ASSERT_NE(attributes, nullptr);
    ASSERT_NE(original, nullptr);
    EXPECT_EQ(attributes->attn_scale_value().has_value(), original->attn_scale_value().has_value());
    EXPECT_EQ(attributes->attn_scale_value().value_or(-1.0F),
              original->attn_scale_value().value_or(-1.0F));
    EXPECT_EQ(attributes->q_tensor_uid(), original->q_tensor_uid());
    EXPECT_EQ(attributes->o_tensor_uid(), original->o_tensor_uid());
    EXPECT_FALSE(flydsl_sdpa::needsHeadPad(context));
}

TEST(TestFlydslSdpaHeadPad, PacksTheCopiesTokenMajorThroughTheSameTables)
{
    auto spec = compactSpec(53);
    setPacked(spec, {37, 11}, {61, 29});
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto padded = flydsl_sdpa::buildHeadPadGraph(fixture.context());
    ASSERT_NE(padded, nullptr);
    const auto context = padded->contextFor(fixture.context());
    for(const auto& [uid, heads] : {std::pair<int64_t, int64_t>{FLYDSL_SDPA_RDNA_Q_UID, 4},
                                    {FLYDSL_SDPA_RDNA_K_UID, 2},
                                    {FLYDSL_SDPA_RDNA_O_UID, 4}})
    {
        const auto& before = tensorOf(fixture.context(), uid);
        const auto& after = tensorOf(context, uid);
        EXPECT_EQ(after.ragged_offset_tensor_uid(), before.ragged_offset_tensor_uid()) << uid;
        // Offsets still count tokens: the multiplier is the new sequence stride.
        EXPECT_EQ(after.ragged_offset_multiplier(), heads * 56) << uid;
        EXPECT_EQ(after.strides()->Get(2), heads * 56) << uid;
    }
}

TEST(TestFlydslSdpaHeadPad, DeclinesOperandsTheCopyCannotRead)
{
    {
        SCOPED_TRACE("Q's head-dim stride other than 1");
        auto spec = compactSpec(37);
        spec.queryHeadDimStride = 2;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        EXPECT_EQ(flydsl_sdpa::buildHeadPadGraph(fixture.context()), nullptr);
    }
    {
        SCOPED_TRACE("Q virtual");
        auto spec = compactSpec(37);
        spec.virtualTensors = {FLYDSL_SDPA_RDNA_Q_UID};
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        EXPECT_EQ(flydsl_sdpa::buildHeadPadGraph(fixture.context()), nullptr);
    }
    {
        SCOPED_TRACE("O's rows overlapping");
        auto spec = compactSpec(37);
        spec.outputSequenceStride = 5;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        EXPECT_EQ(flydsl_sdpa::buildHeadPadGraph(fixture.context()), nullptr);
    }
    {
        SCOPED_TRACE("packed offsets that do not count tokens");
        auto spec = compactSpec(37);
        setPacked(spec, {37, 11}, {61, 29});
        spec.raggedMultiplierOf[FLYDSL_SDPA_RDNA_Q_UID] = 1;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        EXPECT_EQ(flydsl_sdpa::buildHeadPadGraph(fixture.context()), nullptr);
    }
    {
        SCOPED_TRACE("served: a gap past the head dim");
        auto spec = compactSpec(37);
        spec.querySequenceStride = spec.heads * 37 + 3;
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        EXPECT_NE(flydsl_sdpa::buildHeadPadGraph(fixture.context()), nullptr);
    }
}

// Every engine that pads serves the AOTriton form, which its graph match declines unpadded:
// the RDNA prefill and aiw engines the width, the CDNA one the width and the pitch.
TEST(TestFlydslSdpaHeadPad, TheEnginesServeTheCompactOddForm)
{
    for(const int64_t headDim : {7, 23, 241})
    {
        SCOPED_TRACE("head_dim " + std::to_string(headDim));
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(compactSpec(headDim)));
        EXPECT_TRUE(matchesGraph(FLYDSL_SDPA_RDNA, fixture.context()).has_value());
        EXPECT_TRUE(matchesGraph(FLYDSL_SDPA_AIW, fixture.context()).has_value());
        EXPECT_TRUE(matchesGraph(FLYDSL_SDPA_CDNA, fixture.context()).has_value());
    }
}

// ---------------------------------------------------------------------------
// Routing: every graph with both head dims multiples of 8 reaches the engine's own functions
// with its own context, tokens and kernel, and its own handler's workspace, prepared dispatch
// and launch.
// ---------------------------------------------------------------------------

/// What the stand-in engine functions saw last.
struct Seen
{
    const MatchContext* context = nullptr;
    int64_t headDim = 0;
    int calls = 0;
};

Seen& seen()
{
    static Seen s_seen;
    return s_seen;
}

void record(const MatchContext& context)
{
    auto& s = seen();
    s.context = &context;
    const auto* attributes = flydsl_sdpa::sdpaNode(context);
    s.headDim = attributes == nullptr
                    ? -1
                    : flydsl_sdpa::findTensor(context, attributes->q_tensor_uid())->dims()->Get(3);
    ++s.calls;
}

std::optional<BoundTokens> standInGraphMatch(const MatchContext& context)
{
    record(context);
    return BoundTokens{{"token", int64_t{7}}};
}

bool standInKernelMatch(const MatchContext& context,
                        const BoundTokens& /*bound*/,
                        const KernelDefinition& /*kernel*/)
{
    record(context);
    return true;
}

double standInScore(const MatchContext& context,
                    const BoundTokens& /*bound*/,
                    const KernelDefinition& /*kernel*/)
{
    record(context);
    return 42.0;
}

using StandInPadded
    = flydsl_sdpa::HeadPadMatchers<&standInGraphMatch, &standInKernelMatch, &standInScore>;

class StandInPrepared : public PreparedDispatch
{
};

/// The engine's handler, recording what reaches it.
class StandInHandler : public IKernelDispatchHandler<Handle>
{
public:
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        record(context);
        return INNER_BYTES;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& /*kernel*/) const override
    {
        record(context);
        return std::make_unique<StandInPrepared>();
    }

    void launch(const Handle& /*handle*/,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        launchedWith = &prepared;
        buffersSeen = deviceBuffers;
        countSeen = numDeviceBuffers;
        workspaceSeen = workspace;
    }

    static constexpr size_t INNER_BYTES = 123;
    mutable const PreparedDispatch* launchedWith = nullptr;
    mutable const hipdnnPluginDeviceBuffer_t* buffersSeen = nullptr;
    mutable uint32_t countSeen = 0;
    mutable void* workspaceSeen = nullptr;
};

TEST(TestFlydslSdpaHeadPad, AGraphOnMultiplesOfEightReachesTheEngineUntouched)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(compactSpec(64)));
    const auto context = fixture.context();
    const KernelDefinition kernel{};
    const BoundTokens bound{{"token", int64_t{1}}};

    seen() = {};
    EXPECT_TRUE(StandInPadded::graphMatches(context).has_value());
    EXPECT_EQ(seen().context, &context);
    EXPECT_TRUE(StandInPadded::kernelMatches(context, bound, kernel));
    EXPECT_EQ(seen().context, &context);
    EXPECT_EQ(StandInPadded::score(context, bound, kernel), 42.0);
    EXPECT_EQ(seen().context, &context);
    EXPECT_EQ(seen().calls, 3);

    compilation::KpackModuleCache cache;
    const compilation::KpackKernelLoader loader(cache);
    const StandInHandler inner;
    const flydsl_sdpa::HeadPadDispatchHandler handler(inner, loader);
    EXPECT_EQ(handler.workspaceBytes(context, bound, kernel), StandInHandler::INNER_BYTES);
    EXPECT_EQ(seen().context, &context);
    const auto prepared = handler.prepare(context, bound, kernel);
    EXPECT_EQ(seen().context, &context);
    ASSERT_NE(dynamic_cast<const StandInPrepared*>(prepared.get()), nullptr)
        << "the engine's own prepared dispatch, not a padded one";

    const std::array<hipdnnPluginDeviceBuffer_t, 2> buffers{
        {{FLYDSL_SDPA_RDNA_Q_UID, nullptr}, {FLYDSL_SDPA_RDNA_O_UID, nullptr}}};
    int workspace = 0;
    const Handle handle;
    handler.launch(handle, *prepared, buffers.data(), 2, &workspace);
    EXPECT_EQ(inner.launchedWith, prepared.get());
    EXPECT_EQ(inner.buffersSeen, buffers.data()) << "the caller's buffers, not a copy";
    EXPECT_EQ(inner.countSeen, 2U);
    EXPECT_EQ(inner.workspaceSeen, &workspace);
}

TEST(TestFlydslSdpaHeadPad, AGraphOffEightReachesTheEnginePadded)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(compactSpec(37)));
    const auto context = fixture.context();
    const BoundTokens bound;

    seen() = {};
    EXPECT_TRUE(StandInPadded::graphMatches(context).has_value());
    EXPECT_NE(seen().context, &context);
    EXPECT_EQ(seen().headDim, 40);
    EXPECT_EQ(StandInPadded::score(context, bound, KernelDefinition{}), 42.0);
    EXPECT_EQ(seen().headDim, 40);
    // A candidate whose shard ships no padding copy declines, whatever the engine says.
    seen() = {};
    EXPECT_FALSE(StandInPadded::kernelMatches(context, bound, KernelDefinition{}));
    EXPECT_EQ(seen().calls, 0);

    // The copies first, each 256-byte aligned, then the engine's own workspace at the end.
    compilation::KpackModuleCache cache;
    const compilation::KpackKernelLoader loader(cache);
    const StandInHandler inner;
    const flydsl_sdpa::HeadPadDispatchHandler handler(inner, loader);
    const auto plan = flydsl_sdpa::headPadPlan(context, StandInHandler::INNER_BYTES);
    ASSERT_EQ(plan.inputs.size(), 3U);
    const size_t q = size_t{2} * 37 * 4 * 40 * 2;
    const size_t kv = size_t{2} * 61 * 2 * 40 * 2;
    EXPECT_EQ(plan.inputs[0].offset, 0U);
    EXPECT_EQ(plan.inputs[0].bytes, q);
    EXPECT_EQ(plan.inputs[1].offset, flydsl_sdpa::headPadAlignUp(q));
    EXPECT_EQ(plan.inputs[1].bytes, kv);
    EXPECT_EQ(plan.inputs[2].bytes, kv);
    EXPECT_EQ(plan.output.bytes, q);
    EXPECT_EQ(plan.output.offset % 256, 0U);
    EXPECT_GE(plan.innerOffset, plan.output.offset + plan.output.bytes);
    EXPECT_EQ(plan.totalBytes, plan.innerOffset + StandInHandler::INNER_BYTES);
    EXPECT_EQ(handler.workspaceBytes(context, bound, KernelDefinition{}), plan.totalBytes);
    // The copy moves the head dim and zeroes the rest; O's copy back moves only the head dim.
    EXPECT_EQ(plan.inputs[0].units, 37);
    EXPECT_EQ(plan.inputs[0].paddedUnits, 40);
    EXPECT_EQ(plan.inputs[0].planeThreads, int64_t{37} * 40);
    EXPECT_EQ(plan.inputs[0].planes, int64_t{2} * 4);
    EXPECT_EQ(plan.output.planeThreads, int64_t{37} * 37);
}

TEST(TestFlydslSdpaHeadPad, CopiesF32AsTwoUnitsPerElement)
{
    auto spec = compactSpec(7);
    spec.dataType = DataType::FLOAT;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto plan = flydsl_sdpa::headPadPlan(fixture.context(), 0);
    EXPECT_EQ(plan.inputs[0].units, 14);
    EXPECT_EQ(plan.inputs[0].paddedUnits, 16);
    EXPECT_EQ(plan.inputs[0].strideSequence, int64_t{2} * spec.heads * 7);
    EXPECT_EQ(plan.inputs[0].padded.sequence, int64_t{2} * spec.heads * 8);
    EXPECT_EQ(plan.inputs[0].bytes, size_t{2} * 37 * 4 * 8 * 4);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
