// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "compilation/ICompiledProgram.hpp"
#include "compilation/IRunnableKernel.hpp"
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

// A packed one-token sequence keeps the real token stride instead of the extent-1 zero every
// other axis gets: the copy's source and destination rows would otherwise collapse every
// sequence onto token 0.
TEST(TestFlydslSdpaHeadPad, KeepsThePackedTokenStrideForAOneTokenSequence)
{
    auto spec = compactSpec(7);
    spec.seqLenQ = 1;
    spec.seqLenKv = 1;
    setPacked(spec, {1, 1}, {1, 1});
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto padded = flydsl_sdpa::buildHeadPadGraph(fixture.context());
    ASSERT_NE(padded, nullptr);
    const auto plan = flydsl_sdpa::headPadPlan(fixture.context(), *padded, 0);
    ASSERT_EQ(plan.inputs.size(), 3U);
    EXPECT_EQ(plan.inputs[0].uid, FLYDSL_SDPA_RDNA_Q_UID);
    EXPECT_EQ(plan.inputs[0].strideSequence, spec.heads * spec.headDim);
    EXPECT_EQ(plan.inputs[1].uid, FLYDSL_SDPA_RDNA_K_UID);
    EXPECT_EQ(plan.inputs[1].strideSequence, spec.kvHeads * spec.headDim);
    EXPECT_EQ(plan.inputs[2].uid, FLYDSL_SDPA_RDNA_V_UID);
    EXPECT_EQ(plan.inputs[2].strideSequence, spec.kvHeads * spec.headDim);
    EXPECT_EQ(plan.output.strideSequence, spec.heads * spec.headDim);
}

// A packed one-token sequence whose token stride is zero cannot be addressed: the copy has
// nothing to tell one sequence's row from another's.
TEST(TestFlydslSdpaHeadPad, DeclinesAPackedOneTokenSequenceWithAZeroStride)
{
    auto spec = compactSpec(7);
    spec.seqLenQ = 1;
    spec.seqLenKv = 1;
    setPacked(spec, {1, 1}, {1, 1});
    spec.querySequenceStride = 0;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    EXPECT_EQ(flydsl_sdpa::buildHeadPadGraph(fixture.context()), nullptr);
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

/// @p spec's graph with K bound to Q's tensor (self-attention over one packed tensor), and V
/// packed through Q's table, as K now is: K and V share one table, as the kernels require.
flatbuffers::FlatBufferBuilder withKeyAsQuery(const SdpaGraphSpec& spec)
{
    const auto built = buildFlydslSdpaRdnaGraph(spec);
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT> graph(
        hipdnn_flatbuffers_sdk::data_objects::GetGraph(built.GetBufferPointer())->UnPack());
    auto* sdpa = graph->nodes.front()->attributes.AsSdpaAttributes();
    EXPECT_NE(sdpa, nullptr);
    if(sdpa != nullptr)
    {
        sdpa->k_tensor_uid = sdpa->q_tensor_uid;
        decltype(hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT::ragged_offset_tensor_uid)
            queryTable{};
        for(const auto& tensor : graph->tensors)
        {
            if(tensor != nullptr && tensor->uid == sdpa->q_tensor_uid)
            {
                queryTable = tensor->ragged_offset_tensor_uid;
            }
        }
        for(auto& tensor : graph->tensors)
        {
            if(tensor != nullptr && tensor->uid == sdpa->v_tensor_uid)
            {
                tensor->ragged_offset_tensor_uid = queryTable;
            }
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::Graph::Pack(builder, graph.get()));
    return builder;
}

// The copies read a packed tensor's sequences for its role's lengths (seq_len_q for Q and O,
// seq_len_kv for K and V): one tensor bound as both Q and K, with lengths that differ, has no
// one set of sequences to copy, so it declines; without lengths it is copied through its table.
TEST(TestFlydslSdpaHeadPad, DeclinesATensorInTwoRolesWithDifferentLengths)
{
    auto spec = compactSpec(37);
    spec.seqLenKv = spec.seqLenQ;
    spec.kvHeads = spec.heads;
    setPacked(spec, {37, 11}, {37, 11});
    {
        SCOPED_TRACE("control: no lengths");
        const GraphFixture fixture(withKeyAsQuery(spec));
        EXPECT_NE(flydsl_sdpa::buildHeadPadGraph(fixture.context()), nullptr);
    }
    setPadding(spec, {30, 11}, {37, 5});
    const GraphFixture fixture(withKeyAsQuery(spec));
    EXPECT_EQ(flydsl_sdpa::buildHeadPadGraph(fixture.context()), nullptr);
}

// The copies read the lengths the attention kernels read: the binding their matchers take,
// servablePackedVarlenWithLengths(), on the graph as given. Q and O its query lengths, unless
// they are Q's own table (seq_len_kv alone), K and V its key lengths. A side with lengths is
// rebased (its copies' workspace side compact), a side without keeps its table; the LSE is
// copied out only when Q is rebased.
TEST(TestFlydslSdpaHeadPad, CopiesReadTheLengthBindingTheKernelsRead)
{
    struct Case
    {
        const char* name;
        bool lengths;
        bool omitSeqLenQ;
        int64_t query;
        int64_t key;
        bool queryRebased;
        bool keyRebased;
    };
    for(const auto& form : {Case{"both lengths",
                                 true,
                                 false,
                                 FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID,
                                 FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID,
                                 true,
                                 true},
                            Case{"seq_len_kv alone",
                                 true,
                                 true,
                                 flydsl_sdpa::NO_SEQ_LEN,
                                 FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID,
                                 false,
                                 true},
                            Case{"no lengths",
                                 false,
                                 false,
                                 flydsl_sdpa::NO_SEQ_LEN,
                                 flydsl_sdpa::NO_SEQ_LEN,
                                 false,
                                 false}})
    {
        SCOPED_TRACE(form.name);
        auto spec = compactSpec(53);
        setPacked(spec, {37, 11}, {61, 29});
        if(form.lengths)
        {
            setPadding(spec, {30, 11}, {50, 5});
            spec.omitSeqLenQ = form.omitSeqLenQ;
        }
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto padded = flydsl_sdpa::buildHeadPadGraph(fixture.context());
        ASSERT_NE(padded, nullptr);
        const auto plan = flydsl_sdpa::headPadPlan(fixture.context(), *padded, 0);
        ASSERT_EQ(plan.inputs.size(), 3U);
        EXPECT_EQ(plan.inputs[0].uid, FLYDSL_SDPA_RDNA_Q_UID);
        EXPECT_EQ(plan.inputs[0].lengths, form.query);
        EXPECT_EQ(plan.inputs[1].lengths, form.key);
        EXPECT_EQ(plan.inputs[2].lengths, form.key);
        EXPECT_EQ(plan.output.lengths, form.query);
        const auto compact = [](bool rebased, int32_t side) { return rebased ? side : 0; };
        EXPECT_EQ(plan.inputs[0].compact,
                  compact(form.queryRebased, flydsl_sdpa::HEAD_PAD_COMPACT_DESTINATION));
        EXPECT_EQ(plan.inputs[1].compact,
                  compact(form.keyRebased, flydsl_sdpa::HEAD_PAD_COMPACT_DESTINATION));
        EXPECT_EQ(plan.inputs[2].compact,
                  compact(form.keyRebased, flydsl_sdpa::HEAD_PAD_COMPACT_DESTINATION));
        EXPECT_EQ(plan.output.compact,
                  compact(form.queryRebased, flydsl_sdpa::HEAD_PAD_COMPACT_SOURCE));
        EXPECT_EQ(plan.stats.has_value(), form.queryRebased);
        EXPECT_EQ(plan.tables.size(),
                  static_cast<size_t>(form.queryRebased) + static_cast<size_t>(form.keyRebased));
    }
}

/// @p spec's graph with one tensor's head dim set to @p headDim, its strides contiguous BSHD at
/// that width: a graph the frontend would reject, as a native criterion must still judge.
flatbuffers::FlatBufferBuilder withHeadDim(const SdpaGraphSpec& spec, int64_t uid, int64_t headDim)
{
    const auto built = buildFlydslSdpaRdnaGraph(spec);
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT> graph(
        hipdnn_flatbuffers_sdk::data_objects::GetGraph(built.GetBufferPointer())->UnPack());
    for(auto& tensor : graph->tensors)
    {
        if(tensor != nullptr && tensor->uid == uid)
        {
            const int64_t heads = tensor->dims.at(1);
            const int64_t sequence = tensor->dims.at(2);
            tensor->dims.at(3) = headDim;
            tensor->strides = {sequence * heads * headDim, headDim, heads * headDim, 1};
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::Graph::Pack(builder, graph.get()));
    return builder;
}

// K must have Q's head dim and O V's, as the engines' own graph match requires. Padding writes
// one width over each pair, so a mismatch inside one multiple of 8 must be declined before it:
// padded, it would match, and the copies would drop K's columns or read past O's.
TEST(TestFlydslSdpaHeadPad, DeclinesAHeadDimMismatchWithinOneMultipleOfEight)
{
    struct Mismatch
    {
        const char* name;
        int64_t headDim;
        int64_t valueHeadDim;
        int64_t uid;
        int64_t width;
    };
    for(const auto& mismatch : {Mismatch{"K 5 under Q 7", 7, 8, FLYDSL_SDPA_RDNA_K_UID, 5},
                                Mismatch{"O 9 over V 7", 8, 7, FLYDSL_SDPA_RDNA_O_UID, 9},
                                Mismatch{"O 6 under V 7", 8, 7, FLYDSL_SDPA_RDNA_O_UID, 6}})
    {
        SCOPED_TRACE(mismatch.name);
        const auto spec = compactSpec(mismatch.headDim, mismatch.valueHeadDim);
        const GraphFixture fixture(withHeadDim(spec, mismatch.uid, mismatch.width));
        EXPECT_TRUE(flydsl_sdpa::needsHeadPad(fixture.context()));
        EXPECT_EQ(flydsl_sdpa::buildHeadPadGraph(fixture.context()), nullptr);
        EXPECT_FALSE(matchesGraph(FLYDSL_SDPA_RDNA, fixture.context()).has_value());
        EXPECT_FALSE(matchesGraph(FLYDSL_SDPA_AIW, fixture.context()).has_value());
        EXPECT_FALSE(matchesGraph(FLYDSL_SDPA_CDNA, fixture.context()).has_value());

        // Control: the same graph with the pair's dims equal is padded and served.
        const auto same
            = mismatch.uid == FLYDSL_SDPA_RDNA_K_UID ? mismatch.headDim : mismatch.valueHeadDim;
        const GraphFixture control(withHeadDim(spec, mismatch.uid, same));
        EXPECT_NE(flydsl_sdpa::buildHeadPadGraph(control.context()), nullptr);
        EXPECT_TRUE(matchesGraph(FLYDSL_SDPA_RDNA, control.context()).has_value());
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
        buffersCopy.assign(deviceBuffers, deviceBuffers + numDeviceBuffers);
        if(onLaunch)
        {
            onLaunch(deviceBuffers, numDeviceBuffers, workspace);
        }
    }

    static constexpr size_t INNER_BYTES = 123;
    mutable const PreparedDispatch* launchedWith = nullptr;
    mutable const hipdnnPluginDeviceBuffer_t* buffersSeen = nullptr;
    mutable uint32_t countSeen = 0;
    mutable void* workspaceSeen = nullptr;
    /// The buffers of the last launch, by value (the caller's array may be a temporary).
    mutable std::vector<hipdnnPluginDeviceBuffer_t> buffersCopy;
    /// Run inside launch(), when set.
    std::function<void(const hipdnnPluginDeviceBuffer_t*, uint32_t, void*)> onLaunch;
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
    const auto padded = flydsl_sdpa::buildHeadPadGraph(context);
    ASSERT_NE(padded, nullptr);
    const auto plan = flydsl_sdpa::headPadPlan(context, *padded, StandInHandler::INNER_BYTES);
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
    const auto padded = flydsl_sdpa::buildHeadPadGraph(fixture.context());
    ASSERT_NE(padded, nullptr);
    const auto plan = flydsl_sdpa::headPadPlan(fixture.context(), *padded, 0);
    EXPECT_EQ(plan.inputs[0].units, 14);
    EXPECT_EQ(plan.inputs[0].paddedUnits, 16);
    EXPECT_EQ(plan.inputs[0].strideSequence, int64_t{2} * spec.heads * 7);
    EXPECT_EQ(plan.inputs[0].padded.sequence, int64_t{2} * spec.heads * 8);
    EXPECT_EQ(plan.inputs[0].bytes, size_t{2} * 37 * 4 * 8 * 4);
}

// ---------------------------------------------------------------------------
// Rebasing: a packed operand whose sequences the graph gives lengths for goes through a compact
// token table in the workspace, whatever origins the graph's table holds.
// ---------------------------------------------------------------------------

/// A packed graph at head dim 61 with per-batch lengths: both (cuDNN's form 2), or `seq_len_kv`
/// alone (form 3, PyTorch's seqused_k) when @p seqLenQ is false. K's and V's sequence extent is
/// @p seqLenKv.
SdpaGraphSpec rebaseSpec(int64_t heads, bool seqLenQ, int64_t seqLenKv = 61)
{
    auto spec = compactSpec(61);
    spec.heads = heads;
    spec.kvHeads = heads;
    spec.seqLenKv = seqLenKv;
    setPacked(spec, {37, 11}, {static_cast<int32_t>(seqLenKv), 29});
    setPadding(spec, {30, 11}, {static_cast<int32_t>(seqLenKv) - 11, 5});
    spec.omitSeqLenQ = !seqLenQ;
    return spec;
}

int64_t highestUid(const MatchContext& context)
{
    int64_t top = 0;
    for(const auto& entry : context.graph.getTensorMap())
    {
        top = std::max(top, entry.first);
    }
    return top;
}

int64_t raggedOf(const MatchContext& context, int64_t uid)
{
    return tensorOf(context, uid).ragged_offset_tensor_uid().value_or(-1);
}

void expectCompactTable(const MatchContext& context, const flydsl_sdpa::HeadPadTable& table)
{
    const auto& tensor = tensorOf(context, table.uid);
    EXPECT_EQ(dimsOf(tensor), (std::vector<int64_t>{table.batch + 1, 1, 1, 1})) << table.uid;
    EXPECT_EQ(stridesOf(tensor), (std::vector<int64_t>{1, 1, 1, 1})) << table.uid;
    EXPECT_EQ(tensor.data_type(), DataType::INT32) << table.uid;
    EXPECT_EQ(tensor.alignment(), 16) << table.uid;
    EXPECT_FALSE(tensor.virtual_()) << table.uid;
    EXPECT_FALSE(tensor.ragged_offset_tensor_uid().has_value()) << table.uid;
}

// Form 2 rebases Q, K, V and O (each side through a table of its own) and rewrites the LSE as a
// scratch tensor packed through Q's compact table; form 3 rebases K and V only and leaves Q, O
// and the LSE as the graph has them; a graph without lengths rebases nothing.
TEST(TestFlydslSdpaHeadPad, RebasesExactlyTheSidesWhoseLengthsTheGraphGives)
{
    constexpr int64_t HEADS = 4;
    constexpr int64_t Q_TABLE = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
    constexpr int64_t KV_TABLE = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID;
    constexpr int64_t STATS = FLYDSL_SDPA_RDNA_STATS_UID;
    {
        SCOPED_TRACE("seq_len_q and seq_len_kv");
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(rebaseSpec(HEADS, true)));
        const auto context = fixture.context();
        const auto padded = flydsl_sdpa::buildHeadPadGraph(context);
        ASSERT_NE(padded, nullptr);
        const int64_t top = highestUid(context);
        ASSERT_EQ(padded->tables.size(), 2U);
        EXPECT_EQ(padded->tables[0].uid, top + 1);
        EXPECT_EQ(padded->tables[0].original, Q_TABLE);
        EXPECT_EQ(padded->tables[0].batch, 2);
        EXPECT_EQ(padded->tables[0].sequence, 37);
        EXPECT_EQ(padded->tables[1].uid, top + 2);
        EXPECT_EQ(padded->tables[1].original, KV_TABLE);
        EXPECT_EQ(padded->tables[1].batch, 2);
        EXPECT_EQ(padded->tables[1].sequence, 61);
        EXPECT_EQ(padded->rebased,
                  (std::set<int64_t>{FLYDSL_SDPA_RDNA_Q_UID,
                                     FLYDSL_SDPA_RDNA_K_UID,
                                     FLYDSL_SDPA_RDNA_V_UID,
                                     FLYDSL_SDPA_RDNA_O_UID}));
        const auto inner = padded->contextFor(context);
        EXPECT_EQ(inner.graph.getTensorMap().size(), context.graph.getTensorMap().size() + 2);
        expectCompactTable(inner, padded->tables[0]);
        expectCompactTable(inner, padded->tables[1]);
        // Q and O through Q's compact table, K and V through K's; multipliers still the padded
        // sequence strides, and the graph's own tables still there.
        for(const int64_t uid : {FLYDSL_SDPA_RDNA_Q_UID, FLYDSL_SDPA_RDNA_O_UID})
        {
            EXPECT_EQ(raggedOf(inner, uid), top + 1) << uid;
            EXPECT_EQ(tensorOf(inner, uid).ragged_offset_multiplier(), HEADS * 64) << uid;
        }
        for(const int64_t uid : {FLYDSL_SDPA_RDNA_K_UID, FLYDSL_SDPA_RDNA_V_UID})
        {
            EXPECT_EQ(raggedOf(inner, uid), top + 2) << uid;
            EXPECT_EQ(tensorOf(inner, uid).ragged_offset_multiplier(), HEADS * 64) << uid;
        }
        EXPECT_EQ(raggedOf(context, FLYDSL_SDPA_RDNA_Q_UID), Q_TABLE);
        EXPECT_NE(flydsl_sdpa::findTensor(inner, Q_TABLE), nullptr);
        EXPECT_NE(flydsl_sdpa::findTensor(inner, KV_TABLE), nullptr);
        // The LSE: token-major through Q's compact table, one float per (token, head).
        EXPECT_EQ(padded->scratchStats, STATS);
        const auto& stats = tensorOf(inner, STATS);
        EXPECT_EQ(dimsOf(stats), dimsOf(tensorOf(context, STATS)));
        EXPECT_EQ(stridesOf(stats), (std::vector<int64_t>{37 * HEADS, 1, HEADS, 1}));
        EXPECT_EQ(raggedOf(inner, STATS), top + 1);
        EXPECT_EQ(stats.ragged_offset_multiplier(), HEADS);
        EXPECT_EQ(stats.alignment(), 4);
        EXPECT_EQ(stats.data_type(), DataType::FLOAT);
    }
    {
        SCOPED_TRACE("seq_len_kv alone");
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(rebaseSpec(HEADS, false)));
        const auto context = fixture.context();
        const auto padded = flydsl_sdpa::buildHeadPadGraph(context);
        ASSERT_NE(padded, nullptr);
        const int64_t top = highestUid(context);
        ASSERT_EQ(padded->tables.size(), 1U);
        EXPECT_EQ(padded->tables[0].uid, top + 1);
        EXPECT_EQ(padded->tables[0].original, KV_TABLE);
        EXPECT_EQ(padded->tables[0].sequence, 61);
        EXPECT_EQ(padded->rebased,
                  (std::set<int64_t>{FLYDSL_SDPA_RDNA_K_UID, FLYDSL_SDPA_RDNA_V_UID}));
        EXPECT_EQ(padded->scratchStats, flydsl_sdpa::NO_SEQ_LEN);
        const auto inner = padded->contextFor(context);
        expectCompactTable(inner, padded->tables[0]);
        for(const int64_t uid : {FLYDSL_SDPA_RDNA_K_UID, FLYDSL_SDPA_RDNA_V_UID})
        {
            EXPECT_EQ(raggedOf(inner, uid), top + 1) << uid;
        }
        // Q, O and the LSE keep Q's own table, and the LSE its strides.
        for(const int64_t uid : {FLYDSL_SDPA_RDNA_Q_UID, FLYDSL_SDPA_RDNA_O_UID, STATS})
        {
            EXPECT_EQ(raggedOf(inner, uid), Q_TABLE) << uid;
        }
        EXPECT_EQ(stridesOf(tensorOf(inner, STATS)), stridesOf(tensorOf(context, STATS)));
        EXPECT_EQ(tensorOf(inner, STATS).alignment(), tensorOf(context, STATS).alignment());
    }
    {
        SCOPED_TRACE("no lengths");
        auto spec = rebaseSpec(HEADS, true);
        spec.paddingMask = false;
        spec.seqLensQ.clear();
        spec.seqLensKv.clear();
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
        const auto context = fixture.context();
        const auto padded = flydsl_sdpa::buildHeadPadGraph(context);
        ASSERT_NE(padded, nullptr);
        EXPECT_TRUE(padded->tables.empty());
        EXPECT_TRUE(padded->rebased.empty());
        EXPECT_EQ(padded->scratchStats, flydsl_sdpa::NO_SEQ_LEN);
        const auto inner = padded->contextFor(context);
        EXPECT_EQ(inner.graph.getTensorMap().size(), context.graph.getTensorMap().size());
        for(const int64_t uid : {FLYDSL_SDPA_RDNA_Q_UID,
                                 FLYDSL_SDPA_RDNA_O_UID,
                                 STATS,
                                 FLYDSL_SDPA_RDNA_K_UID,
                                 FLYDSL_SDPA_RDNA_V_UID})
        {
            EXPECT_EQ(raggedOf(inner, uid), raggedOf(context, uid)) << uid;
        }
    }
}

// Q and K sharing one table: one compact table when their sequence extents agree, two when they
// do not (each table is `[0, S, ..., B * S]` for its own S).
TEST(TestFlydslSdpaHeadPad, GivesQAndKSharingATableTablesOfTheirOwnWhenTheirSequencesDiffer)
{
    const auto sharedSpec = [](int64_t seqLenKv) {
        auto spec = rebaseSpec(2, true, seqLenKv);
        spec.raggedOffsetOf[FLYDSL_SDPA_RDNA_K_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
        spec.raggedOffsetOf[FLYDSL_SDPA_RDNA_V_UID] = FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID;
        return spec;
    };
    {
        SCOPED_TRACE("equal extents");
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(sharedSpec(37)));
        const auto context = fixture.context();
        const auto padded = flydsl_sdpa::buildHeadPadGraph(context);
        ASSERT_NE(padded, nullptr);
        ASSERT_EQ(padded->tables.size(), 1U);
        const auto inner = padded->contextFor(context);
        for(const int64_t uid : {FLYDSL_SDPA_RDNA_Q_UID,
                                 FLYDSL_SDPA_RDNA_K_UID,
                                 FLYDSL_SDPA_RDNA_V_UID,
                                 FLYDSL_SDPA_RDNA_O_UID})
        {
            EXPECT_EQ(raggedOf(inner, uid), padded->tables[0].uid) << uid;
        }
        const auto plan = flydsl_sdpa::headPadPlan(context, *padded, 0);
        ASSERT_EQ(plan.inputs.size(), 3U);
        // One copy writes the table: the first through it, Q's.
        EXPECT_EQ(plan.inputs[0].tableIndex, 0);
        EXPECT_EQ(plan.inputs[1].tableIndex, -1);
        EXPECT_EQ(plan.inputs[2].tableIndex, -1);
        EXPECT_EQ(plan.output.tableIndex, -1);
    }
    {
        SCOPED_TRACE("different extents");
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(sharedSpec(61)));
        const auto context = fixture.context();
        const auto padded = flydsl_sdpa::buildHeadPadGraph(context);
        ASSERT_NE(padded, nullptr);
        ASSERT_EQ(padded->tables.size(), 2U);
        EXPECT_EQ(padded->tables[0].original, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID);
        EXPECT_EQ(padded->tables[1].original, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID);
        EXPECT_EQ(padded->tables[0].sequence, 37);
        EXPECT_EQ(padded->tables[1].sequence, 61);
        EXPECT_NE(padded->tables[0].uid, padded->tables[1].uid);
        const auto inner = padded->contextFor(context);
        EXPECT_EQ(raggedOf(inner, FLYDSL_SDPA_RDNA_Q_UID), padded->tables[0].uid);
        EXPECT_EQ(raggedOf(inner, FLYDSL_SDPA_RDNA_O_UID), padded->tables[0].uid);
        EXPECT_EQ(raggedOf(inner, FLYDSL_SDPA_RDNA_K_UID), padded->tables[1].uid);
        EXPECT_EQ(raggedOf(inner, FLYDSL_SDPA_RDNA_V_UID), padded->tables[1].uid);
        const auto plan = flydsl_sdpa::headPadPlan(context, *padded, 0);
        ASSERT_EQ(plan.inputs.size(), 3U);
        EXPECT_EQ(plan.inputs[0].tableIndex, 0);
        EXPECT_EQ(plan.inputs[1].tableIndex, 1);
        EXPECT_EQ(plan.inputs[2].tableIndex, -1);
    }
}

// The rebased form-2 graph (LSE included) still matches the CDNA and RDNA matchers. At one head
// every LSE layout is the same, so H = 2 and 4 are what pin its token-major strides.
TEST(TestFlydslSdpaHeadPad, TheRebasedFormTwoGraphStillMatchesTheEngines)
{
    for(const int64_t heads : {1, 2, 4})
    {
        SCOPED_TRACE("heads " + std::to_string(heads));
        const GraphFixture fixture(buildFlydslSdpaRdnaGraph(rebaseSpec(heads, true)));
        const auto context = fixture.context();
        const auto padded = flydsl_sdpa::buildHeadPadGraph(context);
        ASSERT_NE(padded, nullptr);
        ASSERT_EQ(padded->tables.size(), 2U);
        ASSERT_EQ(padded->scratchStats, FLYDSL_SDPA_RDNA_STATS_UID);
        for(const auto* engine : {&FLYDSL_SDPA_RDNA, &FLYDSL_SDPA_CDNA})
        {
            EXPECT_TRUE(matchesGraph(*engine, context).has_value()) << engine->engineName;
        }
        // The RDNA matcher on the rebased graph itself binds the compact tables and the scratch
        // LSE.
        const auto inner = padded->contextFor(context);
        EXPECT_FALSE(flydsl_sdpa::needsHeadPad(inner));
        const auto bound = matchesGraph(FLYDSL_SDPA_RDNA, inner);
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(
                      *bound, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_TOKEN),
                  padded->tables[0].uid);
        EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(
                      *bound, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_TOKEN),
                  padded->tables[1].uid);
        EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, FLYDSL_SDPA_RDNA_STATS_TOKEN),
                  FLYDSL_SDPA_RDNA_STATS_UID);
    }
}

// The workspace: each distinct operand's copy, O's, the LSE scratch, each compact table (B + 1
// int32 entries), then the engine's own workspace, every part 256-byte aligned.
TEST(TestFlydslSdpaHeadPad, LaysOutTheWorkspaceWithTheLseScratchAndTheCompactTables)
{
    constexpr int64_t HEADS = 4;
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(rebaseSpec(HEADS, true)));
    const auto context = fixture.context();
    const auto padded = flydsl_sdpa::buildHeadPadGraph(context);
    ASSERT_NE(padded, nullptr);
    const auto plan = flydsl_sdpa::headPadPlan(context, *padded, StandInHandler::INNER_BYTES);
    ASSERT_EQ(plan.inputs.size(), 3U);
    const size_t q = size_t{2} * 37 * HEADS * 64 * 2;
    const size_t kv = size_t{2} * 61 * HEADS * 64 * 2;
    EXPECT_EQ(plan.inputs[0].offset, 0U);
    EXPECT_EQ(plan.inputs[0].bytes, q);
    EXPECT_EQ(plan.inputs[1].offset, flydsl_sdpa::headPadAlignUp(q));
    EXPECT_EQ(plan.inputs[1].bytes, kv);
    EXPECT_EQ(plan.inputs[2].offset, flydsl_sdpa::headPadAlignUp(plan.inputs[1].offset + kv));
    EXPECT_EQ(plan.output.offset, flydsl_sdpa::headPadAlignUp(plan.inputs[2].offset + kv));
    EXPECT_EQ(plan.output.bytes, q);
    // The LSE scratch: one float per (token, head) of B * Sq tokens.
    ASSERT_TRUE(plan.stats.has_value());
    const auto& stats = *plan.stats;
    EXPECT_EQ(stats.uid, FLYDSL_SDPA_RDNA_STATS_UID);
    EXPECT_EQ(stats.offset, flydsl_sdpa::headPadAlignUp(plan.output.offset + q));
    EXPECT_EQ(stats.bytes, size_t{2} * 37 * HEADS * 4);
    EXPECT_EQ(stats.rowOffset, FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID);
    EXPECT_EQ(stats.lengths, FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID);
    EXPECT_EQ(stats.batch, 2);
    EXPECT_EQ(stats.heads, HEADS);
    EXPECT_EQ(stats.sequence, 37);
    EXPECT_EQ(stats.units, 2);
    EXPECT_EQ(stats.paddedUnits, 2);
    // The graph's LSE strides in units; the scratch's token-major ones.
    EXPECT_EQ(stats.strideBatch, 2 * 37 * HEADS);
    EXPECT_EQ(stats.strideHead, 2);
    EXPECT_EQ(stats.strideSequence, 2 * HEADS);
    EXPECT_EQ(stats.padded.batch, 2 * 37 * HEADS);
    EXPECT_EQ(stats.padded.head, 2);
    EXPECT_EQ(stats.padded.sequence, 2 * HEADS);
    EXPECT_EQ(stats.compact, flydsl_sdpa::HEAD_PAD_COMPACT_SOURCE);
    EXPECT_EQ(stats.planeThreads, int64_t{37} * 2);
    EXPECT_EQ(stats.planes, int64_t{2} * HEADS);
    // The tables: B + 1 = 3 int32 entries each.
    ASSERT_EQ(plan.tables.size(), 2U);
    EXPECT_EQ(plan.tables[0].uid, padded->tables[0].uid);
    EXPECT_EQ(plan.tables[0].bytes, 12U);
    EXPECT_EQ(plan.tables[0].offset, flydsl_sdpa::headPadAlignUp(stats.offset + stats.bytes));
    EXPECT_EQ(plan.tables[1].uid, padded->tables[1].uid);
    EXPECT_EQ(plan.tables[1].bytes, 12U);
    EXPECT_EQ(plan.tables[1].offset, flydsl_sdpa::headPadAlignUp(plan.tables[0].offset + 12));
    EXPECT_EQ(plan.innerOffset, flydsl_sdpa::headPadAlignUp(plan.tables[1].offset + 12));
    EXPECT_EQ(plan.totalBytes, plan.innerOffset + StandInHandler::INNER_BYTES);
    // The copies: the workspace side compact, Q's and K's the writers of their tables.
    for(size_t index = 0; index < 3; ++index)
    {
        EXPECT_EQ(plan.inputs[index].compact, flydsl_sdpa::HEAD_PAD_COMPACT_DESTINATION) << index;
    }
    EXPECT_EQ(plan.output.compact, flydsl_sdpa::HEAD_PAD_COMPACT_SOURCE);
    EXPECT_EQ(plan.inputs[0].tableIndex, 0);
    EXPECT_EQ(plan.inputs[1].tableIndex, 1);
    EXPECT_EQ(plan.inputs[2].tableIndex, -1);
    EXPECT_EQ(plan.output.tableIndex, -1);
    EXPECT_EQ(stats.tableIndex, -1);
}

// ---------------------------------------------------------------------------
// The whole launch on host memory: the copy kernel's contract (sdpa_head_pad.py), executed from
// the arguments the wrapper gives it, around a stand-in engine that reads and writes the
// rebased graph's tensors through the buffers it is handed.
// ---------------------------------------------------------------------------

/// One launch of a copy, decoded as headPadSignature() lays its arguments out.
struct CopyCall
{
    size_t copy = 0;
    void* source = nullptr;
    void* destination = nullptr;
    void* rowOffset = nullptr;
    void* lengths = nullptr;
    void* tableOut = nullptr;
    int32_t batch = 0;
    int32_t heads = 0;
    int32_t sequence = 0;
    int32_t copyUnits = 0;
    int32_t dstUnits = 0;
    int32_t compact = 0;
    std::array<int64_t, 3> sourceStrides{};
    std::array<int64_t, 3> destinationStrides{};
};

class RecordingCopyKernel : public compilation::IRunnableKernel
{
public:
    RecordingCopyKernel(size_t index, std::function<void(const CopyCall&)> sink)
        : _index(index)
        , _sink(std::move(sink))
    {
    }

    void setBlockSize(unsigned int /*x*/, unsigned int /*y*/, unsigned int /*z*/) override {}
    void setGridSize(unsigned int /*x*/, unsigned int /*y*/, unsigned int /*z*/) override {}
    void setSharedMemBytes(unsigned int /*bytes*/) override {}

protected:
    void launchImpl(hipStream_t /*stream*/, void** params) const override
    {
        const auto pointer = [&](size_t i) { return *static_cast<void**>(params[i]); };
        const auto i32 = [&](size_t i) { return *static_cast<const int32_t*>(params[i]); };
        const auto i64 = [&](size_t i) { return *static_cast<const int64_t*>(params[i]); };
        CopyCall call;
        call.copy = _index;
        call.source = pointer(0);
        call.destination = pointer(1);
        call.rowOffset = pointer(2);
        call.lengths = pointer(3);
        call.tableOut = pointer(4);
        call.batch = i32(5);
        call.heads = i32(6);
        call.sequence = i32(7);
        call.copyUnits = i32(8);
        call.dstUnits = i32(9);
        call.compact = i32(10);
        call.sourceStrides = {i64(11), i64(12), i64(13)};
        call.destinationStrides = {i64(14), i64(15), i64(16)};
        _sink(call);
    }

private:
    size_t _index;
    std::function<void(const CopyCall&)> _sink;
};

class NoProgram : public compilation::ICompiledProgram
{
public:
    std::unique_ptr<compilation::IRunnableKernel>
        getKernel(const std::string& /*name*/) const override
    {
        return nullptr;
    }
};

/// The copy kernel's contract, run on host memory: which rows move, where from and to, and the
/// compact token table. Any access outside the registered allocations fails the test.
class CopyModel
{
public:
    template <typename T>
    void allow(const std::vector<T>& memory)
    {
        allow(memory.data(), memory.size() * sizeof(T));
    }

    void allow(const void* begin, size_t bytes)
    {
        _ranges.emplace_back(reinterpret_cast<uintptr_t>(begin), bytes);
    }

    void run(const CopyCall& call) const
    {
        const auto* source = static_cast<const uint16_t*>(call.source);
        auto* destination = static_cast<uint16_t*>(call.destination);
        const auto* rowOffset = static_cast<const int32_t*>(call.rowOffset);
        const auto* lengths = static_cast<const int32_t*>(call.lengths);
        if(call.tableOut != nullptr)
        {
            auto* table = static_cast<int32_t*>(call.tableOut);
            for(int32_t entry = 0; entry <= call.batch; ++entry)
            {
                store(table + entry, static_cast<int32_t>(entry * call.sequence));
            }
        }
        const bool packed = rowOffset != nullptr;
        for(int64_t b = 0; b < call.batch; ++b)
        {
            for(int64_t h = 0; h < call.heads; ++h)
            {
                for(int64_t s = 0; s < call.sequence; ++s)
                {
                    int64_t first = 0;
                    int64_t rows = call.sequence;
                    if(packed)
                    {
                        first = load(rowOffset + b);
                        rows = lengths != nullptr ? load(lengths + b)
                                                  : load(rowOffset + b + 1) - first;
                    }
                    if(s >= rows)
                    {
                        continue;
                    }
                    const int64_t token = first + s;
                    const int64_t compactToken = b * call.sequence + s;
                    const auto& ss = call.sourceStrides;
                    const auto& ds = call.destinationStrides;
                    const int64_t sourceToken = (call.compact & 1) != 0 ? compactToken : token;
                    const int64_t destinationToken = (call.compact & 2) != 0 ? compactToken : token;
                    const int64_t sourceRow = packed ? sourceToken * ss[2] : b * ss[0] + s * ss[2];
                    const int64_t destinationRow
                        = packed ? destinationToken * ds[2] : b * ds[0] + s * ds[2];
                    for(int64_t c = 0; c < call.dstUnits; ++c)
                    {
                        const uint16_t value = c < call.copyUnits
                                                   ? load(source + sourceRow + h * ss[1] + c)
                                                   : uint16_t{0};
                        store(destination + destinationRow + h * ds[1] + c, value);
                    }
                }
            }
        }
    }

private:
    bool covered(const void* pointer, size_t bytes) const
    {
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        for(const auto& [begin, size] : _ranges)
        {
            if(address >= begin && address + bytes <= begin + size)
            {
                return true;
            }
        }
        return false;
    }

    template <typename T>
    T load(const T* pointer) const
    {
        if(!covered(pointer, sizeof(T)))
        {
            ADD_FAILURE() << "the copy read outside its allocations";
            return T{};
        }
        return *pointer;
    }

    template <typename T>
    void store(T* pointer, T value) const
    {
        if(!covered(pointer, sizeof(T)))
        {
            ADD_FAILURE() << "the copy wrote outside its allocations";
            return;
        }
        *pointer = value;
    }

    std::vector<std::pair<uintptr_t, size_t>> _ranges;
};

/**
 * Launches the wrapper over a packed f16 graph (B 2, H 2, D 7 padded to 8) whose origins are
 * gapped past B * S tokens, with a stand-in engine that reads Q, K and V through the inner
 * graph's tables, sets O to Q and the LSE to a per-(token, head) value, and checks what it was
 * given. Form 2 gives both lengths and K, V and Q, O at origins {30, 0} and {20, 3}; form 3
 * gives seq_len_kv alone, K at {30, 0} and Q end to end.
 */
void expectGappedOriginsLaunchCorrectly(bool bothLengths)
{
    constexpr int64_t B = 2;
    constexpr int64_t H = 2;
    constexpr int64_t D = 7;
    constexpr int64_t W = 8;
    constexpr int64_t SQ = 6;
    constexpr int64_t SKV = 8;
    constexpr int64_t QUERY_TOKENS = 24;
    constexpr int64_t KEY_TOKENS = 37;
    constexpr uint16_t UNWRITTEN = 0x7777;

    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.batch = B;
    spec.heads = H;
    spec.kvHeads = H;
    spec.seqLenQ = SQ;
    spec.seqLenKv = SKV;
    spec.headDim = D;
    spec.stats = SdpaStats::LSE;
    const std::vector<int32_t> lensQ{4, 5};
    const std::vector<int32_t> lensKv{7, 3};
    setPacked(spec, lensQ, lensKv);
    setPadding(spec, lensQ, lensKv);
    spec.omitSeqLenQ = !bothLengths;
    // Q's regions end at token 24 and K's at 37, past B * Sq = 12 and B * Skv = 16 (form 3: Q's
    // table lists its sequences end to end).
    std::vector<int32_t> tableQ
        = bothLengths ? std::vector<int32_t>{20, 3, 0} : std::vector<int32_t>{0, 4, 9};
    std::vector<int32_t> tableKv{30, 0, 0};
    std::vector<int32_t> lengthsQ = lensQ;
    std::vector<int32_t> lengthsKv = lensKv;

    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec));
    const auto context = fixture.context();
    const auto padded = flydsl_sdpa::buildHeadPadGraph(context);
    ASSERT_NE(padded, nullptr);
    const auto plan = flydsl_sdpa::headPadPlan(context, *padded, StandInHandler::INNER_BYTES);
    ASSERT_EQ(plan.tables.size(), bothLengths ? 2U : 1U);
    ASSERT_EQ(plan.stats.has_value(), bothLengths);

    std::vector<uint16_t> q(QUERY_TOKENS * H * D);
    std::vector<uint16_t> k(KEY_TOKENS * H * D);
    std::vector<uint16_t> v(KEY_TOKENS * H * D);
    for(size_t i = 0; i < q.size(); ++i)
    {
        q[i] = static_cast<uint16_t>(1 + i);
    }
    for(size_t i = 0; i < k.size(); ++i)
    {
        k[i] = static_cast<uint16_t>(1000 + i);
        v[i] = static_cast<uint16_t>(5000 + i);
    }
    std::vector<uint16_t> o(q.size(), UNWRITTEN);
    std::vector<float> lse(QUERY_TOKENS * H, -1.0F);
    std::vector<uint8_t> workspace(plan.totalBytes, 0xCD);

    std::vector<hipdnnPluginDeviceBuffer_t> device{
        {FLYDSL_SDPA_RDNA_Q_UID, q.data()},
        {FLYDSL_SDPA_RDNA_K_UID, k.data()},
        {FLYDSL_SDPA_RDNA_V_UID, v.data()},
        {FLYDSL_SDPA_RDNA_O_UID, o.data()},
        {FLYDSL_SDPA_RDNA_STATS_UID, lse.data()},
        {FLYDSL_SDPA_RDNA_SEQ_LEN_KV_UID, lengthsKv.data()},
        {FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID, tableQ.data()},
        {FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID, tableKv.data()}};
    if(bothLengths)
    {
        device.push_back({FLYDSL_SDPA_RDNA_SEQ_LEN_Q_UID, lengthsQ.data()});
    }

    CopyModel model;
    model.allow(q);
    model.allow(k);
    model.allow(v);
    model.allow(o);
    model.allow(lse);
    model.allow(tableQ);
    model.allow(tableKv);
    model.allow(lengthsQ);
    model.allow(lengthsKv);
    model.allow(workspace);

    std::vector<CopyCall> calls;
    std::vector<std::string> events;
    const size_t copyCount = plan.inputs.size() + 1 + (plan.stats.has_value() ? 1 : 0);
    std::vector<IngestorKernelCode> copies;
    copies.reserve(copyCount);
    for(size_t index = 0; index < copyCount; ++index)
    {
        copies.emplace_back(std::make_unique<NoProgram>(),
                            std::make_unique<RecordingCopyKernel>(index, [&](const CopyCall& call) {
                                calls.push_back(call);
                                events.push_back("copy " + std::to_string(call.copy));
                                model.run(call);
                            }));
    }
    const flydsl_sdpa::HeadPadPrepared prepared(
        std::make_unique<StandInPrepared>(), plan, std::move(copies));

    const auto innerContext = padded->contextFor(context);
    StandInHandler inner;
    inner.onLaunch = [&](const hipdnnPluginDeviceBuffer_t* buffers, uint32_t count, void*) {
        events.emplace_back("inner");
        const auto find = [&](int64_t uid) {
            return hipdnn_plugin_sdk::findDeviceBuffer(uid, buffers, count).ptr;
        };
        const auto tableOf = [&](int64_t operand) {
            return static_cast<const int32_t*>(find(raggedOf(innerContext, operand)));
        };
        const auto* queryTable = tableOf(FLYDSL_SDPA_RDNA_Q_UID);
        const auto* keyTable = tableOf(FLYDSL_SDPA_RDNA_K_UID);
        const auto* queryCopy = static_cast<const uint16_t*>(find(FLYDSL_SDPA_RDNA_Q_UID));
        const auto* keyCopy = static_cast<const uint16_t*>(find(FLYDSL_SDPA_RDNA_K_UID));
        const auto* valueCopy = static_cast<const uint16_t*>(find(FLYDSL_SDPA_RDNA_V_UID));
        auto* outputCopy = static_cast<uint16_t*>(find(FLYDSL_SDPA_RDNA_O_UID));
        auto* stats = static_cast<float*>(find(FLYDSL_SDPA_RDNA_STATS_UID));
        for(int64_t b = 0; b < B; ++b)
        {
            // Q's rows: its lengths, or its table's differences (form 3).
            const int64_t queryRows = bothLengths ? lengthsQ[static_cast<size_t>(b)]
                                                  : queryTable[b + 1] - queryTable[b];
            for(int64_t s = 0; s < queryRows; ++s)
            {
                const int64_t token = queryTable[b] + s;
                ASSERT_GE(token, 0);
                ASSERT_LT(token, B * SQ) << "Q's token is outside the workspace copy";
                const int64_t origin = tableQ[static_cast<size_t>(b)] + s;
                for(int64_t h = 0; h < H; ++h)
                {
                    for(int64_t d = 0; d < W; ++d)
                    {
                        const auto at = static_cast<size_t>((token * H + h) * W + d);
                        const uint16_t expected
                            = d < D ? q[static_cast<size_t>((origin * H + h) * D + d)]
                                    : uint16_t{0};
                        EXPECT_EQ(queryCopy[at], expected) << b << " " << s << " " << h << " " << d;
                        outputCopy[at] = queryCopy[at];
                    }
                    stats[token * H + h] = static_cast<float>(100 * b + 10 * s + h) + 0.5F;
                }
            }
            for(int64_t s = 0; s < lengthsKv[static_cast<size_t>(b)]; ++s)
            {
                const int64_t token = keyTable[b] + s;
                ASSERT_GE(token, 0);
                ASSERT_LT(token, B * SKV) << "K's token is outside the workspace copy";
                const int64_t origin = tableKv[static_cast<size_t>(b)] + s;
                for(int64_t h = 0; h < H; ++h)
                {
                    for(int64_t d = 0; d < W; ++d)
                    {
                        const auto at = static_cast<size_t>((token * H + h) * W + d);
                        const auto from = static_cast<size_t>((origin * H + h) * D + d);
                        EXPECT_EQ(keyCopy[at], d < D ? k[from] : uint16_t{0});
                        EXPECT_EQ(valueCopy[at], d < D ? v[from] : uint16_t{0});
                    }
                }
            }
        }
    };

    compilation::KpackModuleCache cache;
    const compilation::KpackKernelLoader loader(cache);
    const flydsl_sdpa::HeadPadDispatchHandler handler(inner, loader);
    const Handle handle;
    handler.launch(
        handle, prepared, device.data(), static_cast<uint32_t>(device.size()), workspace.data());

    // O and the LSE hold the engine's values at the graph's own origins and nothing anywhere else.
    std::vector<uint16_t> expectedO(o.size(), UNWRITTEN);
    std::vector<float> expectedLse(lse.size(), -1.0F);
    for(int64_t b = 0; b < B; ++b)
    {
        for(int64_t s = 0; s < lensQ[static_cast<size_t>(b)]; ++s)
        {
            const int64_t origin = tableQ[static_cast<size_t>(b)] + s;
            for(int64_t h = 0; h < H; ++h)
            {
                for(int64_t d = 0; d < D; ++d)
                {
                    const auto at = static_cast<size_t>((origin * H + h) * D + d);
                    expectedO[at] = q[at];
                }
                expectedLse[static_cast<size_t>(origin * H + h)]
                    = static_cast<float>(100 * b + 10 * s + h) + 0.5F;
            }
        }
    }
    EXPECT_EQ(o, expectedO);
    EXPECT_EQ(lse, expectedLse);

    // The launch: the copies in, the engine, then O's and the LSE's copies out.
    std::vector<std::string> expectedEvents;
    for(size_t index = 0; index < plan.inputs.size(); ++index)
    {
        expectedEvents.push_back("copy " + std::to_string(index));
    }
    expectedEvents.emplace_back("inner");
    for(size_t index = plan.inputs.size(); index < copyCount; ++index)
    {
        expectedEvents.push_back("copy " + std::to_string(index));
    }
    EXPECT_EQ(events, expectedEvents);
    ASSERT_EQ(calls.size(), copyCount);

    // Each compact table is written by exactly one copy, the first through it, and every other
    // launch gets null.
    auto* ws = workspace.data();
    size_t writers = 0;
    for(const auto& call : calls)
    {
        writers += call.tableOut != nullptr ? 1 : 0;
    }
    EXPECT_EQ(writers, plan.tables.size());
    if(bothLengths)
    {
        EXPECT_EQ(calls[0].tableOut, static_cast<void*>(ws + plan.tables[0].offset));
        EXPECT_EQ(calls[1].tableOut, static_cast<void*>(ws + plan.tables[1].offset));
        EXPECT_EQ(calls[2].tableOut, nullptr);
        EXPECT_EQ(calls[3].tableOut, nullptr);
        EXPECT_EQ(calls[4].tableOut, nullptr);
        for(size_t index = 0; index < 3; ++index)
        {
            EXPECT_EQ(calls[index].compact, flydsl_sdpa::HEAD_PAD_COMPACT_DESTINATION) << index;
            EXPECT_EQ(calls[index].lengths,
                      static_cast<void*>(index == 0 ? lengthsQ.data() : lengthsKv.data()))
                << index;
        }
        EXPECT_EQ(calls[3].compact, flydsl_sdpa::HEAD_PAD_COMPACT_SOURCE);
        EXPECT_EQ(calls[4].compact, flydsl_sdpa::HEAD_PAD_COMPACT_SOURCE);
        EXPECT_EQ(calls[3].lengths, static_cast<void*>(lengthsQ.data()));
        EXPECT_EQ(calls[4].lengths, static_cast<void*>(lengthsQ.data()));
        // The copies read and write the graph's own tables, and the LSE goes out through Q's.
        EXPECT_EQ(calls[0].rowOffset, static_cast<void*>(tableQ.data()));
        EXPECT_EQ(calls[3].rowOffset, static_cast<void*>(tableQ.data()));
        EXPECT_EQ(calls[4].rowOffset, static_cast<void*>(tableQ.data()));
        EXPECT_EQ(calls[4].destination, static_cast<void*>(lse.data()));
        EXPECT_EQ(calls[4].source, static_cast<void*>(ws + plan.stats->offset));
    }
    else
    {
        // Q has no lengths: its copies are the unrebased ones, and the LSE is the engine's.
        EXPECT_EQ(calls[0].tableOut, nullptr);
        EXPECT_EQ(calls[1].tableOut, static_cast<void*>(ws + plan.tables[0].offset));
        EXPECT_EQ(calls[2].tableOut, nullptr);
        EXPECT_EQ(calls[3].tableOut, nullptr);
        EXPECT_EQ(calls[0].compact, 0);
        EXPECT_EQ(calls[1].compact, flydsl_sdpa::HEAD_PAD_COMPACT_DESTINATION);
        EXPECT_EQ(calls[2].compact, flydsl_sdpa::HEAD_PAD_COMPACT_DESTINATION);
        EXPECT_EQ(calls[3].compact, 0);
        EXPECT_EQ(calls[0].lengths, nullptr);
        EXPECT_EQ(calls[1].lengths, static_cast<void*>(lengthsKv.data()));
        EXPECT_EQ(calls[3].lengths, nullptr);
    }
    EXPECT_EQ(calls[1].rowOffset, static_cast<void*>(tableKv.data()));
    EXPECT_EQ(calls[2].rowOffset, static_cast<void*>(tableKv.data()));
    // The workspace side of each copy carries the compact strides of its padded layout.
    EXPECT_EQ(calls[1].destinationStrides[2], H * W);
    EXPECT_EQ(calls[1].sequence, SKV);

    // The engine's buffers: Q, K, V and O at their workspace copies, the LSE at its scratch
    // (form 2), each new table at its workspace place, and the caller's other buffers as given.
    const auto bufferOf = [&](int64_t uid) -> void* {
        for(const auto& buffer : inner.buffersCopy)
        {
            if(buffer.uid == uid)
            {
                return buffer.ptr;
            }
        }
        return nullptr;
    };
    EXPECT_EQ(inner.buffersCopy.size(), device.size() + plan.tables.size());
    EXPECT_EQ(bufferOf(FLYDSL_SDPA_RDNA_Q_UID), static_cast<void*>(ws + plan.inputs[0].offset));
    EXPECT_EQ(bufferOf(FLYDSL_SDPA_RDNA_K_UID), static_cast<void*>(ws + plan.inputs[1].offset));
    EXPECT_EQ(bufferOf(FLYDSL_SDPA_RDNA_V_UID), static_cast<void*>(ws + plan.inputs[2].offset));
    EXPECT_EQ(bufferOf(FLYDSL_SDPA_RDNA_O_UID), static_cast<void*>(ws + plan.output.offset));
    EXPECT_EQ(bufferOf(FLYDSL_SDPA_RDNA_STATS_UID),
              plan.stats.has_value() ? static_cast<void*>(ws + plan.stats->offset)
                                     : static_cast<void*>(lse.data()));
    EXPECT_EQ(bufferOf(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_Q_UID), static_cast<void*>(tableQ.data()));
    EXPECT_EQ(bufferOf(FLYDSL_SDPA_RDNA_RAGGED_OFFSET_KV_UID), static_cast<void*>(tableKv.data()));
    for(const auto& table : plan.tables)
    {
        EXPECT_EQ(bufferOf(table.uid), static_cast<void*>(ws + table.offset)) << table.uid;
    }
}

TEST(TestFlydslSdpaHeadPad, FormTwoWithOriginsPastTheTokenCountLaunchesInsideItsWorkspace)
{
    expectGappedOriginsLaunchCorrectly(true);
}

TEST(TestFlydslSdpaHeadPad, FormThreeWithKeyOriginsPastTheTokenCountLaunchesInsideItsWorkspace)
{
    expectGappedOriginsLaunchCorrectly(false);
}

/// @p spec's graph with Q, K, V and O packed head-major: each `[B, H, S, D]` at strides
/// `[H*S*D, S*D, D, 1]`, its ragged multiplier its sequence stride D. Every stride is positive
/// and the multiplier counts the sequence stride, as headPadCopyable() asks, but one token's
/// head x head-dim block (H > 1) does not fit within that stride.
flatbuffers::FlatBufferBuilder withHeadMajorPacking(const SdpaGraphSpec& spec)
{
    const auto built = buildFlydslSdpaRdnaGraph(spec);
    std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::GraphT> graph(
        hipdnn_flatbuffers_sdk::data_objects::GetGraph(built.GetBufferPointer())->UnPack());
    for(auto& tensor : graph->tensors)
    {
        if(tensor == nullptr
           || (tensor->uid != FLYDSL_SDPA_RDNA_Q_UID && tensor->uid != FLYDSL_SDPA_RDNA_K_UID
               && tensor->uid != FLYDSL_SDPA_RDNA_V_UID && tensor->uid != FLYDSL_SDPA_RDNA_O_UID))
        {
            continue;
        }
        const int64_t heads = tensor->dims.at(1);
        const int64_t sequence = tensor->dims.at(2);
        const int64_t width = tensor->dims.at(3);
        tensor->strides = {heads * sequence * width, sequence * width, width, 1};
        tensor->ragged_offset_multiplier = width;
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(hipdnn_flatbuffers_sdk::data_objects::Graph::Pack(builder, graph.get()));
    return builder;
}

// A head-major packed graph (a token's heads spread over the sequence stride, which RFC 0014's
// token-major layout and isPackedThrough() decline) is declined, with or without lengths: the
// binding is read on the graph as given, not on the layout the padding would give it.
TEST(TestFlydslSdpaHeadPad, DeclinesAHeadMajorPackedGraph)
{
    for(const bool lengths : {false, true})
    {
        SCOPED_TRACE(lengths ? "seq_len_q and seq_len_kv" : "no lengths");
        auto spec = rebaseSpec(4, true);
        if(!lengths)
        {
            spec.paddingMask = false;
            spec.seqLensQ.clear();
            spec.seqLensKv.clear();
        }
        // Control: the PyTorch layout, token-major with the multiplier its token stride, is
        // built, and its LSE is [sq*hq, 1, hq, 1] as the graph has it or as the scratch is.
        {
            const GraphFixture control(buildFlydslSdpaRdnaGraph(spec));
            const auto padded = flydsl_sdpa::buildHeadPadGraph(control.context());
            ASSERT_NE(padded, nullptr);
            const auto inner = padded->contextFor(control.context());
            const auto& q = tensorOf(inner, FLYDSL_SDPA_RDNA_Q_UID);
            EXPECT_EQ(stridesOf(q), (std::vector<int64_t>{37 * 4 * 64, 64, 4 * 64, 1}));
            EXPECT_EQ(q.ragged_offset_multiplier(), q.strides()->Get(2));
            EXPECT_EQ(stridesOf(tensorOf(inner, FLYDSL_SDPA_RDNA_STATS_UID)),
                      (std::vector<int64_t>{37 * 4, 1, 4, 1}));
        }
        const GraphFixture fixture(withHeadMajorPacking(spec));
        EXPECT_EQ(flydsl_sdpa::buildHeadPadGraph(fixture.context()), nullptr);
        EXPECT_FALSE(matchesGraph(FLYDSL_SDPA_RDNA, fixture.context()).has_value());
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
