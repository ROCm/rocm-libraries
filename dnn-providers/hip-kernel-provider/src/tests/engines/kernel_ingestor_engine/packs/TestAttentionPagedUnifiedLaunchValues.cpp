// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/packs/AttentionPagedUnifiedLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanTestUtilities.hpp"

/**
 * @file TestAttentionPagedUnifiedLaunchValues.cpp
 * @brief The saved launch values of hipkernel:AttentionPagedUnified: every member of the
 *        launch state maps to the value of its own name and type, reads back unchanged
 *        for each binding the engine serves (dense, ragged, separate K and V page tables,
 *        sinks in the Q data type and FLOAT, every finite scale), and the reader refuses
 *        values outside the v1 contract. No device needed.
 *
 * Each member gets a distinct value, so a swapped or dropped mapping shows. The expected
 * names are written out again here: they are the stored contract, and a test that shared
 * the constants would not see a rename.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using hipdnn_plugin_sdk::ingestor::MetadataType;
using hipdnn_plugin_sdk::ingestor::MetadataValues;
using hipdnn_plugin_sdk::ingestor::SavedLaunchInputs;
using hipdnn_plugin_sdk::ingestor::toString;
using serialization::expectCommonReadRefusals;

constexpr int64_t NO_UID = ATTENTION_PAGED_UNIFIED_NO_UID;

/// A dense (uniform decode) binding: one page table for K and V, no sink.
AttentionPagedUnifiedLaunchInputs denseInputs()
{
    AttentionPagedUnifiedLaunchInputs launch;
    launch.uids.q = 101;
    launch.uids.k = 102;
    launch.uids.v = 103;
    launch.uids.o = 104;
    launch.uids.pageTableK = 105;
    launch.uids.pageTableV = 105;
    launch.uids.seqLenKv = 107;
    launch.uids.seqLenQ = 108;
    launch.uids.raggedOffset = NO_UID;
    launch.uids.sink = NO_UID;
    auto& s = launch.scalars;
    s.scale = 1.0F / std::sqrt(128.0F);
    s.numSeqs = 6;
    s.blockTableStride = 24;
    s.numQueryHeads = 32;
    s.numKvHeads = 8;
    s.slidingWindow = 0x3FFFFFFF;
    s.qUniformLen = 3;
    s.q = {12288, 4096, 128};
    s.o = {12416, 4224, 136};
    s.kvBlockStride = 65536;
    s.kvTokStride = 256;
    s.kvHeadStride = 8192;
    s.kvBlockSpanBytes = 131072;
    s.qBlockGrid = 18;
    launch.blockThreads = 256;
    return launch;
}

/// A ragged Q over paged K/V with a sliding window.
AttentionPagedUnifiedLaunchInputs raggedInputs()
{
    auto launch = denseInputs();
    launch.uids.raggedOffset = 109;
    launch.scalars.numSeqs = 5;
    launch.scalars.qUniformLen = 0;
    launch.scalars.slidingWindow = 33;
    launch.scalars.q = {4096, 4096, 128};
    launch.scalars.o = {4224, 4224, 136};
    launch.scalars.qBlockGrid = 21;
    return launch;
}

/// Separate page_table_k and page_table_v tensors.
AttentionPagedUnifiedLaunchInputs separateTablesInputs()
{
    auto launch = denseInputs();
    launch.uids.pageTableV = 110;
    return launch;
}

/// An attention sink in the I/O dtype.
AttentionPagedUnifiedLaunchInputs sinkInputs()
{
    auto launch = raggedInputs();
    launch.uids.sink = 111;
    launch.blockThreads = 128;
    return launch;
}

/// A FLOAT attention sink (cuDNN's form).
AttentionPagedUnifiedLaunchInputs floatSinkInputs()
{
    auto launch = sinkInputs();
    launch.scalars.sinkIsF32 = 1;
    return launch;
}

SavedLaunchInputs saved(const AttentionPagedUnifiedLaunchInputs& launch)
{
    return {std::string(ATTENTION_PAGED_UNIFIED_DISPATCH_SYMBOL_V1),
            attentionPagedUnifiedLaunchValues(launch)};
}

uint32_t bitsOf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void expectSameLaunch(const AttentionPagedUnifiedLaunchInputs& read,
                      const AttentionPagedUnifiedLaunchInputs& expected)
{
    EXPECT_EQ(read.uids.q, expected.uids.q);
    EXPECT_EQ(read.uids.k, expected.uids.k);
    EXPECT_EQ(read.uids.v, expected.uids.v);
    EXPECT_EQ(read.uids.o, expected.uids.o);
    EXPECT_EQ(read.uids.pageTableK, expected.uids.pageTableK);
    EXPECT_EQ(read.uids.pageTableV, expected.uids.pageTableV);
    EXPECT_EQ(read.uids.seqLenKv, expected.uids.seqLenKv);
    EXPECT_EQ(read.uids.seqLenQ, expected.uids.seqLenQ);
    EXPECT_EQ(read.uids.raggedOffset, expected.uids.raggedOffset);
    EXPECT_EQ(read.uids.sink, expected.uids.sink);
    const auto& r = read.scalars;
    const auto& e = expected.scalars;
    EXPECT_EQ(bitsOf(r.scale), bitsOf(e.scale));
    EXPECT_EQ(r.numSeqs, e.numSeqs);
    EXPECT_EQ(r.blockTableStride, e.blockTableStride);
    EXPECT_EQ(r.numQueryHeads, e.numQueryHeads);
    EXPECT_EQ(r.numKvHeads, e.numKvHeads);
    EXPECT_EQ(r.slidingWindow, e.slidingWindow);
    EXPECT_EQ(r.qUniformLen, e.qUniformLen);
    EXPECT_EQ(r.q.seqStride, e.q.seqStride);
    EXPECT_EQ(r.q.tokenStride, e.q.tokenStride);
    EXPECT_EQ(r.q.headStride, e.q.headStride);
    EXPECT_EQ(r.o.seqStride, e.o.seqStride);
    EXPECT_EQ(r.o.tokenStride, e.o.tokenStride);
    EXPECT_EQ(r.o.headStride, e.o.headStride);
    EXPECT_EQ(r.kvBlockStride, e.kvBlockStride);
    EXPECT_EQ(r.kvTokStride, e.kvTokStride);
    EXPECT_EQ(r.kvHeadStride, e.kvHeadStride);
    EXPECT_EQ(r.kvBlockSpanBytes, e.kvBlockSpanBytes);
    EXPECT_EQ(r.qBlockGrid, e.qBlockGrid);
    EXPECT_EQ(r.sinkIsF32, e.sinkIsF32);
    EXPECT_EQ(read.blockThreads, expected.blockThreads);
}

// Expects the reader to refuse `inputs` as a valid plan that cannot be restored, with
// `phrase` in the message.
void expectRefusal(const SavedLaunchInputs& inputs, const std::string& phrase)
{
    serialization::expectReadRefusal(&readAttentionPagedUnifiedLaunchInputs, inputs, phrase);
}

SavedLaunchInputs with(SavedLaunchInputs inputs,
                       const std::string& name,
                       hipdnn_plugin_sdk::ingestor::MetadataValue value)
{
    inputs.values[name] = std::move(value);
    return inputs;
}

TEST(TestAttentionPagedUnifiedLaunchValues, ValuesMatchTheV1Contract)
{
    const auto launch = denseInputs();

    const MetadataValues values = attentionPagedUnifiedLaunchValues(launch);

    const MetadataValues expected{{"q.uid", int64_t{101}},
                                  {"k.uid", int64_t{102}},
                                  {"v.uid", int64_t{103}},
                                  {"o.uid", int64_t{104}},
                                  {"page_table_k.uid", int64_t{105}},
                                  {"page_table_v.uid", int64_t{105}},
                                  {"seq_len_kv.uid", int64_t{107}},
                                  {"seq_len_q.uid", int64_t{108}},
                                  {"ragged_offset.uid", int64_t{-1}},
                                  {"sink.uid", int64_t{-1}},
                                  {"scale", static_cast<double>(launch.scalars.scale)},
                                  {"num_seqs", int64_t{6}},
                                  {"block_table_stride", int64_t{24}},
                                  {"num_query_heads", int64_t{32}},
                                  {"num_kv_heads", int64_t{8}},
                                  {"sliding_window", int64_t{0x3FFFFFFF}},
                                  {"q_uniform_len", int64_t{3}},
                                  {"q_seq_stride", int64_t{12288}},
                                  {"q_tok_stride", int64_t{4096}},
                                  {"q_head_stride", int64_t{128}},
                                  {"o_seq_stride", int64_t{12416}},
                                  {"o_tok_stride", int64_t{4224}},
                                  {"o_head_stride", int64_t{136}},
                                  {"kv_block_stride", int64_t{65536}},
                                  {"kv_tok_stride", int64_t{256}},
                                  {"kv_head_stride", int64_t{8192}},
                                  {"kv_block_span_bytes", int64_t{131072}},
                                  {"q_block_grid", int64_t{18}},
                                  {"sink_is_f32", int64_t{0}},
                                  {"block_threads", int64_t{256}}};
    EXPECT_EQ(values, expected);
    EXPECT_EQ(ATTENTION_PAGED_UNIFIED_DISPATCH_SYMBOL_V1,
              "hipkernel.attention_paged_unified.dispatch.v1");
}

TEST(TestAttentionPagedUnifiedLaunchValues, DenseInputsRoundTripThroughTheirValues)
{
    expectSameLaunch(readAttentionPagedUnifiedLaunchInputs(saved(denseInputs())), denseInputs());
}

TEST(TestAttentionPagedUnifiedLaunchValues, RaggedInputsRoundTripThroughTheirValues)
{
    expectSameLaunch(readAttentionPagedUnifiedLaunchInputs(saved(raggedInputs())), raggedInputs());
}

TEST(TestAttentionPagedUnifiedLaunchValues, SeparateKAndVTablesRoundTripThroughTheirValues)
{
    const auto read = readAttentionPagedUnifiedLaunchInputs(saved(separateTablesInputs()));
    expectSameLaunch(read, separateTablesInputs());
    EXPECT_NE(read.uids.pageTableK, read.uids.pageTableV);
}

TEST(TestAttentionPagedUnifiedLaunchValues, SinkInputsRoundTripThroughTheirValues)
{
    expectSameLaunch(readAttentionPagedUnifiedLaunchInputs(saved(sinkInputs())), sinkInputs());
    const auto read = readAttentionPagedUnifiedLaunchInputs(saved(floatSinkInputs()));
    expectSameLaunch(read, floatSinkInputs());
    EXPECT_EQ(read.scalars.sinkIsF32, 1);
}

// graph_match serves every finite attn_scale, so a plan with a zero or negative scale
// saves and must load back.
TEST(TestAttentionPagedUnifiedLaunchValues, ZeroAndNegativeScalesRoundTripThroughTheirValues)
{
    for(const float scale : {0.0F, -0.0F, -0.125F, -std::numeric_limits<float>::max()})
    {
        SCOPED_TRACE(scale);
        auto launch = denseInputs();
        launch.scalars.scale = scale;
        expectSameLaunch(readAttentionPagedUnifiedLaunchInputs(saved(launch)), launch);
    }
}

TEST(TestAttentionPagedUnifiedLaunchValues, RefusesWhatEveryReaderRefuses)
{
    expectCommonReadRefusals(&readAttentionPagedUnifiedLaunchInputs,
                             saved(denseInputs()),
                             "hipkernel.attention_paged_unified.dispatch",
                             "num_seqs");
}

TEST(TestAttentionPagedUnifiedLaunchValues, RefusesAnotherContractName)
{
    for(const char* name : {"hipkernel.attention_paged_unified.dispatch.v2",
                            "hipkernel.gfx950_attention_dense.dispatch.v2"})
    {
        SCOPED_TRACE(name);
        auto inputs = saved(denseInputs());
        inputs.dispatchSymbol = name;
        expectRefusal(inputs, "names no launch-value contract");
    }
}

TEST(TestAttentionPagedUnifiedLaunchValues, RefusesAMissingOptionalOrFloatValue)
{
    for(const char* name : {"sink.uid", "ragged_offset.uid", "scale", "sink_is_f32"})
    {
        SCOPED_TRACE(name);
        auto missing = saved(denseInputs());
        missing.values.erase(name);
        expectRefusal(missing, "needs launch value '" + std::string(name) + "'");
    }
}

TEST(TestAttentionPagedUnifiedLaunchValues, RefusesAValueOfAnotherType)
{
    const auto valid = saved(denseInputs());
    expectRefusal(with(valid, "k.uid", 102.0), "launch value 'k.uid'");

    const std::string mustBeFloat
        = "must be " + toString(MetadataType::FLOAT) + ", and the plan holds ";
    expectRefusal(with(valid, "scale", int64_t{1}), mustBeFloat + toString(MetadataType::INT));
    expectRefusal(with(valid, "scale", std::string("0.088")),
                  mustBeFloat + toString(MetadataType::STRING));
}

TEST(TestAttentionPagedUnifiedLaunchValues, RefusesAnIntegerOutsideWhatTheKernelTakes)
{
    const auto dense = saved(denseInputs());
    const auto ragged = saved(raggedInputs());
    struct OutOfRange
    {
        const SavedLaunchInputs* base;
        const char* name;
        int64_t value;
    };
    const std::vector<OutOfRange> cases = {
        {&dense, "num_seqs", 0},
        {&dense, "num_seqs", int64_t{2147483648}},
        {&dense, "num_query_heads", 0},
        {&dense, "num_kv_heads", 0},
        {&dense, "sliding_window", 0},
        {&dense, "sliding_window", int64_t{0x40000000}},
        {&dense, "q_uniform_len", 0}, // a dense Q has at least one row per sequence
        {&dense, "q_uniform_len", -3},
        {&ragged, "q_uniform_len", 1}, // a ragged Q reads its rows from the offsets
        {&ragged, "q_uniform_len", -1},
        {&dense, "sink_is_f32", -1},
        {&dense, "sink_is_f32", 2},
        {&dense, "block_table_stride", -1},
        {&dense, "q_tok_stride", -1},
        {&dense, "o_head_stride", int64_t{2147483648}},
        {&dense, "kv_block_stride", -1},
        {&dense, "kv_block_span_bytes", 0},
        {&dense, "kv_block_span_bytes", int64_t{0x7FFF0001}},
        {&dense, "q_block_grid", 0},
        {&dense, "q_block_grid", 65536},
        {&dense, "block_threads", 32},
        {&dense, "block_threads", 2048},
    };
    for(const auto& c : cases)
    {
        SCOPED_TRACE(std::string(c.name) + " = " + std::to_string(c.value));
        expectRefusal(with(*c.base, c.name, c.value), "launch value '" + std::string(c.name) + "'");
    }

    // In range, but not a whole number of waves.
    expectRefusal(with(dense, "block_threads", int64_t{96}), "a multiple of 64");
    // The query heads must share the KV heads evenly.
    expectRefusal(with(dense, "num_query_heads", int64_t{12}), "a multiple of 'num_kv_heads'");
}

TEST(TestAttentionPagedUnifiedLaunchValues, RefusesAScaleNoFiniteFloatWasSavedAs)
{
    const auto valid = saved(denseInputs());
    // 0.1 as a double is not a float value, so no float scale was saved as it.
    for(const double scale : {0.1,
                              std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN(),
                              1.0e39,
                              -1.0e39})
    {
        SCOPED_TRACE(scale);
        expectRefusal(with(valid, "scale", scale), "launch value 'scale'");
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
