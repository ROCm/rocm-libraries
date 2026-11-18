// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdVerdictTable.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdSplitEngine.cpp
 * @brief The slotted flyDSL SDPA backward engine across its native seams, over synthetic
 *        slot rows: no staged shard, no device.
 *
 * A graph is built from the shared matcher's test graphs and run through graph_match; its
 * bound tokens and a row written in the test go to a slot's kernel_match and, through
 * describeFlydslSdpaBwdSplitLaunchForTesting, to the arguments the slot handler's launch()
 * passes without touching a device. The merged engine's launch of the same graph is the
 * reference for every buffer, stride and scalar the two engines share.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

using data_objects::DataType;
using Slot = FlydslSdpaBwdSplitSlot;

/// The dq slot's multi-launch dispatch, over the same graph match and kernel match.
const PackSymbols DQ_REDUCE{"hipkernel:flydsl_sdpa_bwd_split",
                            "hipkernel.flydsl_sdpa_bwd_split.graph_match",
                            "",
                            "hipkernel.flydsl_sdpa_bwd_split.dq.kernel_match",
                            "",
                            "hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch",
                            "",
                            "",
                            ""};

/// The dkdv slot's two-launch dispatch for V heads that differ from K's.
const PackSymbols DKDV_U2{"hipkernel:flydsl_sdpa_bwd_split",
                          "hipkernel.flydsl_sdpa_bwd_split.graph_match",
                          "",
                          "hipkernel.flydsl_sdpa_bwd_split.dkdv.kernel_match",
                          "",
                          "hipkernel.flydsl_sdpa_bwd_split.dkdv_u2.dispatch",
                          "",
                          "",
                          ""};

/// The dq slot's dispatch for V heads that differ from K's: the V-head dQ and tail-zero kernels.
const PackSymbols DQ_U2{"hipkernel:flydsl_sdpa_bwd_split",
                        "hipkernel.flydsl_sdpa_bwd_split.graph_match",
                        "",
                        "hipkernel.flydsl_sdpa_bwd_split.dq.kernel_match",
                        "",
                        "hipkernel.flydsl_sdpa_bwd_split.dq_u2.dispatch",
                        "",
                        "",
                        ""};

const PackSymbols MERGED{"hipkernel:flydsl_sdpa_bwd_merged",
                         "hipkernel.flydsl_sdpa_bwd_merged.graph_match",
                         "",
                         "hipkernel.flydsl_sdpa_bwd_merged.kernel_match",
                         "hipkernel.flydsl_sdpa_bwd_merged.score",
                         "hipkernel.flydsl_sdpa_bwd_merged.dispatch",
                         "",
                         "",
                         ""};

const char* const REDUCE_SYMBOL = "fmha_bwd_dbias_reduce_gfx950_kernel_1";
const std::array<Slot, 3> SLOTS{Slot::PREP, Slot::DKDV, Slot::DQ};

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

SdpaBwdSpec dimsSpec(int64_t batch, int64_t heads, int64_t seqQ, int64_t seqKv, int64_t headDim)
{
    SdpaBwdSpec spec;
    spec.batch = batch;
    spec.heads = heads;
    spec.kvHeads = heads;
    spec.seqQ = seqQ;
    spec.seqKv = seqKv;
    spec.headDimQk = headDim;
    spec.headDimV = headDim;
    return spec;
}

SdpaBwdSpec withWindow(const SdpaBwdSpec& spec)
{
    return spec.withEdit([](SdpaBwdGraph& g) {
        g.attributes.causalMask = true;
        g.attributes.leftBound = 4;
    });
}

SdpaBwdSpec withDropout(const SdpaBwdSpec& spec,
                        DataType seedType = DataType::INT64,
                        bool passByValue = false,
                        float probability = 0.2F)
{
    return spec.withEdit(
        [=](SdpaBwdGraph& g) { addDropout(g, probability, seedType, passByValue); });
}

SdpaBwdSpec withScale(const SdpaBwdSpec& spec)
{
    return spec.withEdit([](SdpaBwdGraph& g) { addScaleTensor(g, 0.5F); });
}

SdpaBwdSpec withPaddedVarlen(const SdpaBwdSpec& spec)
{
    return spec.withEdit([](SdpaBwdGraph& g) { addSeqLens(g); });
}

/// @p spec padded, with the tensors named in @p names ragged (multiplier @p multiplier), each
/// with an offset tensor of its own; a ragged Q brings ragged stats packed (H, T).
SdpaBwdSpec
    raggedOf(const SdpaBwdSpec& spec, const std::vector<std::string>& names, int64_t multiplier = 1)
{
    return spec.withEdit([names, multiplier](SdpaBwdGraph& g) {
        addSeqLens(g);
        for(const auto& name : names)
        {
            addRagged(g, name, multiplier);
            if(name == "q")
            {
                addRagged(g, "stats");
                g.tensor("stats").strides = {int64_t{1} << 20, g.batch * g.seqQ, 1, 1};
            }
        }
    });
}

SdpaBwdSpec withHeadDims(const SdpaBwdSpec& spec, int64_t qk, int64_t v)
{
    auto copy = spec;
    copy.headDimQk = qk;
    copy.headDimV = v;
    return copy;
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

/// A slot object's record, as the descriptor generator writes it. Fields a slot does not carry
/// are left out of its record by makeRow().
struct RowSpec
{
    std::string dtype = "f16";
    int64_t tile = 64;
    int64_t causal = 0;
    int64_t hasBias = 0;
    int64_t dropout = 0;
    int64_t padded = 0;
    std::string floorValue = "tail";
    std::string dform = "bk";
    int64_t sink = -1;
    int64_t gqaPack = -1;
    int64_t blockQ = -1;
    int64_t storeDb = -1;
    /// dq and dkdv: the bias is FLOAT beside a 16-bit query.
    int64_t biasF32 = 0;
    /// dq: `lds`, `direct`, `direct_only`, or `na` for a row without a bias.
    std::string biasStageResolved = "na";
    /// dq and dkdv: `none`, `dk` (dkdv) or `dq` (dq).
    std::string u2Role = "none";
    std::string reduceSymbol;
    int64_t blockThreads = 256;
    int64_t blockKv = 64;
    int64_t blockM = 64;
    /// The launch grid's spelling; empty is the slot's gfx950 one.
    std::string grid;
};

const char* const GRID_PREP = "qblocks_by_heads_by_seqs";
const char* const GRID_DKDV_HEADS_FIRST = "kheads_by_kvblocks_by_seqs";
const char* const GRID_DKDV_BLOCKS_FIRST = "kvblocks_by_kheads_by_seqs";
const char* const GRID_DQ = "qheads_by_qblocks_by_seqs";

std::string defaultGrid(Slot slot)
{
    if(slot == Slot::PREP)
    {
        return GRID_PREP;
    }
    return slot == Slot::DKDV ? GRID_DKDV_HEADS_FIRST : GRID_DQ;
}

KernelDefinition makeRow(Slot slot, const RowSpec& row)
{
    KernelDefinition kernel;
    kernel.name = "sdpa_bwd_split_row";
    kernel.metadata = {{"dtype", row.dtype},
                       {"head_dim", row.tile},
                       {"head_dim_v", row.tile},
                       {"padded", row.padded},
                       {"floor", row.floorValue},
                       {"hints", int64_t{1}},
                       {"grid", row.grid.empty() ? defaultGrid(slot) : row.grid}};
    if(slot == Slot::PREP)
    {
        kernel.metadata["sink"] = row.sink;
        return kernel;
    }
    kernel.metadata["causal"] = row.causal;
    kernel.metadata["has_bias"] = row.hasBias;
    kernel.metadata["has_dropout"] = row.dropout;
    kernel.metadata["dform"] = row.dform;
    kernel.metadata["block_threads"] = row.blockThreads;
    kernel.metadata["bias_f32"] = row.biasF32;
    kernel.metadata["u2_role"] = row.u2Role;
    if(slot == Slot::DKDV)
    {
        kernel.metadata["gqa_pack"] = row.gqaPack;
        kernel.metadata["block_q"] = row.blockQ;
        kernel.metadata["block_kv"] = row.blockKv;
        return kernel;
    }
    kernel.metadata["store_db"] = row.storeDb;
    kernel.metadata["bias_stage_resolved"] = row.biasStageResolved;
    kernel.metadata["block_m"] = row.blockM;
    kernel.metadata["reduce_symbol"] = row.reduceSymbol;
    return kernel;
}

RowSpec plainRow(int64_t tile = 64)
{
    RowSpec row;
    row.tile = tile;
    return row;
}

RowSpec causalRow(int64_t tile = 64)
{
    RowSpec row = plainRow(tile);
    row.causal = 1;
    return row;
}

/// A bias row that stages the bias through LDS, as the gfx950 16-bit bias rows with a staged
/// alternative do not (those are `direct`) but the ones this default stands for do.
RowSpec biasRow(int64_t tile = 64)
{
    RowSpec row = plainRow(tile);
    row.hasBias = 1;
    row.biasStageResolved = "lds";
    row.reduceSymbol = REDUCE_SYMBOL;
    return row;
}

/// The merged engine's record of a plain, causal or bias row at @p tile, for reference
/// launches.
KernelDefinition makeMergedRow(const RowSpec& row)
{
    KernelDefinition kernel;
    kernel.name = "sdpa_bwd_merged_row";
    kernel.metadata = {{"dtype", row.dtype},
                       {"head_dim", row.tile},
                       {"head_dim_v", row.tile},
                       {"causal", row.causal},
                       {"has_bias", row.hasBias},
                       {"has_dropout", row.dropout},
                       {"padded", row.padded},
                       {"hints", int64_t{1}},
                       {"floor", std::string("tail")},
                       {"dform", std::string("bk")},
                       {"role_order", std::string("concat")},
                       {"store_db", int64_t{-1}},
                       {"gqa_pack", int64_t{-1}},
                       {"block_threads", row.blockThreads},
                       {"block_kv", row.blockKv},
                       {"block_m", row.blockM},
                       {"reduce_symbol", row.reduceSymbol}};
    return kernel;
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

std::optional<BoundTokens> graphMatch(const SdpaBwdSpec& spec)
{
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    return matchesGraph(splitSymbols(Slot::PREP), fixture.context());
}

/// Whether the @p slot row serves the graph of @p spec; the graph itself must be served.
bool serves(Slot slot, const SdpaBwdSpec& spec, const RowSpec& row)
{
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    const auto bound = matchesGraph(splitSymbols(slot), fixture.context());
    if(!bound.has_value())
    {
        throw std::logic_error("test graph is declined by graph_match");
    }
    return matchesKernel(splitSymbols(slot), fixture.context(), makeRow(slot, row), *bound);
}

// ---------------------------------------------------------------------------
// Launches
// ---------------------------------------------------------------------------

std::vector<hipdnnPluginDeviceBuffer_t> buffersFor()
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers;
    for(int64_t uid = sdpa_bwd_uid::Q; uid <= sdpa_bwd_uid::RAGGED_OFFSET_STATS; ++uid)
    {
        buffers.push_back({uid, asPointer(ptrOf(uid))});
    }
    return buffers;
}

FlydslSdpaBwdSplitLaunchDump dumpOf(Slot slot,
                                    const SdpaBwdSpec& spec,
                                    const RowSpec& row,
                                    const std::function<void(BoundTokens&)>& edit = {})
{
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    auto bound = matchesGraph(splitSymbols(slot), fixture.context());
    if(!bound.has_value())
    {
        throw std::logic_error("test graph is declined by graph_match");
    }
    if(edit)
    {
        edit(*bound);
    }
    return describeFlydslSdpaBwdSplitLaunchForTesting(
        slot, fixture.context(), *bound, makeRow(slot, row), buffersFor(), asPointer(WORKSPACE));
}

FlydslSdpaBwdMergedLaunchDump mergedDumpOf(const SdpaBwdSpec& spec, const RowSpec& row)
{
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    return describeFlydslSdpaBwdMergedLaunchForTesting(
        fixture.context(), makeMergedRow(row), buffersFor(), asPointer(WORKSPACE));
}

int64_t tokenOf(const BoundTokens& bound, const std::string& name)
{
    const auto value = tryGetBoundInt(bound, name);
    EXPECT_TRUE(value.has_value()) << name;
    return value.value_or(-1);
}

// ---------------------------------------------------------------------------
// Kernarg images
// ---------------------------------------------------------------------------

/// One kernel parameter: its name and kind -- p pointer, i int32, f float32, l int64.
struct Param
{
    std::string name;
    char kind;
};

size_t paramBytes(char kind)
{
    return kind == 'p' || kind == 'l' ? 8 : 4;
}

using ParamList = std::vector<Param>;

void addParams(ParamList& list, char kind, std::initializer_list<const char*> names)
{
    for(const char* name : names)
    {
        list.push_back({name, kind});
    }
}

void addStrides(ParamList& list, std::initializer_list<const char*> tensors)
{
    for(const char* tensor : tensors)
    {
        const std::string stem = std::string("stride_") + tensor;
        const bool rowAxis = std::string(tensor) == "b" || std::string(tensor) == "db";
        list.push_back({stem + "_batch", 'l'});
        list.push_back({stem + "_head", 'l'});
        list.push_back({stem + (rowAxis ? "_seq_q" : "_seq"), 'l'});
    }
}

/// The offset arguments a kernel takes after its seqinfo pointers: the multipliers of Q and K
/// when @p queryKey (their offsets are `seqinfo_q1` / `seqinfo_k1`), an offset array and a
/// multiplier for each of @p tensors, and, with @p stats, the stats' offset, multiplier and
/// strides.
void addOffsetBlock(ParamList& list,
                    bool queryKey,
                    std::initializer_list<const char*> tensors,
                    bool stats)
{
    if(queryKey)
    {
        addParams(list, 'l', {"mul_Q", "mul_K"});
    }
    for(const char* tensor : tensors)
    {
        list.push_back({std::string("off_") + tensor, 'p'});
        list.push_back({std::string("mul_") + tensor, 'l'});
    }
    if(stats)
    {
        addParams(list, 'p', {"off_Stats"});
        addParams(list,
                  'l',
                  {"mul_Stats", "stride_stats_batch", "stride_stats_head", "stride_stats_token"});
    }
}

/// The scalar block every dkdv, dq and merged argument list shares, from `seqinfo_q0` to
/// `sm_scale`, with the offset block @p offsets after `seqinfo_k1`, and, with @p valueHeads,
/// `num_head_v` right after `num_head_k` (the V-head variants).
void addScalarBlock(ParamList& list,
                    const std::function<void(ParamList&)>& offsets,
                    bool valueHeads = false)
{
    addParams(list, 'p', {"seqinfo_q0", "seqinfo_q1", "seqinfo_k0", "seqinfo_k1"});
    offsets(list);
    addParams(list,
              'i',
              {"varlen_bits",
               "num_seqlens",
               "max_seqlen_q",
               "max_seqlen_k",
               "window_left",
               "window_right"});
    addParams(list, 'p', {"philox_seed_ptr", "philox_offset1"});
    addParams(list, 'l', {"philox_offset2"});
    addParams(list, 'i', {"idropout_p"});
    addParams(list, 'f', {"dropout_scale"});
    addParams(list, 'i', {"num_head_q", "num_head_k"});
    if(valueHeads)
    {
        addParams(list, 'i', {"num_head_v"});
    }
    addParams(list, 'i', {"hdim_qk", "hdim_vo"});
    addParams(list, 'f', {"sm_scale"});
}

/// `_PARAMS["prep"]`, in order.
const ParamList& prepParams()
{
    static const ParamList s_params = [] {
        ParamList list;
        addParams(list, 'p', {"O", "DO", "LSE", "LSE_ws", "Delta", "seqinfo_q0", "seqinfo_q1"});
        addOffsetBlock(list, false, {"O", "DO"}, true);
        addParams(
            list, 'i', {"varlen_bits", "num_seqlens", "max_seqlen_q", "num_head_q", "hdim_vo"});
        addStrides(list, {"o", "do"});
        return list;
    }();
    return s_params;
}

/// `_PARAMS["dkdv"]`, in order.
const ParamList& dkdvParams()
{
    static const ParamList s_params = [] {
        ParamList list;
        addParams(list, 'p', {"Q", "K", "V", "B", "DO", "DK", "DV", "LSE", "Delta"});
        addScalarBlock(list, [](ParamList& offsets) {
            addOffsetBlock(offsets, true, {"V", "DO", "DK", "DV"}, false);
        });
        addStrides(list, {"q", "k", "v", "do", "dk", "dv", "b"});
        addParams(list, 'i', {"rt_flags"});
        return list;
    }();
    return s_params;
}

/// `_PARAMS["dkdv_u2"]`: the dkdv list with `num_head_v` after `num_head_k`.
const ParamList& dkdvU2Params()
{
    static const ParamList s_params = [] {
        ParamList list;
        addParams(list, 'p', {"Q", "K", "V", "B", "DO", "DK", "DV", "LSE", "Delta"});
        addScalarBlock(
            list,
            [](ParamList& offsets) {
                addOffsetBlock(offsets, true, {"V", "DO", "DK", "DV"}, false);
            },
            true);
        addStrides(list, {"q", "k", "v", "do", "dk", "dv", "b"});
        addParams(list, 'i', {"rt_flags"});
        return list;
    }();
    return s_params;
}

/// `_PARAMS["dq_u2"]`: the dq list with `num_head_v` after `num_head_k`.
const ParamList& dqU2Params()
{
    static const ParamList s_params = [] {
        ParamList list;
        addParams(list, 'p', {"Q", "K", "V", "B", "DO", "DQ", "DB", "LSE", "Delta"});
        addScalarBlock(
            list,
            [](ParamList& offsets) { addOffsetBlock(offsets, true, {"V", "DO", "DQ"}, false); },
            true);
        addStrides(list, {"q", "k", "v", "do", "dq", "b", "db"});
        addParams(list, 'i', {"rt_flags"});
        return list;
    }();
    return s_params;
}

/// `_PARAMS["dq"]`, in order.
const ParamList& dqParams()
{
    static const ParamList s_params = [] {
        ParamList list;
        addParams(list, 'p', {"Q", "K", "V", "B", "DO", "DQ", "DB", "LSE", "Delta"});
        addScalarBlock(list, [](ParamList& offsets) {
            addOffsetBlock(offsets, true, {"V", "DO", "DQ"}, false);
        });
        addStrides(list, {"q", "k", "v", "do", "dq", "b", "db"});
        addParams(list, 'i', {"rt_flags"});
        return list;
    }();
    return s_params;
}

/// `_PARAMS["merged"]`, in order.
const ParamList& mergedParams()
{
    static const ParamList s_params = [] {
        ParamList list;
        addParams(list, 'p', {"Q", "K", "V", "B", "O", "DO", "DK", "DV", "DQ", "DB", "LSE"});
        addParams(list, 'i', {"batch_size"});
        addScalarBlock(list, [](ParamList& offsets) {
            addOffsetBlock(offsets, true, {"V", "O", "DO", "DQ", "DK", "DV"}, true);
        });
        addStrides(list, {"q", "k", "v", "o", "do", "dk", "dv", "dq", "b", "db"});
        addParams(list, 'i', {"rt_flags"});
        return list;
    }();
    return s_params;
}

/// `_PARAMS["dbias_reduce"]`, in order.
const ParamList& reduceParams()
{
    static const ParamList s_params = [] {
        ParamList list;
        addParams(list, 'p', {"DB", "DBias"});
        addParams(list,
                  'i',
                  {"num_batch", "num_head", "seqlen_q", "seqlen_k", "reduce_batch", "reduce_head"});
        addStrides(list, {"ws"});
        addStrides(list, {"out"});
        return list;
    }();
    return s_params;
}

/// `PARAMS_HV` of the tail-zero generator: `_PARAMS["tailzero"]` with `Hv` after `Hkv`.
const ParamList& tailzeroHvParams()
{
    static const ParamList s_params = [] {
        ParamList list;
        addParams(list, 'p', {"DQ", "DK", "DV", "seq_len_q", "seq_len_kv"});
        addParams(list, 'i', {"B", "Hq", "Hkv", "Hv", "Sq", "Skv", "Dqk", "Dv"});
        addStrides(list, {"dq", "dk", "dv"});
        addParams(list, 'i', {"mask"});
        return list;
    }();
    return s_params;
}

/// `_PARAMS["tailzero"]`, in order.
const ParamList& tailzeroParams()
{
    static const ParamList s_params = [] {
        ParamList list;
        addParams(list, 'p', {"DQ", "DK", "DV", "seq_len_q", "seq_len_kv"});
        addParams(list, 'i', {"B", "Hq", "Hkv", "Sq", "Skv", "Dqk", "Dv"});
        addStrides(list, {"dq", "dk", "dv"});
        addParams(list, 'i', {"mask"});
        return list;
    }();
    return s_params;
}

const ParamList& paramsOf(Slot slot)
{
    if(slot == Slot::PREP)
    {
        return prepParams();
    }
    return slot == Slot::DKDV ? dkdvParams() : dqParams();
}

size_t imageBytes(const ParamList& params)
{
    size_t offset = 0;
    for(const auto& param : params)
    {
        const size_t size = paramBytes(param.kind);
        offset = (offset + size - 1) / size * size + size;
    }
    return offset;
}

/// The parameter values a kernarg image holds, by name, read at each parameter's natural
/// offset.
std::map<std::string, uint64_t> decode(const std::vector<unsigned char>& image,
                                       const ParamList& params)
{
    std::map<std::string, uint64_t> values;
    size_t offset = 0;
    for(const auto& param : params)
    {
        const size_t size = paramBytes(param.kind);
        offset = (offset + size - 1) / size * size;
        uint64_t raw = 0;
        if(offset + size <= image.size())
        {
            std::memcpy(&raw, image.data() + offset, size);
        }
        values[param.name] = raw;
        offset += size;
    }
    return values;
}

/// The values of @p dump's launch of @p slot, with the image checked against the parameter
/// list's size.
std::map<std::string, uint64_t> valuesOf(Slot slot, const FlydslSdpaBwdSplitLaunchDump& dump)
{
    EXPECT_EQ(dump.kernarg.size(), imageBytes(paramsOf(slot)));
    return decode(dump.kernarg, paramsOf(slot));
}

// ---------------------------------------------------------------------------
// graph_match
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdSplitGraph, ServesThePlainGraphTheSharedMatcherAccepts)
{
    EXPECT_TRUE(std::holds_alternative<flydsl_sdpa_bwd::SdpaBwdProblem>(
        matchSdpaBwdSpec(baseSpec(), flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS)));
    EXPECT_TRUE(graphMatch(baseSpec()).has_value());
}

TEST(TestFlydslSdpaBwdSplitGraph, DeclinesExactlyWhatTheSharedMatcherDeclines)
{
    const std::vector<SdpaBwdSpec> specs{
        baseSpec(),
        withCausal(baseSpec()),
        withWindow(baseSpec()),
        withBias(baseSpec(), true),
        withBias(baseSpec(), false, {1, 1, 16, 16}),
        withBias(withCausal(baseSpec()), false),
        withDropout(baseSpec()),
        withDropout(baseSpec(), DataType::INT32, true),
        withScale(baseSpec()),
        withHeadDims(baseSpec(), 520, 520),
        withHeadDims(baseSpec(), 96, 160),
        withPaddedVarlen(baseSpec()),
        withPaddedVarlen(withCausal(baseSpec())),
        fp32Spec(),
        withBias(fp32Spec(), true),
        baseSpec().withEdit([](SdpaBwdGraph& g) { g.attributes.alibiMask = true; }),
        baseSpec().withEdit([](SdpaBwdGraph& g) { g.overrideShape = true; }),
        baseSpec().withEdit([](SdpaBwdGraph& g) { g.nodeKind = SdpaBwdNodeKind::FORWARD; }),
    };
    for(size_t index = 0; index < specs.size(); ++index)
    {
        SCOPED_TRACE("spec " + std::to_string(index));
        const auto matched = matchSdpaBwdSpec(specs[index], flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS);
        EXPECT_EQ(graphMatch(specs[index]).has_value(),
                  std::holds_alternative<flydsl_sdpa_bwd::SdpaBwdProblem>(matched));
    }
}

TEST(TestFlydslSdpaBwdSplitGraph, ServesPaddedVarlenWithEverySlotsObject)
{
    const std::array<std::pair<SdpaBwdSpec, RowSpec>, 2> cells{
        {{withPaddedVarlen(baseSpec()), plainRow()},
         {withPaddedVarlen(withCausal(baseSpec())), causalRow()}}};
    for(const auto& [spec, row] : cells)
    {
        EXPECT_TRUE(graphMatch(spec).has_value());
        for(const Slot slot : SLOTS)
        {
            EXPECT_TRUE(serves(slot, spec, row));
        }
    }
}

TEST(TestFlydslSdpaBwdSplitGraphFp32, ServesTheGraphsTheMergedEngineDeclines)
{
    for(const auto& spec : {fp32Spec(), withBias(fp32Spec(), true), withCausal(fp32Spec())})
    {
        const GraphFixture fixture(buildSdpaBwdGraph(spec));
        EXPECT_TRUE(matchesGraph(splitSymbols(Slot::PREP), fixture.context()).has_value());
        EXPECT_FALSE(matchesGraph(MERGED, fixture.context()).has_value());
    }
}

TEST(TestFlydslSdpaBwdSplitGraph, DeclinesAForwardGraphWithoutBindingTokens)
{
    const auto spec
        = baseSpec().withEdit([](SdpaBwdGraph& g) { g.nodeKind = SdpaBwdNodeKind::FORWARD; });
    EXPECT_FALSE(graphMatch(spec).has_value());
}

// ---------------------------------------------------------------------------
// The shared workspace tokens
// ---------------------------------------------------------------------------

/// The graphs whose workspaces differ in shape: no dropout, seeds read in place or uploaded
/// from either form, a broadcast dBias in either dtype.
std::vector<SdpaBwdSpec> workspaceSpecs()
{
    return {baseSpec(),
            withDropout(baseSpec(), DataType::INT64),
            withDropout(baseSpec(), DataType::INT32, true),
            withDropout(baseSpec(), DataType::INT64, true),
            withBias(baseSpec(), true),
            withBias(baseSpec(), true, {1, 1, 16, 16}),
            withBias(withDropout(baseSpec(), DataType::INT32, true), true, {1, 8, 16, 16}),
            withBias(fp32Spec(), true, {1, 1, 16, 16}),
            dimsSpec(3, 5, 77, 91, 64)};
}

TEST(TestFlydslSdpaBwdSplitWorkspace, TokensSatisfyTheRuntimesSharedWorkspaceChecks)
{
    for(const auto& spec : workspaceSpecs())
    {
        const auto bound = graphMatch(spec);
        ASSERT_TRUE(bound.has_value());
        const int64_t bytes = tokenOf(*bound, "ws.bytes");
        EXPECT_GE(bytes, 0);
        // The runtime's rule: ws.scratch_offset is ws.bytes rounded up to 256 bytes.
        EXPECT_EQ(tokenOf(*bound, "ws.scratch_offset"),
                  static_cast<int64_t>(alignSharedWorkspace(static_cast<size_t>(bytes))));
        for(const auto& [name, value] : *bound)
        {
            if(name.rfind("ws.", 0) == 0 && name.size() > 10
               && name.compare(name.size() - 7, 7, ".offset") == 0)
            {
                ASSERT_TRUE(std::holds_alternative<int64_t>(value)) << name;
                EXPECT_GE(std::get<int64_t>(value), 0) << name;
            }
        }
    }
}

TEST(TestFlydslSdpaBwdSplitWorkspace, EveryRegionEndsWithinTheSharedBytesAndNoneOverlap)
{
    for(const auto& spec : workspaceSpecs())
    {
        const auto bound = graphMatch(spec);
        ASSERT_TRUE(bound.has_value());
        const int64_t bytes = tokenOf(*bound, "ws.bytes");
        const auto problem = matchSdpaBwdSpec(spec, flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS);
        const auto& shape = std::get<flydsl_sdpa_bwd::SdpaBwdProblem>(problem);
        const int64_t stats = shape.batch * shape.numHeads * shape.seqLenQ * 4;

        std::vector<std::pair<int64_t, int64_t>> regions{
            {tokenOf(*bound, "ws.delta.offset"), stats}, {tokenOf(*bound, "ws.lse.offset"), stats}};
        if(bound->count("ws.seed.offset") != 0)
        {
            regions.emplace_back(tokenOf(*bound, "ws.seed.offset"), 16);
        }
        // The seed is uploaded exactly when it is not a 64-bit device tensor.
        const bool uploads = shape.seedForm != flydsl_sdpa_bwd::SeedForm::NONE
                             && (shape.seedForm != flydsl_sdpa_bwd::SeedForm::DEVICE
                                 || shape.seedWidth != flydsl_sdpa_bwd::SeedWidth::INT64);
        EXPECT_EQ(bound->count("ws.seed.offset") != 0, uploads);

        std::sort(regions.begin(), regions.end());
        int64_t end = 0;
        for(const auto& [offset, extent] : regions)
        {
            EXPECT_GE(offset, end);
            EXPECT_EQ(offset % 256, 0);
            end = offset + extent;
        }
        EXPECT_LE(end, bytes);
        EXPECT_EQ(end, bytes);
    }
}

TEST(TestFlydslSdpaBwdSplitWorkspace, SPWSDeclinesSharedRegionsWhoseEndOverflowsInt64)
{
    // Delta and LSE' are 2^62 bytes each: each fits int64, the end of the second does not.
    const auto spec = dimsSpec(int64_t{1} << 60, 1, 1, 1, 1);
    EXPECT_TRUE(std::holds_alternative<flydsl_sdpa_bwd::SdpaBwdProblem>(
        matchSdpaBwdSpec(spec, flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS)));
    EXPECT_FALSE(graphMatch(spec).has_value());
    // One bit less: each is 2^61 bytes and the second ends at 2^62, inside int64.
    EXPECT_TRUE(graphMatch(dimsSpec(int64_t{1} << 59, 1, 1, 1, 1)).has_value());
}

TEST(TestFlydslSdpaBwdSplitWorkspace, SPWSDeclinesABroadcastDbiasWorkspaceThatOverflowsInt64)
{
    // The dBias workspace holds B * Sq * Sk elements of 2 bytes, 2^30 * B bytes: past int64 at
    // B = 2^33, inside it at B = 2^31. The statistics (2^50 bytes) fit either way.
    const auto broadcast = [](int64_t batch) {
        constexpr int64_t SEQ_Q = int64_t{1} << 15;
        constexpr int64_t SEQ_KV = int64_t{1} << 14;
        return withBias(dimsSpec(batch, 1, SEQ_Q, SEQ_KV, 1), true, {1, 1, SEQ_Q, SEQ_KV});
    };
    const auto overflowing = broadcast(int64_t{1} << 33);
    const auto matched = matchSdpaBwdSpec(overflowing, flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS);
    const auto* decline = std::get_if<flydsl_sdpa_bwd::Decline>(&matched);
    ASSERT_EQ(decline, nullptr) << decline->cause << ": " << decline->detail;
    EXPECT_FALSE(graphMatch(overflowing).has_value());
    EXPECT_TRUE(graphMatch(broadcast(int64_t{1} << 31)).has_value());
}

TEST(TestFlydslSdpaBwdSplitWorkspace, ServesAGraphWhoseWorkspaceIsLargeButFitsInt64)
{
    const auto spec = dimsSpec(int64_t{1} << 40, 2, 16, 16, 64);
    const auto bound = graphMatch(spec);
    ASSERT_TRUE(bound.has_value());
    // Delta, then LSE' after it: no seed, so no region ahead of them.
    EXPECT_EQ(tokenOf(*bound, "ws.bytes"), 2 * (int64_t{1} << 40) * 2 * 16 * 4);
}

TEST(TestFlydslSdpaBwdSplitGraph, DeclinesADeviceInt32SeedAndServesAPassByValueInt32One)
{
    for(const auto& spec :
        {withDropout(baseSpec(), DataType::INT32), withDropout(fp32Spec(), DataType::INT32)})
    {
        const auto matched = matchSdpaBwdSpec(spec, flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS);
        const auto* decline = std::get_if<flydsl_sdpa_bwd::Decline>(&matched);
        ASSERT_NE(decline, nullptr);
        EXPECT_EQ(decline->cause, "dropout") << decline->detail;
        EXPECT_FALSE(graphMatch(spec).has_value());
    }
    EXPECT_TRUE(graphMatch(withDropout(baseSpec(), DataType::INT32, true)).has_value());
    EXPECT_TRUE(graphMatch(withDropout(baseSpec(), DataType::INT64)).has_value());
}

TEST(TestFlydslSdpaBwdSplitGraph, ProbabilityZeroWithSeedAndOffsetIsTheGraphWithoutDropout)
{
    const auto plain = graphMatch(baseSpec());
    ASSERT_TRUE(plain.has_value());
    for(const auto& spec : {withDropout(baseSpec(), DataType::INT64, false, 0.0F),
                            withDropout(baseSpec(), DataType::INT64, true, 0.0F),
                            withDropout(baseSpec(), DataType::INT32, true, 0.0F)})
    {
        EXPECT_EQ(graphMatch(spec), plain);
        for(const Slot slot : {Slot::DKDV, Slot::DQ})
        {
            EXPECT_TRUE(serves(slot, spec, plainRow()));
        }
    }
    EXPECT_FALSE(graphMatch(withDropout(baseSpec(), DataType::INT32, false, 0.0F)).has_value());
}

TEST(TestFlydslSdpaBwdSplitWorkspace, PublishesNoSeedOffsetForASeedReadInPlace)
{
    const auto inPlace = graphMatch(withDropout(baseSpec(), DataType::INT64));
    const auto uploaded = graphMatch(withDropout(baseSpec(), DataType::INT32, true));
    ASSERT_TRUE(inPlace.has_value() && uploaded.has_value());
    EXPECT_EQ(inPlace->count("ws.seed.offset"), 0U);
    EXPECT_EQ(tokenOf(*uploaded, "ws.seed.offset"), 0);
    EXPECT_EQ(tokenOf(*uploaded, "ws.delta.offset"), 256);
    EXPECT_EQ(tokenOf(*inPlace, "ws.delta.offset"), 0);
}

// ---------------------------------------------------------------------------
// kernel_match
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdSplitKernel, PrepServesEveryClassAndDropoutStateWithOneObject)
{
    for(const auto& spec :
        {baseSpec(), withCausal(baseSpec()), withBias(baseSpec(), true), withDropout(baseSpec())})
    {
        EXPECT_TRUE(serves(Slot::PREP, spec, plainRow(64)));
    }
}

TEST(TestFlydslSdpaBwdSplitKernel, PrepReadsTheValueHeadDimOnly)
{
    RowSpec padded = plainRow(128);
    padded.padded = 1;
    const auto spec = withHeadDims(baseSpec(), 64, 100);
    EXPECT_TRUE(serves(Slot::PREP, spec, padded));
    EXPECT_FALSE(serves(Slot::PREP, spec, plainRow(64)));
    EXPECT_TRUE(serves(Slot::PREP, withHeadDims(baseSpec(), 100, 64), plainRow(64)));
}

TEST(TestFlydslSdpaBwdSplitKernel, PrepDeclinesTheSinkTwin)
{
    RowSpec twin = plainRow(64);
    twin.sink = 1;
    EXPECT_FALSE(serves(Slot::PREP, baseSpec(), twin));
    EXPECT_TRUE(serves(Slot::PREP, baseSpec(), plainRow(64)));
}

TEST(TestFlydslSdpaBwdSplitKernel, ServesDimsEqualToAnExactTileAndAbovePaddedFloors)
{
    for(const Slot slot : {Slot::DKDV, Slot::DQ})
    {
        RowSpec padded = plainRow(128);
        padded.padded = 1;
        EXPECT_TRUE(serves(slot, baseSpec(), plainRow(64)));
        EXPECT_FALSE(serves(slot, baseSpec(), plainRow(96)));
        // tail: the floor is the rung below the tile.
        EXPECT_TRUE(serves(slot, withHeadDims(baseSpec(), 100, 100), padded));
        EXPECT_FALSE(serves(slot, withHeadDims(baseSpec(), 96, 96), padded));
        EXPECT_FALSE(serves(slot, withHeadDims(baseSpec(), 100, 129), padded));
        EXPECT_TRUE(serves(slot, withHeadDims(baseSpec(), 100, 128), padded));
    }
}

TEST(TestFlydslSdpaBwdSplitKernel, ReadsTheFloorAsG1AsANumberOrAsAnInteger)
{
    RowSpec padded = plainRow(128);
    padded.padded = 1;
    const auto narrow = withHeadDims(baseSpec(), 40, 40);
    padded.floorValue = "g1";
    EXPECT_TRUE(serves(Slot::DKDV, narrow, padded));
    padded.floorValue = "tail";
    EXPECT_FALSE(serves(Slot::DKDV, narrow, padded));
    padded.floorValue = "32";
    EXPECT_TRUE(serves(Slot::DKDV, narrow, padded));
    padded.floorValue = "64";
    EXPECT_FALSE(serves(Slot::DKDV, narrow, padded));

    // An integer field serves as the floor too.
    auto row = makeRow(Slot::DKDV, padded);
    row.metadata["floor"] = int64_t{32};
    const GraphFixture fixture(buildSdpaBwdGraph(narrow));
    const auto bound = matchesGraph(splitSymbols(Slot::DKDV), fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_TRUE(matchesKernel(splitSymbols(Slot::DKDV), fixture.context(), row, *bound));
}

TEST(TestFlydslSdpaBwdSplitKernel, TheTailFloorOfTheLargestTileFollowsTheDtypesLadder)
{
    RowSpec sixteen = plainRow(512);
    sixteen.padded = 1;
    RowSpec single = sixteen;
    single.dtype = "fp32";
    const auto spec = [](DataType dtype) {
        auto result = withHeadDims(baseSpec(), 300, 300);
        result.dtype = dtype;
        return result;
    };
    // 16-bit: the rung below 512 is 384, so a tail-floor t512 row serves 385..512 and 300 rides the
    // g1 t512 row (floor 0). fp32 has no 384 rung: its tail-floor t512 row serves 257..512, so 300
    // rides it.
    EXPECT_FALSE(serves(Slot::DKDV, spec(DataType::HALF), sixteen));
    EXPECT_TRUE(serves(Slot::DKDV, spec(DataType::FLOAT), single));
    EXPECT_FALSE(serves(Slot::DKDV, spec(DataType::FLOAT), sixteen));
}

TEST(TestFlydslSdpaBwdSplitKernel, ThePrepG1RowOfTheLargestTileServesWhatItsTailFloorLeaves)
{
    for(const auto& dtypePair :
        {std::pair{DataType::HALF, "f16"}, std::pair{DataType::BFLOAT16, "bf16"}})
    {
        const DataType dtype = dtypePair.first;
        const std::string name = dtypePair.second;
        SCOPED_TRACE(name);
        RowSpec g1 = plainRow(512);
        g1.dtype = name;
        g1.padded = 1;
        g1.floorValue = "g1";
        RowSpec tail = g1;
        tail.floorValue = "tail";
        RowSpec smaller = plainRow(224);
        smaller.dtype = name;
        smaller.padded = 1;
        const auto graph = [&](int64_t headDimV) {
            auto result = withHeadDims(baseSpec(), headDimV, headDimV);
            result.dtype = dtype;
            return result;
        };
        // 16-bit: a tail-floor t512 row serves 385..512, so 300 rides only the g1 row.
        EXPECT_TRUE(serves(Slot::PREP, graph(300), g1));
        EXPECT_FALSE(serves(Slot::PREP, graph(300), tail));
        EXPECT_FALSE(serves(Slot::PREP, graph(300), smaller));
        // 200 is served by both its own tile and the g1 row; the catalog ranks the tile first.
        EXPECT_TRUE(serves(Slot::PREP, graph(200), smaller));
        EXPECT_TRUE(serves(Slot::PREP, graph(200), g1));
        EXPECT_FALSE(serves(Slot::PREP, graph(200), tail));
    }
}

TEST(TestFlydslSdpaBwdSplitKernel, ClassFollowsTheGraphForTheClassedSlots)
{
    for(const Slot slot : {Slot::DKDV, Slot::DQ})
    {
        EXPECT_TRUE(serves(slot, withCausal(baseSpec()), causalRow()));
        EXPECT_TRUE(serves(slot, withWindow(baseSpec()), causalRow()));
        EXPECT_FALSE(serves(slot, withCausal(baseSpec()), plainRow()));
        EXPECT_FALSE(serves(slot, baseSpec(), causalRow()));
        EXPECT_TRUE(serves(slot, withBias(baseSpec(), false), biasRow()));
        EXPECT_FALSE(serves(slot, withBias(baseSpec(), false), plainRow()));
        EXPECT_FALSE(serves(slot, baseSpec(), biasRow()));
    }
}

TEST(TestFlydslSdpaBwdSplitKernel, ABakedDropoutObjectServesItsOwnStateAndARuntimeOneBoth)
{
    for(const Slot slot : {Slot::DKDV, Slot::DQ})
    {
        RowSpec baked = plainRow();
        baked.dropout = 1;
        RowSpec runtime = baked;
        runtime.dform = "rt";
        EXPECT_TRUE(serves(slot, withDropout(baseSpec()), baked));
        EXPECT_FALSE(serves(slot, baseSpec(), baked));
        EXPECT_FALSE(serves(slot, withDropout(baseSpec()), plainRow()));
        EXPECT_TRUE(serves(slot, withDropout(baseSpec()), runtime));
        EXPECT_TRUE(serves(slot, baseSpec(), runtime));
    }
}

TEST(TestFlydslSdpaBwdSplitKernel, DtypeMustMatchTheRow)
{
    RowSpec single = plainRow();
    single.dtype = "fp32";
    RowSpec bf16 = plainRow();
    bf16.dtype = "bf16";
    for(const Slot slot : SLOTS)
    {
        EXPECT_TRUE(serves(slot, baseSpec(), plainRow()));
        EXPECT_FALSE(serves(slot, baseSpec(), single));
        EXPECT_FALSE(serves(slot, baseSpec(), bf16));
        EXPECT_TRUE(serves(slot, fp32Spec(), single));
    }
}

TEST(TestFlydslSdpaBwdSplitKernel, APackedDkdvObjectServesDenseGqaOnly)
{
    RowSpec packed = plainRow();
    packed.gqaPack = 1;
    packed.blockQ = 32;
    auto grouped = baseSpec();
    grouped.kvHeads = 2;
    EXPECT_TRUE(serves(Slot::DKDV, grouped, packed));
    EXPECT_FALSE(serves(Slot::DKDV, baseSpec(), packed));
    EXPECT_TRUE(serves(Slot::DKDV, grouped, plainRow()));
    // Packing is dense only: padded varlen is served unpacked.
    EXPECT_FALSE(serves(Slot::DKDV, withPaddedVarlen(grouped), packed));
    EXPECT_TRUE(serves(Slot::DKDV, withPaddedVarlen(grouped), plainRow()));
    // A query tile past the span bound is not served, and a packed row must record its tile
    // (C2): none, a non-positive one and one above the bound serve nothing, the bound itself
    // is served.
    for(const int64_t blockQ : {int64_t{-1}, int64_t{0}, int64_t{65}, int64_t{128}})
    {
        SCOPED_TRACE(blockQ);
        packed.blockQ = blockQ;
        EXPECT_FALSE(serves(Slot::DKDV, grouped, packed));
    }
    packed.blockQ = 64;
    EXPECT_TRUE(serves(Slot::DKDV, grouped, packed));
    // An unpacked row has no packed tile to bound.
    RowSpec unpacked = plainRow();
    unpacked.blockQ = -1;
    EXPECT_TRUE(serves(Slot::DKDV, grouped, unpacked));
}

/// A 16-bit bias whose row stride of 17 is not a multiple of 8 elements: outside the LDS
/// staging contract, though the shared matcher accepts the graph.
SdpaBwdSpec unstagedBiasSpec()
{
    return baseSpec().withEdit([](SdpaBwdGraph& g) {
        auto& bias = addBias(g);
        bias.strides = {int64_t{8} * 16 * 17, int64_t{16} * 17, 17, 1};
    });
}

RowSpec dqRowStage(const char* biasStageResolved)
{
    RowSpec row = biasRow();
    row.biasStageResolved = biasStageResolved;
    return row;
}

TEST(TestFlydslSdpaBwdSplitKernel, TheChunkContractPicksTheDqRowByHowItReadsTheBias)
{
    const auto unstaged = unstagedBiasSpec();
    ASSERT_TRUE(std::holds_alternative<flydsl_sdpa_bwd::SdpaBwdProblem>(
        matchSdpaBwdSpec(unstaged, flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS)));
    // The graph is served; each dq row decides by its own bias_stage_resolved: a bias outside
    // the staging contract goes to a row that loads it directly, never to a staged one.
    EXPECT_TRUE(graphMatch(unstaged).has_value());
    EXPECT_FALSE(serves(Slot::DQ, unstaged, dqRowStage("lds")));
    EXPECT_TRUE(serves(Slot::DQ, unstaged, dqRowStage("direct")));
    EXPECT_TRUE(serves(Slot::DQ, unstaged, dqRowStage("direct_only")));
    // A bias inside it goes to a staged row, to a direct row only where no staged row serves it
    // (direct_only), and never to a direct row that has a staged alternative (B-1).
    const auto staged = withBias(baseSpec(), false);
    EXPECT_TRUE(serves(Slot::DQ, staged, dqRowStage("lds")));
    EXPECT_FALSE(serves(Slot::DQ, staged, dqRowStage("direct")));
    EXPECT_TRUE(serves(Slot::DQ, staged, dqRowStage("direct_only")));
    // Only the dq objects read the bias by stage.
    EXPECT_TRUE(serves(Slot::PREP, unstaged, plainRow()));
    EXPECT_TRUE(serves(Slot::DKDV, unstaged, biasRow()));
}

TEST(TestFlydslSdpaBwdSplitKernel, ARowBroadcastBiasIsStagedOnlyAtAKeyLengthOfWholeVectors)
{
    // A row-broadcast bias has no padding behind its single row, so a staged object's last
    // 16-byte chunk stays inside the allocation only when Sk is a multiple of 8 elements.
    const auto rowBroadcast = [](SdpaBwdSpec spec, int64_t seqKv) {
        spec.seqKv = seqKv;
        return spec.withEdit([=](SdpaBwdGraph& g) {
            auto& bias = addBias(g, {g.batch, g.heads, 1, seqKv});
            bias.strides = {g.heads * int64_t{24}, 24, 24, 1};
        });
    };
    EXPECT_FALSE(serves(Slot::DQ, rowBroadcast(baseSpec(), 20), dqRowStage("lds")));
    EXPECT_TRUE(serves(Slot::DQ, rowBroadcast(baseSpec(), 20), dqRowStage("direct")));
    EXPECT_TRUE(serves(Slot::DQ, rowBroadcast(baseSpec(), 24), dqRowStage("lds")));
    EXPECT_FALSE(serves(Slot::DQ, rowBroadcast(baseSpec(), 24), dqRowStage("direct")));
    // fp32 dQ objects load the bias directly and no staged row backs them (direct_only): they
    // serve any layout.
    RowSpec single = biasRow();
    single.dtype = "fp32";
    single.biasStageResolved = "direct_only";
    EXPECT_TRUE(serves(Slot::DQ, rowBroadcast(fp32Spec(), 18), single));
    EXPECT_TRUE(serves(Slot::DQ, rowBroadcast(fp32Spec(), 24), single));
}

TEST(TestFlydslSdpaBwdSplitKernel, ADbiasGraphNeedsADqObjectThatStoresDb)
{
    RowSpec withoutStore = biasRow();
    withoutStore.storeDb = 0;
    EXPECT_FALSE(serves(Slot::DQ, withBias(baseSpec(), true), withoutStore));
    EXPECT_TRUE(serves(Slot::DQ, withBias(baseSpec(), false), withoutStore));
    EXPECT_TRUE(serves(Slot::DQ, withBias(baseSpec(), true), biasRow()));
}

TEST(TestFlydslSdpaBwdSplitKernel, ABroadcastDbiasNeedsADqObjectThatNamesTheReduceKernel)
{
    RowSpec withoutReduce = biasRow();
    withoutReduce.reduceSymbol.clear();
    const auto broadcast = withBias(baseSpec(), true, {1, 1, 16, 16});
    EXPECT_FALSE(serves(Slot::DQ, broadcast, withoutReduce));
    EXPECT_TRUE(serves(Slot::DQ, broadcast, biasRow()));
    // A dBias that is not broadcast is written in place: no reduce kernel is called for.
    EXPECT_TRUE(serves(Slot::DQ, withBias(baseSpec(), true), withoutReduce));
}

TEST(TestFlydslSdpaBwdSplitKernel, DeclinesALaunchGridPastTheLimitsOfAnAxis)
{
    const auto manyBatches = dimsSpec(70000, 1, 16, 16, 64);
    for(const Slot slot : SLOTS)
    {
        EXPECT_TRUE(serves(slot, baseSpec(), plainRow()));
        EXPECT_FALSE(serves(slot, manyBatches, plainRow()));
    }
    RowSpec narrow = plainRow();
    narrow.blockKv = 1;
    narrow.blockM = 1;
    const auto longSequence = dimsSpec(1, 1, 70000, 70000, 64);
    EXPECT_FALSE(serves(Slot::DKDV, longSequence, narrow));
    EXPECT_FALSE(serves(Slot::DQ, longSequence, narrow));
    EXPECT_TRUE(serves(Slot::DKDV, longSequence, plainRow()));
}

TEST(TestFlydslSdpaBwdSplitLaunch, TheDkdvGridFollowsTheRowsGridOrder)
{
    auto spec = dimsSpec(3, 6, 100, 91, 64);
    spec.kvHeads = 2;
    RowSpec row = plainRow();
    row.blockKv = 32;
    // 91 keys at 32 per block: 3 key blocks, over 2 key heads and 3 sequences.
    row.grid = GRID_DKDV_HEADS_FIRST;
    EXPECT_EQ(dumpOf(Slot::DKDV, spec, row).grid, (std::array<unsigned int, 3>{2, 3, 3}));
    row.grid = GRID_DKDV_BLOCKS_FIRST;
    EXPECT_EQ(dumpOf(Slot::DKDV, spec, row).grid, (std::array<unsigned int, 3>{3, 2, 3}));
}

TEST(TestFlydslSdpaBwdSplitKernel, TheAxisLimitsApplyToWhateverLandsOnYAndZ)
{
    RowSpec headsFirst = plainRow();
    headsFirst.grid = GRID_DKDV_HEADS_FIRST;
    RowSpec blocksFirst = plainRow();
    blocksFirst.grid = GRID_DKDV_BLOCKS_FIRST;

    // 70000 key blocks: y in the heads-first order, x in the other.
    headsFirst.blockKv = 1;
    blocksFirst.blockKv = 1;
    const auto longKeys = dimsSpec(1, 1, 16, 70000, 64);
    EXPECT_FALSE(serves(Slot::DKDV, longKeys, headsFirst));
    EXPECT_TRUE(serves(Slot::DKDV, longKeys, blocksFirst));

    // 70000 key heads: x in the heads-first order, y in the other.
    headsFirst.blockKv = 64;
    blocksFirst.blockKv = 64;
    const auto manyHeads = dimsSpec(1, 70000, 16, 16, 64);
    EXPECT_TRUE(serves(Slot::DKDV, manyHeads, headsFirst));
    EXPECT_FALSE(serves(Slot::DKDV, manyHeads, blocksFirst));

    // The batch is z in both.
    const auto manyBatches = dimsSpec(70000, 1, 16, 16, 64);
    EXPECT_FALSE(serves(Slot::DKDV, manyBatches, headsFirst));
    EXPECT_FALSE(serves(Slot::DKDV, manyBatches, blocksFirst));
}

TEST(TestFlydslSdpaBwdSplitKernel, ARowWithAnUnknownOrMissingGridIsDeclined)
{
    const GraphFixture fixture(buildSdpaBwdGraph(baseSpec()));
    const auto bound = matchesGraph(splitSymbols(Slot::DQ), fixture.context());
    ASSERT_TRUE(bound.has_value());
    for(const Slot slot : SLOTS)
    {
        SCOPED_TRACE(static_cast<int>(slot));
        EXPECT_TRUE(serves(slot, baseSpec(), plainRow()));
        for(const char* grid : {"", "heads_by_blocks", "kvblocks_by_kheads_by_seqs_x"})
        {
            auto row = makeRow(slot, plainRow());
            row.metadata["grid"] = std::string(grid);
            EXPECT_FALSE(matchesKernel(splitSymbols(slot), fixture.context(), row, *bound)) << grid;
        }
        auto missing = makeRow(slot, plainRow());
        missing.metadata.erase("grid");
        EXPECT_FALSE(matchesKernel(splitSymbols(slot), fixture.context(), missing, *bound));
    }
    // A spelling of another slot is not this slot's.
    RowSpec foreign = plainRow();
    foreign.grid = GRID_DQ;
    EXPECT_FALSE(serves(Slot::PREP, baseSpec(), foreign));
    EXPECT_FALSE(serves(Slot::DKDV, baseSpec(), foreign));
    foreign.grid = GRID_PREP;
    EXPECT_FALSE(serves(Slot::DQ, baseSpec(), foreign));
}

TEST(TestFlydslSdpaBwdSplitKernel, DeclinesARecordWithoutTheFieldsItsSlotReads)
{
    const GraphFixture fixture(buildSdpaBwdGraph(baseSpec()));
    const auto bound = matchesGraph(splitSymbols(Slot::DQ), fixture.context());
    ASSERT_TRUE(bound.has_value());
    for(const Slot slot : SLOTS)
    {
        auto row = makeRow(slot, plainRow());
        row.metadata.erase("dtype");
        EXPECT_FALSE(matchesKernel(splitSymbols(slot), fixture.context(), row, *bound));
    }
    auto missingGeometry = makeRow(Slot::DQ, plainRow());
    missingGeometry.metadata.erase("block_m");
    EXPECT_FALSE(matchesKernel(splitSymbols(Slot::DQ), fixture.context(), missingGeometry, *bound));
}

// ---------------------------------------------------------------------------
// Launches
// ---------------------------------------------------------------------------

/// The names every slot's list and the merged list share whose values must agree on a graph
/// both serve; LSE and DB are the buffers the engines place differently.
void expectSharedValuesEqual(const std::map<std::string, uint64_t>& split,
                             const std::map<std::string, uint64_t>& merged,
                             const std::vector<std::string>& skip,
                             size_t minimum)
{
    size_t compared = 0;
    for(const auto& [name, value] : split)
    {
        if(merged.count(name) == 0 || std::find(skip.begin(), skip.end(), name) != skip.end())
        {
            continue;
        }
        EXPECT_EQ(value, merged.at(name)) << name;
        ++compared;
    }
    EXPECT_GE(compared, minimum);
}

/// The graphs both engines serve at tile 64, with the rows that serve them.
struct SharedCell
{
    const char* name;
    SdpaBwdSpec spec;
    RowSpec row;
};

std::vector<SharedCell> sharedCells()
{
    RowSpec dropout = plainRow();
    dropout.dropout = 1;
    auto grouped = baseSpec();
    grouped.kvHeads = 2;
    return {{"plain", baseSpec(), plainRow()},
            {"causal", withCausal(baseSpec()), causalRow()},
            {"window", withWindow(baseSpec()), causalRow()},
            {"bias", withBias(baseSpec(), false), biasRow()},
            {"dbias", withBias(baseSpec(), true), biasRow()},
            {"dbiasBroadcast", withBias(baseSpec(), true, {1, 1, 16, 16}), biasRow()},
            {"dropoutDevice", withDropout(baseSpec(), DataType::INT64), dropout},
            {"dropoutByValue32", withDropout(baseSpec(), DataType::INT32, true), dropout},
            {"dropoutByValue", withDropout(baseSpec(), DataType::INT64, true), dropout},
            {"scale", withScale(baseSpec()), plainRow()},
            {"grouped", grouped, plainRow()},
            {"oddShape", dimsSpec(3, 5, 77, 91, 64), plainRow()},
            {"unitAxes", dimsSpec(1, 1, 1, 1, 64), plainRow()},
            {"paddedVarlen", withPaddedVarlen(baseSpec()), plainRow()},
            {"paddedVarlenCausal", withPaddedVarlen(withCausal(baseSpec())), causalRow()},
            {"raggedAll",
             raggedOf(baseSpec(), {"q", "k", "v", "o", "do", "dq", "dk", "dv"}, 8),
             plainRow()},
            {"raggedQueries", raggedOf(baseSpec(), {"q", "o", "do", "dq"}), plainRow()},
            {"raggedKeys", raggedOf(baseSpec(), {"k", "v", "dk", "dv"}), plainRow()}};
}

TEST(TestFlydslSdpaBwdSplitLaunch, SharedBuffersStridesAndScalarsEqualTheMergedEngines)
{
    for(const auto& cell : sharedCells())
    {
        SCOPED_TRACE(cell.name);
        const auto merged = mergedDumpOf(cell.spec, cell.row);
        const auto mergedValues = decode(merged.kernarg, mergedParams());
        const bool broadcast = merged.reduces;
        // A unit-Sq graph gives the merged kernels a unit stats sequence stride, which they bound
        // their LSE reads by; the split engine passes the stats as recorded (0).
        const bool unitQuery = cell.spec.seqQ == 1;

        for(const Slot slot : {Slot::DKDV, Slot::DQ})
        {
            const auto dump = dumpOf(slot, cell.spec, cell.row);
            std::vector<std::string> skip{"LSE", "Delta"};
            if(unitQuery)
            {
                skip.emplace_back("stride_stats_token");
            }
            if(broadcast)
            {
                // The broadcast dBias lives at the scratch offset, not the front.
                skip.emplace_back("DB");
            }
            expectSharedValuesEqual(valuesOf(slot, dump), mergedValues, skip, 40);
        }
        const auto prep = valuesOf(Slot::PREP, dumpOf(Slot::PREP, cell.spec, cell.row));
        std::vector<std::string> prepSkip{"LSE", "LSE_ws", "Delta"};
        if(unitQuery)
        {
            prepSkip.emplace_back("stride_stats_token");
        }
        expectSharedValuesEqual(prep, mergedValues, prepSkip, 10);
        // prep reads the raw statistics, as the merged kernel does.
        EXPECT_EQ(prep.at("LSE"), mergedValues.at("LSE"));
    }
}

TEST(TestFlydslSdpaBwdSplitLaunch, PaddedVarlenPassesTheLengthTensorsAndBothSidesOfVarlenBits)
{
    const auto spec = withPaddedVarlen(baseSpec());
    const uint64_t lengthsQ = ptrOf(sdpa_bwd_uid::SEQ_LEN_Q);
    const uint64_t lengthsKv = ptrOf(sdpa_bwd_uid::SEQ_LEN_KV);
    const uint64_t unused = ptrOf(sdpa_bwd_uid::O);
    for(const Slot slot : SLOTS)
    {
        const auto values = valuesOf(slot, dumpOf(slot, spec, plainRow()));
        EXPECT_EQ(values.at("seqinfo_q0"), lengthsQ);
        // Dense tensors: no offset array, and length-only bits on both sides.
        EXPECT_EQ(values.at("seqinfo_q1"), 0U);
        EXPECT_EQ(values.at("varlen_bits"), 0x0404U);
        EXPECT_EQ(values.at("num_seqlens"), 0U);
        EXPECT_EQ(values.at("max_seqlen_q"), 16U);
        if(slot != Slot::PREP)
        {
            EXPECT_EQ(values.at("seqinfo_k0"), lengthsKv);
            EXPECT_EQ(values.at("seqinfo_k1"), 0U);
            EXPECT_EQ(values.at("max_seqlen_k"), 16U);
            EXPECT_EQ(values.at("mul_K"), 0U);
        }
    }
    // A dense graph reads no lengths.
    const auto dense = valuesOf(Slot::DKDV, dumpOf(Slot::DKDV, baseSpec(), plainRow()));
    EXPECT_EQ(dense.at("seqinfo_q0"), unused);
    EXPECT_EQ(dense.at("varlen_bits"), 0U);
}

TEST(TestFlydslSdpaBwdSplitLaunch,
     EveryRaggedTensorPassesItsOffsetArrayAndMultiplierToTheSlotsThatAddressIt)
{
    const auto spec = raggedOf(baseSpec(), {"q", "k", "v", "o", "do", "dq", "dk", "dv"}, 8);
    namespace uid = sdpa_bwd_uid;
    const auto prep = valuesOf(Slot::PREP, dumpOf(Slot::PREP, spec, plainRow()));
    EXPECT_EQ(prep.at("varlen_bits"), 0x1515U);
    EXPECT_EQ(prep.at("seqinfo_q1"), ptrOf(uid::RAGGED_OFFSET_Q));
    EXPECT_EQ(prep.at("off_O"), ptrOf(uid::RAGGED_OFFSET_O));
    EXPECT_EQ(prep.at("mul_O"), 8U);
    EXPECT_EQ(prep.at("off_DO"), ptrOf(uid::RAGGED_OFFSET_DO));
    EXPECT_EQ(prep.at("mul_DO"), 8U);
    // The caller's stats: their own offset and units, read through their own strides.
    EXPECT_EQ(prep.at("off_Stats"), ptrOf(uid::RAGGED_OFFSET_STATS));
    EXPECT_EQ(prep.at("mul_Stats"), 1U);
    EXPECT_EQ(prep.at("stride_stats_batch"), 0U);
    EXPECT_EQ(prep.at("stride_stats_head"), 32U);
    EXPECT_EQ(prep.at("stride_stats_token"), 1U);
    EXPECT_EQ(prep.at("stride_o_batch"), 0U);

    const auto dkdv = valuesOf(Slot::DKDV, dumpOf(Slot::DKDV, spec, plainRow()));
    EXPECT_EQ(dkdv.at("varlen_bits"), 0x1515U);
    EXPECT_EQ(dkdv.at("seqinfo_q1"), ptrOf(uid::RAGGED_OFFSET_Q));
    EXPECT_EQ(dkdv.at("seqinfo_k1"), ptrOf(uid::RAGGED_OFFSET_K));
    EXPECT_EQ(dkdv.at("mul_Q"), 8U);
    EXPECT_EQ(dkdv.at("mul_K"), 8U);
    EXPECT_EQ(dkdv.at("off_V"), ptrOf(uid::RAGGED_OFFSET_V));
    EXPECT_EQ(dkdv.at("off_DO"), ptrOf(uid::RAGGED_OFFSET_DO));
    EXPECT_EQ(dkdv.at("off_DK"), ptrOf(uid::RAGGED_OFFSET_DK));
    EXPECT_EQ(dkdv.at("off_DV"), ptrOf(uid::RAGGED_OFFSET_DV));
    // dkdv reads Delta and LSE' from the workspace: no stats arguments, no O.
    EXPECT_EQ(dkdv.count("off_Stats"), 0U);
    EXPECT_EQ(dkdv.count("off_O"), 0U);

    const auto dq = valuesOf(Slot::DQ, dumpOf(Slot::DQ, spec, plainRow()));
    EXPECT_EQ(dq.at("seqinfo_q1"), ptrOf(uid::RAGGED_OFFSET_Q));
    EXPECT_EQ(dq.at("off_V"), ptrOf(uid::RAGGED_OFFSET_V));
    EXPECT_EQ(dq.at("off_DO"), ptrOf(uid::RAGGED_OFFSET_DO));
    EXPECT_EQ(dq.at("off_DQ"), ptrOf(uid::RAGGED_OFFSET_DQ));
    EXPECT_EQ(dq.at("mul_DQ"), 8U);
    EXPECT_EQ(dq.count("off_Stats"), 0U);
}

TEST(TestFlydslSdpaBwdSplitLaunch, DenseTensorsBesideRaggedOnesPassNullOffsetsAndMultiplierZero)
{
    // Only the key side is ragged: the query side is length-only.
    const auto spec = raggedOf(baseSpec(), {"k", "v", "dk", "dv"});
    const auto dkdv = valuesOf(Slot::DKDV, dumpOf(Slot::DKDV, spec, plainRow()));
    EXPECT_EQ(dkdv.at("varlen_bits"), 0x1504U);
    EXPECT_EQ(dkdv.at("seqinfo_q1"), 0U);
    EXPECT_EQ(dkdv.at("mul_Q"), 0U);
    EXPECT_EQ(dkdv.at("off_DO"), 0U);
    EXPECT_EQ(dkdv.at("mul_DO"), 0U);
    EXPECT_EQ(dkdv.at("off_DK"), ptrOf(sdpa_bwd_uid::RAGGED_OFFSET_DK));
    // The dense tensors keep their batch strides; the ragged ones do not read theirs.
    EXPECT_EQ(dkdv.at("stride_q_batch"), 8192U);
    EXPECT_EQ(dkdv.at("stride_k_batch"), 0U);
    // Dense stats stay dense: no offset, the graph's own strides.
    const auto prep = valuesOf(Slot::PREP, dumpOf(Slot::PREP, spec, plainRow()));
    EXPECT_EQ(prep.at("off_Stats"), 0U);
    EXPECT_EQ(prep.at("mul_Stats"), 0U);
    EXPECT_EQ(prep.at("stride_stats_batch"), 128U);
}

TEST(TestFlydslSdpaBwdSplitLaunch, StatsStridesFollowTheGraphsLayoutWithNoLayoutBit)
{
    const auto head = valuesOf(Slot::PREP, dumpOf(Slot::PREP, baseSpec(), plainRow()));
    EXPECT_EQ(head.at("stride_stats_batch"), 128U);
    EXPECT_EQ(head.at("stride_stats_head"), 16U);
    EXPECT_EQ(head.at("stride_stats_token"), 1U);
    EXPECT_EQ(head.at("varlen_bits"), 0U);

    const auto token = baseSpec().withEdit(
        [](SdpaBwdGraph& g) { g.tensor("stats").strides = {g.seqQ * g.heads, 1, g.heads, 1}; });
    const auto values = valuesOf(Slot::PREP, dumpOf(Slot::PREP, token, plainRow()));
    EXPECT_EQ(values.at("stride_stats_batch"), 128U);
    EXPECT_EQ(values.at("stride_stats_head"), 1U);
    EXPECT_EQ(values.at("stride_stats_token"), 8U);
    EXPECT_EQ(values.at("varlen_bits"), 0U);
}

TEST(TestFlydslSdpaBwdSplitLaunch, PaddedDenseStatsPassTheirActualStrides)
{
    // 8 padding floats behind every head's rows and 3 behind every batch's.
    const auto padded = baseSpec().withEdit(
        [](SdpaBwdGraph& g) { g.tensor("stats").strides = {8 * 24 + 3, 24, 1, 1}; });
    const auto values = valuesOf(Slot::PREP, dumpOf(Slot::PREP, padded, plainRow()));
    EXPECT_EQ(values.at("off_Stats"), 0U);
    EXPECT_EQ(values.at("stride_stats_batch"), 195U);
    EXPECT_EQ(values.at("stride_stats_head"), 24U);
    EXPECT_EQ(values.at("stride_stats_token"), 1U);
    // The compact workspace is unchanged: Delta and LSE' do not follow the caller's padding.
    const GraphFixture fixture(buildSdpaBwdGraph(padded));
    const auto bound = matchesGraph(splitSymbols(Slot::DQ), fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tokenOf(*bound, "ws.bytes"), 1024 + 1024);
}

TEST(TestFlydslSdpaBwdSplitKernel,
     APackedDkdvObjectDeclinesWhenThePaddedStatsGroupSpansMoreThan32Bits)
{
    // Eight query heads over two key heads: a packed group is four heads. Padding the stats
    // heads 2^29 floats apart puts the group's last head at 3 * 2^31 bytes, past a 32-bit
    // descriptor, though each head's rows and the matcher's per-head bound fit. Only the packed
    // object declines; the unpacked row is served.
    RowSpec packed = plainRow();
    packed.gqaPack = 1;
    packed.blockQ = 32;
    auto grouped = baseSpec();
    grouped.kvHeads = 2;
    const auto farStats = grouped.withEdit([](SdpaBwdGraph& g) {
        const int64_t headStride = int64_t{1} << 29;
        g.tensor("stats").strides = {g.heads * headStride, headStride, 1, 1};
    });
    EXPECT_TRUE(serves(Slot::DKDV, grouped, packed));
    EXPECT_FALSE(serves(Slot::DKDV, farStats, packed));
    EXPECT_TRUE(serves(Slot::DKDV, farStats, plainRow()));
    EXPECT_TRUE(serves(Slot::PREP, farStats, plainRow()));
}

TEST(TestFlydslSdpaBwdSplitLaunch, PackedStatsInEitherOrderPassTheirOwnOffsetAndStrides)
{
    const auto packedStats = [](const std::vector<int64_t>& strides) {
        return raggedOf(baseSpec(), {"q", "o", "do", "dq"}).withEdit([strides](SdpaBwdGraph& g) {
            g.tensor("stats").strides = strides;
        });
    };
    // (H, T): the head stride is the token count; the batch stride is never read.
    const auto head
        = valuesOf(Slot::PREP, dumpOf(Slot::PREP, packedStats({7, 100, 1, 1}), plainRow()));
    EXPECT_EQ(head.at("off_Stats"), ptrOf(sdpa_bwd_uid::RAGGED_OFFSET_STATS));
    EXPECT_EQ(head.at("stride_stats_batch"), 0U);
    EXPECT_EQ(head.at("stride_stats_head"), 100U);
    EXPECT_EQ(head.at("stride_stats_token"), 1U);
    // (T, H): the token stride is H.
    const auto token
        = valuesOf(Slot::PREP, dumpOf(Slot::PREP, packedStats({7, 1, 8, 1}), plainRow()));
    EXPECT_EQ(token.at("off_Stats"), ptrOf(sdpa_bwd_uid::RAGGED_OFFSET_STATS));
    EXPECT_EQ(token.at("stride_stats_batch"), 0U);
    EXPECT_EQ(token.at("stride_stats_head"), 1U);
    EXPECT_EQ(token.at("stride_stats_token"), 8U);
}

TEST(TestFlydslSdpaBwdSplitWorkspace, RaggedAndDenseGraphsShareOneCompactStatisticsWorkspace)
{
    const auto bytesOf = [](const SdpaBwdSpec& spec) {
        const GraphFixture fixture(buildSdpaBwdGraph(spec));
        const auto bound = matchesGraph(splitSymbols(Slot::DQ), fixture.context());
        EXPECT_TRUE(bound.has_value());
        return std::array<int64_t, 3>{tokenOf(*bound, "ws.bytes"),
                                      tokenOf(*bound, "ws.delta.offset"),
                                      tokenOf(*bound, "ws.lse.offset")};
    };
    // Delta and LSE' are (batch, head, in-sequence row) at pitch Sq: B * H * Sq floats each,
    // whatever the layout of the caller's stats.
    const auto dense = bytesOf(baseSpec());
    EXPECT_EQ(dense[0], 1024 + 1024);
    EXPECT_EQ(bytesOf(raggedOf(baseSpec(), {"q", "o", "do", "dq"})), dense);
    EXPECT_EQ(bytesOf(baseSpec().withEdit([](SdpaBwdGraph& g) {
                  g.tensor("stats").strides = {g.seqQ * g.heads, 1, g.heads, 1};
              })),
              dense);
}

TEST(TestFlydslSdpaBwdSplitLaunch, OnlyTheDqLaunchZeroesTheTailsOfAPaddedGraphsDenseGradients)
{
    const auto padded = withPaddedVarlen(baseSpec());
    for(const Slot slot : {Slot::PREP, Slot::DKDV})
    {
        EXPECT_FALSE(dumpOf(slot, padded, plainRow()).tailzero);
    }
    const auto dq = dumpOf(Slot::DQ, padded, plainRow());
    ASSERT_TRUE(dq.tailzero);
    EXPECT_EQ(dq.tailzeroBlock, 256U);
    EXPECT_EQ(dq.tailzeroGrid, (std::array<unsigned int, 3>{1, 48, 1}));
    ASSERT_EQ(dq.tailzeroKernarg.size(), imageBytes(tailzeroParams()));
    const auto values = decode(dq.tailzeroKernarg, tailzeroParams());
    EXPECT_EQ(values.at("DQ"), ptrOf(sdpa_bwd_uid::DQ));
    EXPECT_EQ(values.at("DK"), ptrOf(sdpa_bwd_uid::DK));
    EXPECT_EQ(values.at("DV"), ptrOf(sdpa_bwd_uid::DV));
    EXPECT_EQ(values.at("seq_len_q"), ptrOf(sdpa_bwd_uid::SEQ_LEN_Q));
    EXPECT_EQ(values.at("seq_len_kv"), ptrOf(sdpa_bwd_uid::SEQ_LEN_KV));
    EXPECT_EQ(values.at("mask"), 7U);
    // The merged engine's launch of the same graph is the same one.
    const auto merged = mergedDumpOf(padded, plainRow());
    ASSERT_TRUE(merged.tailzero);
    EXPECT_EQ(dq.tailzeroKernarg, merged.tailzeroKernarg);
    EXPECT_EQ(dq.tailzeroGrid, merged.tailzeroGrid);
    // No padding, no launch.
    EXPECT_FALSE(dumpOf(Slot::DQ, baseSpec(), plainRow()).tailzero);
}

TEST(TestFlydslSdpaBwdSplitLaunch, TheTailZeroMaskNamesTheDenseGradientsOnly)
{
    const auto maskOf = [](const std::vector<std::string>& ragged) {
        const auto dump = dumpOf(Slot::DQ, raggedOf(baseSpec(), ragged), plainRow());
        return dump.tailzero ? decode(dump.tailzeroKernarg, tailzeroParams()).at("mask") : 0U;
    };
    EXPECT_EQ(maskOf({"dk", "dv"}), 1U);
    EXPECT_EQ(maskOf({"dq", "dv"}), 2U);
    EXPECT_EQ(maskOf({"dq", "dk"}), 4U);
    EXPECT_EQ(maskOf({"q", "dq"}), 6U);
    // Every gradient ragged: their storage past a length is not ours.
    EXPECT_EQ(maskOf({"dq", "dk", "dv"}), 0U);
}

TEST(TestFlydslSdpaBwdSplitLaunch, SeedWordsAndPointersEqualTheMergedEnginesAndOnlyPrepUploads)
{
    RowSpec dropout = plainRow();
    dropout.dropout = 1;
    for(const auto& spec : {withDropout(baseSpec(), DataType::INT64),
                            withDropout(baseSpec(), DataType::INT64, true),
                            withDropout(baseSpec(), DataType::INT32, true)})
    {
        const auto merged = mergedDumpOf(spec, dropout);
        const auto prep = dumpOf(Slot::PREP, spec, dropout);
        EXPECT_EQ(prep.seedUploaded, merged.seedUploaded);
        EXPECT_EQ(prep.seedWords, merged.seedWords);
        for(const Slot slot : {Slot::DKDV, Slot::DQ})
        {
            const auto dump = dumpOf(slot, spec, dropout);
            EXPECT_FALSE(dump.seedUploaded);
        }
    }
    EXPECT_FALSE(dumpOf(Slot::PREP, baseSpec(), plainRow()).seedUploaded);
}

TEST(TestFlydslSdpaBwdSplitLaunch, ARankOneSeedAndOffsetAreReadAsTheRankFourOnesAre)
{
    // The frontend's constant form has dims [1].
    const auto rankOne = [](const SdpaBwdSpec& spec) {
        return spec.withEdit([](SdpaBwdGraph& g) {
            for(const char* name : {"seed", "offset"})
            {
                g.tensor(name).dims = {1};
                g.tensor(name).strides = {1};
            }
        });
    };
    RowSpec dropout = plainRow();
    dropout.dropout = 1;
    for(const auto& spec : {withDropout(baseSpec(), DataType::INT64, true),
                            withDropout(baseSpec(), DataType::INT32, true),
                            withDropout(baseSpec(), DataType::INT64)})
    {
        const auto expected = dumpOf(Slot::PREP, spec, dropout);
        const auto actual = dumpOf(Slot::PREP, rankOne(spec), dropout);
        EXPECT_EQ(actual.seedUploaded, expected.seedUploaded);
        EXPECT_EQ(actual.seedWords, expected.seedWords);
        for(const Slot slot : {Slot::DKDV, Slot::DQ})
        {
            EXPECT_EQ(valuesOf(slot, dumpOf(slot, rankOne(spec), dropout)),
                      valuesOf(slot, dumpOf(slot, spec, dropout)));
        }
    }
    // The constant 7 is the value uploaded.
    const auto uploaded
        = dumpOf(Slot::PREP, rankOne(withDropout(baseSpec(), DataType::INT64, true)), dropout);
    EXPECT_EQ(uploaded.seedWords, (std::vector<uint32_t>{7, 0, 7, 0}));
}

TEST(TestFlydslSdpaBwdSplitLaunch, RowStatisticsAreAddressedThroughTheWorkspaceTokens)
{
    const auto spec = withDropout(baseSpec(), DataType::INT32, true);
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    const auto bound = matchesGraph(splitSymbols(Slot::PREP), fixture.context());
    ASSERT_TRUE(bound.has_value());
    const uint64_t delta = WORKSPACE + static_cast<uint64_t>(tokenOf(*bound, "ws.delta.offset"));
    const uint64_t lse = WORKSPACE + static_cast<uint64_t>(tokenOf(*bound, "ws.lse.offset"));
    const uint64_t seed = WORKSPACE + static_cast<uint64_t>(tokenOf(*bound, "ws.seed.offset"));
    RowSpec dropout = plainRow();
    dropout.dropout = 1;

    const auto prep = valuesOf(Slot::PREP, dumpOf(Slot::PREP, spec, dropout));
    EXPECT_EQ(prep.at("Delta"), delta);
    EXPECT_EQ(prep.at("LSE_ws"), lse);
    EXPECT_EQ(prep.at("LSE"), ptrOf(sdpa_bwd_uid::STATS));
    for(const Slot slot : {Slot::DKDV, Slot::DQ})
    {
        const auto values = valuesOf(slot, dumpOf(slot, spec, dropout));
        EXPECT_EQ(values.at("Delta"), delta);
        EXPECT_EQ(values.at("LSE"), lse);
        EXPECT_EQ(values.at("philox_seed_ptr"), seed);
        EXPECT_EQ(values.at("philox_offset1"), seed + 8);
    }
}

TEST(TestFlydslSdpaBwdSplitLaunch, PrepLaunchValuesAreTheKernelsOwn)
{
    // Sq = 100 at tile 64 in 16-bit: 8 elements per access, 8 lanes per row, 32 rows per block.
    const auto spec = dimsSpec(3, 5, 100, 91, 64);
    const auto dump = dumpOf(Slot::PREP, spec, plainRow());
    EXPECT_EQ(dump.block, 256U);
    EXPECT_EQ(dump.grid, (std::array<unsigned int, 3>{4, 5, 3}));
    const auto values = valuesOf(Slot::PREP, dump);
    EXPECT_EQ(values.at("max_seqlen_q"), 100U);
    EXPECT_EQ(values.at("num_head_q"), 5U);
    EXPECT_EQ(values.at("hdim_vo"), 64U);
    EXPECT_EQ(values.at("num_seqlens"), 0U);
    EXPECT_EQ(values.at("O"), ptrOf(sdpa_bwd_uid::O));
    EXPECT_EQ(values.at("DO"), ptrOf(sdpa_bwd_uid::DO));

    // fp32: 4 elements per access, so 16 lanes per row cap at gcd(16, 16) = 16, 16 rows per block.
    RowSpec single = plainRow();
    single.dtype = "fp32";
    auto fp32 = dimsSpec(3, 5, 100, 91, 64);
    fp32.dtype = DataType::FLOAT;
    EXPECT_EQ(dumpOf(Slot::PREP, fp32, single).grid, (std::array<unsigned int, 3>{7, 5, 3}));
}

TEST(TestFlydslSdpaBwdSplitLaunch, DkdvAndDqGridsFollowTheRowsGeometry)
{
    const auto spec = dimsSpec(3, 6, 100, 91, 64);
    RowSpec row = plainRow();
    row.blockKv = 32;
    row.blockM = 16;
    row.blockThreads = 512;
    const auto dkdv = dumpOf(Slot::DKDV, spec, row);
    EXPECT_EQ(dkdv.block, 512U);
    EXPECT_EQ(dkdv.grid, (std::array<unsigned int, 3>{6, 3, 3}));
    const auto dq = dumpOf(Slot::DQ, spec, row);
    EXPECT_EQ(dq.block, 512U);
    EXPECT_EQ(dq.grid, (std::array<unsigned int, 3>{6, 7, 3}));
}

TEST(TestFlydslSdpaBwdSplitLaunch, HeadDimBelowTheTileSetsThePaddedFlag)
{
    RowSpec padded = plainRow(128);
    padded.padded = 1;
    const auto spec = withHeadDims(baseSpec(), 100, 100);
    for(const Slot slot : {Slot::DKDV, Slot::DQ})
    {
        const auto values = valuesOf(slot, dumpOf(slot, spec, padded));
        EXPECT_EQ(values.at("rt_flags"), 1U);
        EXPECT_EQ(values.at("hdim_qk"), 100U);
    }
    EXPECT_EQ(valuesOf(Slot::DKDV, dumpOf(Slot::DKDV, baseSpec(), plainRow())).at("rt_flags"), 0U);
}

// The flag is set when either head dim is below the tile, not only when both are.
TEST(TestFlydslSdpaBwdSplitLaunch, OneHeadDimBelowTheTileSetsThePaddedFlag)
{
    RowSpec padded = plainRow(128);
    padded.padded = 1;
    for(const auto& [qk, v] : {std::pair<int64_t, int64_t>{128, 100}, {100, 128}})
    {
        SCOPED_TRACE(std::to_string(qk) + "x" + std::to_string(v));
        const auto spec = withHeadDims(baseSpec(), qk, v);
        for(const Slot slot : {Slot::DKDV, Slot::DQ})
        {
            EXPECT_EQ(valuesOf(slot, dumpOf(slot, spec, padded)).at("rt_flags"), 1U);
        }
    }
}

TEST(TestFlydslSdpaBwdSplitLaunch, TheDqReduceRunsOnlyForABroadcastDbiasAndTheScratchIsItsBuffer)
{
    const auto none = dumpOf(Slot::DQ, withBias(baseSpec(), false), biasRow());
    EXPECT_FALSE(none.reduces);
    EXPECT_EQ(none.scratchBytes, 0U);
    // A bias without dBias: DB is O over zero strides, so the kernel stores nothing.
    const auto noneValues = valuesOf(Slot::DQ, none);
    EXPECT_EQ(noneValues.at("DB"), ptrOf(sdpa_bwd_uid::O));
    EXPECT_EQ(noneValues.at("stride_db_batch"), 0U);

    const auto inPlace = dumpOf(Slot::DQ, withBias(baseSpec(), true), biasRow());
    EXPECT_FALSE(inPlace.reduces);
    EXPECT_EQ(inPlace.scratchBytes, 0U);
    EXPECT_EQ(valuesOf(Slot::DQ, inPlace).at("DB"), ptrOf(sdpa_bwd_uid::DBIAS));

    const auto spec = withBias(baseSpec(), true, {1, 1, 16, 16});
    const auto reduced = dumpOf(Slot::DQ, spec, biasRow());
    ASSERT_TRUE(reduced.reduces);
    EXPECT_EQ(reduced.reduceSymbol, REDUCE_SYMBOL);
    EXPECT_EQ(reduced.reduceBlock, 256U);
    EXPECT_EQ(reduced.scratchBytes, size_t{2} * 8 * 16 * 16 * 2);
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    const auto bound = matchesGraph(splitSymbols(Slot::DQ), fixture.context());
    ASSERT_TRUE(bound.has_value());
    const uint64_t scratch
        = WORKSPACE + static_cast<uint64_t>(tokenOf(*bound, "ws.scratch_offset"));
    EXPECT_EQ(valuesOf(Slot::DQ, reduced).at("DB"), scratch);
    const auto reduce = decode(reduced.reduceKernarg, reduceParams());
    EXPECT_EQ(reduce.at("DB"), scratch);
    EXPECT_EQ(reduce.at("DBias"), ptrOf(sdpa_bwd_uid::DBIAS));

    // The merged engine's reduce, on the same graph, differs only by where the buffer is.
    const auto merged = mergedDumpOf(spec, biasRow());
    ASSERT_TRUE(merged.reduces);
    const auto expected = decode(merged.reduceKernarg, reduceParams());
    for(const auto& param : reduceParams())
    {
        if(param.name != "DB")
        {
            EXPECT_EQ(reduce.at(param.name), expected.at(param.name)) << param.name;
        }
    }
    EXPECT_EQ(reduced.reduceGrid, merged.reduceGrid);

    // prep and dkdv never carry scratch or a reduce.
    for(const Slot slot : {Slot::PREP, Slot::DKDV})
    {
        const auto dump = dumpOf(slot, spec, biasRow());
        EXPECT_FALSE(dump.reduces);
        EXPECT_EQ(dump.scratchBytes, 0U);
    }
}

TEST(TestFlydslSdpaBwdSplitLaunchFp32, DbiasScratchAndReduceGridAreSizedByTheElement)
{
    RowSpec single = biasRow();
    single.dtype = "fp32";
    auto spec = withBias(fp32Spec(), true, {1, 1, 16, 20});
    spec.seqKv = 20;
    const auto dump = dumpOf(Slot::DQ, spec, single);
    ASSERT_TRUE(dump.reduces);
    // fp32 elements in the workspace, and a reduce thread owns 4 columns: 16 * 5 items.
    EXPECT_EQ(dump.scratchBytes, size_t{2} * 8 * 16 * 20 * 4);
    EXPECT_EQ(dump.reduceGrid, (std::array<unsigned int, 3>{1, 1, 1}));
    const auto values = valuesOf(Slot::DQ, dump);
    EXPECT_EQ(values.at("stride_db_seq_q"), 20U);
}

TEST(TestFlydslSdpaBwdSplitLaunch, ARowThatServesABroadcastDbiasWithoutAReduceKernelIsRefused)
{
    RowSpec withoutReduce = biasRow();
    withoutReduce.reduceSymbol.clear();
    expectBadParam(
        [&] { dumpOf(Slot::DQ, withBias(baseSpec(), true, {1, 1, 16, 16}), withoutReduce); });
}

TEST(TestFlydslSdpaBwdSplitLaunch, TokensThatDisagreeWithTheGraphsLayoutAreRefused)
{
    const auto edits = std::vector<std::function<void(BoundTokens&)>>{
        [](BoundTokens& bound) { bound["ws.bytes"] = int64_t{4096}; },
        [](BoundTokens& bound) { bound["ws.delta.offset"] = int64_t{512}; },
        [](BoundTokens& bound) { bound.erase("ws.lse.offset"); },
        [](BoundTokens& bound) { bound["ws.scratch_offset"] = int64_t{0}; }};
    for(const auto& edit : edits)
    {
        expectBadParam([&] { dumpOf(Slot::DKDV, baseSpec(), plainRow(), edit); });
    }
}

TEST(TestFlydslSdpaBwdSplitLaunch, ALaunchWithoutItsWorkspaceIsRefused)
{
    const GraphFixture fixture(buildSdpaBwdGraph(baseSpec()));
    const auto bound = matchesGraph(splitSymbols(Slot::DKDV), fixture.context());
    ASSERT_TRUE(bound.has_value());
    expectBadParam([&] {
        describeFlydslSdpaBwdSplitLaunchForTesting(Slot::DKDV,
                                                   fixture.context(),
                                                   *bound,
                                                   makeRow(Slot::DKDV, plainRow()),
                                                   buffersFor(),
                                                   nullptr);
    });
}

// ---------------------------------------------------------------------------
// Dispatch handlers
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdSplitDispatch, NoSlotHandlerSupportsSaving)
{
    for(const Slot slot : SLOTS)
    {
        EXPECT_FALSE(dispatchHandler(splitSymbols(slot)).supportsSaving());
    }
}

TEST(TestFlydslSdpaBwdSplitDispatch, EachSlotsWorkspaceBytesAreItsScratchOnly)
{
    const auto spec = withBias(baseSpec(), true, {1, 1, 16, 16});
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    const auto bound = matchesGraph(splitSymbols(Slot::DQ), fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_GT(tokenOf(*bound, "ws.bytes"), 0);
    const auto bytes = [&](Slot slot) {
        return dispatchHandler(splitSymbols(slot))
            .workspaceBytes(fixture.context(), *bound, makeRow(slot, biasRow()));
    };
    EXPECT_EQ(bytes(Slot::PREP), 0U);
    EXPECT_EQ(bytes(Slot::DKDV), 0U);
    EXPECT_EQ(bytes(Slot::DQ), size_t{2} * 8 * 16 * 16 * 2);

    const GraphFixture plain(buildSdpaBwdGraph(baseSpec()));
    const auto plainBound = matchesGraph(splitSymbols(Slot::DQ), plain.context());
    ASSERT_TRUE(plainBound.has_value());
    EXPECT_EQ(dispatchHandler(splitSymbols(Slot::DQ))
                  .workspaceBytes(plain.context(), *plainBound, makeRow(Slot::DQ, plainRow())),
              0U);
}

// ---------------------------------------------------------------------------
// The dq_reduce dispatch
// ---------------------------------------------------------------------------

/// The argument list a kpack entry records for @p params: kind and size per argument.
std::vector<KernelArgument> recordedSignature(const ParamList& params)
{
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    for(const auto& param : params)
    {
        const auto size = static_cast<uint32_t>(paramBytes(param.kind));
        offset = (offset + size - 1) / size * size;
        signature.push_back(
            {param.kind == 'p' ? "global_buffer" : "by_value", size, offset, param.name});
        offset += size;
    }
    return signature;
}

const char* const DQ_ENTRY_SYMBOL = "dq_entry_symbol";
const char* const REDUCE_ENTRY_SYMBOL = "reduce_entry_symbol";
const char* const TAILZERO_ENTRY_SYMBOL = "tailzero_entry_symbol";

/// A dq row of @p row whose launches are the `sources` entries "dq" and, with @p reduce, the
/// "dbias_reduce" one and, with @p tailzero, the "tailzero" one, in archives that do not
/// exist: prepare() gets as far as loading the first launch's code object and no further.
KernelDefinition sourcesRow(const RowSpec& row,
                            std::optional<std::vector<KernelArgument>> reduce
                            = recordedSignature(reduceParams()),
                            std::optional<std::vector<KernelArgument>> tailzero
                            = recordedSignature(tailzeroParams()))
{
    auto kernel = makeRow(Slot::DQ, row);
    const auto tree = std::filesystem::temp_directory_path().string();
    kernel.originDirectory = tree;
    kernel.treeRoot = tree;
    KernelSource dq;
    dq.kind = KernelSourceKind::KPACK;
    dq.library = "there-is-no-archive-here.kpack";
    dq.tocKey = "toc#0";
    dq.symbol = DQ_ENTRY_SYMBOL;
    dq.signature = recordedSignature(dqParams());
    kernel.sources = {{"dq", dq}};
    if(reduce.has_value())
    {
        KernelSource reduceSource = dq;
        reduceSource.symbol = REDUCE_ENTRY_SYMBOL;
        reduceSource.signature = std::move(*reduce);
        kernel.sources["dbias_reduce"] = std::move(reduceSource);
    }
    if(tailzero.has_value())
    {
        KernelSource tailzeroSource = dq;
        tailzeroSource.symbol = TAILZERO_ENTRY_SYMBOL;
        tailzeroSource.signature = std::move(*tailzero);
        kernel.sources["tailzero"] = std::move(tailzeroSource);
    }
    return kernel;
}

/// What @p pack's prepare() throws for @p kernel on the graph of @p spec; empty when it
/// throws nothing, which no case expects (no archive exists).
std::string
    prepareFailure(const PackSymbols& pack, const SdpaBwdSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    const auto bound = matchesGraph(pack, fixture.context());
    EXPECT_TRUE(bound.has_value());
    if(!bound.has_value())
    {
        return {};
    }
    try
    {
        dispatchHandler(pack).prepare(fixture.context(), *bound, kernel);
    }
    catch(const std::exception& error)
    {
        return error.what();
    }
    return {};
}

bool contains(const std::string& text, const std::string& part)
{
    return text.find(part) != std::string::npos;
}

RowSpec bareDqRow()
{
    RowSpec row = biasRow();
    row.reduceSymbol = "fmha_reduce_by_symbol";
    return row;
}

TEST(TestFlydslSdpaBwdSplitDispatch, TheDqHandlersDeclareTheirLaunchesAndTheOthersNone)
{
    EXPECT_EQ(dispatchHandler(DQ_REDUCE).launchNames(),
              (std::set<std::string>{"dq", "dbias_reduce", "tailzero"}));
    EXPECT_EQ(dispatchHandler(splitSymbols(Slot::DQ)).launchNames(),
              (std::set<std::string>{"dq", "tailzero"}));
    for(const Slot slot : {Slot::PREP, Slot::DKDV})
    {
        EXPECT_TRUE(dispatchHandler(splitSymbols(slot)).launchNames().empty());
    }
    EXPECT_FALSE(dispatchHandler(DQ_REDUCE).supportsSaving());
    EXPECT_EQ(dispatchHandler(DQ_U2).launchNames(), (std::set<std::string>{"dq", "tailzero"}));
    EXPECT_FALSE(dispatchHandler(DQ_U2).supportsSaving());
}

TEST(TestFlydslSdpaBwdSplitDispatch, TheDqReduceHandlerHasTheDqHandlersScratch)
{
    const auto spec = withBias(baseSpec(), true, {1, 1, 16, 16});
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    const auto bound = matchesGraph(DQ_REDUCE, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto row = makeRow(Slot::DQ, bareDqRow());
    EXPECT_EQ(
        dispatchHandler(DQ_REDUCE).workspaceBytes(fixture.context(), *bound, row),
        dispatchHandler(splitSymbols(Slot::DQ)).workspaceBytes(fixture.context(), *bound, row));
    EXPECT_EQ(dispatchHandler(DQ_REDUCE).workspaceBytes(fixture.context(), *bound, row),
              size_t{2} * 8 * 16 * 16 * 2);
}

// The checks below read only refusals prepare() raises on the host before any code object
// loads (a missing launch, a reduce argument list that disagrees with the pack's), so they
// do not depend on what the loader reports for an archive that does not exist, which
// differs with and without a device.

TEST(TestFlydslSdpaBwdSplitDispatch, TheReduceLaunchIsBuiltOnlyForABroadcastDbias)
{
    const auto broadcast = withBias(baseSpec(), true, {1, 1, 16, 16});

    // A broadcast dBias needs the reduce launch: a row without it is refused...
    const auto missing
        = prepareFailure(DQ_REDUCE, broadcast, sourcesRow(bareDqRow(), std::nullopt));
    EXPECT_TRUE(contains(missing, "no launch 'dbias_reduce'")) << missing;

    // ...and its entry is the one consumed, whose argument list is checked.
    auto dropped = recordedSignature(reduceParams());
    dropped.pop_back();
    const auto refused = prepareFailure(DQ_REDUCE, broadcast, sourcesRow(bareDqRow(), dropped));
    EXPECT_TRUE(contains(refused, "is packaged with arguments")) << refused;
    EXPECT_TRUE(contains(refused, REDUCE_ENTRY_SYMBOL)) << refused;

    // Without a broadcast dBias that launch is neither required nor looked at: the same
    // two rows are not refused for it.
    for(const auto& spec :
        {baseSpec().withEdit([](SdpaBwdGraph& g) { addBias(g); }), withBias(baseSpec(), true)})
    {
        for(const auto& kernel :
            {sourcesRow(bareDqRow(), std::nullopt), sourcesRow(bareDqRow(), dropped)})
        {
            const auto failure = prepareFailure(DQ_REDUCE, spec, kernel);
            EXPECT_FALSE(contains(failure, "no launch")) << failure;
            EXPECT_FALSE(contains(failure, "is packaged with arguments")) << failure;
            EXPECT_FALSE(failure.empty());
        }
    }
}

TEST(TestFlydslSdpaBwdSplitDispatch, TheDqHandlerRefusesARowThatNamesAReduceKernel)
{
    const auto broadcast = withBias(baseSpec(), true, {1, 1, 16, 16});
    const auto failure = prepareFailure(splitSymbols(Slot::DQ), broadcast, sourcesRow(bareDqRow()));
    EXPECT_TRUE(contains(failure, "dq_reduce handler")) << failure;
    // The dq_reduce dispatch consumes the `dbias_reduce` entry, not the row's `reduce_symbol`.
    auto dropped = recordedSignature(reduceParams());
    dropped.pop_back();
    const auto refused = prepareFailure(DQ_REDUCE, broadcast, sourcesRow(bareDqRow(), dropped));
    EXPECT_TRUE(contains(refused, "is packaged with arguments")) << refused;
    EXPECT_TRUE(contains(refused, REDUCE_ENTRY_SYMBOL)) << refused;
    EXPECT_FALSE(contains(refused, "fmha_reduce_by_symbol")) << refused;
}

TEST(TestFlydslSdpaBwdSplitDispatch, EveryDqRowIsMultiLaunchSoASingleLaunchRowIsRefused)
{
    auto single = makeRow(Slot::DQ, plainRow());
    const auto failure = prepareFailure(splitSymbols(Slot::DQ), baseSpec(), single);
    EXPECT_TRUE(contains(failure, "is a single-launch kernel, so it has no launch 'dq'"))
        << failure;
}

TEST(TestFlydslSdpaBwdSplitDispatch, TheTailZeroLaunchIsRequiredAndCheckedOnlyForAPaddedGraph)
{
    const auto padded = withPaddedVarlen(baseSpec());
    // A padded graph with a dense gradient needs the launch: a row without it is refused...
    const auto missing = prepareFailure(
        splitSymbols(Slot::DQ), padded, sourcesRow(plainRow(), std::nullopt, std::nullopt));
    EXPECT_TRUE(contains(missing, "no launch 'tailzero'")) << missing;
    // ...and its entry's argument list is checked against the pack's.
    auto dropped = recordedSignature(tailzeroParams());
    dropped.pop_back();
    const auto refused = prepareFailure(
        splitSymbols(Slot::DQ), padded, sourcesRow(plainRow(), std::nullopt, dropped));
    EXPECT_TRUE(contains(refused, "is packaged with arguments")) << refused;
    EXPECT_TRUE(contains(refused, TAILZERO_ENTRY_SYMBOL)) << refused;
    // The dq_reduce dispatch has the same launch.
    const auto reduceMissing
        = prepareFailure(DQ_REDUCE, padded, sourcesRow(plainRow(), std::nullopt, std::nullopt));
    EXPECT_TRUE(contains(reduceMissing, "no launch 'tailzero'")) << reduceMissing;
    // A graph that zeroes nothing neither requires nor looks at it.
    for(const auto& kernel : {sourcesRow(plainRow(), std::nullopt, std::nullopt),
                              sourcesRow(plainRow(), std::nullopt, dropped)})
    {
        const auto failure = prepareFailure(splitSymbols(Slot::DQ), baseSpec(), kernel);
        EXPECT_FALSE(contains(failure, "no launch")) << failure;
        EXPECT_FALSE(contains(failure, "is packaged with arguments")) << failure;
        EXPECT_FALSE(failure.empty());
    }
    // Every gradient ragged: nothing to zero either.
    const auto raggedGradients = raggedOf(baseSpec(), {"dq", "dk", "dv"});
    const auto none = prepareFailure(splitSymbols(Slot::DQ),
                                     raggedGradients,
                                     sourcesRow(plainRow(), std::nullopt, std::nullopt));
    EXPECT_FALSE(contains(none, "no launch")) << none;
}

// ---------------------------------------------------------------------------
// An f32 bias beside a 16-bit query, and V heads of their own
// ---------------------------------------------------------------------------

/// A graph whose bias, and its dBias when @p withDbias, is FLOAT beside @p spec's 16-bit Q.
SdpaBwdSpec
    withF32Bias(const SdpaBwdSpec& spec, bool withDbias, const std::vector<int64_t>& dims = {})
{
    return spec.withEdit([=](SdpaBwdGraph& g) {
        addBias(g, dims).dtype = DataType::FLOAT;
        if(withDbias)
        {
            addDbias(g);
        }
    });
}

/// A graph over @p kvHeads key heads whose V and dV have @p valueHeads heads, laid out
/// head-major over those heads at V's row pitch (as the device harness lays them out).
SdpaBwdSpec withValueHeads(SdpaBwdSpec spec, int64_t kvHeads, int64_t valueHeads)
{
    spec.kvHeads = kvHeads;
    return spec.withEdit([=](SdpaBwdGraph& g) {
        for(const char* name : {"v", "dv"})
        {
            auto& tensor = g.tensor(name);
            tensor.dims[1] = valueHeads;
            tensor.strides = headMajorStrides(valueHeads, g.seqKv, tensor.strides[2]);
        }
    });
}

RowSpec dkRow(int64_t tile = 64)
{
    RowSpec row = plainRow(tile);
    row.u2Role = "dk";
    return row;
}

RowSpec dqU2Row(int64_t tile = 64)
{
    RowSpec row = plainRow(tile);
    row.u2Role = "dq";
    return row;
}

/// A dq or dkdv row of an f32 bias beside a 16-bit query: no staged row backs it.
RowSpec f32BiasRow(int64_t tile = 64)
{
    RowSpec row = biasRow(tile);
    row.biasF32 = 1;
    row.biasStageResolved = "direct_only";
    return row;
}

TEST(TestFlydslSdpaBwdSplitGraph, BindsTheF32BiasAndValueHeadTokens)
{
    const auto token = [](const BoundTokens& bound, const char* name) {
        return tokenOf(bound, std::string("flydsl_sdpa_bwd_split.") + name);
    };
    const auto plain = graphMatch(baseSpec());
    ASSERT_TRUE(plain.has_value());
    EXPECT_EQ(token(*plain, "bias_f32"), 0);
    EXPECT_EQ(token(*plain, "v_heads_differ"), 0);
    EXPECT_EQ(token(*plain, "v_heads"), 8);

    const auto f32 = graphMatch(withF32Bias(baseSpec(), true));
    ASSERT_TRUE(f32.has_value());
    EXPECT_EQ(token(*f32, "bias_f32"), 1);
    EXPECT_EQ(token(*f32, "dbias"), 1);
    // The staging contract is a 16-bit bias's; an f32 bias is never "outside" it.
    const auto f32Unaligned = graphMatch(baseSpec().withEdit([](SdpaBwdGraph& g) {
        auto& bias = addBias(g);
        bias.dtype = DataType::FLOAT;
        bias.strides = {int64_t{8} * 16 * 17, int64_t{16} * 17, 17, 1};
    }));
    ASSERT_TRUE(f32Unaligned.has_value());
    EXPECT_EQ(token(*f32Unaligned, "bias_f32"), 1);
    EXPECT_EQ(token(*f32Unaligned, "bias_unstaged"), 0);

    // A FLOAT bias beside a FLOAT query is the query's dtype, not an f32-bias graph.
    const auto fp32 = graphMatch(withBias(fp32Spec(), true));
    ASSERT_TRUE(fp32.has_value());
    EXPECT_EQ(token(*fp32, "bias_f32"), 0);

    const auto valueHeads = graphMatch(withValueHeads(baseSpec(), 4, 2));
    ASSERT_TRUE(valueHeads.has_value());
    EXPECT_EQ(token(*valueHeads, "v_heads_differ"), 1);
    EXPECT_EQ(token(*valueHeads, "kv_heads"), 4);
    EXPECT_EQ(token(*valueHeads, "v_heads"), 2);
    // V heads that equal K's are no variant, grouped or not.
    const auto grouped = graphMatch(withValueHeads(baseSpec(), 4, 4));
    ASSERT_TRUE(grouped.has_value());
    EXPECT_EQ(token(*grouped, "v_heads_differ"), 0);
}

TEST(TestFlydslSdpaBwdSplitGraph, DeclinesTheValueHeadGraphsNoShippedRowCouldServe)
{
    const auto causal = withCausal(withValueHeads(baseSpec(), 4, 2));
    struct Case
    {
        const char* name;
        SdpaBwdSpec spec;
        const char* cause;
    };
    const std::array<Case, 6> cases{{
        {"bias", withValueHeads(withBias(baseSpec(), false), 4, 2), "shape"},
        {"dbias", withValueHeads(withBias(baseSpec(), true), 4, 2), "shape"},
        {"fp32", withValueHeads(fp32Spec(), 4, 2), "shape"},
        {"ragged keys",
         withValueHeads(raggedOf(baseSpec(), {"k", "v", "dk", "dv"}), 4, 2),
         "shape"},
        {"query heads not a multiple of V's", withValueHeads(baseSpec(), 4, 3), "gqa"},
        {"query heads not a multiple of K's", withValueHeads(baseSpec(), 3, 2), "gqa"},
    }};
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        const auto matched = matchSdpaBwdSpec(c.spec, flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS);
        const auto* decline = std::get_if<flydsl_sdpa_bwd::Decline>(&matched);
        ASSERT_NE(decline, nullptr);
        EXPECT_EQ(decline->cause, c.cause) << decline->detail;
        EXPECT_FALSE(graphMatch(c.spec).has_value());
    }
    // Under a mask and under padding the variant is served.
    EXPECT_TRUE(graphMatch(causal).has_value());
    EXPECT_TRUE(graphMatch(withPaddedVarlen(withValueHeads(baseSpec(), 4, 2))).has_value());
    // V heads above K's are a group of K's heads over V's: served while Hq divides by both.
    EXPECT_TRUE(graphMatch(withValueHeads(baseSpec(), 2, 4)).has_value());
}

TEST(TestFlydslSdpaBwdSplitGraph, TheMergedEngineDeclinesAnF32BiasAndValueHeadsWithTheirCauses)
{
    const std::array<std::pair<SdpaBwdSpec, const char*>, 2> graphs{
        {{withF32Bias(baseSpec(), true), "attn_mask"},
         {withValueHeads(baseSpec(), 4, 2), "shape"}}};
    for(const auto& [spec, cause] : graphs)
    {
        SCOPED_TRACE(cause);
        const GraphFixture fixture(buildSdpaBwdGraph(spec));
        EXPECT_FALSE(matchesGraph(MERGED, fixture.context()).has_value());
        EXPECT_TRUE(matchesGraph(splitSymbols(Slot::PREP), fixture.context()).has_value());
        const auto merged = matchSdpaBwdSpec(spec);
        const auto* decline = std::get_if<flydsl_sdpa_bwd::Decline>(&merged);
        ASSERT_NE(decline, nullptr);
        EXPECT_EQ(decline->cause, cause) << decline->detail;
    }
}

TEST(TestFlydslSdpaBwdSplitGraph, BothEnginesServeAMaskedBiasedOrDropoutHeadDim192Graph)
{
    const auto atDtype = [](DataType dtype, int64_t headDim) {
        auto spec = withHeadDims(baseSpec(), headDim, headDim);
        spec.dtype = dtype;
        return spec;
    };
    const auto splitServes = [](const SdpaBwdSpec& spec) {
        const GraphFixture fixture(buildSdpaBwdGraph(spec));
        return matchesGraph(splitSymbols(Slot::PREP), fixture.context()).has_value();
    };
    const auto mergedServes = [](const SdpaBwdSpec& spec) {
        const GraphFixture fixture(buildSdpaBwdGraph(spec));
        return matchesGraph(MERGED, fixture.context()).has_value();
    };
    for(const DataType dtype : {DataType::BFLOAT16, DataType::HALF})
    {
        const auto d192 = atDtype(dtype, 192);
        // Any diagonal bound: top-left causal, a window, and bottom-right causal.
        const auto bottomRight
            = d192.withEdit([](SdpaBwdGraph& g) { g.attributes.causalMaskBottomRight = true; });
        std::vector<SdpaBwdSpec> specs{
            d192, withCausal(d192), withWindow(d192), bottomRight, withDropout(d192)};
        // Any bias, with and without dBias; dropout, masked or not.
        for(const bool dbias : {false, true})
        {
            specs.push_back(withBias(d192, dbias));
        }
        specs.push_back(withCausal(withDropout(d192)));
        for(const auto& spec : specs)
        {
            EXPECT_TRUE(mergedServes(spec));
            EXPECT_TRUE(splitServes(spec));
        }
    }
}

TEST(TestFlydslSdpaBwdSplitGraph, BothEnginesDeclineEveryLeftBoundOfZero)
{
    const auto splitServes = [](const SdpaBwdSpec& spec) {
        const GraphFixture fixture(buildSdpaBwdGraph(spec));
        return matchesGraph(splitSymbols(Slot::PREP), fixture.context()).has_value();
    };
    const auto mergedServes = [](const SdpaBwdSpec& spec) {
        const GraphFixture fixture(buildSdpaBwdGraph(spec));
        return matchesGraph(MERGED, fixture.context()).has_value();
    };
    const auto leftZero = [](std::optional<int64_t> right) {
        return baseSpec().withEdit([right](SdpaBwdGraph& g) {
            g.attributes.leftBound = 0;
            g.attributes.rightBound = right;
        });
    };
    for(const auto& spec : {leftZero(0), leftZero(std::nullopt), leftZero(1)})
    {
        EXPECT_FALSE(mergedServes(spec));
        EXPECT_FALSE(splitServes(spec));
    }
    // The smallest valid left bound is served by both.
    const auto leftOne = baseSpec().withEdit([](SdpaBwdGraph& g) { g.attributes.leftBound = 1; });
    EXPECT_TRUE(mergedServes(leftOne));
    EXPECT_TRUE(splitServes(leftOne));
}

TEST(TestFlydslSdpaBwdSplitGraph, TheSplitEngineOutranksTheMergedEngineInTheDefaultOrder)
{
    // The backend orders applicable engines by engine id (the FNV-1a hash of the engine name),
    // so a graph both engines serve goes to the split engine unless the merged engine is
    // selected explicitly. COVERAGE 9.2 states the order; a rename of either engine reorders
    // them, and this test fails first.
    const auto splitName = splitSymbols(Slot::PREP).engineName;
    EXPECT_EQ(splitName, "hipkernel:flydsl_sdpa_bwd_split");
    EXPECT_EQ(MERGED.engineName, "hipkernel:flydsl_sdpa_bwd_merged");
    const int64_t splitId = hipdnn_data_sdk::utilities::engineNameToId(splitName);
    const int64_t mergedId = hipdnn_data_sdk::utilities::engineNameToId(MERGED.engineName);
    EXPECT_EQ(splitId, int64_t{-1036064246992822851});
    EXPECT_EQ(mergedId, int64_t{1872578828357153731});
    EXPECT_LT(splitId, mergedId);
}

TEST(TestFlydslSdpaBwdSplitKernel, ARowReadsTheBiasAtTheWidthItWasBuiltFor)
{
    const auto f32 = withF32Bias(baseSpec(), true);
    const auto half = withBias(baseSpec(), true);
    for(const Slot slot : {Slot::DKDV, Slot::DQ})
    {
        SCOPED_TRACE(static_cast<int>(slot));
        EXPECT_TRUE(serves(slot, f32, f32BiasRow()));
        EXPECT_FALSE(serves(slot, f32, biasRow()));
        EXPECT_FALSE(serves(slot, half, f32BiasRow()));
        EXPECT_TRUE(serves(slot, half, biasRow()));
        // A graph without a bias is served by the rows that carry none.
        EXPECT_FALSE(serves(slot, baseSpec(), [] {
            auto row = plainRow();
            row.biasF32 = 1;
            return row;
        }()));
    }
    // An f32 bias is served by a direct row whatever its layout; a staged row never serves it.
    RowSpec staged = f32BiasRow();
    staged.biasStageResolved = "lds";
    EXPECT_FALSE(serves(Slot::DQ, f32, staged));
    // prep reads neither.
    EXPECT_TRUE(serves(Slot::PREP, f32, plainRow()));
    // A FLOAT bias beside a FLOAT query is the query's width: a row of bias_f32 0.
    RowSpec fp32Row = biasRow();
    fp32Row.dtype = "fp32";
    fp32Row.biasStageResolved = "direct_only";
    EXPECT_TRUE(serves(Slot::DQ, withBias(fp32Spec(), true), fp32Row));
    fp32Row.biasF32 = 1;
    EXPECT_FALSE(serves(Slot::DQ, withBias(fp32Spec(), true), fp32Row));
}

TEST(TestFlydslSdpaBwdSplitKernel, ARowServesValueHeadsOfItsOwnExactlyWhenItsRoleSaysSo)
{
    const auto valueHeads = withValueHeads(baseSpec(), 4, 2);
    const auto causal = withCausal(valueHeads);
    EXPECT_TRUE(serves(Slot::DKDV, valueHeads, dkRow()));
    EXPECT_FALSE(serves(Slot::DKDV, valueHeads, plainRow()));
    EXPECT_FALSE(serves(Slot::DKDV, baseSpec(), dkRow()));
    EXPECT_TRUE(serves(Slot::DKDV, baseSpec(), plainRow()));
    EXPECT_TRUE(serves(Slot::DQ, valueHeads, dqU2Row()));
    EXPECT_FALSE(serves(Slot::DQ, valueHeads, plainRow()));
    EXPECT_FALSE(serves(Slot::DQ, baseSpec(), dqU2Row()));
    EXPECT_TRUE(serves(Slot::DQ, baseSpec(), plainRow()));
    // The class still routes: a causal graph needs the causal row of its role.
    EXPECT_FALSE(serves(Slot::DKDV, causal, dkRow()));
    auto causalDk = causalRow();
    causalDk.u2Role = "dk";
    EXPECT_TRUE(serves(Slot::DKDV, causal, causalDk));
    // prep serves both.
    EXPECT_TRUE(serves(Slot::PREP, valueHeads, plainRow()));
}

TEST(TestFlydslSdpaBwdSplitKernel, ARecordWithAnUnreadableRoleWidthOrStageIsDeclined)
{
    const GraphFixture fixture(buildSdpaBwdGraph(withBias(baseSpec(), false)));
    const auto bound = matchesGraph(splitSymbols(Slot::DQ), fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto declines = [&](Slot slot, const std::function<void(KernelDefinition&)>& damage) {
        auto row = makeRow(slot, biasRow());
        EXPECT_TRUE(matchesKernel(splitSymbols(slot), fixture.context(), row, *bound));
        damage(row);
        return !matchesKernel(splitSymbols(slot), fixture.context(), row, *bound);
    };
    for(const Slot slot : {Slot::DKDV, Slot::DQ})
    {
        SCOPED_TRACE(static_cast<int>(slot));
        EXPECT_TRUE(declines(slot, [](auto& row) { row.metadata.erase("bias_f32"); }));
        EXPECT_TRUE(declines(slot, [](auto& row) { row.metadata["bias_f32"] = int64_t{2}; }));
        EXPECT_TRUE(declines(slot, [](auto& row) { row.metadata.erase("u2_role"); }));
        EXPECT_TRUE(
            declines(slot, [](auto& row) { row.metadata["u2_role"] = std::string("both"); }));
    }
    // A role of the other slot is no role of this one.
    EXPECT_TRUE(
        declines(Slot::DKDV, [](auto& row) { row.metadata["u2_role"] = std::string("dq"); }));
    EXPECT_TRUE(declines(Slot::DQ, [](auto& row) { row.metadata["u2_role"] = std::string("dk"); }));
    EXPECT_TRUE(declines(Slot::DQ, [](auto& row) { row.metadata.erase("bias_stage_resolved"); }));
    EXPECT_TRUE(declines(
        Slot::DQ, [](auto& row) { row.metadata["bias_stage_resolved"] = std::string("auto"); }));
}

TEST(TestFlydslSdpaBwdSplitKernel, ADkdvU2RowIsDeclinedWhenItsValueOnlyLaunchLeavesTheGrid)
{
    // The dK-only launch is 1 x Hk wide in y; the dV-only launch is Hv, past the limit of y.
    auto spec = withValueHeads(baseSpec(), 1, 70000);
    spec.heads = 70000;
    spec.batch = 1;
    RowSpec row = dkRow();
    row.grid = GRID_DKDV_BLOCKS_FIRST;
    EXPECT_FALSE(serves(Slot::DKDV, spec, row));
    // Over the K heads the same extent is x, which may be as wide as int32.
    row.grid = GRID_DKDV_HEADS_FIRST;
    EXPECT_TRUE(serves(Slot::DKDV, spec, row));
}

TEST(TestFlydslSdpaBwdSplitLaunch,
     ADkdvU2RowLaunchesTheDkOnlyObjectOverKeyHeadsThenTheDvOnlyOverValueHeads)
{
    const auto spec = withValueHeads(baseSpec(), 4, 2);
    for(const char* grid : {GRID_DKDV_HEADS_FIRST, GRID_DKDV_BLOCKS_FIRST})
    {
        SCOPED_TRACE(grid);
        const bool headsFirst = std::string(grid) == GRID_DKDV_HEADS_FIRST;
        RowSpec row = dkRow();
        row.grid = grid;
        const auto dump = dumpOf(Slot::DKDV, spec, row);
        using Grid3 = std::array<unsigned int, 3>;
        EXPECT_EQ(dump.grid, headsFirst ? (Grid3{4, 1, 2}) : (Grid3{1, 4, 2}));
        ASSERT_TRUE(dump.dvonly);
        EXPECT_EQ(dump.dvonlyGrid, headsFirst ? (Grid3{2, 1, 2}) : (Grid3{1, 2, 2}));
        EXPECT_EQ(dump.dvonlyBlock, dump.block);
        EXPECT_EQ(dump.kernarg.size(), imageBytes(dkdvU2Params()));
        const auto values = decode(dump.kernarg, dkdvU2Params());
        EXPECT_EQ(values.at("num_head_q"), 8U);
        EXPECT_EQ(values.at("num_head_k"), 4U);
        EXPECT_EQ(values.at("num_head_v"), 2U);
        // V's own strides: the dV-only launch reads V and writes dV at the graph's own.
        EXPECT_EQ(values.at("stride_v_head"), 16U * 64U);
        EXPECT_EQ(values.at("stride_dv_batch"), values.at("stride_v_batch"));
    }
    // The argument adds no bytes to the image (_PARAMS["dkdv_u2"]), and a row of Hk == Hv has
    // neither the argument nor the second launch.
    EXPECT_EQ(imageBytes(dkdvU2Params()), imageBytes(dkdvParams()));
    const auto plain = dumpOf(Slot::DKDV, baseSpec(), plainRow());
    EXPECT_FALSE(plain.dvonly);
    EXPECT_EQ(plain.kernarg.size(), imageBytes(dkdvParams()));
}

TEST(TestFlydslSdpaBwdSplitLaunch, ADqU2RowPassesTheValueHeadCountAndTheVHeadTailZero)
{
    const auto spec = withValueHeads(baseSpec(), 4, 2);
    const auto dump = dumpOf(Slot::DQ, spec, dqU2Row());
    EXPECT_FALSE(dump.dvonly);
    EXPECT_EQ(dump.kernarg.size(), imageBytes(dqU2Params()));
    const auto values = decode(dump.kernarg, dqU2Params());
    EXPECT_EQ(values.at("num_head_q"), 8U);
    EXPECT_EQ(values.at("num_head_k"), 4U);
    EXPECT_EQ(values.at("num_head_v"), 2U);
    EXPECT_EQ(dump.grid[0], 8U);
    EXPECT_FALSE(dump.tailzero);

    // Padded: the tail-zero launch is the V-head object over B * (Hq + Hk + Hv) planes.
    const auto padded = dumpOf(Slot::DQ, withPaddedVarlen(spec), dqU2Row());
    ASSERT_TRUE(padded.tailzero);
    EXPECT_EQ(padded.tailzeroKernarg.size(), imageBytes(tailzeroHvParams()));
    const auto tail = decode(padded.tailzeroKernarg, tailzeroHvParams());
    EXPECT_EQ(tail.at("Hq"), 8U);
    EXPECT_EQ(tail.at("Hkv"), 4U);
    EXPECT_EQ(tail.at("Hv"), 2U);
    EXPECT_EQ(tail.at("mask"), 7U);
    EXPECT_EQ(padded.tailzeroGrid[1], 2U * (8U + 4U + 2U));
    // Without V heads of its own the launch is the one of old: Hq + 2 Hkv planes.
    const auto plain = dumpOf(Slot::DQ, withPaddedVarlen(baseSpec()), plainRow());
    ASSERT_TRUE(plain.tailzero);
    EXPECT_EQ(plain.tailzeroKernarg.size(), imageBytes(tailzeroParams()));
    EXPECT_EQ(plain.tailzeroGrid[1], 2U * (8U + 8U + 8U));
}

TEST(TestFlydslSdpaBwdSplitLaunch, AnF32DbiasIsSizedAndReducedAtFourBytesBesideA16BitQuery)
{
    auto spec = withF32Bias(baseSpec(), true, {1, 1, 16, 20});
    spec.seqKv = 20;
    const auto dump = dumpOf(Slot::DQ, spec, f32BiasRow());
    ASSERT_TRUE(dump.reduces);
    EXPECT_EQ(dump.scratchBytes, size_t{2} * 8 * 16 * 20 * 4);
    // A reduce thread owns 4 columns of f32: 16 rows * 5 vectors = 80 items.
    EXPECT_EQ(dump.reduceGrid, (std::array<unsigned int, 3>{1, 1, 1}));
    EXPECT_EQ(decode(dump.kernarg, dqParams()).at("stride_db_seq_q"), 20U);
    // The workspace token the handler checks follows the same width.
    const auto bound = graphMatch(spec);
    ASSERT_TRUE(bound.has_value());
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    EXPECT_EQ(dispatchHandler(splitSymbols(Slot::DQ))
                  .workspaceBytes(fixture.context(), *bound, makeRow(Slot::DQ, f32BiasRow())),
              size_t{2} * 8 * 16 * 20 * 4);
}

TEST(TestFlydslSdpaBwdSplitMatcher, AnF32DbiasStrideIsHeldToAWholeVectorOfFourElements)
{
    const auto dbiasWithRowPitch = [](int64_t pitch) {
        return baseSpec().withEdit([=](SdpaBwdGraph& g) {
            auto& bias = addBias(g);
            bias.dtype = DataType::FLOAT;
            bias.strides = {int64_t{8} * 16 * pitch, int64_t{16} * pitch, pitch, 1};
            addDbias(g);
        });
    };
    // 20 elements = 80 bytes: a whole number of 16-byte vectors of FLOAT, though not of 16-bit
    // elements; 18 is not.
    EXPECT_TRUE(graphMatch(dbiasWithRowPitch(20)).has_value());
    EXPECT_FALSE(graphMatch(dbiasWithRowPitch(18)).has_value());
}

// The checks below read only refusals prepare() raises on the host before any code object
// loads.

/// A dkdv row of @p row whose launches are the `sources` entries "dkdv" and, with @p dvonly,
/// "dvonly", in archives that do not exist.
KernelDefinition dkdvSourcesRow(const RowSpec& row,
                                std::optional<std::vector<KernelArgument>> dvonly
                                = recordedSignature(dkdvU2Params()))
{
    auto kernel = makeRow(Slot::DKDV, row);
    const auto tree = std::filesystem::temp_directory_path().string();
    kernel.originDirectory = tree;
    kernel.treeRoot = tree;
    KernelSource dk;
    dk.kind = KernelSourceKind::KPACK;
    dk.library = "there-is-no-archive-here.kpack";
    dk.tocKey = "toc#0";
    dk.symbol = "dk_entry_symbol";
    dk.signature = recordedSignature(row.u2Role == "none" ? dkdvParams() : dkdvU2Params());
    kernel.sources = {{"dkdv", dk}};
    if(dvonly.has_value())
    {
        KernelSource dv = dk;
        dv.symbol = "dv_entry_symbol";
        dv.signature = std::move(*dvonly);
        kernel.sources["dvonly"] = std::move(dv);
    }
    return kernel;
}

TEST(TestFlydslSdpaBwdSplitDispatch, TheDkdvU2HandlerLaunchesTheDkOnlyAndTheDvOnlyObject)
{
    EXPECT_EQ(dispatchHandler(DKDV_U2).launchNames(), (std::set<std::string>{"dkdv", "dvonly"}));
    EXPECT_FALSE(dispatchHandler(DKDV_U2).supportsSaving());

    const auto spec = withValueHeads(baseSpec(), 4, 2);
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    const auto bound = matchesGraph(DKDV_U2, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(dispatchHandler(DKDV_U2).workspaceBytes(
                  fixture.context(), *bound, makeRow(Slot::DKDV, dkRow())),
              0U);

    // A row without its dV-only launch is refused...
    const auto missing = prepareFailure(DKDV_U2, spec, dkdvSourcesRow(dkRow(), std::nullopt));
    EXPECT_TRUE(contains(missing, "no launch 'dvonly'")) << missing;
    // ...and so is one whose dV-only entry takes the arguments of the old list.
    const auto refused
        = prepareFailure(DKDV_U2, spec, dkdvSourcesRow(dkRow(), recordedSignature(dkdvParams())));
    EXPECT_TRUE(contains(refused, "is packaged with arguments")) << refused;
    EXPECT_TRUE(contains(refused, "dv_entry_symbol")) << refused;
    // A well-formed pair gets as far as the archive.
    const auto good = prepareFailure(DKDV_U2, spec, dkdvSourcesRow(dkRow()));
    EXPECT_FALSE(contains(good, "no launch")) << good;
    EXPECT_FALSE(contains(good, "is packaged with arguments")) << good;
    EXPECT_FALSE(good.empty());

    // The role and the dispatch agree: a V-head row on the plain handler would launch one
    // object, a plain row on the two-launch handler has no second one.
    const auto onPlain
        = prepareFailure(splitSymbols(Slot::DKDV), spec, makeRow(Slot::DKDV, dkRow()));
    EXPECT_TRUE(contains(onPlain, "u2_role dk but is dispatched by the dkdv handler")) << onPlain;
    const auto onU2 = prepareFailure(DKDV_U2, baseSpec(), makeRow(Slot::DKDV, plainRow()));
    EXPECT_TRUE(contains(onU2, "u2_role none but is dispatched by the dkdv_u2 handler")) << onU2;
}

TEST(TestFlydslSdpaBwdSplitDispatch, TheDqU2HandlerLaunchesTheValueHeadDqAndTailZeroObjects)
{
    const auto spec = withValueHeads(baseSpec(), 4, 2);
    const auto padded = withPaddedVarlen(spec);
    const auto u2Row
        = [](std::optional<std::vector<KernelArgument>> tailzero, bool takesValueHeads = true) {
              auto kernel = sourcesRow(dqU2Row(), std::nullopt, std::move(tailzero));
              kernel.sources["dq"].signature
                  = recordedSignature(takesValueHeads ? dqU2Params() : dqParams());
              return kernel;
          };

    // The scratch of a V-head dq row is none (no bias).
    const GraphFixture fixture(buildSdpaBwdGraph(padded));
    const auto bound = matchesGraph(DQ_U2, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(dispatchHandler(DQ_U2).workspaceBytes(
                  fixture.context(), *bound, makeRow(Slot::DQ, dqU2Row())),
              0U);

    // The tail-zero list without Hv is refused; the one with it is not (the archive is what
    // is missing).
    const auto refused = prepareFailure(DQ_U2, padded, u2Row(recordedSignature(tailzeroParams())));
    EXPECT_TRUE(contains(refused, "is packaged with arguments")) << refused;
    EXPECT_TRUE(contains(refused, TAILZERO_ENTRY_SYMBOL)) << refused;
    // So is a dQ entry that takes the arguments of the old list.
    const auto oldDq
        = prepareFailure(DQ_U2, padded, u2Row(recordedSignature(tailzeroHvParams()), false));
    EXPECT_TRUE(contains(oldDq, "is packaged with arguments")) << oldDq;
    EXPECT_TRUE(contains(oldDq, DQ_ENTRY_SYMBOL)) << oldDq;
    const auto good = prepareFailure(DQ_U2, padded, u2Row(recordedSignature(tailzeroHvParams())));
    EXPECT_FALSE(contains(good, "is packaged with arguments")) << good;
    EXPECT_FALSE(contains(good, "dispatched by")) << good;
    EXPECT_FALSE(good.empty());
    // A row without its tail-zero launch is refused on a padded graph.
    const auto missing = prepareFailure(DQ_U2, padded, u2Row(std::nullopt));
    EXPECT_TRUE(contains(missing, "no launch 'tailzero'")) << missing;

    // The role and the dispatch agree: a V-head row on a standard dq handler, or a plain row
    // on dq_u2, is refused before any archive is read.
    for(const auto& standard : {splitSymbols(Slot::DQ), DQ_REDUCE})
    {
        const auto onStandard
            = prepareFailure(standard, spec, u2Row(recordedSignature(tailzeroHvParams())));
        EXPECT_TRUE(contains(onStandard, "u2_role dq but is dispatched by the dq handler"))
            << onStandard;
    }
    const auto onU2 = prepareFailure(DQ_U2, baseSpec(), sourcesRow(plainRow()));
    EXPECT_TRUE(contains(onU2, "u2_role none but is dispatched by the dq_u2 handler")) << onU2;
    // A plain row on the standard handler refuses the list with Hv.
    const auto plainRefused = prepareFailure(
        splitSymbols(Slot::DQ),
        withPaddedVarlen(baseSpec()),
        sourcesRow(plainRow(), std::nullopt, recordedSignature(tailzeroHvParams())));
    EXPECT_TRUE(contains(plainRefused, TAILZERO_ENTRY_SYMBOL)) << plainRefused;
}

// ---------------------------------------------------------------------------
// Graphs the gfx11 objects serve
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdSplitLaunch, ABiasBroadcastOverBatchAndOrHeadsReducesTheKeptPlanes)
{
    struct Case
    {
        bool overBatch;
        bool overHead;
        unsigned int planes;
    };
    // B = 2, H = 8: the reduce launch has one grid row per (batch, head) plane the dBias keeps.
    for(const auto& [overBatch, overHead, planes] :
        {Case{true, false, 8}, Case{false, true, 2}, Case{true, true, 1}})
    {
        SCOPED_TRACE(std::to_string(overBatch) + std::to_string(overHead));
        const auto spec = withBroadcastBias(baseSpec(), overBatch, overHead, true);
        ASSERT_TRUE(serves(Slot::DQ, spec, biasRow()));
        const auto dump = dumpOf(Slot::DQ, spec, biasRow());
        ASSERT_TRUE(dump.reduces);
        EXPECT_EQ(dump.reduceGrid[1], planes);
        EXPECT_EQ(dump.scratchBytes, size_t{2} * 8 * 16 * 16 * 2);
    }
    // Without a dBias nothing is reduced, whatever the bias broadcasts over.
    const auto bias = withBroadcastBias(baseSpec(), true, true, false);
    ASSERT_TRUE(serves(Slot::DQ, bias, biasRow()));
    EXPECT_FALSE(dumpOf(Slot::DQ, bias, biasRow()).reduces);
}

TEST(TestFlydslSdpaBwdSplitLaunch, AnOddHeadDimLaunchesAtTheRowStrideOfWholeVectors)
{
    // D = 77 in 16-bit rows 80 elements apart (ceil8); fp32 rows are ceil4 = 80 too.
    RowSpec padded = plainRow(96);
    padded.padded = 1;
    for(const Slot slot : {Slot::DKDV, Slot::DQ})
    {
        const auto values
            = valuesOf(slot, dumpOf(slot, oddHeadDimSpec(DataType::HALF, 77, 77), padded));
        EXPECT_EQ(values.at("hdim_qk"), 77U);
        EXPECT_EQ(values.at("stride_q_seq"), 80U);
        EXPECT_EQ(values.at("stride_k_seq"), 80U);
        EXPECT_EQ(values.at("stride_v_seq"), 80U);
        EXPECT_EQ(values.at("stride_do_seq"), 80U);
        EXPECT_EQ(values.at("rt_flags"), 1U);
    }
    RowSpec fp32 = padded;
    fp32.dtype = "fp32";
    const auto values
        = valuesOf(Slot::DQ, dumpOf(Slot::DQ, oddHeadDimSpec(DataType::FLOAT, 77, 77), fp32));
    EXPECT_EQ(values.at("stride_q_seq"), 80U);
    EXPECT_EQ(values.at("stride_dq_seq"), 80U);
}

TEST(TestFlydslSdpaBwdSplitLaunch, DropoutWithUnequalAndUnalignedLengthsLaunchesTheRuntimeFlags)
{
    const auto spec = dropoutSpecOfLengths(37, 29);
    RowSpec dropout = plainRow();
    dropout.dropout = 1;
    for(const Slot slot : {Slot::DKDV, Slot::DQ})
    {
        ASSERT_TRUE(serves(slot, spec, dropout));
        const auto values = valuesOf(slot, dumpOf(slot, spec, dropout));
        EXPECT_EQ(values.at("max_seqlen_q"), 37U);
        EXPECT_EQ(values.at("max_seqlen_k"), 29U);
        EXPECT_NE(values.at("idropout_p"), 0U);
        EXPECT_EQ(values.at("rt_flags") & 2U, 2U);
    }
}

// ---------------------------------------------------------------------------
// Rows shaped like gfx11-generic's
// ---------------------------------------------------------------------------

/// The gfx11-generic head-dim rungs. Every row is a padded object.
const std::array<int64_t, 11> GFX11_RUNGS{16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256};
const std::array<const char*, 3> GFX11_DTYPES{"f16", "bf16", "fp32"};

/// The rows of a gfx11-generic catalog for @p slot, as the generator writes them: prep rows
/// carry the rung below the tile as a numeric floor ("0" for the lowest), dkdv and dq rows the
/// padded "g1" floor; dropout is baked ("bk") in both states; dq rows never stage the bias
/// (`bias_stage_resolved` `direct_only`) and store dB only for bias; dkdv packs nothing.
std::vector<RowSpec> gfx11Rows(Slot slot)
{
    std::vector<RowSpec> rows;
    for(const char* dtype : GFX11_DTYPES)
    {
        for(size_t rung = 0; rung < GFX11_RUNGS.size(); ++rung)
        {
            RowSpec row;
            row.dtype = dtype;
            row.tile = GFX11_RUNGS[rung];
            row.padded = 1;
            row.dform = "bk";
            if(slot == Slot::PREP)
            {
                row.floorValue = rung == 0 ? "0" : std::to_string(GFX11_RUNGS[rung - 1]);
                row.sink = 0;
                rows.push_back(row);
                continue;
            }
            row.floorValue = "g1";
            row.grid = slot == Slot::DKDV ? GRID_DKDV_BLOCKS_FIRST : GRID_DQ;
            row.gqaPack = slot == Slot::DKDV ? 0 : -1;
            for(const int64_t dropout : {int64_t{0}, int64_t{1}})
            {
                for(const char cls : {'p', 'c', 'b'})
                {
                    row.dropout = dropout;
                    row.causal = cls == 'c' ? 1 : 0;
                    row.hasBias = cls == 'b' ? 1 : 0;
                    row.biasStageResolved = row.hasBias != 0 ? "direct_only" : "na";
                    row.storeDb = row.hasBias != 0 ? 1 : -1;
                    row.reduceSymbol = row.hasBias != 0 ? REDUCE_SYMBOL : "";
                    rows.push_back(row);
                }
            }
        }
    }
    return rows;
}

/// The rows of @p catalog that serve the graph of @p spec in @p slot; the graph itself must be
/// served.
std::vector<RowSpec>
    servingRows(Slot slot, const SdpaBwdSpec& spec, const std::vector<RowSpec>& catalog)
{
    std::vector<RowSpec> served;
    for(const auto& row : catalog)
    {
        if(serves(slot, spec, row))
        {
            served.push_back(row);
        }
    }
    return served;
}

SdpaBwdSpec withDtype(SdpaBwdSpec spec, DataType dtype)
{
    spec.dtype = dtype;
    return spec;
}

TEST(TestFlydslSdpaBwdSplitGfx11Rows, TheNumericPrepFloorServesEveryHeadDimWithExactlyOneRow)
{
    const auto catalog = gfx11Rows(Slot::PREP);
    const std::array<std::pair<DataType, const char*>, 3> dtypes{
        {{DataType::HALF, "f16"}, {DataType::BFLOAT16, "bf16"}, {DataType::FLOAT, "fp32"}}};
    for(const auto& [dtype, name] : dtypes)
    {
        for(int64_t headDim = 1; headDim <= 256; ++headDim)
        {
            SCOPED_TRACE(std::string(name) + " D=" + std::to_string(headDim));
            // prep reads the value width only: the key width does not change the verdict.
            const auto spec = withDtype(withHeadDims(baseSpec(), 257 - headDim, headDim), dtype);
            const auto served = servingRows(Slot::PREP, spec, catalog);
            ASSERT_EQ(served.size(), 1U);
            EXPECT_EQ(served.front().dtype, name);
            const auto rung = std::find_if(
                GFX11_RUNGS.begin(), GFX11_RUNGS.end(), [&](auto tile) { return tile >= headDim; });
            EXPECT_EQ(served.front().tile, *rung);
        }
    }
}

TEST(TestFlydslSdpaBwdSplitGfx11Rows, ADropoutGraphNeverMatchesABakedOffRowAndViceVersa)
{
    const std::array<SdpaBwdSpec, 3> classes{
        baseSpec(), withCausal(baseSpec()), withBias(baseSpec(), true)};
    for(const Slot slot : {Slot::DKDV, Slot::DQ})
    {
        const auto catalog = gfx11Rows(slot);
        for(const auto& spec : classes)
        {
            SCOPED_TRACE(static_cast<int>(slot));
            const auto runtime = servingRows(slot, withDropout(spec), catalog);
            ASSERT_FALSE(runtime.empty());
            for(const auto& row : runtime)
            {
                EXPECT_EQ(row.dropout, 1);
                EXPECT_EQ(row.dform, "bk");
            }
            const auto off = servingRows(slot, spec, catalog);
            ASSERT_FALSE(off.empty());
            for(const auto& row : off)
            {
                EXPECT_EQ(row.dropout, 0);
            }
        }
    }
}

TEST(TestFlydslSdpaBwdSplitGfx11Rows,
     OnlyRowsThatStageTheBiasDeclineALayoutOutsideTheStagingContract)
{
    const auto unstaged = unstagedBiasSpec();
    // The same layout in FLOAT: 17 elements apart is outside a 16-byte vector for both.
    const auto unstagedFp32 = fp32Spec().withEdit([](SdpaBwdGraph& g) {
        auto& bias = addBias(g);
        bias.strides = {int64_t{8} * 16 * 17, int64_t{16} * 17, 17, 1};
    });
    ASSERT_TRUE(graphMatch(unstaged).has_value());
    ASSERT_TRUE(graphMatch(unstagedFp32).has_value());

    // gfx11 rows load the bias directly: every bias row of the dtype serves the layout.
    auto catalog = gfx11Rows(Slot::DQ);
    const auto f16Rows = servingRows(Slot::DQ, unstaged, catalog);
    ASSERT_FALSE(f16Rows.empty());
    for(const auto& row : f16Rows)
    {
        EXPECT_EQ(row.dtype, "f16");
        EXPECT_EQ(row.biasStageResolved, "direct_only");
        EXPECT_EQ(row.hasBias, 1);
    }
    const auto fp32Rows = servingRows(Slot::DQ, unstagedFp32, catalog);
    ASSERT_FALSE(fp32Rows.empty());
    for(const auto& row : fp32Rows)
    {
        EXPECT_EQ(row.dtype, "fp32");
        EXPECT_EQ(row.biasStageResolved, "direct_only");
    }

    // A staging 16-bit row beside them declines the 16-bit layout.
    RowSpec staging = biasRow(64);
    staging.padded = 1;
    staging.floorValue = "g1";
    staging.dform = "bk";
    staging.grid = GRID_DQ;
    catalog.push_back(staging);
    for(const auto& row : servingRows(Slot::DQ, unstaged, catalog))
    {
        EXPECT_EQ(row.biasStageResolved, "direct_only");
    }
    EXPECT_FALSE(serves(Slot::DQ, unstaged, staging));
    // Inside the contract the staging row serves too: the decline is the layout's.
    EXPECT_TRUE(serves(Slot::DQ, withBias(baseSpec(), false), staging));
}

TEST(TestFlydslSdpaBwdSplitGfx11Rows, AHeadDimAboveTheLadderIsDeclinedWithANamedCause)
{
    // Above the shared matcher's bound the graph is declined [head_size] (DT-4), whichever of
    // Dqk and Dv is larger.
    for(const auto& dims :
        {std::pair<int64_t, int64_t>{513, 513}, {600, 64}, {64, 600}, {512, 1024}})
    {
        SCOPED_TRACE(std::to_string(dims.first) + "/" + std::to_string(dims.second));
        const auto spec = withHeadDims(baseSpec(), dims.first, dims.second);
        const auto matched = matchSdpaBwdSpec(spec, flydsl_sdpa_bwd::SPLIT_MATCH_OPTIONS);
        const auto* decline = std::get_if<flydsl_sdpa_bwd::Decline>(&matched);
        ASSERT_NE(decline, nullptr);
        EXPECT_EQ(decline->cause, "head_size");
        EXPECT_FALSE(graphMatch(spec).has_value());
    }
}

TEST(TestFlydslSdpaBwdSplitGfx11Rows,
     AHeadDimBetweenTheLadderAnd512PassesTheGraphButHasNoDkdvOrDqRow)
{
    // 257..512 passes graph_match (the shared bound is 512); the gfx11 catalog ends at 256.
    const std::array<std::pair<int64_t, int64_t>, 6> dims{
        {{257, 257}, {384, 384}, {512, 512}, {96, 300}, {300, 96}, {256, 257}}};
    for(const auto& [qk, v] : dims)
    {
        SCOPED_TRACE(std::to_string(qk) + "/" + std::to_string(v));
        const auto spec = withHeadDims(baseSpec(), qk, v);
        ASSERT_TRUE(graphMatch(spec).has_value());
        // prep reads the value width only, so a wide key with a narrow value still has a prep
        // row; dkdv and dq have none, and the graph is served by no complete plan.
        for(const Slot slot : SLOTS)
        {
            const bool unserved = slot == Slot::PREP ? v > 256 : true;
            EXPECT_EQ(servingRows(slot, spec, gfx11Rows(slot)).empty(), unserved);
        }
    }
    // The ladder's own top rung still serves.
    const auto top = withHeadDims(baseSpec(), 256, 256);
    for(const Slot slot : SLOTS)
    {
        EXPECT_FALSE(servingRows(slot, top, gfx11Rows(slot)).empty());
    }
}

TEST(TestFlydslSdpaBwdSplitGfx11Rows, EveryTableClassHasUniqueEnrolledKnobTuplesInEverySlot)
{
    // The classes of the verdict table's served rows: dtype, mask form, dropout, bias form and
    // dBias, and the rungs max(Dqk, Dv) (dkdv, dq) and Dv (prep) route to.
    const auto rungOf = [](int64_t headDim) {
        return *std::find_if(
            GFX11_RUNGS.begin(), GFX11_RUNGS.end(), [&](auto tile) { return tile >= headDim; });
    };
    using ClassKey = std::
        tuple<std::string, std::string, bool, int64_t, std::string, int64_t, int64_t, int64_t>;
    std::map<ClassKey, VerdictRow> classes;
    for(const auto& row : loadVerdictRows())
    {
        if(!row.served)
        {
            continue;
        }
        const bool windowed = row.windowLeft != -1 || row.windowRight != -1;
        classes.emplace(ClassKey{row.dtype,
                                 row.mask,
                                 windowed,
                                 row.dropout,
                                 row.bias,
                                 row.dbias,
                                 rungOf(std::max(row.dqk, row.dv)),
                                 rungOf(row.dv)},
                        row);
    }
    ASSERT_FALSE(classes.empty());

    // A gfx11-shaped catalog per slot, as the shipped gfx11 rows write it.
    std::array<std::vector<KernelDefinition>, 3> catalog;
    for(size_t slot = 0; slot < SLOTS.size(); ++slot)
    {
        for(const auto& row : gfx11Rows(SLOTS[slot]))
        {
            catalog[slot].push_back(makeRow(SLOTS[slot], row));
        }
    }

    std::vector<std::string> failures;
    for(const auto& [key, row] : classes)
    {
        const GraphFixture fixture(buildSdpaBwdGraph(specOfRow(row)));
        const auto context = fixture.context();
        const auto bound = matchesGraph(splitSymbols(Slot::PREP), context);
        if(!bound.has_value())
        {
            failures.push_back(row.id + ": graph_match declined a served row's graph");
            continue;
        }
        for(size_t slot = 0; slot < SLOTS.size(); ++slot)
        {
            std::vector<const KernelDefinition*> matched;
            for(const auto& kernel : catalog[slot])
            {
                if(matchesKernel(splitSymbols(SLOTS[slot]), context, kernel, *bound))
                {
                    matched.push_back(&kernel);
                }
            }
            const std::string label = slotName(SLOTS[slot]);
            if(matched.empty())
            {
                failures.push_back(row.id + " " + label + ": no row of the catalog serves it");
            }
            else if(const auto pair = ambiguousEnrolledTuple(matched, enrolledKnobs(label)))
            {
                failures.push_back(row.id + " " + label + ": ambiguous tuple between "
                                   + std::to_string(matched.size()) + " rows");
            }
        }
    }
    std::string report;
    for(const auto& failure : failures)
    {
        report += "\n  " + failure;
    }
    EXPECT_TRUE(failures.empty()) << failures.size() << " of " << classes.size()
                                  << " classes:" << report;
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
