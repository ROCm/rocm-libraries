// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdTestGraphs.hpp"
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

PackSymbols splitSymbols(Slot slot)
{
    switch(slot)
    {
    case Slot::PREP:
        return {"hipkernel:flydsl_sdpa_bwd_split",
                "hipkernel.flydsl_sdpa_bwd_split.graph_match",
                "",
                "hipkernel.flydsl_sdpa_bwd_split.prep.kernel_match",
                "",
                "hipkernel.flydsl_sdpa_bwd_split.prep.dispatch",
                "",
                "",
                ""};
    case Slot::DKDV:
        return {"hipkernel:flydsl_sdpa_bwd_split",
                "hipkernel.flydsl_sdpa_bwd_split.graph_match",
                "",
                "hipkernel.flydsl_sdpa_bwd_split.dkdv.kernel_match",
                "",
                "hipkernel.flydsl_sdpa_bwd_split.dkdv.dispatch",
                "",
                "",
                ""};
    default:
        return {"hipkernel:flydsl_sdpa_bwd_split",
                "hipkernel.flydsl_sdpa_bwd_split.graph_match",
                "",
                "hipkernel.flydsl_sdpa_bwd_split.dq.kernel_match",
                "",
                "hipkernel.flydsl_sdpa_bwd_split.dq.dispatch",
                "",
                "",
                ""};
    }
}

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

SdpaBwdSpec baseSpec()
{
    return SdpaBwdSpec{};
}

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

SdpaBwdSpec withCausal(const SdpaBwdSpec& spec)
{
    return spec.withEdit([](SdpaBwdGraph& g) { g.attributes.causalMask = true; });
}

SdpaBwdSpec withWindow(const SdpaBwdSpec& spec)
{
    return spec.withEdit([](SdpaBwdGraph& g) {
        g.attributes.causalMask = true;
        g.attributes.leftBound = 4;
    });
}

SdpaBwdSpec withBias(const SdpaBwdSpec& spec, bool withDbias, const std::vector<int64_t>& dims = {})
{
    return spec.withEdit([=](SdpaBwdGraph& g) {
        addBias(g, dims);
        if(withDbias)
        {
            addDbias(g);
        }
    });
}

SdpaBwdSpec withDropout(const SdpaBwdSpec& spec,
                        DataType seedType = DataType::INT64,
                        bool passByValue = false)
{
    return spec.withEdit([=](SdpaBwdGraph& g) { addDropout(g, 0.2F, seedType, passByValue); });
}

SdpaBwdSpec withScale(const SdpaBwdSpec& spec)
{
    return spec.withEdit([](SdpaBwdGraph& g) { addScaleTensor(g, 0.5F); });
}

SdpaBwdSpec withPaddedVarlen(const SdpaBwdSpec& spec)
{
    return spec.withEdit([](SdpaBwdGraph& g) { addSeqLens(g); });
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
    std::string biasStage = "auto";
    std::string reduceSymbol;
    int64_t blockThreads = 256;
    int64_t blockKv = 64;
    int64_t blockM = 64;
};

KernelDefinition makeRow(Slot slot, const RowSpec& row)
{
    KernelDefinition kernel;
    kernel.name = "sdpa_bwd_split_row";
    kernel.metadata = {{"dtype", row.dtype},
                       {"head_dim", row.tile},
                       {"head_dim_v", row.tile},
                       {"padded", row.padded},
                       {"floor", row.floorValue},
                       {"hints", int64_t{1}}};
    if(slot == Slot::PREP)
    {
        kernel.metadata["sink"] = row.sink;
        return kernel;
    }
    kernel.metadata["causal"] = row.causal;
    kernel.metadata["has_bias"] = row.hasBias;
    kernel.metadata["dropout"] = row.dropout;
    kernel.metadata["dform"] = row.dform;
    kernel.metadata["block_threads"] = row.blockThreads;
    if(slot == Slot::DKDV)
    {
        kernel.metadata["gqa_pack"] = row.gqaPack;
        kernel.metadata["block_q"] = row.blockQ;
        kernel.metadata["block_kv"] = row.blockKv;
        return kernel;
    }
    kernel.metadata["store_db"] = row.storeDb;
    kernel.metadata["bias_stage"] = row.biasStage;
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

RowSpec biasRow(int64_t tile = 64)
{
    RowSpec row = plainRow(tile);
    row.hasBias = 1;
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
                       {"dropout", row.dropout},
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

/// A fake device address per tensor uid, and a workspace base; only ever compared.
uint64_t ptrOf(int64_t uid)
{
    return static_cast<uint64_t>(uid) * 0x100000;
}

constexpr uint64_t WORKSPACE = 0x70000000;

void* asPointer(uint64_t address)
{
    // A fake address, only ever compared and offset into, never dereferenced.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(static_cast<uintptr_t>(address));
}

std::vector<hipdnnPluginDeviceBuffer_t> buffersFor()
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers;
    for(int64_t uid = sdpa_bwd_uid::Q; uid <= sdpa_bwd_uid::DROPOUT_SCALE_INV; ++uid)
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

void expectBadParam(const std::function<void()>& action)
{
    try
    {
        action();
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM) << e.what();
        return;
    }
    FAIL() << "expected a BAD_PARAM exception";
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

/// The scalar block every dkdv and dq argument list shares, from `varlen_bits` to `sm_scale`.
void addScalarBlock(ParamList& list)
{
    addParams(list, 'p', {"seqinfo_q0", "seqinfo_q1", "seqinfo_k0", "seqinfo_k1"});
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
    addParams(list, 'i', {"num_head_q", "num_head_k", "hdim_qk", "hdim_vo"});
    addParams(list, 'f', {"sm_scale"});
}

/// `_PARAMS["prep"]`, in order.
const ParamList& prepParams()
{
    static const ParamList s_params = [] {
        ParamList list;
        addParams(list, 'p', {"O", "DO", "LSE", "LSE_ws", "Delta", "seqinfo_q0", "seqinfo_q1"});
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
        addScalarBlock(list);
        addStrides(list, {"q", "k", "v", "do", "dk", "dv", "b"});
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
        addScalarBlock(list);
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
        addScalarBlock(list);
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
        matchSdpaBwdSpec(baseSpec(), flydsl_sdpa_bwd::MatchOptions{true})));
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
        const auto matched = matchSdpaBwdSpec(specs[index], flydsl_sdpa_bwd::MatchOptions{true});
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
            withDropout(baseSpec(), DataType::INT32),
            withDropout(baseSpec(), DataType::INT64, true),
            withBias(baseSpec(), true),
            withBias(baseSpec(), true, {1, 1, 16, 16}),
            withBias(withDropout(baseSpec(), DataType::INT32), true, {1, 8, 16, 16}),
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
        const auto problem = matchSdpaBwdSpec(spec, flydsl_sdpa_bwd::MatchOptions{true});
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
        matchSdpaBwdSpec(spec, flydsl_sdpa_bwd::MatchOptions{true})));
    EXPECT_FALSE(graphMatch(spec).has_value());
    // One bit less ends at 2^62 + 2^61, inside int64.
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
    const auto matched = matchSdpaBwdSpec(overflowing, flydsl_sdpa_bwd::MatchOptions{true});
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

TEST(TestFlydslSdpaBwdSplitWorkspace, PublishesNoSeedOffsetForASeedReadInPlace)
{
    const auto inPlace = graphMatch(withDropout(baseSpec(), DataType::INT64));
    const auto uploaded = graphMatch(withDropout(baseSpec(), DataType::INT32));
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
    // 16-bit: the rung below 512 is 384, so 300 rides the t384 row. fp32 has no 384: 300 rides
    // the padded t512 object.
    EXPECT_FALSE(serves(Slot::DKDV, spec(DataType::HALF), sixteen));
    EXPECT_TRUE(serves(Slot::DKDV, spec(DataType::FLOAT), single));
    EXPECT_FALSE(serves(Slot::DKDV, spec(DataType::FLOAT), sixteen));
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
    // A query tile past the span bound is not served.
    packed.blockQ = 128;
    EXPECT_FALSE(serves(Slot::DKDV, grouped, packed));
}

TEST(TestFlydslSdpaBwdSplitGraph, BI8DeclinesA16BitBiasOutsideTheStagingContract)
{
    // A bias row stride of 17 is not a multiple of 8 elements.
    const auto unstaged = baseSpec().withEdit([](SdpaBwdGraph& g) {
        auto& bias = addBias(g);
        bias.strides = {int64_t{8} * 16 * 17, int64_t{16} * 17, 17, 1};
    });
    ASSERT_TRUE(std::holds_alternative<flydsl_sdpa_bwd::SdpaBwdProblem>(
        matchSdpaBwdSpec(unstaged, flydsl_sdpa_bwd::MatchOptions{true})));
    EXPECT_FALSE(graphMatch(unstaged).has_value());
    EXPECT_TRUE(graphMatch(withBias(baseSpec(), false)).has_value());
}

TEST(TestFlydslSdpaBwdSplitGraph, BI8StagesARowBroadcastBiasOnlyAtAKeyLengthOfWholeVectors)
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
    EXPECT_FALSE(graphMatch(rowBroadcast(baseSpec(), 20)).has_value());
    EXPECT_TRUE(graphMatch(rowBroadcast(baseSpec(), 24)).has_value());
    // fp32 dQ objects load the bias directly: no staging contract.
    RowSpec single = biasRow();
    single.dtype = "fp32";
    EXPECT_TRUE(serves(Slot::DQ, rowBroadcast(fp32Spec(), 18), single));
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
            {"dropoutDevice32", withDropout(baseSpec(), DataType::INT32), dropout},
            {"dropoutByValue", withDropout(baseSpec(), DataType::INT64, true), dropout},
            {"scale", withScale(baseSpec()), plainRow()},
            {"grouped", grouped, plainRow()},
            {"oddShape", dimsSpec(3, 5, 77, 91, 64), plainRow()},
            {"unitAxes", dimsSpec(1, 1, 1, 1, 64), plainRow()},
            {"paddedVarlen", withPaddedVarlen(baseSpec()), plainRow()},
            {"paddedVarlenCausal", withPaddedVarlen(withCausal(baseSpec())), causalRow()}};
}

TEST(TestFlydslSdpaBwdSplitLaunch, SharedBuffersStridesAndScalarsEqualTheMergedEngines)
{
    for(const auto& cell : sharedCells())
    {
        SCOPED_TRACE(cell.name);
        const auto merged = mergedDumpOf(cell.spec, cell.row);
        const auto mergedValues = decode(merged.kernarg, mergedParams());
        const bool broadcast = merged.reduces;

        for(const Slot slot : {Slot::DKDV, Slot::DQ})
        {
            const auto dump = dumpOf(slot, cell.spec, cell.row);
            std::vector<std::string> skip{"LSE", "Delta"};
            if(broadcast)
            {
                // The broadcast dBias lives at the scratch offset, not the front.
                skip.emplace_back("DB");
            }
            expectSharedValuesEqual(valuesOf(slot, dump), mergedValues, skip, 40);
        }
        const auto prep = valuesOf(Slot::PREP, dumpOf(Slot::PREP, cell.spec, cell.row));
        expectSharedValuesEqual(prep, mergedValues, {"LSE", "LSE_ws", "Delta"}, 10);
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
        EXPECT_EQ(values.at("seqinfo_q1"), unused);
        EXPECT_EQ(values.at("varlen_bits"), 0x0404U);
        EXPECT_EQ(values.at("num_seqlens"), 0U);
        EXPECT_EQ(values.at("max_seqlen_q"), 16U);
        if(slot != Slot::PREP)
        {
            EXPECT_EQ(values.at("seqinfo_k0"), lengthsKv);
            EXPECT_EQ(values.at("seqinfo_k1"), unused);
            EXPECT_EQ(values.at("max_seqlen_k"), 16U);
        }
    }
    // A dense graph reads no lengths.
    const auto dense = valuesOf(Slot::DKDV, dumpOf(Slot::DKDV, baseSpec(), plainRow()));
    EXPECT_EQ(dense.at("seqinfo_q0"), unused);
    EXPECT_EQ(dense.at("varlen_bits"), 0U);
}

TEST(TestFlydslSdpaBwdSplitLaunch, SeedWordsAndPointersEqualTheMergedEnginesAndOnlyPrepUploads)
{
    RowSpec dropout = plainRow();
    dropout.dropout = 1;
    for(const auto& spec : {withDropout(baseSpec(), DataType::INT64),
                            withDropout(baseSpec(), DataType::INT32),
                            withDropout(baseSpec(), DataType::INT64, true),
                            withDropout(baseSpec(), DataType::INT32, true)})
    {
        const auto merged = mergedDumpOf(spec, dropout);
        const auto prep = dumpOf(Slot::PREP, spec, dropout);
        EXPECT_EQ(prep.seedUploaded, merged.seedUploaded);
        EXPECT_EQ(prep.seedCopiedFromDevice, merged.seedCopiedFromDevice);
        EXPECT_EQ(prep.seedWords, merged.seedWords);
        EXPECT_EQ(prep.seedSource, merged.seedSource);
        EXPECT_EQ(prep.offsetSource, merged.offsetSource);
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
                            withDropout(baseSpec(), DataType::INT32),
                            withDropout(baseSpec(), DataType::INT64)})
    {
        const auto expected = dumpOf(Slot::PREP, spec, dropout);
        const auto actual = dumpOf(Slot::PREP, rankOne(spec), dropout);
        EXPECT_EQ(actual.seedUploaded, expected.seedUploaded);
        EXPECT_EQ(actual.seedCopiedFromDevice, expected.seedCopiedFromDevice);
        EXPECT_EQ(actual.seedWords, expected.seedWords);
        EXPECT_EQ(actual.seedSource, expected.seedSource);
        EXPECT_EQ(actual.offsetSource, expected.offsetSource);
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
    const auto spec = withDropout(baseSpec(), DataType::INT32);
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

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
