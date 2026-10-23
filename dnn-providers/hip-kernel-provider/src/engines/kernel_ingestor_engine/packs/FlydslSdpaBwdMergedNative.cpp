// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslSdpaCdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"

/**
 * @file FlydslSdpaBwdMergedNative.cpp
 * @brief The flyDSL SDPA backward engine of the merged kernel: graph match, kernel match,
 *        score, dispatch, and the one function that registers them.
 *
 * Every object is one launch of `fmha_bwd_merged_gfx950`, which computes dK, dV, dQ and, on
 * a bias graph, dBias in one grid, with no prep kernel and no Delta buffer. A bias object
 * also carries `fmha_bwd_dbias_reduce_gfx950`, the sum over a broadcast bias's batch and
 * head axes, as a second symbol of the same code object; it runs only when the graph's
 * dBias is broadcast.
 *
 * The graph contract is the shared backward matcher's (FlydslSdpaBwdCommon.hpp). The merged
 * engine adds what only it knows: padded varlen is declined, a row serves the head sizes its
 * tile covers, every bias object stages the bias through LDS, and `gqa_pack` rows serve only
 * grouped-query graphs.
 *
 * Launch arguments are a pure function of the plan prepare() resolves and the execute-time
 * buffers (launchArguments()), so one description of the 452-byte merged kernarg list feeds
 * the launch, the signature the descriptor is checked against, and the tests' golden dumps.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

namespace bwd = flydsl_sdpa_bwd;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view ENGINE_NAME = "hipkernel:flydsl_sdpa_bwd_merged";
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_bwd_merged.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_bwd_merged.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa_bwd_merged.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_bwd_merged.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
/// The head-dim tile the object is built at.
constexpr std::string_view HEAD_DIM_FIELD = "head_dim";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view HAS_BIAS_FIELD = "has_bias";
constexpr std::string_view DROPOUT_FIELD = "dropout";
/// 0: serves head dims equal to the tile; 1: serves the dims between the rung below and it.
constexpr std::string_view PADDED_FIELD = "padded";
/// -1 leaves the dB store to the policy (it exists); 0 builds a dQ role without it.
constexpr std::string_view STORE_DB_FIELD = "store_db";
/// 1 packs the GQA group into the dK/dV role's q-tile row axis; -1 leaves it unset.
constexpr std::string_view GQA_PACK_FIELD = "gqa_pack";
/// The launch geometry of the build: threads per block, and the key and query tile heights.
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";
constexpr std::string_view BLOCK_KV_FIELD = "block_kv";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
/// The dBias reduce kernel's symbol in the same object; empty on an object without one.
constexpr std::string_view REDUCE_SYMBOL_FIELD = "reduce_symbol";

// What graph_match binds for kernel_match.
constexpr std::string_view CLASS_TOKEN = "flydsl_sdpa_bwd_merged.class";
constexpr std::string_view DTYPE_TOKEN = "flydsl_sdpa_bwd_merged.dtype";
constexpr std::string_view HEAD_DIM_QK_TOKEN = "flydsl_sdpa_bwd_merged.head_dim_qk";
constexpr std::string_view HEAD_DIM_V_TOKEN = "flydsl_sdpa_bwd_merged.head_dim_v";
constexpr std::string_view DROPOUT_TOKEN = "flydsl_sdpa_bwd_merged.dropout";
constexpr std::string_view DBIAS_TOKEN = "flydsl_sdpa_bwd_merged.dbias";
constexpr std::string_view GQA_PACK_TOKEN = "flydsl_sdpa_bwd_merged.gqa_pack";
constexpr std::string_view BIAS_STAGING_TOKEN = "flydsl_sdpa_bwd_merged.bias_staging";
constexpr std::string_view BATCH_TOKEN = "flydsl_sdpa_bwd_merged.batch";
constexpr std::string_view HEADS_TOKEN = "flydsl_sdpa_bwd_merged.heads";
constexpr std::string_view KV_HEADS_TOKEN = "flydsl_sdpa_bwd_merged.kv_heads";
constexpr std::string_view SEQ_Q_TOKEN = "flydsl_sdpa_bwd_merged.seq_q";
constexpr std::string_view SEQ_KV_TOKEN = "flydsl_sdpa_bwd_merged.seq_kv";
/// The reduce launch's grid y extent; 0 when the graph's dBias is not broadcast.
constexpr std::string_view REDUCE_PLANES_TOKEN = "flydsl_sdpa_bwd_merged.reduce_planes";

/// The graph classes an object is built for: `causal` serves every masked graph, `bias`
/// every bias graph (masked graphs with a bias are declined by the matcher), `plain` the rest.
constexpr int64_t CLASS_PLAIN = 0;
constexpr int64_t CLASS_CAUSAL = 1;
constexpr int64_t CLASS_BIAS = 2;

constexpr int64_t INT32_LIMIT = 2147483647LL;

/// The largest y or z extent of a launch grid; the same limit as `GRID_YZ_LIMIT` in
/// FlydslSdpaAiwNative.cpp.
constexpr int64_t GRID_YZ_LIMIT = 65535;

/// The head-dim tiles `_sdpa_bwd_spec.LADDER` builds objects at.
constexpr std::array<int64_t, 10> LADDER{32, 64, 96, 128, 160, 192, 224, 256, 384, 512};

/// The mask sentinels the kernel resolves per sequence (`fmha_common_gfx1201.py`,
/// WINDOW_TOPLEFT and WINDOW_BOTRIGHT).
constexpr int32_t WINDOW_TOPLEFT = -2147483647;
constexpr int32_t WINDOW_BOTRIGHT = -2147483646;

/// `varlen_bits` names the layout of the LSE: bit 16 is the (B, Sq, H) packing.
constexpr int32_t VARLEN_LSE_LAYOUT_TH = 1 << 16;

/// `rt_flags`: bit 0 serves a head dim below the tile, bit 1 turns dropout on.
constexpr int32_t RT_FLAG_PADDED = 1;
constexpr int32_t RT_FLAG_DROPOUT = 2;

/// The workspace's seed region, `[seed u64 | offset u64]`, and the alignment of the region
/// after it.
constexpr size_t SEED_REGION_BYTES = 16;
constexpr size_t SEED_WORD_BYTES = 8;
constexpr size_t WORKSPACE_ALIGNMENT = 256;
constexpr int64_t WORKSPACE_ELEMENT_BYTES = 2; // bf16 and f16 only

/// The reduce kernel's launch: 256 threads, each owning 8 consecutive columns of one row
/// (`fmha_bwd_dbias_reduce_gfx950.py`, BLOCK_THREADS and VEC).
constexpr unsigned int REDUCE_THREADS = 256;
constexpr int64_t REDUCE_VECTOR = 8;

[[noreturn]] void failBadParam(const std::string& message)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "flydsl_sdpa_bwd_merged: " + message);
}

void requireHip(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       std::string("flydsl_sdpa_bwd_merged: ")
                                                           + what + " failed: "
                                                           + hipGetErrorString(status));
    }
}

int32_t toInt32(int64_t value, const char* what)
{
    if(value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
    {
        failBadParam(std::string(what) + " " + std::to_string(value) + " does not fit in int32");
    }
    return static_cast<int32_t>(value);
}

// ---------------------------------------------------------------------------
// Kernel records
// ---------------------------------------------------------------------------

/// What the engine reads of a merged object's KMD record.
struct Row
{
    std::string dtype;
    int64_t tile = 0;
    int64_t causal = 0;
    int64_t hasBias = 0;
    int64_t dropout = 0;
    int64_t padded = 0;
    int64_t storeDb = 0;
    int64_t gqaPack = 0;
    int64_t blockThreads = 0;
    int64_t blockKv = 0;
    int64_t blockM = 0;
};

/// The record of @p kernel, or nullopt for one that lacks a field or holds it as another
/// type: a malformed record declines rather than throwing out of a match.
std::optional<Row> readRow(const KernelDefinition& kernel)
{
    using flydsl_sdpa::integerMetadata;
    const auto dtype = flydsl_sdpa::stringMetadata(kernel, DTYPE_FIELD);
    const auto tile = integerMetadata(kernel, HEAD_DIM_FIELD);
    const auto causal = integerMetadata(kernel, CAUSAL_FIELD);
    const auto hasBias = integerMetadata(kernel, HAS_BIAS_FIELD);
    const auto dropout = integerMetadata(kernel, DROPOUT_FIELD);
    const auto padded = integerMetadata(kernel, PADDED_FIELD);
    const auto storeDb = integerMetadata(kernel, STORE_DB_FIELD);
    const auto gqaPack = integerMetadata(kernel, GQA_PACK_FIELD);
    const auto blockThreads = integerMetadata(kernel, BLOCK_THREADS_FIELD);
    const auto blockKv = integerMetadata(kernel, BLOCK_KV_FIELD);
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    if(!dtype || !tile || !causal || !hasBias || !dropout || !padded || !storeDb || !gqaPack
       || !blockThreads || !blockKv || !blockM)
    {
        return std::nullopt;
    }
    return Row{*dtype,
               *tile,
               *causal,
               *hasBias,
               *dropout,
               *padded,
               *storeDb,
               *gqaPack,
               *blockThreads,
               *blockKv,
               *blockM};
}

int64_t rungBelow(int64_t tile)
{
    int64_t below = 0;
    for(const int64_t rung : LADDER)
    {
        if(rung < tile)
        {
            below = rung;
        }
    }
    return below;
}

/// Whether an object of @p row's tile serves a graph of head dims @p headDimQk and
/// @p headDimV: an exact object serves dims equal to its tile; a padded one serves dims
/// above the rung below the tile (the kernel's mask floor) and at most the tile.
bool servesHeadDims(const Row& row, int64_t headDimQk, int64_t headDimV)
{
    if(row.padded == 0)
    {
        return headDimQk == row.tile && headDimV == row.tile;
    }
    return std::min(headDimQk, headDimV) > rungBelow(row.tile)
           && std::max(headDimQk, headDimV) <= row.tile;
}

/// The launch grid's x extent, one block per (batch, key tile, key head) and per (batch,
/// query tile, query head), or nullopt past int32 or for a record without geometry.
std::optional<int64_t> mergedGridBlocks(int64_t batch,
                                        int64_t numHeads,
                                        int64_t numKvHeads,
                                        int64_t seqLenQ,
                                        int64_t seqLenKv,
                                        const Row& row)
{
    if(row.blockThreads <= 0 || row.blockKv <= 0 || row.blockM <= 0)
    {
        return std::nullopt;
    }
    const int64_t keyTiles = (seqLenKv + row.blockKv - 1) / row.blockKv;
    const int64_t queryTiles = (seqLenQ + row.blockM - 1) / row.blockM;
    const auto keyBlocks = bwd::detail::checkedProduct({numKvHeads, keyTiles});
    const auto queryBlocks = bwd::detail::checkedProduct({numHeads, queryTiles});
    if(!keyBlocks || !queryBlocks || *keyBlocks > INT32_LIMIT || *queryBlocks > INT32_LIMIT)
    {
        return std::nullopt;
    }
    const auto blocks = bwd::detail::checkedProduct({batch, *keyBlocks + *queryBlocks});
    if(!blocks || *blocks > INT32_LIMIT)
    {
        return std::nullopt;
    }
    return *blocks;
}

// ---------------------------------------------------------------------------
// Graph match and kernel match
// ---------------------------------------------------------------------------

/// Whether any node of the graph is an SDPA-backward node. Only such a graph hears why it
/// was declined: graph_match sees every graph the catalog misses on, and a line for each
/// convolution or normalization graph would bury the attention ones.
bool hasSdpaBackwardNode(const MatchContext& context)
{
    for(const auto& node : context.graph.nodeWrappers())
    {
        if(node->attributesType() == data_objects::NodeAttributes::SdpaBackwardAttributes)
        {
            return true;
        }
    }
    return false;
}

/// The output (batch, head) planes of the dBias reduce launch, one grid row each: the axes
/// the dBias keeps. 0 when the graph's dBias is not broadcast; int64 max on overflow.
int64_t reducePlanes(const bwd::SdpaBwdProblem& problem)
{
    if(problem.dbiasUid == bwd::NO_UID
       || !(problem.biasBroadcastBatch || problem.biasBroadcastHead))
    {
        return 0;
    }
    const auto planes
        = bwd::detail::checkedProduct({problem.biasBroadcastBatch ? int64_t{1} : problem.batch,
                                       problem.biasBroadcastHead ? int64_t{1} : problem.numHeads});
    return planes.value_or(std::numeric_limits<int64_t>::max());
}

/// "<engine> declined the graph [<cause>]: <detail>", at INFO: hipDNN reports a declined
/// graph to its caller only as "No engine configurations available".
void logDecline(std::string_view cause, const std::string& detail)
{
    HIPDNN_PLUGIN_LOG_INFO(ENGINE_NAME << " declined the graph [" << cause << "]: " << detail);
}

/// The bias's LDS staging rule: every non-unit stride a multiple of 8 elements (the matcher
/// records 0 on a broadcast or unit axis) and a 16-byte aligned base.
bool stagesBias(const bwd::SdpaBwdProblem& problem)
{
    constexpr int64_t VECTOR_ELEMENTS = 8;
    constexpr int64_t ALIGNMENT_BYTES = 16;
    return problem.biasStrides.batch % VECTOR_ELEMENTS == 0
           && problem.biasStrides.head % VECTOR_ELEMENTS == 0
           && problem.biasStrides.seq % VECTOR_ELEMENTS == 0
           && problem.biasAlignmentBytes % ALIGNMENT_BYTES == 0;
}

/**
 * @brief Graph-scoped applicability: the shared backward matcher's contract, less padded
 *        varlen, with what kernel_match needs bound.
 *
 * Each decline of a graph with an SDPA-backward node logs its cause at INFO; any other
 * graph declines silently.
 */
std::optional<BoundTokens> flydslSdpaBwdMergedGraphMatches(const MatchContext& context)
{
    if(!hasSdpaBackwardNode(context))
    {
        return std::nullopt;
    }
    const auto matched = bwd::matchSdpaBackward(context);
    if(const auto* decline = std::get_if<bwd::Decline>(&matched))
    {
        logDecline(decline->cause, decline->detail);
        return std::nullopt;
    }
    const auto& problem = std::get<bwd::SdpaBwdProblem>(matched);

    // The matcher serves padded varlen; the merged kernel's varlen path is not offered.
    if(problem.paddingMask || problem.seqLenQUid != bwd::NO_UID
       || problem.seqLenKvUid != bwd::NO_UID)
    {
        logDecline("varlen", "padded varlen (padding_mask with seq_len tensors) is not served");
        return std::nullopt;
    }

    // The merged table (`_instances.py`, `sdpa_bwd_merged`) builds exact objects at tiles 64 to
    // 192 and, at tile 32 only, a padded object: a head size is served when both dims are at
    // most 32 or both equal one of those tiles. kernel_match keeps its per-row check.
    const int64_t largest = std::max(problem.headDimQk, problem.headDimV);
    const bool exactTile = problem.headDimQk == problem.headDimV && largest >= 64 && largest <= 192
                           && largest % 32 == 0;
    if(largest > 32 && !exactTile)
    {
        logDecline(
            "head_size",
            "head dims Qk " + std::to_string(problem.headDimQk) + ", V "
                + std::to_string(problem.headDimV)
                + " are covered by no merged object (<= 32, or equal at 64, 96, 128, 160 or 192)");
        return std::nullopt;
    }

    const auto dtype = flydsl_sdpa::kernelDataTypeName(problem.dtype);
    if(!dtype.has_value())
    {
        logDecline("data_type", "the matcher served a dtype this engine has no spelling for");
        return std::nullopt;
    }
    const int64_t graphClass = problem.biasUid != bwd::NO_UID ? CLASS_BIAS
                               : problem.masked               ? CLASS_CAUSAL
                                                              : CLASS_PLAIN;

    BoundTokens bound;
    bound[std::string(CLASS_TOKEN)] = graphClass;
    bound[std::string(DTYPE_TOKEN)] = *dtype;
    bound[std::string(HEAD_DIM_QK_TOKEN)] = problem.headDimQk;
    bound[std::string(HEAD_DIM_V_TOKEN)] = problem.headDimV;
    bound[std::string(DROPOUT_TOKEN)] = int64_t{problem.dropoutP > 0.0F ? 1 : 0};
    bound[std::string(DBIAS_TOKEN)] = int64_t{problem.dbiasUid != bwd::NO_UID ? 1 : 0};
    // The matcher guarantees Hq % Hk == 0, so more query heads than key heads is a group.
    bound[std::string(GQA_PACK_TOKEN)] = int64_t{problem.numHeads > problem.numKvHeads ? 1 : 0};
    bound[std::string(BIAS_STAGING_TOKEN)] = int64_t{stagesBias(problem) ? 1 : 0};
    bound[std::string(BATCH_TOKEN)] = problem.batch;
    bound[std::string(HEADS_TOKEN)] = problem.numHeads;
    bound[std::string(KV_HEADS_TOKEN)] = problem.numKvHeads;
    bound[std::string(SEQ_Q_TOKEN)] = problem.seqLenQ;
    bound[std::string(SEQ_KV_TOKEN)] = problem.seqLenKv;
    bound[std::string(REDUCE_PLANES_TOKEN)] = reducePlanes(problem);
    return bound;
}

/// What kernel_match reads of the bound tokens.
struct GraphFacts
{
    std::string dtype;
    int64_t graphClass = 0;
    int64_t headDimQk = 0;
    int64_t headDimV = 0;
    bool dropout = false;
    bool dbias = false;
    bool gqaPack = false;
    bool biasStaging = false;
    int64_t batch = 0;
    int64_t numHeads = 0;
    int64_t numKvHeads = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t reducePlanes = 0;
};

std::optional<GraphFacts> readFacts(const BoundTokens& bound)
{
    const auto dtype = bound.find(std::string(DTYPE_TOKEN));
    if(dtype == bound.end() || !std::holds_alternative<std::string>(dtype->second))
    {
        return std::nullopt;
    }
    const auto graphClass = tryGetBoundInt(bound, CLASS_TOKEN);
    const auto headDimQk = tryGetBoundInt(bound, HEAD_DIM_QK_TOKEN);
    const auto headDimV = tryGetBoundInt(bound, HEAD_DIM_V_TOKEN);
    const auto dropout = tryGetBoundInt(bound, DROPOUT_TOKEN);
    const auto dbias = tryGetBoundInt(bound, DBIAS_TOKEN);
    const auto gqaPack = tryGetBoundInt(bound, GQA_PACK_TOKEN);
    const auto biasStaging = tryGetBoundInt(bound, BIAS_STAGING_TOKEN);
    const auto batch = tryGetBoundInt(bound, BATCH_TOKEN);
    const auto numHeads = tryGetBoundInt(bound, HEADS_TOKEN);
    const auto numKvHeads = tryGetBoundInt(bound, KV_HEADS_TOKEN);
    const auto seqLenQ = tryGetBoundInt(bound, SEQ_Q_TOKEN);
    const auto seqLenKv = tryGetBoundInt(bound, SEQ_KV_TOKEN);
    const auto planes = tryGetBoundInt(bound, REDUCE_PLANES_TOKEN);
    if(!graphClass || !headDimQk || !headDimV || !dropout || !dbias || !gqaPack || !biasStaging
       || !batch || !numHeads || !numKvHeads || !seqLenQ || !seqLenKv || !planes)
    {
        return std::nullopt;
    }
    return GraphFacts{std::get<std::string>(dtype->second),
                      *graphClass,
                      *headDimQk,
                      *headDimV,
                      *dropout != 0,
                      *dbias != 0,
                      *gqaPack != 0,
                      *biasStaging != 0,
                      *batch,
                      *numHeads,
                      *numKvHeads,
                      *seqLenQ,
                      *seqLenKv,
                      *planes};
}

/**
 * @brief Kernel-scoped applicability: the object's class, dtype, dropout, head-dim tile,
 *        bias staging, dB store and GQA packing against the graph, and a launch grid that
 *        fits int32.
 */
bool flydslSdpaBwdMergedKernelMatches(const MatchContext& /*context*/,
                                      const BoundTokens& bound,
                                      const KernelDefinition& kernel)
{
    const auto facts = readFacts(bound);
    const auto row = readRow(kernel);
    if(!facts || !row)
    {
        return false;
    }
    if(row->dtype != facts->dtype || (row->dropout != 0) != facts->dropout)
    {
        return false;
    }
    if(row->causal != (facts->graphClass == CLASS_CAUSAL ? 1 : 0)
       || row->hasBias != (facts->graphClass == CLASS_BIAS ? 1 : 0))
    {
        return false;
    }
    if(!servesHeadDims(*row, facts->headDimQk, facts->headDimV))
    {
        return false;
    }
    // Every bias object reads the bias through LDS.
    if(row->hasBias != 0 && !facts->biasStaging)
    {
        return false;
    }
    // store_db 0 builds the dQ role without its dB store.
    if(facts->dbias && row->storeDb == 0)
    {
        return false;
    }
    if(row->gqaPack == 1 && !facts->gqaPack)
    {
        return false;
    }
    // The reduce launch puts one plane on each grid row.
    if(facts->reducePlanes > GRID_YZ_LIMIT)
    {
        return false;
    }
    return mergedGridBlocks(facts->batch,
                            facts->numHeads,
                            facts->numKvHeads,
                            facts->seqLenQ,
                            facts->seqLenKv,
                            *row)
        .has_value();
}

/// Every row's position is its descriptor order, which is the priority order (D9).
double flydslSdpaBwdMergedScore(const MatchContext& /*context*/,
                                const BoundTokens& /*bound*/,
                                const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

// ---------------------------------------------------------------------------
// Launch values
// ---------------------------------------------------------------------------

/// The drop threshold of probability @p p: a uniform u32 read as i32 is kept when above
/// `(p - 0.5) * 2^32`. A port of `philox.dropout_threshold`, which truncates toward zero
/// and clamps to int32; the same value AOTriton computes.
int32_t dropoutThreshold(double p)
{
    const auto threshold = static_cast<int64_t>((p - 0.5) * 4294967295.0);
    return static_cast<int32_t>(std::clamp<int64_t>(
        threshold, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()));
}

/// The kernel's window bounds: `column j is kept for row i iff i - left <= j <= i + right`.
struct Window
{
    int32_t left = 0;
    int32_t right = 0;
};

/// The bounds a graph asks for in the kernel's convention. A graph without a mask passes 0, 0
/// (the object has no window). Plain causal, no left bound and right 0, passes the
/// sentinel of its alignment for both bounds, which the kernel resolves against each
/// sequence's own lengths. Every other mask passes literals: cuDNN's right bound plus the
/// diagonal offset, clamped to the key length; cuDNN's left bound less one and less the
/// offset, clamped to the query length plus the offset's magnitude. A side the graph leaves
/// unbounded is its clamp, and both clamps keep every value int32-safe.
Window windowFor(const bwd::SdpaBwdProblem& problem)
{
    if(!problem.masked)
    {
        return {};
    }
    if(problem.windowLeft == bwd::NO_WINDOW && problem.windowRight == 0)
    {
        const int32_t sentinel = problem.alignBottomRight ? WINDOW_BOTRIGHT : WINDOW_TOPLEFT;
        return {sentinel, sentinel};
    }
    const int64_t offset = problem.diagonalOffset;
    const int64_t leftClamp = problem.seqLenQ + (offset < 0 ? -offset : offset);
    const int64_t rightClamp = problem.seqLenKv;

    int64_t left = leftClamp;
    if(problem.windowLeft != bwd::NO_WINDOW && problem.windowLeft <= leftClamp + 1 + offset)
    {
        left = problem.windowLeft - 1 - offset;
    }
    int64_t right = rightClamp;
    if(problem.windowRight != bwd::NO_WINDOW && problem.windowRight <= rightClamp - offset)
    {
        right = problem.windowRight + offset;
    }
    return {toInt32(left, "window_left"), toInt32(right, "window_right")};
}

/// Where the workspace's regions are: the one place its size and its offsets are decided,
/// read by workspaceBytes() and by launch.
///
/// `[seed u64 | offset u64]` at offset 0 when the seed and offset are not 64-bit device
/// tensors; the full-shape dBias buffer at the next 256-byte boundary when dBias is
/// broadcast over batch or head. The dBias region is never zeroed: the dQ role writes
/// every in-range element of it exactly once.
struct WorkspaceLayout
{
    bool uploadsSeed = false;
    bool reduces = false;
    size_t dbiasOffset = 0;
    /// (B, Hq, Sq, Sk) strides of the dBias region, in elements.
    bwd::Strides dbiasStrides;
    size_t totalBytes = 0;
};

WorkspaceLayout workspaceLayout(const bwd::SdpaBwdProblem& problem)
{
    WorkspaceLayout layout;
    // A 64-bit device tensor is read in place; every other form is written here.
    layout.uploadsSeed = problem.seedForm != bwd::SeedForm::NONE
                         && !(problem.seedForm == bwd::SeedForm::DEVICE
                              && problem.seedWidth == bwd::SeedWidth::INT64);
    layout.totalBytes = layout.uploadsSeed ? SEED_REGION_BYTES : 0;

    layout.reduces = problem.dbiasUid != bwd::NO_UID
                     && (problem.biasBroadcastBatch || problem.biasBroadcastHead);
    if(!layout.reduces)
    {
        return layout;
    }
    const auto plane = bwd::detail::checkedProduct({problem.seqLenQ, problem.seqLenKv});
    const auto head = plane ? bwd::detail::checkedProduct({problem.numHeads, *plane})
                            : std::optional<int64_t>{};
    const auto elements
        = head ? bwd::detail::checkedProduct({problem.batch, *head}) : std::optional<int64_t>{};
    const auto bytes = elements ? bwd::detail::checkedProduct({*elements, WORKSPACE_ELEMENT_BYTES})
                                : std::optional<int64_t>{};
    if(!bytes)
    {
        failBadParam("the broadcast dBias workspace overflows int64");
    }
    layout.dbiasStrides = {*head, *plane, problem.seqLenKv};
    layout.dbiasOffset
        = (layout.totalBytes + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
    layout.totalBytes = layout.dbiasOffset + static_cast<size_t>(*bytes);
    return layout;
}

/// The tensor uids a matched graph binds.
struct Uids
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t dO = 0;
    int64_t dQ = 0;
    int64_t dK = 0;
    int64_t dV = 0;
    int64_t stats = 0;
};

/// Everything graph- and kernel-derived, resolved once at prepare(). Holds no reference to
/// the match context.
struct MergedPlan
{
    bwd::SdpaBwdProblem problem;
    Uids uids;
    unsigned int blockThreads = 0;
    unsigned int gridBlocks = 0;
    Window window;
    int32_t dropoutThreshold = 0;
    float dropoutScale = 1.0F;
    int32_t runtimeFlags = 0;
    int32_t varlenBits = 0;
    WorkspaceLayout workspace;
    /// Set for a tensor-form scale; the value form is problem.scaleValue.
    std::optional<hipdnn_plugin_sdk::ScalarOperand> scale;
    /// Set for a pass-by-value seed and offset.
    std::optional<hipdnn_plugin_sdk::ScalarOperand> seed;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> offset;
    /// The reduce kernel's symbol and launch, when the graph's dBias is broadcast.
    std::string reduceSymbol;
    unsigned int reduceGridX = 0;
    unsigned int reduceGridY = 0;
};

bwd::SdpaBwdProblem problemOf(const MatchContext& context)
{
    auto matched = bwd::matchSdpaBackward(context);
    if(auto* problem = std::get_if<bwd::SdpaBwdProblem>(&matched))
    {
        return std::move(*problem);
    }
    const auto& decline = std::get<bwd::Decline>(matched);
    failBadParam("the graph no longer matches [" + decline.cause + "]: " + decline.detail);
}

MergedPlan planFor(const MatchContext& context, const KernelDefinition& kernel)
{
    MergedPlan plan;
    plan.problem = problemOf(context);
    const auto& problem = plan.problem;

    const auto row = readRow(kernel);
    if(!row)
    {
        failBadParam("kernel '" + kernel.name + "' lacks a field of the merged record");
    }
    const auto blocks = mergedGridBlocks(problem.batch,
                                         problem.numHeads,
                                         problem.numKvHeads,
                                         problem.seqLenQ,
                                         problem.seqLenKv,
                                         *row);
    if(!blocks)
    {
        failBadParam("launch grid exceeds int32 for kernel '" + kernel.name + "'");
    }
    plan.blockThreads = static_cast<unsigned int>(row->blockThreads);
    plan.gridBlocks = static_cast<unsigned int>(*blocks);

    const auto& attributes
        = context.graph.getNodeWrapper(0).attributesAs<data_objects::SdpaBackwardAttributes>();
    plan.uids = Uids{attributes.q_tensor_uid(),
                     attributes.k_tensor_uid(),
                     attributes.v_tensor_uid(),
                     attributes.o_tensor_uid(),
                     attributes.do_tensor_uid(),
                     attributes.dq_tensor_uid(),
                     attributes.dk_tensor_uid(),
                     attributes.dv_tensor_uid(),
                     attributes.stats_tensor_uid()};

    plan.window = windowFor(problem);
    if(problem.dropoutP > 0.0F)
    {
        const double p = static_cast<double>(problem.dropoutP);
        plan.dropoutThreshold = dropoutThreshold(p);
        plan.dropoutScale = static_cast<float>(1.0 / (1.0 - p));
    }
    const bool paddedHead = problem.headDimQk != row->tile || problem.headDimV != row->tile;
    plan.runtimeFlags
        = (paddedHead ? RT_FLAG_PADDED : 0) | (problem.dropoutP > 0.0F ? RT_FLAG_DROPOUT : 0);
    plan.varlenBits = problem.statsLayout == bwd::StatsLayout::TH ? VARLEN_LSE_LAYOUT_TH : 0;

    const auto& tensors = context.graph.getTensorMap();
    if(problem.scaleSource == bwd::ScaleSource::TENSOR)
    {
        plan.scale = hipdnn_plugin_sdk::makeScalarOperand(tensors, problem.scaleUid, "scale");
    }
    if(problem.seedForm == bwd::SeedForm::PASS_BY_VALUE)
    {
        plan.seed = hipdnn_plugin_sdk::makeScalarOperand(tensors, problem.seedUid, "seed");
        plan.offset = hipdnn_plugin_sdk::makeScalarOperand(tensors, problem.offsetUid, "offset");
    }

    plan.workspace = workspaceLayout(problem);
    if(plan.workspace.reduces)
    {
        // Read only now: an object without a broadcast dBias never names a reduce kernel.
        plan.reduceSymbol
            = flydsl_sdpa::stringMetadata(kernel, REDUCE_SYMBOL_FIELD).value_or(std::string());
        if(plan.reduceSymbol.empty())
        {
            failBadParam("kernel '" + kernel.name
                         + "' serves a broadcast dBias but names no reduce kernel");
        }
        const bool reduceBatch = problem.biasBroadcastBatch;
        const bool reduceHead = problem.biasBroadcastHead;
        const int64_t vectors = (problem.seqLenKv + REDUCE_VECTOR - 1) / REDUCE_VECTOR;
        const auto items = bwd::detail::checkedProduct({problem.seqLenQ, vectors});
        const auto planes = bwd::detail::checkedProduct(
            {reduceBatch ? int64_t{1} : problem.batch, reduceHead ? int64_t{1} : problem.numHeads});
        if(!items || !planes)
        {
            failBadParam("the dBias reduce launch overflows int64");
        }
        const int64_t gridX = (*items + REDUCE_THREADS - 1) / REDUCE_THREADS;
        if(gridX > INT32_LIMIT || *planes > GRID_YZ_LIMIT)
        {
            failBadParam("the dBias reduce launch grid exceeds its limits");
        }
        plan.reduceGridX = static_cast<unsigned int>(gridX);
        plan.reduceGridY = static_cast<unsigned int>(*planes);
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Kernel argument lists
// ---------------------------------------------------------------------------

/**
 * @brief The merged kernel's arguments, in parameter order (`_PARAMS["merged"]` in
 *        `_sdpa_bwd_cdna_build.py`).
 *
 * A pointer is 8 bytes, a count or flag 4 (i32, f32) and a stride or offset 8 (i64), each at
 * its natural offset: 63 arguments, 452 bytes. tie() is the one listing: launch passes it
 * through, mergedSignature() reads its types, and the tests' golden dumps serialize it.
 */
struct MergedKernargs
{
    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* bias = nullptr;
    void* o = nullptr;
    void* dO = nullptr;
    void* dK = nullptr;
    void* dV = nullptr;
    void* dQ = nullptr;
    void* dBias = nullptr;
    void* lse = nullptr;
    int32_t batchSize = 0;
    void* seqinfoQ0 = nullptr;
    void* seqinfoQ1 = nullptr;
    void* seqinfoK0 = nullptr;
    void* seqinfoK1 = nullptr;
    int32_t varlenBits = 0;
    int32_t numSeqlens = 0;
    int32_t maxSeqlenQ = 0;
    int32_t maxSeqlenK = 0;
    int32_t windowLeft = 0;
    int32_t windowRight = 0;
    void* philoxSeed = nullptr;
    void* philoxOffset1 = nullptr;
    int64_t philoxOffset2 = 0;
    int32_t idropoutP = 0;
    float dropoutScale = 1.0F;
    int32_t numHeadQ = 0;
    int32_t numHeadK = 0;
    int32_t hdimQk = 0;
    int32_t hdimVo = 0;
    float smScale = 0.0F;
    bwd::Strides strideQ;
    bwd::Strides strideK;
    bwd::Strides strideV;
    bwd::Strides strideO;
    bwd::Strides strideDO;
    bwd::Strides strideDK;
    bwd::Strides strideDV;
    bwd::Strides strideDQ;
    bwd::Strides strideB;
    bwd::Strides strideDB;
    int32_t rtFlags = 0;

    auto tie() const
    {
        return std::tie(q,
                        k,
                        v,
                        bias,
                        o,
                        dO,
                        dK,
                        dV,
                        dQ,
                        dBias,
                        lse,
                        batchSize,
                        seqinfoQ0,
                        seqinfoQ1,
                        seqinfoK0,
                        seqinfoK1,
                        varlenBits,
                        numSeqlens,
                        maxSeqlenQ,
                        maxSeqlenK,
                        windowLeft,
                        windowRight,
                        philoxSeed,
                        philoxOffset1,
                        philoxOffset2,
                        idropoutP,
                        dropoutScale,
                        numHeadQ,
                        numHeadK,
                        hdimQk,
                        hdimVo,
                        smScale,
                        strideQ.batch,
                        strideQ.head,
                        strideQ.seq,
                        strideK.batch,
                        strideK.head,
                        strideK.seq,
                        strideV.batch,
                        strideV.head,
                        strideV.seq,
                        strideO.batch,
                        strideO.head,
                        strideO.seq,
                        strideDO.batch,
                        strideDO.head,
                        strideDO.seq,
                        strideDK.batch,
                        strideDK.head,
                        strideDK.seq,
                        strideDV.batch,
                        strideDV.head,
                        strideDV.seq,
                        strideDQ.batch,
                        strideDQ.head,
                        strideDQ.seq,
                        strideB.batch,
                        strideB.head,
                        strideB.seq,
                        strideDB.batch,
                        strideDB.head,
                        strideDB.seq,
                        rtFlags);
    }
};

/// The dBias reduce kernel's arguments (`_PARAMS["dbias_reduce"]`): the full-shape workspace
/// and the user's dBias, the problem's extents, which axes are summed, and the strides of
/// each tensor: 14 arguments, 88 bytes.
struct ReduceKernargs
{
    void* workspace = nullptr;
    void* dBias = nullptr;
    int32_t numBatch = 0;
    int32_t numHead = 0;
    int32_t seqLenQ = 0;
    int32_t seqLenK = 0;
    int32_t reduceBatch = 0;
    int32_t reduceHead = 0;
    bwd::Strides strideWorkspace;
    bwd::Strides strideOut;

    auto tie() const
    {
        return std::tie(workspace,
                        dBias,
                        numBatch,
                        numHead,
                        seqLenQ,
                        seqLenK,
                        reduceBatch,
                        reduceHead,
                        strideWorkspace.batch,
                        strideWorkspace.head,
                        strideWorkspace.seq,
                        strideOut.batch,
                        strideOut.head,
                        strideOut.seq);
    }
};

/// The natural offset of the next argument of type @p T after @p offset.
template <typename T>
size_t alignedOffset(size_t offset)
{
    return (offset + alignof(T) - 1) / alignof(T) * alignof(T);
}

template <typename T>
KernelArgument describeArgument(size_t& offset)
{
    const size_t at = alignedOffset<T>(offset);
    offset = at + sizeof(T);
    return KernelArgument{std::is_pointer_v<T> ? "global_buffer" : "by_value",
                          static_cast<uint32_t>(sizeof(T)),
                          static_cast<uint32_t>(at),
                          ""};
}

template <typename Tuple, size_t... Index>
std::vector<KernelArgument> signatureFromTuple(std::index_sequence<Index...>)
{
    size_t offset = 0;
    // A braced list is evaluated left to right, so each offset follows the one before.
    return std::vector<KernelArgument>{describeArgument<
        std::remove_cv_t<std::remove_reference_t<std::tuple_element_t<Index, Tuple>>>>(offset)...};
}

template <typename Kernargs>
std::vector<KernelArgument> signatureOf()
{
    using Tuple = decltype(std::declval<const Kernargs&>().tie());
    return signatureFromTuple<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

const std::vector<KernelArgument>& mergedSignature()
{
    static const std::vector<KernelArgument> s_signature = signatureOf<MergedKernargs>();
    return s_signature;
}

const std::vector<KernelArgument>& reduceSignature()
{
    static const std::vector<KernelArgument> s_signature = signatureOf<ReduceKernargs>();
    return s_signature;
}

template <typename Kernargs>
std::vector<unsigned char> kernargImage(const Kernargs& kernargs)
{
    std::vector<unsigned char> image;
    std::apply(
        [&image](const auto&... argument) {
            const auto append = [&image](const auto& value) {
                using T = std::remove_cv_t<std::remove_reference_t<decltype(value)>>;
                const size_t at = alignedOffset<T>(image.size());
                image.resize(at + sizeof(T));
                std::memcpy(image.data() + at, &value, sizeof(T));
            };
            (append(argument), ...);
        },
        kernargs.tie());
    return image;
}

// ---------------------------------------------------------------------------
// Launch arguments
// ---------------------------------------------------------------------------

/// How a seed and offset that are not 64-bit device tensors reach the workspace: the four
/// 32-bit words of `[seed u64 | offset u64]` written as values (pass-by-value, either
/// width), or, for 32-bit device tensors, the sources whose 4 bytes are copied into the low
/// half of each zeroed word.
struct SeedUpload
{
    std::array<uint32_t, 4> words{};
    const void* seedSource = nullptr;
    const void* offsetSource = nullptr;
};

struct LaunchArguments
{
    MergedKernargs merged;
    std::optional<ReduceKernargs> reduce;
    std::optional<SeedUpload> seedUpload;
};

/// A pass-by-value integer as the low and high words of its 64-bit extension. A 32-bit value
/// is zero-extended, so a negative int32 seed is its unsigned bit pattern.
std::array<uint32_t, 2> wordsOf(const hipdnn_plugin_sdk::ScalarValue& value, const char* what)
{
    if(const auto* narrow = std::get_if<int32_t>(&value))
    {
        return {static_cast<uint32_t>(*narrow), 0U};
    }
    if(const auto* wide = std::get_if<int64_t>(&value))
    {
        const auto bits = static_cast<uint64_t>(*wide);
        return {static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)};
    }
    failBadParam(std::string("the ") + what + " is not an integer");
}

float resolveScale(const MergedPlan& plan,
                   const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                   uint32_t numDeviceBuffers)
{
    if(!plan.scale.has_value())
    {
        return plan.problem.scaleValue;
    }
    const double value = hipdnn_plugin_sdk::toDouble(
        hipdnn_plugin_sdk::resolveScalarOperand(*plan.scale, deviceBuffers, numDeviceBuffers));
    // A pass-by-value scalar read from host memory: a constant scale was checked at match
    // time, this one arrives now.
    if(!std::isfinite(value) || std::fabs(value) > bwd::detail::MAX_SOFTMAX_SCALE)
    {
        failBadParam("the runtime softmax scale must be finite and at most 2^20 in magnitude");
    }
    return static_cast<float>(value);
}

/// Every argument of the launch, from the plan and the execute-time buffers; touches no
/// device. @p workspace is the caller's, laid out as plan.workspace says.
LaunchArguments launchArguments(const MergedPlan& plan,
                                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                uint32_t numDeviceBuffers,
                                void* workspace)
{
    const auto& problem = plan.problem;
    const auto& layout = plan.workspace;
    if(workspace == nullptr && layout.totalBytes > 0)
    {
        failBadParam("the launch needs its " + std::to_string(layout.totalBytes)
                     + " byte workspace");
    }
    const auto buffer = [&](int64_t uid) {
        return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
    };
    const auto inWorkspace
        = [&](size_t offset) { return static_cast<void*>(static_cast<char*>(workspace) + offset); };

    LaunchArguments arguments;
    auto& m = arguments.merged;
    m.q = buffer(plan.uids.q);
    m.k = buffer(plan.uids.k);
    m.v = buffer(plan.uids.v);
    m.o = buffer(plan.uids.o);
    m.dO = buffer(plan.uids.dO);
    m.dK = buffer(plan.uids.dK);
    m.dV = buffer(plan.uids.dV);
    m.dQ = buffer(plan.uids.dQ);
    // The raw statistics of the forward: the kernel turns -inf into +inf itself.
    m.lse = buffer(plan.uids.stats);
    // A tensor the graph does not use is passed as O, as the harness does.
    m.bias = problem.biasUid != bwd::NO_UID ? buffer(problem.biasUid) : m.o;

    m.batchSize = toInt32(problem.batch, "batch");
    m.seqinfoQ0 = m.o;
    m.seqinfoQ1 = m.o;
    m.seqinfoK0 = m.o;
    m.seqinfoK1 = m.o;
    m.varlenBits = plan.varlenBits;
    m.numSeqlens = 0;
    m.maxSeqlenQ = toInt32(problem.seqLenQ, "seqlen_q");
    m.maxSeqlenK = toInt32(problem.seqLenKv, "seqlen_k");
    m.windowLeft = plan.window.left;
    m.windowRight = plan.window.right;

    m.philoxSeed = m.o;
    m.philoxOffset1 = m.o;
    if(problem.seedForm != bwd::SeedForm::NONE)
    {
        m.idropoutP = plan.dropoutThreshold;
        m.dropoutScale = plan.dropoutScale;
        if(layout.uploadsSeed)
        {
            m.philoxSeed = inWorkspace(0);
            m.philoxOffset1 = inWorkspace(SEED_WORD_BYTES);
            SeedUpload upload;
            if(problem.seedForm == bwd::SeedForm::PASS_BY_VALUE)
            {
                const auto seed = wordsOf(hipdnn_plugin_sdk::resolveScalarOperand(
                                              *plan.seed, deviceBuffers, numDeviceBuffers),
                                          "seed");
                const auto offset = wordsOf(hipdnn_plugin_sdk::resolveScalarOperand(
                                                *plan.offset, deviceBuffers, numDeviceBuffers),
                                            "offset");
                upload.words = {seed[0], seed[1], offset[0], offset[1]};
            }
            else
            {
                upload.seedSource = buffer(problem.seedUid);
                upload.offsetSource = buffer(problem.offsetUid);
            }
            arguments.seedUpload = upload;
        }
        else
        {
            m.philoxSeed = buffer(problem.seedUid);
            m.philoxOffset1 = buffer(problem.offsetUid);
        }
    }
    // DR-5: the one-offset API, the second Philox scalar is zero.
    m.philoxOffset2 = 0;

    m.numHeadQ = toInt32(problem.numHeads, "num_head_q");
    m.numHeadK = toInt32(problem.numKvHeads, "num_head_k");
    m.hdimQk = toInt32(problem.headDimQk, "hdim_qk");
    m.hdimVo = toInt32(problem.headDimV, "hdim_vo");
    m.smScale = resolveScale(plan, deviceBuffers, numDeviceBuffers);

    // The matcher records 0 on a unit axis. A unit sequence axis keeps a row pitch instead:
    // the kernels drop padding rows through the buffer bound rows * stride_seq, which a zero
    // stride collapses onto row 0. The pitch is the head dim rounded to 8 elements, which
    // keeps the 8-element alignment of every other stride.
    const auto rowPitch = [](bwd::Strides strides, int64_t headDim) {
        if(strides.seq == 0)
        {
            strides.seq = (headDim + 7) / 8 * 8;
        }
        return strides;
    };
    m.strideQ = rowPitch(problem.q, problem.headDimQk);
    m.strideK = rowPitch(problem.k, problem.headDimQk);
    m.strideV = rowPitch(problem.v, problem.headDimV);
    m.strideO = rowPitch(problem.o, problem.headDimV);
    m.strideDO = rowPitch(problem.dO, problem.headDimV);
    m.strideDK = rowPitch(problem.dK, problem.headDimQk);
    m.strideDV = rowPitch(problem.dV, problem.headDimV);
    m.strideDQ = rowPitch(problem.dQ, problem.headDimQk);
    m.strideB = problem.biasUid != bwd::NO_UID ? problem.biasStrides : bwd::Strides{};

    // dBias: none requested leaves DB as O over zero strides, with which the kernel drops
    // every store; a broadcast one lands in the workspace and is summed after; any other
    // is written in place.
    m.dBias = m.o;
    if(problem.dbiasUid != bwd::NO_UID)
    {
        if(layout.reduces)
        {
            m.dBias = inWorkspace(layout.dbiasOffset);
            m.strideDB = layout.dbiasStrides;

            ReduceKernargs reduce;
            reduce.workspace = m.dBias;
            reduce.dBias = buffer(problem.dbiasUid);
            reduce.numBatch = m.batchSize;
            reduce.numHead = m.numHeadQ;
            reduce.seqLenQ = m.maxSeqlenQ;
            reduce.seqLenK = m.maxSeqlenK;
            reduce.reduceBatch = problem.biasBroadcastBatch ? 1 : 0;
            reduce.reduceHead = problem.biasBroadcastHead ? 1 : 0;
            reduce.strideWorkspace = layout.dbiasStrides;
            reduce.strideOut = problem.dbiasStrides;
            arguments.reduce = reduce;
        }
        else
        {
            m.dBias = buffer(problem.dbiasUid);
            m.strideDB = problem.dbiasStrides;
            // The kernel bounds dB by Sq * stride_seq_q, so a zero row stride (a unit Sq)
            // drops every store: the row pitch stands in.
            if(m.strideDB.seq == 0)
            {
                m.strideDB.seq = problem.seqLenKv;
            }
        }
    }
    m.rtFlags = plan.runtimeFlags;
    return arguments;
}

/// Writes the seed and offset words into the front of @p workspace, ahead of the launch on
/// @p stream: four word writes for a pass-by-value form; for a 32-bit device form the region
/// is zeroed and the 4 bytes of each tensor copied into the low half of its word.
void uploadSeed(const SeedUpload& upload, void* workspace, hipStream_t stream)
{
    auto* base = static_cast<char*>(workspace);
    if(upload.seedSource == nullptr)
    {
        for(size_t word = 0; word < upload.words.size(); ++word)
        {
            requireHip(
                hipMemsetD32Async(static_cast<hipDeviceptr_t>(base + word * sizeof(uint32_t)),
                                  static_cast<int>(upload.words[word]),
                                  1,
                                  stream),
                "hipMemsetD32Async of the seed words");
        }
        return;
    }
    requireHip(
        hipMemsetD32Async(
            static_cast<hipDeviceptr_t>(base), 0, SEED_REGION_BYTES / sizeof(uint32_t), stream),
        "hipMemsetD32Async of the seed region");
    requireHip(
        hipMemcpyAsync(base, upload.seedSource, sizeof(uint32_t), hipMemcpyDeviceToDevice, stream),
        "hipMemcpyAsync of the seed");
    requireHip(hipMemcpyAsync(base + SEED_WORD_BYTES,
                              upload.offsetSource,
                              sizeof(uint32_t),
                              hipMemcpyDeviceToDevice,
                              stream),
               "hipMemcpyAsync of the offset");
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The merged kernel, the reduce kernel when the graph needs one, and the plan.
class PreparedFlydslSdpaBwdMerged : public PreparedDispatch
{
public:
    PreparedFlydslSdpaBwdMerged(MergedPlan plan,
                                IngestorKernelCode mergedCode,
                                std::optional<IngestorKernelCode> reduceCode)
        : _plan(std::move(plan))
        , _merged(std::move(mergedCode))
        , _reduce(std::move(reduceCode))
    {
    }

    const MergedPlan& plan() const
    {
        return _plan;
    }

    compilation::IRunnableKernel& mergedForStream(hipStream_t stream) const
    {
        return _merged.kernelForStream(stream);
    }

    compilation::IRunnableKernel& reduceForStream(hipStream_t stream) const
    {
        return _reduce->kernelForStream(stream);
    }

private:
    MergedPlan _plan;
    IngestorKernelCode _merged;
    std::optional<IngestorKernelCode> _reduce;
};

/**
 * @brief The native dispatch behind this pack's UDD: sizes the workspace, uploads the
 *        seed, launches the merged kernel and, for a broadcast dBias, the reduce kernel.
 */
class FlydslSdpaBwdMergedDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    FlydslSdpaBwdMergedDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                       const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return workspaceLayout(problemOf(context)).totalBytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& /*bound*/,
                                              const KernelDefinition& kernel) const override
    {
        auto plan = planFor(context, kernel);

        // Built from the arch alone; see FlydslRmsNormNative.cpp's prepare().
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto mergedCode = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, mergedSignature());
        mergedCode.setBlockSize(plan.blockThreads, 1, 1);
        mergedCode.setGridSize(plan.gridBlocks, 1, 1);

        // The reduce kernel is the same code object under another symbol: the same archive
        // entry, with that symbol and its own argument list.
        std::optional<IngestorKernelCode> reduceCode;
        if(plan.workspace.reduces)
        {
            KernelDefinition reduceKernel = kernel;
            reduceKernel.source.symbol = plan.reduceSymbol;
            reduceKernel.source.signature = reduceSignature();
            reduceCode.emplace(buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, reduceKernel, options, reduceSignature()));
            reduceCode->setBlockSize(REDUCE_THREADS, 1, 1);
            reduceCode->setGridSize(plan.reduceGridX, plan.reduceGridY, 1);
        }
        return std::make_unique<PreparedFlydslSdpaBwdMerged>(
            std::move(plan), std::move(mergedCode), std::move(reduceCode));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedBwd = dynamic_cast<const PreparedFlydslSdpaBwdMerged&>(prepared);
        const auto arguments
            = launchArguments(preparedBwd.plan(), deviceBuffers, numDeviceBuffers, workspace);
        const auto stream = handle.getStream();

        if(arguments.seedUpload.has_value())
        {
            uploadSeed(*arguments.seedUpload, workspace, stream);
        }
        // Changing either argument list means changing its struct with it: the signature the
        // descriptor is checked against is read from the same tie().
        std::apply(
            [&](const auto&... argument) {
                preparedBwd.mergedForStream(stream).launch(stream, argument...);
            },
            arguments.merged.tie());
        if(arguments.reduce.has_value())
        {
            std::apply(
                [&](const auto&... argument) {
                    preparedBwd.reduceForStream(stream).launch(stream, argument...);
                },
                arguments.reduce->tie());
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& flydslSdpaBwdMergedKpackModuleCache()
{
    // This pack's own; see flydslSdpaCdnaKpackModuleCache().
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

/// Process-lifetime: the registry holds a non-owning pointer to it.
const FlydslSdpaBwdMergedDispatchHandler& flydslSdpaBwdMergedDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        flydslSdpaBwdMergedKpackModuleCache());
    static const FlydslSdpaBwdMergedDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                      s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void resetFlydslSdpaBwdMergedModuleCache()
{
    flydslSdpaBwdMergedKpackModuleCache().clear();
}

void registerFlydslSdpaBwdMergedSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslSdpaBwdMergedGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSdpaBwdMergedKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSdpaBwdMergedScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaBwdMergedDispatchHandler());
}

FlydslSdpaBwdMergedLaunchDump describeFlydslSdpaBwdMergedLaunchForTesting(
    const MatchContext& context,
    const KernelDefinition& kernel,
    const std::vector<hipdnnPluginDeviceBuffer_t>& buffers,
    void* workspace)
{
    const auto plan = planFor(context, kernel);
    const auto arguments
        = launchArguments(plan, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);

    FlydslSdpaBwdMergedLaunchDump dump;
    dump.kernarg = kernargImage(arguments.merged);
    dump.block = plan.blockThreads;
    dump.grid = {plan.gridBlocks, 1, 1};
    dump.workspaceBytes = plan.workspace.totalBytes;
    if(arguments.reduce.has_value())
    {
        dump.reduces = true;
        dump.reduceSymbol = plan.reduceSymbol;
        dump.reduceKernarg = kernargImage(*arguments.reduce);
        dump.reduceBlock = REDUCE_THREADS;
        dump.reduceGrid = {plan.reduceGridX, plan.reduceGridY, 1};
    }
    if(arguments.seedUpload.has_value())
    {
        dump.seedUploaded = true;
        dump.seedCopiedFromDevice = arguments.seedUpload->seedSource != nullptr;
        if(!dump.seedCopiedFromDevice)
        {
            dump.seedWords.assign(arguments.seedUpload->words.begin(),
                                  arguments.seedUpload->words.end());
        }
    }
    return dump;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
