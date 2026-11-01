// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see FlydslSdpaCdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/LaunchKernelDefinition.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdNativeShared.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"

/**
 * @file FlydslSdpaBwdSplitNative.cpp
 * @brief The slotted flyDSL SDPA backward engine (RFC 0021): graph match, one kernel match and
 *        one dispatch handler per slot, and the one function that registers them.
 *
 * The backward pass is three launches, one per slot, in this order: `prep` (Delta = sum of
 * O * dO and LSE', the statistics with -inf turned into +inf, both into the shared
 * workspace region), `dkdv` (dK and dV) and `dq` (dQ and, on a bias graph, dBias; a broadcast
 * dBias lands in the dq slot's scratch and is summed by the dBias reduce kernel: a second
 * symbol of the same code object that the row's `reduce_symbol` names, or, on a row served
 * by the `dq_reduce` dispatch, the `dbias_reduce` launch of the row's `sources`). The seed
 * and offset of a dropout graph reach the dkdv and dq kernels as pointers: 64-bit device
 * tensors are read in place, every other form is written into the shared region by the prep
 * slot.
 *
 * The graph contract is the shared backward matcher's (FlydslSdpaBwdCommon.hpp) with FLOAT
 * accepted, padded varlen included: the graph's `seq_len_q` and `seq_len_kv` tensors are the
 * kernels' per-batch lengths. The engine adds what only it knows: a workspace whose regions
 * cannot be proven to fit is declined (SP-WS). Which slot rows serve a graph is the kernel
 * match of each slot, over the fields the slot's KMD carries: the launch grid follows the
 * row's `grid`, and a 16-bit bias outside the LDS staging contract (BI-8) declines the dq
 * rows that stage the bias. The graph match publishes the shared workspace as bound tokens
 * (`ws.*`); the handlers read them at prepare() and check them against the layout they
 * recompute.
 *
 * Launch arguments are a pure function of the plan prepare() resolves and the execute-time
 * buffers (launchArguments()), so one description of each kernel's argument list feeds the
 * launch, the signature the descriptor is checked against, and the tests' dumps.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

namespace bwd = flydsl_sdpa_bwd;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace native = flydsl_sdpa_bwd_native;
using native::GRID_YZ_LIMIT;
using native::INT32_LIMIT;
using native::toInt32;
using Slot = FlydslSdpaBwdSplitSlot;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view ENGINE_NAME = "hipkernel:flydsl_sdpa_bwd_split";
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_bwd_split.graph_match";
constexpr std::string_view PREP_KERNEL_MATCHER_SYMBOL
    = "hipkernel.flydsl_sdpa_bwd_split.prep.kernel_match";
constexpr std::string_view DKDV_KERNEL_MATCHER_SYMBOL
    = "hipkernel.flydsl_sdpa_bwd_split.dkdv.kernel_match";
constexpr std::string_view DQ_KERNEL_MATCHER_SYMBOL
    = "hipkernel.flydsl_sdpa_bwd_split.dq.kernel_match";
constexpr std::string_view PREP_DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_bwd_split.prep.dispatch";
constexpr std::string_view DKDV_DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_bwd_split.dkdv.dispatch";
constexpr std::string_view DQ_DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_bwd_split.dq.dispatch";
/// The dq slot's multi-launch dispatch: the rows that carry the dQ kernel and the dBias reduce
/// kernel as separate launches (`sources`).
constexpr std::string_view DQ_REDUCE_DISPATCH_SYMBOL
    = "hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch";

/// The launch names of a `dq_reduce` row's `sources`.
constexpr std::string_view DQ_LAUNCH = "dq";
constexpr std::string_view REDUCE_LAUNCH = "dbias_reduce";

// Kernel metadata fields, declared by each slot's KMD. Every slot carries the first group; the
// rest belong to the slot that names them.
constexpr std::string_view DTYPE_FIELD = "dtype";
/// The head-dim tile the object is built at.
constexpr std::string_view HEAD_DIM_FIELD = "head_dim";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view HAS_BIAS_FIELD = "has_bias";
/// The compiled dropout flag; with `dform` rt, dropout is a runtime branch on rt_flags.
constexpr std::string_view DROPOUT_FIELD = "dropout";
constexpr std::string_view DFORM_FIELD = "dform";
/// 0: serves head dims equal to the tile; 1: serves the dims above the floor and at most the
/// tile.
constexpr std::string_view PADDED_FIELD = "padded";
/// `g1` (floor 0), `tail` (the rung below the tile), or the floor itself.
constexpr std::string_view FLOOR_FIELD = "floor";
/// prep: 1 is the sink twin, which takes arguments a graph without a sink cannot give.
constexpr std::string_view SINK_FIELD = "sink";
/// dkdv: 1 packs the GQA group into the q-tile row axis; -1 leaves it unset. block_q is the
/// query tile of the dK/dV schedule, -1 where the policy decides.
constexpr std::string_view GQA_PACK_FIELD = "gqa_pack";
constexpr std::string_view BLOCK_Q_FIELD = "block_q";
/// dq: -1 leaves the dB store to the policy (it exists); 0 builds a dQ role without it.
constexpr std::string_view STORE_DB_FIELD = "store_db";
/// dq: the dBias reduce kernel's symbol in the same object; empty on an object without one.
constexpr std::string_view REDUCE_SYMBOL_FIELD = "reduce_symbol";
/// The launch geometry of the build: threads per block, the key tile height of dkdv and the
/// query tile height of dq. prep's follows from its tile.
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";
constexpr std::string_view BLOCK_KV_FIELD = "block_kv";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
/// How the object's launch grid maps the problem onto x, y and z; see gridOf.
constexpr std::string_view GRID_FIELD = "grid";
/// dq: "none" loads the bias directly; any other value stages the 16-bit bias through LDS
/// (see native::stagesBias).
constexpr std::string_view BIAS_STAGE_FIELD = "bias_stage";
constexpr std::string_view BIAS_STAGE_NONE = "none";

/// The `grid` values, one launch order per spelling: x, y, z, with the batch always on z.
constexpr std::string_view GRID_PREP = "qblocks_by_heads_by_seqs";
constexpr std::string_view GRID_DKDV_HEADS_FIRST = "kheads_by_kvblocks_by_seqs";
constexpr std::string_view GRID_DKDV_BLOCKS_FIRST = "kvblocks_by_kheads_by_seqs";
constexpr std::string_view GRID_DQ = "qheads_by_qblocks_by_seqs";

// What graph_match binds for kernel_match.
constexpr std::string_view CLASS_TOKEN = "flydsl_sdpa_bwd_split.class";
constexpr std::string_view DTYPE_TOKEN = "flydsl_sdpa_bwd_split.dtype";
constexpr std::string_view HEAD_DIM_QK_TOKEN = "flydsl_sdpa_bwd_split.head_dim_qk";
constexpr std::string_view HEAD_DIM_V_TOKEN = "flydsl_sdpa_bwd_split.head_dim_v";
constexpr std::string_view DROPOUT_TOKEN = "flydsl_sdpa_bwd_split.dropout";
constexpr std::string_view DBIAS_TOKEN = "flydsl_sdpa_bwd_split.dbias";
/// 1 for a 16-bit bias graph outside the LDS staging contract (BI-8): no row that stages the
/// bias serves it.
constexpr std::string_view BIAS_UNSTAGED_TOKEN = "flydsl_sdpa_bwd_split.bias_unstaged";
constexpr std::string_view GQA_PACK_TOKEN = "flydsl_sdpa_bwd_split.gqa_pack";
constexpr std::string_view BATCH_TOKEN = "flydsl_sdpa_bwd_split.batch";
constexpr std::string_view HEADS_TOKEN = "flydsl_sdpa_bwd_split.heads";
constexpr std::string_view KV_HEADS_TOKEN = "flydsl_sdpa_bwd_split.kv_heads";
constexpr std::string_view SEQ_Q_TOKEN = "flydsl_sdpa_bwd_split.seq_q";
constexpr std::string_view SEQ_KV_TOKEN = "flydsl_sdpa_bwd_split.seq_kv";
/// The reduce launch's grid y extent; 0 when the graph's dBias is not broadcast.
constexpr std::string_view REDUCE_PLANES_TOKEN = "flydsl_sdpa_bwd_split.reduce_planes";

// The shared workspace region's tokens (RFC 0021 section 6.1): `ws.bytes` names its size in
// the UED's `shared_workspace`; the runtime checks `ws.scratch_offset` against it.
constexpr std::string_view WS_BYTES_TOKEN = "ws.bytes";
constexpr std::string_view WS_SCRATCH_OFFSET_TOKEN = "ws.scratch_offset";
constexpr std::string_view WS_DELTA_OFFSET_TOKEN = "ws.delta.offset";
constexpr std::string_view WS_LSE_OFFSET_TOKEN = "ws.lse.offset";
/// Published only when the seed and offset are written into the workspace.
constexpr std::string_view WS_SEED_OFFSET_TOKEN = "ws.seed.offset";

/// The graph classes an object is built for: `causal` serves every masked graph, `bias`
/// every bias graph (masked graphs with a bias are declined by the matcher), `plain` the rest.
constexpr int64_t CLASS_PLAIN = 0;
constexpr int64_t CLASS_CAUSAL = 1;
constexpr int64_t CLASS_BIAS = 2;

/// The head-dim tiles `_sdpa_bwd_spec.LADDER` builds 16-bit objects at, and the fp32 ladder
/// (`FP32_LADDER`: no 384, so the t512 tail floor is 256).
constexpr std::array<int64_t, 10> LADDER{32, 64, 96, 128, 160, 192, 224, 256, 384, 512};
constexpr std::array<int64_t, 9> FP32_LADDER{32, 64, 96, 128, 160, 192, 224, 256, 512};

/// `varlen_bits` names the layout of the LSE: bit 16 is the (B, Sq, H) packing.
constexpr int32_t VARLEN_LSE_LAYOUT_TH = 1 << 16;
/// Padded varlen: dense (B, H, S, D) tensors with a length per batch on each side, the
/// layout's position implied by the batch index (`VARLEN_SEQUSED_CACHE_SIDE`, 0x04, for the
/// Q side in bits 0..7 and the K side in bits 8..15; the lengths go in `seqinfo_q0` /
/// `seqinfo_k0`). The prep kernel decodes the Q side only.
constexpr int32_t VARLEN_LENGTHS_BOTH_SIDES = 0x0404;

/// `rt_flags`: bit 0 serves a head dim below the tile, bit 1 turns dropout on.
constexpr int32_t RT_FLAG_PADDED = 1;
constexpr int32_t RT_FLAG_DROPOUT = 2;

/// The prep kernel's threads per block (`BLOCK_THREADS` in `fmha_bwd_prep_gfx950.py`).
constexpr int64_t PREP_BLOCK_THREADS = 256;
constexpr int64_t PREP_MAX_LANES_PER_ROW = 16;
constexpr int64_t VECTOR_BYTES = 16;

/// The largest query tile (block_q) of the dK/dV role among the shipped gqa_pack == 1
/// objects (see packedSpansFit); a row that packs a larger one is not served.
constexpr int64_t PACK_MAX_BLOCK_Q = 64;

[[noreturn]] void failBadParam(const std::string& message)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   "flydsl_sdpa_bwd_split: " + message);
}

constexpr std::string_view slotName(Slot slot)
{
    if(slot == Slot::PREP)
    {
        return "prep";
    }
    return slot == Slot::DKDV ? "dkdv" : "dq";
}

/// "<engine> declined the graph [<rule or cause>]: <detail>", at INFO: hipDNN reports a
/// declined graph to its caller only as "No engine configurations available".
void logDecline(std::string_view cause, const std::string& detail)
{
    HIPDNN_PLUGIN_LOG_INFO(ENGINE_NAME << " declined the graph [" << cause << "]: " << detail);
}

/// The kernel metadata spelling of a graph dtype, fp32 included.
std::optional<std::string> dtypeName(data_objects::DataType dataType)
{
    if(dataType == data_objects::DataType::FLOAT)
    {
        return std::string("fp32");
    }
    return flydsl_sdpa::kernelDataTypeName(dataType);
}

int64_t elementBytesOf(std::string_view dtype)
{
    return dtype == "fp32" ? 4 : 2;
}

/// The rung of @p dtype's ladder below @p tile, 0 below the first.
int64_t rungBelow(int64_t tile, std::string_view dtype)
{
    int64_t below = 0;
    const auto scan = [&](const auto& ladder) {
        for(const int64_t rung : ladder)
        {
            if(rung < tile)
            {
                below = rung;
            }
        }
    };
    if(dtype == "fp32")
    {
        scan(FP32_LADDER);
    }
    else
    {
        scan(LADDER);
    }
    return below;
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

native::WorkspaceRequest requestFor(const bwd::SdpaBwdProblem& problem)
{
    native::WorkspaceRequest request;
    request.rowStatistics = true;
    request.dbiasElementBytes = problem.elementBytes;
    return request;
}

/// The reduce kernel's vector: the columns one thread owns, a 16-byte access of the dtype.
int64_t reduceVectorOf(const bwd::SdpaBwdProblem& problem)
{
    return VECTOR_BYTES / static_cast<int64_t>(problem.elementBytes);
}

/// Why the regions of @p layout cannot be proven to fit, or empty when they can: every offset
/// and extent is an int64, each region starts at or after the end of the one before on a
/// 256-byte boundary, the shared regions end exactly at `sharedBytes` (`ws.bytes`), the
/// scratch offset is `sharedBytes` rounded up to 256, and the scratch region ends within the
/// total, all by checked arithmetic that does not trust the layout's own size_t sums.
std::string proveWorkspace(const native::WorkspaceLayout& layout)
{
    constexpr int64_t MAX = std::numeric_limits<int64_t>::max();
    constexpr auto ALIGNMENT = static_cast<int64_t>(native::WORKSPACE_ALIGNMENT);
    const auto asInt = [](size_t value) {
        return value <= static_cast<size_t>(MAX) ? std::optional<int64_t>(value) : std::nullopt;
    };
    const auto add = [](int64_t lhs, int64_t rhs) {
        return rhs > MAX - lhs ? std::optional<int64_t>{} : std::optional<int64_t>(lhs + rhs);
    };

    struct Named
    {
        const char* name;
        native::Region region;
    };
    std::vector<Named> shared;
    if(layout.uploadsSeed)
    {
        shared.push_back({"seed", layout.seed});
    }
    shared.push_back({"delta", layout.delta});
    shared.push_back({"lse", layout.lse});

    int64_t cursor = 0;
    for(const auto& [name, region] : shared)
    {
        const auto offset = asInt(region.offset);
        const auto bytes = asInt(region.bytes);
        const auto end = offset && bytes ? add(*offset, *bytes) : std::nullopt;
        if(!end)
        {
            return std::string("the ") + name + " region overflows int64";
        }
        if(*offset < cursor || *offset % ALIGNMENT != 0)
        {
            return std::string("the ") + name + " region at " + std::to_string(*offset)
                   + " overlaps its predecessor or is not 256-byte aligned";
        }
        cursor = *end;
    }
    const auto sharedBytes = asInt(layout.sharedBytes);
    if(!sharedBytes || *sharedBytes != cursor)
    {
        return "ws.bytes is not the end of the last shared region (" + std::to_string(cursor) + ")";
    }
    const auto rounded = add(cursor, ALIGNMENT - 1);
    if(!rounded || *rounded / ALIGNMENT * ALIGNMENT != static_cast<int64_t>(layout.scratchOffset))
    {
        return "ws.scratch_offset is not ws.bytes rounded up to 256";
    }
    if(layout.reduces)
    {
        const auto offset = asInt(layout.dbias.offset);
        const auto bytes = asInt(layout.dbias.bytes);
        const auto end = offset && bytes ? add(*offset, *bytes) : std::nullopt;
        if(!end || *offset != static_cast<int64_t>(layout.scratchOffset)
           || static_cast<size_t>(*end) != layout.totalBytes)
        {
            return "the dBias scratch region overflows int64 or leaves the workspace";
        }
    }
    else if(layout.totalBytes != layout.sharedBytes)
    {
        return "the workspace has bytes past its shared regions and no scratch region";
    }
    return {};
}

/// The workspace the graph needs, proven, or the reason it cannot be.
native::WorkspaceLayout provenLayout(const bwd::SdpaBwdProblem& problem, std::string& why)
{
    native::WorkspaceLayout layout;
    try
    {
        layout = native::workspaceLayout(problem, requestFor(problem));
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& exception)
    {
        why = exception.getMessage();
        return {};
    }
    why = proveWorkspace(layout);
    return layout;
}

// ---------------------------------------------------------------------------
// Graph match
// ---------------------------------------------------------------------------

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

/**
 * @brief Graph-scoped applicability: the shared backward matcher's contract with FLOAT
 *        accepted, with the proven shared workspace (SP-WS) and what every kernel_match needs
 *        bound.
 *
 * Each decline of a graph with an SDPA-backward node logs its rule or cause at INFO; any
 * other graph declines silently.
 */
std::optional<BoundTokens> flydslSdpaBwdSplitGraphMatches(const MatchContext& context)
{
    if(!native::hasSdpaBackwardNode(context))
    {
        return std::nullopt;
    }
    const auto matched = bwd::matchSdpaBackward(context, bwd::MatchOptions{true});
    if(const auto* decline = std::get_if<bwd::Decline>(&matched))
    {
        logDecline(decline->cause, decline->detail);
        return std::nullopt;
    }
    const auto& problem = std::get<bwd::SdpaBwdProblem>(matched);

    const auto dtype = dtypeName(problem.dtype);
    if(!dtype.has_value())
    {
        logDecline("data_type", "the matcher served a dtype this engine has no spelling for");
        return std::nullopt;
    }
    // BI-8: a 16-bit dQ object that stages the bias through LDS needs the layout the staging
    // contract describes; kernel_match declines those rows (`bias_stage` other than "none")
    // for a graph outside it. fp32 dQ objects load the bias directly.
    const bool biasUnstaged = problem.biasUid != bwd::NO_UID && problem.elementBytes == 2
                              && !native::stagesBias(problem);
    if(biasUnstaged)
    {
        HIPDNN_PLUGIN_LOG_INFO(ENGINE_NAME
                               << " [BI-8]: the bias layout is outside the LDS staging contract "
                                  "(16-byte base, strides multiples of 16 bytes, last row padded "
                                  "to a whole 16-byte chunk); only dq rows that load the bias "
                                  "directly can serve it");
    }

    // SP-WS: every region of the shared workspace and the dBias scratch is sized by checked
    // arithmetic and proven to lie within the workspace the runtime will allocate.
    std::string why;
    const auto layout = provenLayout(problem, why);
    if(!why.empty())
    {
        logDecline("SP-WS", why);
        return std::nullopt;
    }

    int64_t graphClass = CLASS_PLAIN;
    if(problem.biasUid != bwd::NO_UID)
    {
        graphClass = CLASS_BIAS;
    }
    else if(problem.masked)
    {
        graphClass = CLASS_CAUSAL;
    }

    BoundTokens bound;
    bound[std::string(CLASS_TOKEN)] = graphClass;
    bound[std::string(DTYPE_TOKEN)] = *dtype;
    bound[std::string(HEAD_DIM_QK_TOKEN)] = problem.headDimQk;
    bound[std::string(HEAD_DIM_V_TOKEN)] = problem.headDimV;
    bound[std::string(DROPOUT_TOKEN)] = int64_t{problem.dropoutP > 0.0F ? 1 : 0};
    bound[std::string(DBIAS_TOKEN)] = int64_t{problem.dbiasUid != bwd::NO_UID ? 1 : 0};
    bound[std::string(BIAS_UNSTAGED_TOKEN)] = int64_t{biasUnstaged ? 1 : 0};
    // The matcher guarantees Hq % Hk == 0, so more query heads than key heads is a group; a
    // pack object also needs every packed span inside the 32-bit buffer descriptor, else the
    // graph is served by the non-pack objects, which rebase per head.
    bound[std::string(GQA_PACK_TOKEN)]
        = int64_t{native::packedSpansFit(problem, PACK_MAX_BLOCK_Q) ? 1 : 0};
    bound[std::string(BATCH_TOKEN)] = problem.batch;
    bound[std::string(HEADS_TOKEN)] = problem.numHeads;
    bound[std::string(KV_HEADS_TOKEN)] = problem.numKvHeads;
    bound[std::string(SEQ_Q_TOKEN)] = problem.seqLenQ;
    bound[std::string(SEQ_KV_TOKEN)] = problem.seqLenKv;
    bound[std::string(REDUCE_PLANES_TOKEN)] = reducePlanes(problem);

    bound[std::string(WS_BYTES_TOKEN)] = static_cast<int64_t>(layout.sharedBytes);
    bound[std::string(WS_SCRATCH_OFFSET_TOKEN)] = static_cast<int64_t>(layout.scratchOffset);
    bound[std::string(WS_DELTA_OFFSET_TOKEN)] = static_cast<int64_t>(layout.delta.offset);
    bound[std::string(WS_LSE_OFFSET_TOKEN)] = static_cast<int64_t>(layout.lse.offset);
    if(layout.uploadsSeed)
    {
        bound[std::string(WS_SEED_OFFSET_TOKEN)] = static_cast<int64_t>(layout.seed.offset);
    }
    return bound;
}

// ---------------------------------------------------------------------------
// Kernel records
// ---------------------------------------------------------------------------

/// What the engine reads of a slot object's KMD record; a field a slot does not carry keeps
/// its default.
struct Row
{
    std::string dtype;
    int64_t tile = 0;
    int64_t causal = 0;
    int64_t hasBias = 0;
    int64_t dropout = 0;
    bool runtimeDropout = false;
    int64_t padded = 0;
    int64_t floor = 0;
    int64_t sink = -1;
    int64_t gqaPack = -1;
    int64_t blockQ = -1;
    int64_t storeDb = -1;
    std::string reduceSymbol;
    int64_t blockThreads = 0;
    int64_t blockKv = 0;
    int64_t blockM = 0;
    std::string grid;
    std::string biasStage;
};

/// The floor of an object: `g1` is 0, `tail` the rung below the tile, anything else the
/// number, whether the descriptor spells it as a string or an integer.
std::optional<int64_t> floorOf(const KernelDefinition& kernel, int64_t tile, std::string_view dtype)
{
    if(const auto number = flydsl_sdpa::integerMetadata(kernel, FLOOR_FIELD))
    {
        return number;
    }
    const auto text = flydsl_sdpa::stringMetadata(kernel, FLOOR_FIELD);
    if(!text)
    {
        return std::nullopt;
    }
    if(*text == "g1")
    {
        return 0;
    }
    if(*text == "tail")
    {
        return rungBelow(tile, dtype);
    }
    int64_t value = 0;
    const auto* end = text->data() + text->size();
    const auto parsed = std::from_chars(text->data(), end, value);
    if(parsed.ec != std::errc{} || parsed.ptr != end)
    {
        return std::nullopt;
    }
    return value;
}

/// The record of @p kernel for @p slot, or nullopt for one that lacks a field the slot reads
/// or holds it as another type: a malformed record declines rather than throwing out of a
/// match.
std::optional<Row> readRow(const KernelDefinition& kernel, Slot slot)
{
    using flydsl_sdpa::integerMetadata;
    using flydsl_sdpa::stringMetadata;
    Row row;
    const auto dtype = stringMetadata(kernel, DTYPE_FIELD);
    const auto tile = integerMetadata(kernel, HEAD_DIM_FIELD);
    const auto padded = integerMetadata(kernel, PADDED_FIELD);
    if(!dtype || !tile || !padded)
    {
        return std::nullopt;
    }
    const auto grid = stringMetadata(kernel, GRID_FIELD);
    if(!grid)
    {
        return std::nullopt;
    }
    row.dtype = *dtype;
    row.tile = *tile;
    row.padded = *padded;
    row.grid = *grid;
    const auto floor = floorOf(kernel, row.tile, row.dtype);
    if(!floor)
    {
        return std::nullopt;
    }
    row.floor = *floor;

    if(slot == Slot::PREP)
    {
        // prep is class- and dropout-free: one object per dtype, tile and padded.
        const auto sink = integerMetadata(kernel, SINK_FIELD);
        if(!sink)
        {
            return std::nullopt;
        }
        row.sink = *sink;
        return row;
    }

    const auto causal = integerMetadata(kernel, CAUSAL_FIELD);
    const auto hasBias = integerMetadata(kernel, HAS_BIAS_FIELD);
    const auto dropout = integerMetadata(kernel, DROPOUT_FIELD);
    const auto dform = stringMetadata(kernel, DFORM_FIELD);
    const auto blockThreads = integerMetadata(kernel, BLOCK_THREADS_FIELD);
    if(!causal || !hasBias || !dropout || !dform || !blockThreads)
    {
        return std::nullopt;
    }
    row.causal = *causal;
    row.hasBias = *hasBias;
    row.dropout = *dropout;
    row.runtimeDropout = *dform == "rt";
    row.blockThreads = *blockThreads;

    if(slot == Slot::DKDV)
    {
        const auto gqaPack = integerMetadata(kernel, GQA_PACK_FIELD);
        const auto blockKv = integerMetadata(kernel, BLOCK_KV_FIELD);
        if(!gqaPack || !blockKv)
        {
            return std::nullopt;
        }
        row.gqaPack = *gqaPack;
        row.blockKv = *blockKv;
        row.blockQ = integerMetadata(kernel, BLOCK_Q_FIELD).value_or(-1);
        return row;
    }

    const auto storeDb = integerMetadata(kernel, STORE_DB_FIELD);
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    const auto biasStage = stringMetadata(kernel, BIAS_STAGE_FIELD);
    if(!storeDb || !blockM || !biasStage)
    {
        return std::nullopt;
    }
    row.storeDb = *storeDb;
    row.biasStage = *biasStage;
    row.blockM = *blockM;
    // Read where the graph reduces; an object without a broadcast dBias never names one.
    row.reduceSymbol = stringMetadata(kernel, REDUCE_SYMBOL_FIELD).value_or(std::string());
    return row;
}

/// Whether an object of @p row's tile serves a graph of head dims @p headDimQk and
/// @p headDimV: an exact object serves dims equal to its tile; a padded one serves dims above
/// its floor and at most the tile. prep reads only the V width (`hdim_vo`).
bool servesHeadDims(const Row& row, Slot slot, int64_t headDimQk, int64_t headDimV)
{
    const int64_t low = slot == Slot::PREP ? headDimV : std::min(headDimQk, headDimV);
    const int64_t high = slot == Slot::PREP ? headDimV : std::max(headDimQk, headDimV);
    if(row.padded == 0)
    {
        return low == row.tile && high == row.tile;
    }
    return low > row.floor && high <= row.tile;
}

// ---------------------------------------------------------------------------
// Graph facts and launch geometry
// ---------------------------------------------------------------------------

/// What kernel_match and prepare read of the bound tokens.
struct GraphFacts
{
    std::string dtype;
    int64_t graphClass = 0;
    int64_t headDimQk = 0;
    int64_t headDimV = 0;
    bool dropout = false;
    bool dbias = false;
    bool biasUnstaged = false;
    bool gqaPack = false;
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
    const auto biasUnstaged = tryGetBoundInt(bound, BIAS_UNSTAGED_TOKEN);
    const auto gqaPack = tryGetBoundInt(bound, GQA_PACK_TOKEN);
    const auto batch = tryGetBoundInt(bound, BATCH_TOKEN);
    const auto numHeads = tryGetBoundInt(bound, HEADS_TOKEN);
    const auto numKvHeads = tryGetBoundInt(bound, KV_HEADS_TOKEN);
    const auto seqLenQ = tryGetBoundInt(bound, SEQ_Q_TOKEN);
    const auto seqLenKv = tryGetBoundInt(bound, SEQ_KV_TOKEN);
    const auto planes = tryGetBoundInt(bound, REDUCE_PLANES_TOKEN);
    if(!graphClass || !headDimQk || !headDimV || !dropout || !dbias || !biasUnstaged || !gqaPack
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
                      *biasUnstaged != 0,
                      *gqaPack != 0,
                      *batch,
                      *numHeads,
                      *numKvHeads,
                      *seqLenQ,
                      *seqLenKv,
                      *planes};
}

/// A launch grid.
struct Grid
{
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;
};

/// The prep launch's rows per block: its 256 threads put `gcd(tile / VEC, 16)` lanes on a row,
/// VEC being the elements of a 16-byte access (`prep_threads_per_row`).
int64_t prepRowsPerBlock(int64_t tile, std::string_view dtype)
{
    const int64_t vector = VECTOR_BYTES / elementBytesOf(dtype);
    const int64_t lanes = std::gcd(tile / vector, PREP_MAX_LANES_PER_ROW);
    return lanes > 0 ? PREP_BLOCK_THREADS / lanes : 0;
}

/// The threads per block of @p slot's launch, or 0 for a record without geometry.
int64_t blockThreadsOf(Slot slot, const Row& row)
{
    return slot == Slot::PREP ? PREP_BLOCK_THREADS : row.blockThreads;
}

/// The launch grid of @p slot, laid out as the row's `grid` field names (`_GRID_RULE` in
/// `_sdpa_bwd_cdna_build.py` and the gfx11 builders): x, y and z, the batch always last. prep
/// is (query row blocks, query heads, batch) and dq (query heads, query blocks, batch);
/// dkdv is (key heads, key blocks, batch) or (key blocks, key heads, batch). nullopt for a
/// `grid` the slot does not know, past the limits of an axis, or for a record without
/// geometry.
std::optional<Grid> gridOf(Slot slot, const GraphFacts& facts, const Row& row)
{
    if(blockThreadsOf(slot, row) <= 0)
    {
        return std::nullopt;
    }
    const auto blocks = [](int64_t extent, int64_t tileRows) -> std::optional<int64_t> {
        if(tileRows <= 0)
        {
            return std::nullopt;
        }
        return (extent + tileRows - 1) / tileRows;
    };
    std::optional<int64_t> x;
    std::optional<int64_t> y;
    if(slot == Slot::PREP)
    {
        if(row.grid != GRID_PREP)
        {
            return std::nullopt;
        }
        x = blocks(facts.seqLenQ, prepRowsPerBlock(row.tile, row.dtype));
        y = facts.numHeads;
    }
    else if(slot == Slot::DKDV)
    {
        const auto keyBlocks = blocks(facts.seqLenKv, row.blockKv);
        if(row.grid == GRID_DKDV_HEADS_FIRST)
        {
            x = facts.numKvHeads;
            y = keyBlocks;
        }
        else if(row.grid == GRID_DKDV_BLOCKS_FIRST)
        {
            x = keyBlocks;
            y = facts.numKvHeads;
        }
        else
        {
            return std::nullopt;
        }
    }
    else
    {
        if(row.grid != GRID_DQ)
        {
            return std::nullopt;
        }
        x = facts.numHeads;
        y = blocks(facts.seqLenQ, row.blockM);
    }
    // x may be as wide as int32; y and z are limited to 65535.
    if(!x || !y || *x > INT32_LIMIT || *y > GRID_YZ_LIMIT || facts.batch > GRID_YZ_LIMIT)
    {
        return std::nullopt;
    }
    return Grid{static_cast<unsigned int>(*x),
                static_cast<unsigned int>(*y),
                static_cast<unsigned int>(facts.batch)};
}

// ---------------------------------------------------------------------------
// Kernel match
// ---------------------------------------------------------------------------

/**
 * @brief Kernel-scoped applicability of one slot's object: dtype, head-dim tile, and what the
 *        slot's kernel is built for (class, dropout, packing, dB store, bias staging, the
 *        reduce kernel), against the graph, with a launch grid inside its limits.
 */
template <Slot SLOT>
bool flydslSdpaBwdSplitKernelMatches(const MatchContext& /*context*/,
                                     const BoundTokens& bound,
                                     const KernelDefinition& kernel)
{
    const auto facts = readFacts(bound);
    const auto row = readRow(kernel, SLOT);
    if(!facts || !row || row->dtype != facts->dtype)
    {
        return false;
    }
    if(!servesHeadDims(*row, SLOT, facts->headDimQk, facts->headDimV))
    {
        return false;
    }
    if constexpr(SLOT == Slot::PREP)
    {
        // The sink twin takes arguments no graph here supplies.
        if(row->sink == 1)
        {
            return false;
        }
    }
    else
    {
        // A runtime-dropout object (dform rt) serves both states through rt_flags; a baked
        // one only its own.
        if(!row->runtimeDropout && (row->dropout != 0) != facts->dropout)
        {
            return false;
        }
        if(row->causal != (facts->graphClass == CLASS_CAUSAL ? 1 : 0)
           || row->hasBias != (facts->graphClass == CLASS_BIAS ? 1 : 0))
        {
            return false;
        }
    }
    if constexpr(SLOT == Slot::DKDV)
    {
        // A packed object serves dense GQA whose packed spans fit, at a query tile the span
        // bound covers.
        if(row->gqaPack == 1 && (!facts->gqaPack || row->blockQ > PACK_MAX_BLOCK_Q))
        {
            return false;
        }
    }
    if constexpr(SLOT == Slot::DQ)
    {
        // A row that stages the bias (BI-8) cannot serve a layout outside the staging contract.
        if(facts->biasUnstaged && row->biasStage != BIAS_STAGE_NONE)
        {
            return false;
        }
        // store_db 0 builds the dQ role without its dB store.
        if(facts->dbias && row->storeDb == 0)
        {
            return false;
        }
        // A broadcast dBias is summed by the reduce kernel the row names, one plane to each
        // grid row.
        if(facts->reducePlanes > 0
           && (row->reduceSymbol.empty() || facts->reducePlanes > GRID_YZ_LIMIT))
        {
            return false;
        }
    }
    return gridOf(SLOT, *facts, *row).has_value();
}

// ---------------------------------------------------------------------------
// Launch values
// ---------------------------------------------------------------------------

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
struct SplitPlan
{
    Slot slot = Slot::PREP;
    bwd::SdpaBwdProblem problem;
    Uids uids;
    unsigned int blockThreads = 0;
    Grid grid;
    native::Window window;
    int32_t dropoutThreshold = 0;
    float dropoutScale = 1.0F;
    int32_t runtimeFlags = 0;
    int32_t varlenBits = 0;
    native::WorkspaceLayout workspace;
    native::ScalarOperands scalars;
    /// The reduce kernel's symbol and launch, when the dq slot's graph has a broadcast dBias.
    std::string reduceSymbol;
    native::ReduceGrid reduceGrid;
};

bwd::SdpaBwdProblem problemOf(const MatchContext& context)
{
    auto matched = bwd::matchSdpaBackward(context, bwd::MatchOptions{true});
    if(auto* problem = std::get_if<bwd::SdpaBwdProblem>(&matched))
    {
        return *problem;
    }
    const auto& decline = std::get<bwd::Decline>(matched);
    failBadParam("the graph no longer matches [" + decline.cause + "]: " + decline.detail);
}

/// Checks the tokens graph_match bound against the layout recomputed from the graph: a
/// handler addresses the workspace by the layout, and the runtime sized and checked it by the
/// tokens, so a disagreement is a bug, not a graph.
void requireTokens(const BoundTokens& bound, const native::WorkspaceLayout& layout)
{
    const auto require = [&](std::string_view token, size_t expected) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value || *value < 0 || static_cast<size_t>(*value) != expected)
        {
            failBadParam("workspace token '" + std::string(token) + "' is "
                         + (value ? std::to_string(*value) : std::string("missing"))
                         + ", the layout of the graph has " + std::to_string(expected));
        }
    };
    require(WS_BYTES_TOKEN, layout.sharedBytes);
    require(WS_SCRATCH_OFFSET_TOKEN, layout.scratchOffset);
    require(WS_DELTA_OFFSET_TOKEN, layout.delta.offset);
    require(WS_LSE_OFFSET_TOKEN, layout.lse.offset);
    if(layout.uploadsSeed)
    {
        require(WS_SEED_OFFSET_TOKEN, layout.seed.offset);
    }
}

SplitPlan planFor(Slot slot,
                  const MatchContext& context,
                  const BoundTokens& bound,
                  const KernelDefinition& kernel)
{
    SplitPlan plan;
    plan.slot = slot;
    plan.problem = problemOf(context);
    const auto& problem = plan.problem;

    const auto row = readRow(kernel, slot);
    if(!row)
    {
        failBadParam("kernel '" + kernel.name + "' lacks a field of the "
                     + std::string(slotName(slot)) + " record");
    }
    const auto facts = readFacts(bound);
    if(!facts)
    {
        failBadParam("the bound tokens of the graph are incomplete");
    }
    const auto grid = gridOf(slot, *facts, *row);
    if(!grid)
    {
        failBadParam("launch grid exceeds its limits for kernel '" + kernel.name + "'");
    }
    plan.blockThreads = static_cast<unsigned int>(blockThreadsOf(slot, *row));
    plan.grid = *grid;

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

    plan.window = native::windowFor(problem);
    if(problem.dropoutP > 0.0F)
    {
        const auto p = static_cast<double>(problem.dropoutP);
        plan.dropoutThreshold = native::dropoutThreshold(p);
        plan.dropoutScale = static_cast<float>(1.0 / (1.0 - p));
    }
    const bool paddedHead = problem.headDimQk != row->tile || problem.headDimV != row->tile;
    plan.runtimeFlags
        = (paddedHead ? RT_FLAG_PADDED : 0) | (problem.dropoutP > 0.0F ? RT_FLAG_DROPOUT : 0);
    plan.varlenBits = (problem.statsLayout == bwd::StatsLayout::TH ? VARLEN_LSE_LAYOUT_TH : 0)
                      | (problem.paddingMask ? VARLEN_LENGTHS_BOTH_SIDES : 0);

    plan.scalars = native::scalarOperandsOf(context.graph.getTensorMap(), problem);

    plan.workspace = native::workspaceLayout(problem, requestFor(problem));
    requireTokens(bound, plan.workspace);
    if(slot == Slot::DQ && plan.workspace.reduces)
    {
        if(row->reduceSymbol.empty())
        {
            failBadParam("kernel '" + kernel.name
                         + "' serves a broadcast dBias but names no reduce kernel");
        }
        plan.reduceSymbol = row->reduceSymbol;
        plan.reduceGrid = native::reduceGridOf(problem, reduceVectorOf(problem));
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Kernel argument lists
// ---------------------------------------------------------------------------

/**
 * @brief The prep kernel's arguments, in parameter order (`_PARAMS["prep"]` in
 *        `_sdpa_bwd_cdna_build.py`): 18 arguments.
 *
 * A pointer is 8 bytes, a count or flag 4 and a stride 8 (i64), each at its natural offset.
 * tie() is the one listing: launch passes it through, the signature reads its types, and the
 * tests' dumps serialize it.
 */
struct PrepKernargs
{
    void* o = nullptr;
    void* dO = nullptr;
    void* lse = nullptr;
    void* lseWs = nullptr;
    void* delta = nullptr;
    void* seqinfoQ0 = nullptr;
    void* seqinfoQ1 = nullptr;
    int32_t varlenBits = 0;
    int32_t numSeqlens = 0;
    int32_t maxSeqlenQ = 0;
    int32_t numHeadQ = 0;
    int32_t hdimVo = 0;
    bwd::Strides strideO;
    bwd::Strides strideDO;

    auto tie() const
    {
        return std::tie(o,
                        dO,
                        lse,
                        lseWs,
                        delta,
                        seqinfoQ0,
                        seqinfoQ1,
                        varlenBits,
                        numSeqlens,
                        maxSeqlenQ,
                        numHeadQ,
                        hdimVo,
                        strideO.batch,
                        strideO.head,
                        strideO.seq,
                        strideDO.batch,
                        strideDO.head,
                        strideDO.seq);
    }
};

/// The dK/dV kernel's arguments (`_PARAMS["dkdv"]`): 51 arguments.
struct DkdvKernargs
{
    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* bias = nullptr;
    void* dO = nullptr;
    void* dK = nullptr;
    void* dV = nullptr;
    void* lse = nullptr;
    void* delta = nullptr;
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
    bwd::Strides strideDO;
    bwd::Strides strideDK;
    bwd::Strides strideDV;
    bwd::Strides strideB;
    int32_t rtFlags = 0;

    auto tie() const
    {
        return std::tie(q,
                        k,
                        v,
                        bias,
                        dO,
                        dK,
                        dV,
                        lse,
                        delta,
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
                        strideDO.batch,
                        strideDO.head,
                        strideDO.seq,
                        strideDK.batch,
                        strideDK.head,
                        strideDK.seq,
                        strideDV.batch,
                        strideDV.head,
                        strideDV.seq,
                        strideB.batch,
                        strideB.head,
                        strideB.seq,
                        rtFlags);
    }
};

/// The dQ kernel's arguments (`_PARAMS["dq"]`): 51 arguments.
struct DqKernargs
{
    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* bias = nullptr;
    void* dO = nullptr;
    void* dQ = nullptr;
    void* dBias = nullptr;
    void* lse = nullptr;
    void* delta = nullptr;
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
    bwd::Strides strideDO;
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
                        dO,
                        dQ,
                        dBias,
                        lse,
                        delta,
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
                        strideDO.batch,
                        strideDO.head,
                        strideDO.seq,
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

const std::vector<KernelArgument>& signatureOfSlot(Slot slot)
{
    static const auto s_prep = native::signatureOf<PrepKernargs>();
    static const auto s_dkdv = native::signatureOf<DkdvKernargs>();
    static const auto s_dq = native::signatureOf<DqKernargs>();
    if(slot == Slot::PREP)
    {
        return s_prep;
    }
    return slot == Slot::DKDV ? s_dkdv : s_dq;
}

// ---------------------------------------------------------------------------
// Launch arguments
// ---------------------------------------------------------------------------

struct LaunchArguments
{
    std::variant<PrepKernargs, DkdvKernargs, DqKernargs> kernel;
    std::optional<native::ReduceKernargs> reduce;
    /// Set on the prep launch only: it writes the seed region the other slots read.
    std::optional<native::SeedUpload> seedUpload;
};

/// Every argument of the launch, from the plan and the execute-time buffers; touches no
/// device. @p workspace is the caller's unshifted base: every region is at the offset
/// plan.workspace says.
LaunchArguments launchArguments(const SplitPlan& plan,
                                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                uint32_t numDeviceBuffers,
                                void* workspace)
{
    const auto& problem = plan.problem;
    const auto& layout = plan.workspace;
    native::requireWorkspace(layout, workspace);
    const auto buffer = [&](int64_t uid) {
        return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
    };
    const int64_t vector
        = bwd::detail::REQUIRED_ALIGNMENT_BYTES / static_cast<int64_t>(problem.elementBytes);
    const auto pitch = [&](const bwd::Strides& strides, int64_t rows, int64_t headDim) {
        return native::rowPitchStrides(strides, rows, headDim, vector);
    };

    // A tensor the graph does not use is passed as O, as the harness does.
    void* const unused = buffer(plan.uids.o);
    void* const delta = native::inWorkspace(workspace, layout.delta.offset);
    void* const lseWorkspace = native::inWorkspace(workspace, layout.lse.offset);

    LaunchArguments arguments;
    if(plan.slot == Slot::PREP)
    {
        PrepKernargs prep;
        prep.o = unused;
        prep.dO = buffer(plan.uids.dO);
        // The raw statistics of the forward: the kernel writes LSE' with -inf turned into +inf.
        prep.lse = buffer(plan.uids.stats);
        prep.lseWs = lseWorkspace;
        prep.delta = delta;
        // Unread slots stay O, as the harness passes them; a padded graph's lengths are its
        // `seq_len_q` / `seq_len_kv` tensors, and nothing is packed (num_seqlens stays 0).
        prep.seqinfoQ0 = problem.paddingMask ? buffer(problem.seqLenQUid) : unused;
        prep.seqinfoQ1 = unused;
        prep.varlenBits = plan.varlenBits;
        prep.numSeqlens = 0;
        prep.maxSeqlenQ = toInt32(problem.seqLenQ, "seqlen_q");
        prep.numHeadQ = toInt32(problem.numHeads, "num_head_q");
        prep.hdimVo = toInt32(problem.headDimV, "hdim_vo");
        prep.strideO = pitch(problem.o, problem.seqLenQ, problem.headDimV);
        prep.strideDO = pitch(problem.dO, problem.seqLenQ, problem.headDimV);
        arguments.kernel = prep;
        const auto seed = native::bindSeed(
            problem, layout, plan.scalars, deviceBuffers, numDeviceBuffers, workspace, unused);
        arguments.seedUpload = seed.upload;
        return arguments;
    }

    const auto seed = native::bindSeed(
        problem, layout, plan.scalars, deviceBuffers, numDeviceBuffers, workspace, unused);
    void* const bias = problem.biasUid != bwd::NO_UID ? buffer(problem.biasUid) : unused;
    const bwd::Strides strideB
        = problem.biasUid != bwd::NO_UID ? problem.biasStrides : bwd::Strides{};
    const int32_t idropoutP = problem.seedForm != bwd::SeedForm::NONE ? plan.dropoutThreshold : 0;
    const float dropoutScale = problem.seedForm != bwd::SeedForm::NONE ? plan.dropoutScale : 1.0F;
    const float smScale
        = native::resolveScale(problem, plan.scalars, deviceBuffers, numDeviceBuffers);
    const auto strideQ = pitch(problem.q, problem.seqLenQ, problem.headDimQk);
    const auto strideK = pitch(problem.k, problem.seqLenKv, problem.headDimQk);
    const auto strideV = pitch(problem.v, problem.seqLenKv, problem.headDimV);
    const auto strideDO = pitch(problem.dO, problem.seqLenQ, problem.headDimV);

    if(plan.slot == Slot::DKDV)
    {
        DkdvKernargs d;
        d.q = buffer(plan.uids.q);
        d.k = buffer(plan.uids.k);
        d.v = buffer(plan.uids.v);
        d.bias = bias;
        d.dO = buffer(plan.uids.dO);
        d.dK = buffer(plan.uids.dK);
        d.dV = buffer(plan.uids.dV);
        d.lse = lseWorkspace;
        d.delta = delta;
        d.seqinfoQ0 = problem.paddingMask ? buffer(problem.seqLenQUid) : unused;
        d.seqinfoK0 = problem.paddingMask ? buffer(problem.seqLenKvUid) : unused;
        d.seqinfoQ1 = d.seqinfoK1 = unused;
        d.varlenBits = plan.varlenBits;
        d.numSeqlens = 0;
        d.maxSeqlenQ = toInt32(problem.seqLenQ, "seqlen_q");
        d.maxSeqlenK = toInt32(problem.seqLenKv, "seqlen_k");
        d.windowLeft = plan.window.left;
        d.windowRight = plan.window.right;
        d.philoxSeed = seed.seed;
        d.philoxOffset1 = seed.offset;
        d.philoxOffset2 = 0;
        d.idropoutP = idropoutP;
        d.dropoutScale = dropoutScale;
        d.numHeadQ = toInt32(problem.numHeads, "num_head_q");
        d.numHeadK = toInt32(problem.numKvHeads, "num_head_k");
        d.hdimQk = toInt32(problem.headDimQk, "hdim_qk");
        d.hdimVo = toInt32(problem.headDimV, "hdim_vo");
        d.smScale = smScale;
        d.strideQ = strideQ;
        d.strideK = strideK;
        d.strideV = strideV;
        d.strideDO = strideDO;
        d.strideDK = pitch(problem.dK, problem.seqLenKv, problem.headDimQk);
        d.strideDV = pitch(problem.dV, problem.seqLenKv, problem.headDimV);
        d.strideB = strideB;
        d.rtFlags = plan.runtimeFlags;
        arguments.kernel = d;
        return arguments;
    }

    DqKernargs d;
    d.q = buffer(plan.uids.q);
    d.k = buffer(plan.uids.k);
    d.v = buffer(plan.uids.v);
    d.bias = bias;
    d.dO = buffer(plan.uids.dO);
    d.dQ = buffer(plan.uids.dQ);
    d.lse = lseWorkspace;
    d.delta = delta;
    d.seqinfoQ0 = problem.paddingMask ? buffer(problem.seqLenQUid) : unused;
    d.seqinfoK0 = problem.paddingMask ? buffer(problem.seqLenKvUid) : unused;
    d.seqinfoQ1 = d.seqinfoK1 = unused;
    d.varlenBits = plan.varlenBits;
    d.numSeqlens = 0;
    d.maxSeqlenQ = toInt32(problem.seqLenQ, "seqlen_q");
    d.maxSeqlenK = toInt32(problem.seqLenKv, "seqlen_k");
    d.windowLeft = plan.window.left;
    d.windowRight = plan.window.right;
    d.philoxSeed = seed.seed;
    d.philoxOffset1 = seed.offset;
    d.philoxOffset2 = 0;
    d.idropoutP = idropoutP;
    d.dropoutScale = dropoutScale;
    d.numHeadQ = toInt32(problem.numHeads, "num_head_q");
    d.numHeadK = toInt32(problem.numKvHeads, "num_head_k");
    d.hdimQk = toInt32(problem.headDimQk, "hdim_qk");
    d.hdimVo = toInt32(problem.headDimV, "hdim_vo");
    d.smScale = smScale;
    d.strideQ = strideQ;
    d.strideK = strideK;
    d.strideV = strideV;
    d.strideDO = strideDO;
    d.strideDQ = pitch(problem.dQ, problem.seqLenQ, problem.headDimQk);
    d.strideB = strideB;
    d.rtFlags = plan.runtimeFlags;

    // dBias: none requested leaves DB as O over zero strides, with which the kernel drops
    // every store; a broadcast one lands in the scratch region and is summed after; any other
    // is written in place.
    d.dBias = unused;
    if(problem.dbiasUid != bwd::NO_UID)
    {
        if(layout.reduces)
        {
            d.dBias = native::inWorkspace(workspace, layout.dbias.offset);
            d.strideDB = layout.dbiasStrides;
            arguments.reduce
                = native::reduceKernargsOf(problem, layout, d.dBias, buffer(problem.dbiasUid));
        }
        else
        {
            d.dBias = buffer(problem.dbiasUid);
            d.strideDB = problem.dbiasStrides;
            // The kernel bounds dB by Sq * stride_seq_q, so a zero row stride (a unit Sq)
            // drops every store: the row pitch stands in.
            if(d.strideDB.seq == 0)
            {
                d.strideDB.seq = problem.seqLenKv;
            }
        }
    }
    arguments.kernel = d;
    return arguments;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The slot's kernel, the reduce kernel when the graph needs one, and the plan.
class PreparedFlydslSdpaBwdSplit : public PreparedDispatch
{
public:
    PreparedFlydslSdpaBwdSplit(SplitPlan plan,
                               IngestorKernelCode code,
                               std::optional<IngestorKernelCode> reduceCode)
        : _plan(std::move(plan))
        , _code(std::move(code))
        , _reduce(std::move(reduceCode))
    {
    }

    const SplitPlan& plan() const
    {
        return _plan;
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    compilation::IRunnableKernel& reduceForStream(hipStream_t stream) const
    {
        return _reduce->kernelForStream(stream);
    }

private:
    SplitPlan _plan;
    IngestorKernelCode _code;
    std::optional<IngestorKernelCode> _reduce;
};

/**
 * @brief The native dispatch behind one slot's UDD: reports the slot's scratch, and launches
 *        the slot's kernel; prep first uploads the seed, dq follows its kernel with the reduce
 *        kernel when the graph's dBias is broadcast.
 *
 * With @p SOURCES (dq only) the dispatch is multi-launch: every row carries the dQ kernel
 * and the dBias reduce kernel as the launches `dq` and `dbias_reduce` of its `sources`, and
 * each is built from its own entry. Otherwise the reduce kernel is a second symbol of the
 * row's one code object, the symbol its `reduce_symbol` names. Arguments, workspace and
 * scratch are the same either way.
 *
 * Split handlers do not save plans: supportsSaving() is the SDK default, false.
 */
template <Slot SLOT, bool SOURCES = false>
class FlydslSdpaBwdSplitDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    FlydslSdpaBwdSplitDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                      const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
        static_assert(!SOURCES || SLOT == Slot::DQ, "only dq launches the reduce kernel");
    }

    std::set<std::string> launchNames() const override
    {
        if constexpr(SOURCES)
        {
            return {std::string(DQ_LAUNCH), std::string(REDUCE_LAUNCH)};
        }
        return {};
    }

    /// The scratch of this slot only; the shared region is the graph match's `ws.bytes`.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        if constexpr(SLOT == Slot::DQ)
        {
            const auto problem = problemOf(context);
            const auto layout = native::workspaceLayout(problem, requestFor(problem));
            return layout.reduces ? layout.dbias.bytes : 0;
        }
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        auto plan = planFor(SLOT, context, bound, kernel);

        // Every launch's definition is resolved, and the reduce launch's argument list
        // checked, before any code object loads, so a row that lacks a launch or disagrees on
        // the reduce ABI fails on the host whatever the state of the archive.
        const bool reduces = !plan.reduceSymbol.empty();
        std::optional<KernelDefinition> dqLaunch;
        std::optional<KernelDefinition> reduceKernel;
        if constexpr(SOURCES)
        {
            dqLaunch = launchKernelDefinition(kernel, std::string(DQ_LAUNCH));
            if(reduces)
            {
                reduceKernel = launchKernelDefinition(kernel, std::string(REDUCE_LAUNCH));
                requireSignatureMatch(reduceKernel->source.signature,
                                      native::reduceSignature(),
                                      reduceKernel->source.symbol,
                                      describeDescriptor("kernel", kernel.name, kernel.kernelId));
            }
        }
        else if(reduces)
        {
            // The reduce kernel is the same code object under another symbol: the same
            // archive entry, with that symbol and its own argument list.
            reduceKernel = kernel;
            reduceKernel->source.symbol = plan.reduceSymbol;
            reduceKernel->source.signature = native::reduceSignature();
        }
        const KernelDefinition& slotKernel = dqLaunch.has_value() ? *dqLaunch : kernel;

        // Built from the arch alone; see FlydslRmsNormNative.cpp's prepare().
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, slotKernel, options, signatureOfSlot(SLOT));
        code.setBlockSize(plan.blockThreads, 1, 1);
        code.setGridSize(plan.grid.x, plan.grid.y, plan.grid.z);

        std::optional<IngestorKernelCode> reduceCode;
        if(reduceKernel.has_value())
        {
            reduceCode.emplace(buildIngestorKernelCode(_kernelCompiler,
                                                       _kpackLoader,
                                                       context,
                                                       *reduceKernel,
                                                       options,
                                                       native::reduceSignature()));
            reduceCode->setBlockSize(native::REDUCE_THREADS, 1, 1);
            reduceCode->setGridSize(plan.reduceGrid.x, plan.reduceGrid.y, 1);
        }
        return std::make_unique<PreparedFlydslSdpaBwdSplit>(
            std::move(plan), std::move(code), std::move(reduceCode));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedSplit = dynamic_cast<const PreparedFlydslSdpaBwdSplit&>(prepared);
        const auto arguments
            = launchArguments(preparedSplit.plan(), deviceBuffers, numDeviceBuffers, workspace);
        const auto stream = handle.getStream();

        if(arguments.seedUpload.has_value())
        {
            native::uploadSeed(*arguments.seedUpload, workspace, stream);
        }
        // Changing an argument list means changing its struct with it: the signature the
        // descriptor is checked against is read from the same tie().
        std::visit(
            [&](const auto& kernargs) {
                std::apply(
                    [&](const auto&... argument) {
                        preparedSplit.kernelForStream(stream).launch(stream, argument...);
                    },
                    kernargs.tie());
            },
            arguments.kernel);
        if(arguments.reduce.has_value())
        {
            std::apply(
                [&](const auto&... argument) {
                    preparedSplit.reduceForStream(stream).launch(stream, argument...);
                },
                arguments.reduce->tie());
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& flydslSdpaBwdSplitKpackModuleCache()
{
    // This pack's own; see flydslSdpaCdnaKpackModuleCache().
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

/// Process-lifetime: the registry holds non-owning pointers to the four handlers, which share
/// one compiler, one loader and the pack's module cache.
template <Slot SLOT, bool SOURCES = false>
const FlydslSdpaBwdSplitDispatchHandler<SLOT, SOURCES>& flydslSdpaBwdSplitDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslSdpaBwdSplitKpackModuleCache());
    static const FlydslSdpaBwdSplitDispatchHandler<SLOT, SOURCES> s_dispatchHandler(
        s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void resetFlydslSdpaBwdSplitModuleCache()
{
    flydslSdpaBwdSplitKpackModuleCache().clear();
}

void registerFlydslSdpaBwdSplitSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslSdpaBwdSplitGraphMatches);
    scope.add(std::string(PREP_KERNEL_MATCHER_SYMBOL),
              &flydslSdpaBwdSplitKernelMatches<Slot::PREP>);
    scope.add(std::string(DKDV_KERNEL_MATCHER_SYMBOL),
              &flydslSdpaBwdSplitKernelMatches<Slot::DKDV>);
    scope.add(std::string(DQ_KERNEL_MATCHER_SYMBOL), &flydslSdpaBwdSplitKernelMatches<Slot::DQ>);
    scope.add(std::string(PREP_DISPATCH_SYMBOL), &flydslSdpaBwdSplitDispatchHandler<Slot::PREP>());
    scope.add(std::string(DKDV_DISPATCH_SYMBOL), &flydslSdpaBwdSplitDispatchHandler<Slot::DKDV>());
    scope.add(std::string(DQ_DISPATCH_SYMBOL), &flydslSdpaBwdSplitDispatchHandler<Slot::DQ>());
    scope.add(std::string(DQ_REDUCE_DISPATCH_SYMBOL),
              &flydslSdpaBwdSplitDispatchHandler<Slot::DQ, true>());
}

FlydslSdpaBwdSplitLaunchDump describeFlydslSdpaBwdSplitLaunchForTesting(
    Slot slot,
    const MatchContext& context,
    const BoundTokens& bound,
    const KernelDefinition& kernel,
    const std::vector<hipdnnPluginDeviceBuffer_t>& buffers,
    void* workspace)
{
    const auto plan = planFor(slot, context, bound, kernel);
    const auto arguments
        = launchArguments(plan, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);

    FlydslSdpaBwdSplitLaunchDump dump;
    std::visit([&](const auto& kernargs) { dump.kernarg = native::kernargImage(kernargs); },
               arguments.kernel);
    dump.block = plan.blockThreads;
    dump.grid = {plan.grid.x, plan.grid.y, plan.grid.z};
    dump.scratchBytes = slot == Slot::DQ && plan.workspace.reduces ? plan.workspace.dbias.bytes : 0;
    if(arguments.reduce.has_value())
    {
        dump.reduces = true;
        dump.reduceSymbol = plan.reduceSymbol;
        dump.reduceKernarg = native::kernargImage(*arguments.reduce);
        dump.reduceBlock = native::REDUCE_THREADS;
        dump.reduceGrid = {plan.reduceGrid.x, plan.reduceGrid.y, 1};
    }
    if(arguments.seedUpload.has_value())
    {
        dump.seedUploaded = true;
        dump.seedCopiedFromDevice = arguments.seedUpload->seedSource != nullptr;
        dump.seedSource = arguments.seedUpload->seedSource;
        dump.offsetSource = arguments.seedUpload->offsetSource;
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
