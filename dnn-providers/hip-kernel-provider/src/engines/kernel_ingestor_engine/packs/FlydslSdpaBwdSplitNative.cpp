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
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaDropout.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"

/**
 * @file FlydslSdpaBwdSplitNative.cpp
 * @brief The slotted flyDSL SDPA backward engine (RFC 0021): graph match, one kernel match and
 *        one dispatch handler per slot, and the one function that registers them.
 *
 * The backward pass is three launches, one per slot, in this order: `prep` (Delta = sum of
 * O * dO and LSE', the statistics with -inf turned into +inf, both into the shared
 * workspace region), `dkdv` (dK and dV) and `dq` (dQ and, on a bias graph, dBias; a broadcast
 * dBias lands in the dq slot's scratch and is summed by the dBias reduce kernel, the
 * `dbias_reduce` launch of the `sources` of a row served by the `dq_reduce` dispatch: a
 * standalone object on gfx11, a second kernel of the row's own object on gfx950). The seed
 * and offset of a dropout graph reach the dkdv and dq kernels as pointers: 64-bit device
 * tensors are read in place, every other form is written into the shared region by the prep
 * slot.
 *
 * The graph contract is the shared backward matcher's (FlydslSdpaBwdCommon.hpp) with FLOAT
 * accepted, padded varlen included: the graph's `seq_len_q` and `seq_len_kv` tensors are the
 * kernels' per-batch lengths. The engine adds what only it knows: a workspace whose regions
 * cannot be proven to fit is declined (SP-WS). Which slot rows serve a graph is the kernel
 * match of each slot, over the fields the slot's KMD carries: the launch grid follows the
 * row's `grid`, a dq row serves a bias graph by the chunk-contract rule of its
 * `bias_stage_resolved` (see flydslSdpaBwdSplitKernelMatches), and a row serves a graph by its
 * `bias_f32` and `u2_role`. The graph match publishes the shared workspace as bound tokens
 * (`ws.*`); the handlers read them at prepare() and check them against the layout they
 * recompute.
 *
 * The dq slot's launches are the dQ kernel, the dBias reduce when the graph's dBias is
 * broadcast, and, when the graph has padding_mask and at least one dense gradient output, the
 * tail-zero launch (`tailzero`): it zeroes the rows of those outputs past `seq_len`, so a
 * padded dense gradient is zero beyond its lengths. A graph whose V heads differ from K's
 * (`u2_role` `dq` on the dq row, `dk` on the dkdv row) takes the `num_head_v` variants: the
 * dkdv row's `dkdv_u2` dispatch launches the dK-only object over the K heads and then the
 * dV-only object (`dvonly`) over the V heads, and the dq row's tail-zero launch is the V-head
 * object.
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
using native::BLOCK_KV_FIELD;
using native::BLOCK_M_FIELD;
using native::BLOCK_Q_FIELD;
using native::BLOCK_THREADS_FIELD;
using native::CAUSAL_FIELD;
using native::CLASS_BIAS;
using native::CLASS_CAUSAL;
using native::CLASS_PLAIN;
using native::DROPOUT_FIELD;
using native::DTYPE_FIELD;
using native::GQA_PACK_FIELD;
using native::GRID_YZ_LIMIT;
using native::HAS_BIAS_FIELD;
using native::HEAD_DIM_FIELD;
using native::INT32_LIMIT;
using native::LADDER;
using native::PADDED_FIELD;
using native::REDUCE_LAUNCH;
using native::REDUCE_SYMBOL_FIELD;
using native::reducePlanes;
using native::STORE_DB_FIELD;
using native::TAILZERO_LAUNCH;
using native::toInt32;
using native::Uids;
using Slot = FlydslSdpaBwdSplitSlot;

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view ENGINE_NAME = "hipkernel:flydsl_sdpa_bwd_split";
/// Prefixes the message of every BAD_PARAM this engine raises.
constexpr std::string_view ERROR_SCOPE = "flydsl_sdpa_bwd_split";
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
/// The dq slot's multi-launch dispatches: every dq row carries its launches as `sources`, the
/// dQ kernel and the tail-zero kernel, and the rows of the `dq_reduce` dispatch (every row that
/// names a `reduce_symbol`) also the dBias reduce kernel as a launch of its own.
constexpr std::string_view DQ_REDUCE_DISPATCH_SYMBOL
    = "hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch";
/// The dkdv slot's two-launch dispatch for `Hk != Hv`: every row carries the dK-only object as
/// `dkdv` and the dV-only object as `dvonly`, in that launch order.
constexpr std::string_view DKDV_U2_DISPATCH_SYMBOL
    = "hipkernel.flydsl_sdpa_bwd_split.dkdv_u2.dispatch";
/// The dq slot's dispatch for `Hk != Hv`: every row carries the V-head dQ kernel as `dq` and the
/// V-head tail-zero kernel as `tailzero`.
constexpr std::string_view DQ_U2_DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_bwd_split.dq_u2.dispatch";

/// The launch names of a dkdv row's `sources` (`dkdv_u2` dispatch).
constexpr std::string_view DKDV_LAUNCH = "dkdv";
constexpr std::string_view DVONLY_LAUNCH = "dvonly";
/// The launch names of a dq row's `sources`.
constexpr std::string_view DQ_LAUNCH = "dq";

// Kernel metadata fields beyond the shared ones, declared by each slot's KMD: they belong to the
// slot that names them.
/// The compiled dropout flag's form: with `rt`, dropout is a runtime branch on rt_flags.
constexpr std::string_view DFORM_FIELD = "dform";
/// `g1` (floor 0), `tail` (the rung below the tile), or the floor itself.
constexpr std::string_view FLOOR_FIELD = "floor";
/// prep: 1 is the sink twin, which takes arguments a graph without a sink cannot give.
constexpr std::string_view SINK_FIELD = "sink";
/// How the object's launch grid maps the problem onto x, y and z; see gridOf.
constexpr std::string_view GRID_FIELD = "grid";
/// dq and dkdv: 1 when the row's bias is FLOAT beside a 16-bit Q (fp32-Q rows carry 0).
constexpr std::string_view BIAS_F32_FIELD = "bias_f32";
/// dq: how the object reads the bias: `lds` (staged, 16-bit only), `direct` (loads directly and
/// a staged row exists for the graphs it serves), `direct_only` (loads directly and no staged
/// row serves those graphs), `na` for a row without a bias.
constexpr std::string_view BIAS_STAGE_RESOLVED_FIELD = "bias_stage_resolved";
constexpr std::string_view STAGE_LDS = "lds";
constexpr std::string_view STAGE_DIRECT = "direct";
constexpr std::string_view STAGE_DIRECT_ONLY = "direct_only";
constexpr std::string_view STAGE_NA = "na";
/// dq and dkdv: `none` for a row that serves `Hk == Hv`, `dk` (dkdv) and `dq` (dq) for the rows
/// that serve `Hk != Hv`.
constexpr std::string_view U2_ROLE_FIELD = "u2_role";
constexpr std::string_view U2_NONE = "none";
constexpr std::string_view U2_DK = "dk";
constexpr std::string_view U2_DQ = "dq";

/// The `grid` values, one launch order per spelling: x, y, z, with the batch always on z.
constexpr std::string_view GRID_PREP = "qblocks_by_heads_by_seqs";
constexpr std::string_view GRID_DKDV_HEADS_FIRST = "kheads_by_kvblocks_by_seqs";
constexpr std::string_view GRID_DKDV_BLOCKS_FIRST = "kvblocks_by_kheads_by_seqs";
constexpr std::string_view GRID_DQ = "qheads_by_qblocks_by_seqs";
// A `dkdv_u2` row's `grid` is its dK-only object's; the dV-only object's grid is the same
// spelling over the V heads (`vheads_by_kvblocks_by_seqs`, `kvblocks_by_vheads_by_seqs`).

// What graph_match binds for kernel_match.
constexpr std::string_view CLASS_TOKEN = "flydsl_sdpa_bwd_split.class";
constexpr std::string_view DTYPE_TOKEN = "flydsl_sdpa_bwd_split.dtype";
constexpr std::string_view HEAD_DIM_QK_TOKEN = "flydsl_sdpa_bwd_split.head_dim_qk";
constexpr std::string_view HEAD_DIM_V_TOKEN = "flydsl_sdpa_bwd_split.head_dim_v";
constexpr std::string_view DROPOUT_TOKEN = "flydsl_sdpa_bwd_split.dropout";
constexpr std::string_view DBIAS_TOKEN = "flydsl_sdpa_bwd_split.dbias";
/// 1 for a 16-bit bias graph outside the LDS staging contract (the kernel's bias chunk
/// contract): a dq row whose `bias_stage_resolved` is `lds` does not serve it, one that is
/// `direct` serves only it.
constexpr std::string_view BIAS_UNSTAGED_TOKEN = "flydsl_sdpa_bwd_split.bias_unstaged";
/// 1 when the bias is FLOAT beside a 16-bit Q: a row serves the graph iff its `bias_f32` is 1.
constexpr std::string_view BIAS_F32_TOKEN = "flydsl_sdpa_bwd_split.bias_f32";
/// 1 when V's heads differ from K's: a row serves the graph iff its `u2_role` is not `none`.
constexpr std::string_view V_HEADS_DIFFER_TOKEN = "flydsl_sdpa_bwd_split.v_heads_differ";
constexpr std::string_view GQA_PACK_TOKEN = "flydsl_sdpa_bwd_split.gqa_pack";
constexpr std::string_view BATCH_TOKEN = "flydsl_sdpa_bwd_split.batch";
constexpr std::string_view HEADS_TOKEN = "flydsl_sdpa_bwd_split.heads";
constexpr std::string_view KV_HEADS_TOKEN = "flydsl_sdpa_bwd_split.kv_heads";
constexpr std::string_view V_HEADS_TOKEN = "flydsl_sdpa_bwd_split.v_heads";
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

/// The fp32 ladder: no 384, so the t512 tail floor is 256.
constexpr std::array<int64_t, 9> FP32_LADDER{32, 64, 96, 128, 160, 192, 224, 256, 512};

/// The prep kernel's threads per block (`BLOCK_THREADS` in `fmha_bwd_prep_gfx950.py`).
constexpr int64_t PREP_BLOCK_THREADS = 256;
constexpr int64_t PREP_MAX_LANES_PER_ROW = 16;
constexpr int64_t VECTOR_BYTES = 16;

constexpr std::string_view slotName(Slot slot)
{
    if(slot == Slot::PREP)
    {
        return "prep";
    }
    return slot == Slot::DKDV ? "dkdv" : "dq";
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
    request.dbiasElementBytes = problem.biasElementBytes;
    return request;
}

/// The reduce kernel's vector: the columns one thread owns, a 16-byte access of the dBias's
/// dtype.
int64_t reduceVectorOf(const bwd::SdpaBwdProblem& problem)
{
    return VECTOR_BYTES / static_cast<int64_t>(problem.biasElementBytes);
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

/// The head-dim bands the shipped f32-bias and `Hk != Hv` rows cover on @p arch, empty on an
/// architecture the engine knows no such rows for.
std::vector<bwd::HeadDimBand> parityBandsOf(std::string_view arch)
{
    if(arch.rfind("gfx11", 0) == 0)
    {
        return {bwd::GFX11_PARITY_BANDS.begin(), bwd::GFX11_PARITY_BANDS.end()};
    }
    if(arch.rfind("gfx950", 0) == 0)
    {
        return {bwd::GFX950_PARITY_BANDS.begin(), bwd::GFX950_PARITY_BANDS.end()};
    }
    return {};
}

/// Logs, at INFO, that an f32-bias or `Hk != Hv` graph is admitted and whether its head dims
/// lie in the bands the shipped rows cover. A graph outside them is admitted here and then
/// declined by the ingestor for want of a serving row, as a 16-bit bias outside the staging
/// contract is on a device without a `direct` row.
void logParityCoverage(const MatchContext& context,
                       const bwd::SdpaBwdProblem& problem,
                       bool biasF32,
                       bool valueHeadsDiffer)
{
    const std::string_view arch = context.deviceProperties.gcnArchName;
    const auto bands = parityBandsOf(arch);
    const int64_t low = std::min(problem.headDimQk, problem.headDimV);
    const int64_t high = std::max(problem.headDimQk, problem.headDimV);
    const bool covered = std::any_of(bands.begin(), bands.end(), [&](const bwd::HeadDimBand& band) {
        return low > band.floor && high <= band.tile;
    });
    std::string bandText;
    for(const auto& band : bands)
    {
        bandText += (bandText.empty() ? "" : ", ") + std::string("(") + std::to_string(band.floor)
                    + ", " + std::to_string(band.tile) + "]";
    }
    HIPDNN_PLUGIN_LOG_INFO(ENGINE_NAME
                           << " admits a graph with "
                           << (biasF32 ? "an f32 bias beside a 16-bit query" : "")
                           << (biasF32 && valueHeadsDiffer ? " and " : "")
                           << (valueHeadsDiffer ? "V heads that differ from K's" : "")
                           << " at head dims Qk " << problem.headDimQk << ", V " << problem.headDimV
                           << "; the shipped rows for " << arch << " cover head dims in "
                           << (bandText.empty() ? std::string("no band") : bandText)
                           << ", so the graph is " << (covered ? "inside" : "outside")
                           << " them and is declined by the ingestor unless a row serves it");
}

/**
 * @brief Graph-scoped applicability: the shared backward matcher's contract with every
 *        option the split engine takes (FLOAT, an f32 bias, V heads of their own), with the
 *        proven shared workspace (SP-WS) and what every kernel_match needs bound.
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
    const auto matched = bwd::matchSdpaBackward(context, bwd::SPLIT_MATCH_OPTIONS);
    if(const auto* decline = std::get_if<bwd::Decline>(&matched))
    {
        native::logDecline(ENGINE_NAME, decline->cause, decline->detail);
        return std::nullopt;
    }
    const auto& problem = std::get<bwd::SdpaBwdProblem>(matched);

    const auto dtype = dtypeName(problem.dtype);
    if(!dtype.has_value())
    {
        native::logDecline(
            ENGINE_NAME, "data_type", "the matcher served a dtype this engine has no spelling for");
        return std::nullopt;
    }
    // The bias chunk contract: a 16-bit bias outside the LDS staging contract is served by a dq
    // row that loads the bias directly (`bias_stage_resolved` `direct` or `direct_only`) and
    // by no `lds` row; a bias inside it is served by an `lds` row, or a `direct_only` one.
    // An f32 bias and an fp32 query have no staged row. kernel_match applies the rule.
    const bool biasUnstaged = problem.biasUid != bwd::NO_UID && problem.biasElementBytes == 2
                              && !native::stagesBias(problem);
    if(biasUnstaged)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            ENGINE_NAME
            << " [bias stage]: the 16-bit bias layout is outside the LDS staging contract "
               "(16-byte base, strides multiples of 16 bytes, last row padded to a whole 16-byte "
               "chunk); only dq rows whose bias_stage_resolved is 'direct' or 'direct_only' "
               "serve it, and the graph is declined unless the device has such a row");
    }
    const bool biasF32
        = problem.biasUid != bwd::NO_UID && problem.biasElementBytes != problem.elementBytes;
    const bool valueHeadsDiffer = problem.numVHeads != problem.numKvHeads;
    if(biasF32 || valueHeadsDiffer)
    {
        logParityCoverage(context, problem, biasF32, valueHeadsDiffer);
    }

    // SP-WS: every region of the shared workspace and the dBias scratch is sized by checked
    // arithmetic and proven to lie within the workspace the runtime will allocate.
    std::string why;
    const auto layout = provenLayout(problem, why);
    if(!why.empty())
    {
        native::logDecline(ENGINE_NAME, "SP-WS", why);
        return std::nullopt;
    }
    if(native::tailzeroMaskOf(problem) != 0 && !native::tailzeroGridOf(problem))
    {
        native::logDecline(ENGINE_NAME, "SP-TZ", "the tail-zero launch grid exceeds its limits");
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
    bound[std::string(BIAS_F32_TOKEN)] = int64_t{biasF32 ? 1 : 0};
    bound[std::string(V_HEADS_DIFFER_TOKEN)] = int64_t{valueHeadsDiffer ? 1 : 0};
    // The matcher guarantees Hq % Hk == 0, so more query heads than key heads is a group; a
    // pack object also needs every packed span inside the 32-bit buffer descriptor, else the
    // graph is served by the non-pack objects, which rebase per head.
    bound[std::string(GQA_PACK_TOKEN)]
        = int64_t{native::packedSpansFit(problem, native::PACK_MAX_BLOCK_Q) ? 1 : 0};
    bound[std::string(BATCH_TOKEN)] = problem.batch;
    bound[std::string(HEADS_TOKEN)] = problem.numHeads;
    bound[std::string(KV_HEADS_TOKEN)] = problem.numKvHeads;
    bound[std::string(V_HEADS_TOKEN)] = problem.numVHeads;
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
    /// dq and dkdv: the bias is FLOAT beside a 16-bit Q.
    int64_t biasF32 = 0;
    /// dq: `lds`, `direct`, `direct_only` or `na`.
    std::string biasStageResolved;
    /// dq and dkdv: `none`, `dk` (dkdv) or `dq` (dq).
    std::string u2Role;
};

/// Whether @p role is a `u2_role` spelling @p slot's rows carry.
bool u2RoleOf(Slot slot, const std::string& role)
{
    return role == U2_NONE || role == (slot == Slot::DKDV ? U2_DK : U2_DQ);
}

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
    const auto biasF32 = integerMetadata(kernel, BIAS_F32_FIELD);
    const auto u2Role = stringMetadata(kernel, U2_ROLE_FIELD);
    if(!biasF32 || (*biasF32 != 0 && *biasF32 != 1) || !u2Role || !u2RoleOf(slot, *u2Role))
    {
        return std::nullopt;
    }
    row.causal = *causal;
    row.hasBias = *hasBias;
    row.dropout = *dropout;
    row.runtimeDropout = *dform == "rt";
    row.blockThreads = *blockThreads;
    row.biasF32 = *biasF32;
    row.u2Role = *u2Role;

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
    const auto biasStage = stringMetadata(kernel, BIAS_STAGE_RESOLVED_FIELD);
    if(!storeDb || !blockM || !biasStage
       || (*biasStage != STAGE_LDS && *biasStage != STAGE_DIRECT && *biasStage != STAGE_DIRECT_ONLY
           && *biasStage != STAGE_NA))
    {
        return std::nullopt;
    }
    row.storeDb = *storeDb;
    row.biasStageResolved = *biasStage;
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
    bool biasF32 = false;
    bool valueHeadsDiffer = false;
    bool gqaPack = false;
    int64_t batch = 0;
    int64_t numHeads = 0;
    int64_t numKvHeads = 0;
    int64_t numVHeads = 0;
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
    const auto biasF32 = tryGetBoundInt(bound, BIAS_F32_TOKEN);
    const auto valueHeadsDiffer = tryGetBoundInt(bound, V_HEADS_DIFFER_TOKEN);
    const auto gqaPack = tryGetBoundInt(bound, GQA_PACK_TOKEN);
    const auto batch = tryGetBoundInt(bound, BATCH_TOKEN);
    const auto numHeads = tryGetBoundInt(bound, HEADS_TOKEN);
    const auto numKvHeads = tryGetBoundInt(bound, KV_HEADS_TOKEN);
    const auto numVHeads = tryGetBoundInt(bound, V_HEADS_TOKEN);
    const auto seqLenQ = tryGetBoundInt(bound, SEQ_Q_TOKEN);
    const auto seqLenKv = tryGetBoundInt(bound, SEQ_KV_TOKEN);
    const auto planes = tryGetBoundInt(bound, REDUCE_PLANES_TOKEN);
    if(!graphClass || !headDimQk || !headDimV || !dropout || !dbias || !biasUnstaged || !biasF32
       || !valueHeadsDiffer || !gqaPack || !batch || !numHeads || !numKvHeads || !numVHeads
       || !seqLenQ || !seqLenKv || !planes)
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
                      *biasF32 != 0,
                      *valueHeadsDiffer != 0,
                      *gqaPack != 0,
                      *batch,
                      *numHeads,
                      *numKvHeads,
                      *numVHeads,
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
/// dkdv is (key heads, key blocks, batch) or (key blocks, key heads, batch). With
/// @p valueOnly, a dkdv row's dV-only launch (C6): the same spelling over the V heads, at the
/// block and key tile of the dK-only object. nullopt for a `grid` the slot does not know, past
/// the limits of an axis, or for a record without geometry.
std::optional<Grid>
    gridOf(Slot slot, const GraphFacts& facts, const Row& row, bool valueOnly = false)
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
        const int64_t heads = valueOnly ? facts.numVHeads : facts.numKvHeads;
        if(row.grid == GRID_DKDV_HEADS_FIRST)
        {
            x = heads;
            y = keyBlocks;
        }
        else if(row.grid == GRID_DKDV_BLOCKS_FIRST)
        {
            x = keyBlocks;
            y = heads;
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

/// Whether a dq row of @p stage serves a graph whose 16-bit bias is outside the LDS staging
/// contract when @p unstaged: the chunk-contract rule. An `lds` row serves a bias inside the
/// contract; a `direct` row serves a bias outside it (a staged row serves every other graph
/// the row's geometry covers); a `direct_only` row, which no staged row backs, serves any bias.
/// `na` is a row without a bias. An `lds` row never serves an f32 bias (@p biasF32): staging is
/// 16-bit only, so an f32 bias and an fp32 query are served by `direct_only` rows.
bool servesBiasStage(const std::string& stage, bool unstaged, bool biasF32)
{
    if(stage == STAGE_LDS)
    {
        // A staged row holds a 16-bit bias; an f32 bias never binds to one.
        return !unstaged && !biasF32;
    }
    if(stage == STAGE_DIRECT)
    {
        return unstaged;
    }
    return stage == STAGE_DIRECT_ONLY;
}

/**
 * @brief Kernel-scoped applicability of one slot's object: dtype, head-dim tile, and what the
 *        slot's kernel is built for (class, dropout, packing, dB store, bias width and
 *        staging, V heads, the reduce kernel), against the graph, with a launch grid inside
 *        its limits.
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
        // A row reads the bias at the width it was built for, and a row built for V heads of
        // their own serves exactly the graphs that have them (`dk` on dkdv, `dq` on dq).
        if(row->biasF32 != (facts->biasF32 ? 1 : 0)
           || (row->u2Role != U2_NONE) != facts->valueHeadsDiffer)
        {
            return false;
        }
    }
    if constexpr(SLOT == Slot::DKDV)
    {
        // A packed object serves dense GQA whose packed spans fit, at a query tile the span
        // bound covers: a row that records no tile, or one above the bound, serves nothing.
        if(row->gqaPack == 1
           && (!facts->gqaPack || row->blockQ <= 0 || row->blockQ > native::PACK_MAX_BLOCK_Q))
        {
            return false;
        }
    }
    if constexpr(SLOT == Slot::DQ)
    {
        // The chunk contract: which rows serve a bias graph by how the row reads the bias.
        if(facts->graphClass == CLASS_BIAS
           && !servesBiasStage(row->biasStageResolved, facts->biasUnstaged, facts->biasF32))
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
    if(!gridOf(SLOT, *facts, *row).has_value())
    {
        return false;
    }
    // The dV-only launch of a `dkdv_u2` row runs over the V heads.
    return SLOT != Slot::DKDV || row->u2Role == U2_NONE
           || gridOf(SLOT, *facts, *row, true).has_value();
}

// ---------------------------------------------------------------------------
// Launch values
// ---------------------------------------------------------------------------

/// Everything graph- and kernel-derived, resolved once at prepare(). Holds no reference to
/// the match context.
struct SplitPlan
{
    Slot slot = Slot::PREP;
    bwd::SdpaBwdProblem problem;
    Uids uids;
    unsigned int blockThreads = 0;
    Grid grid;
    /// The row serves a graph whose V heads differ from K's (`u2_role` not `none`): the
    /// kernel takes `num_head_v`, and the dq slot's tail-zero launch is the V-head object.
    bool u2 = false;
    /// A `dkdv` row with `u2`: the dV-only launch's grid, over the V heads. The launch has the
    /// dK-only launch's block and arguments.
    Grid dvonlyGrid;
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
    /// The tail-zero launch of the dq slot: the outputs it zeroes (`mask`, 0 for no launch)
    /// and its grid.
    int32_t tailzeroMask = 0;
    native::TailzeroGrid tailzeroGrid;
};

bwd::SdpaBwdProblem problemOf(const MatchContext& context)
{
    auto matched = bwd::matchSdpaBackward(context, bwd::SPLIT_MATCH_OPTIONS);
    if(auto* problem = std::get_if<bwd::SdpaBwdProblem>(&matched))
    {
        return *problem;
    }
    const auto& decline = std::get<bwd::Decline>(matched);
    native::failBadParam(ERROR_SCOPE,
                         "the graph no longer matches [" + decline.cause + "]: " + decline.detail);
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
            native::failBadParam(ERROR_SCOPE,
                                 "workspace token '" + std::string(token) + "' is "
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
        native::failBadParam(ERROR_SCOPE,
                             "kernel '" + kernel.name + "' lacks a field of the "
                                 + std::string(slotName(slot)) + " record");
    }
    const auto facts = readFacts(bound);
    if(!facts)
    {
        native::failBadParam(ERROR_SCOPE, "the bound tokens of the graph are incomplete");
    }
    const auto grid = gridOf(slot, *facts, *row);
    if(!grid)
    {
        native::failBadParam(ERROR_SCOPE,
                             "launch grid exceeds its limits for kernel '" + kernel.name + "'");
    }
    plan.blockThreads = static_cast<unsigned int>(blockThreadsOf(slot, *row));
    plan.grid = *grid;
    plan.u2 = row->u2Role != U2_NONE;
    if(slot == Slot::DKDV && plan.u2)
    {
        const auto dvonlyGrid = gridOf(slot, *facts, *row, true);
        if(!dvonlyGrid)
        {
            native::failBadParam(ERROR_SCOPE,
                                 "dV-only launch grid exceeds its limits for kernel '" + kernel.name
                                     + "'");
        }
        plan.dvonlyGrid = *dvonlyGrid;
    }

    native::bindGraphOperands(plan, context, row->tile);

    plan.workspace = native::workspaceLayout(problem, requestFor(problem));
    requireTokens(bound, plan.workspace);
    if(slot == Slot::DQ && plan.workspace.reduces)
    {
        if(row->reduceSymbol.empty())
        {
            native::failBadParam(ERROR_SCOPE,
                                 "kernel '" + kernel.name
                                     + "' serves a broadcast dBias but names no reduce kernel");
        }
        plan.reduceSymbol = row->reduceSymbol;
        plan.reduceGrid = native::reduceGridOf(problem, reduceVectorOf(problem));
    }
    if(slot == Slot::DQ)
    {
        plan.tailzeroMask = native::tailzeroMaskOf(problem);
        if(plan.tailzeroMask != 0)
        {
            const auto tailzeroGrid = native::tailzeroGridOf(problem);
            if(!tailzeroGrid)
            {
                native::failBadParam(ERROR_SCOPE, "the tail-zero launch grid exceeds its limits");
            }
            plan.tailzeroGrid = *tailzeroGrid;
        }
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Kernel argument lists
// ---------------------------------------------------------------------------

/**
 * @brief The prep kernel's arguments, in parameter order (`_PARAMS["prep"]` in
 *        `_sdpa_bwd_cdna_build.py`): 27 arguments, the offset and statistics arguments after
 *        `seqinfo_q1`.
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
    native::OffsetArgs offO;
    native::OffsetArgs offDO;
    native::StatsArgs stats;
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
                        offO.offset,
                        offO.multiplier,
                        offDO.offset,
                        offDO.multiplier,
                        stats.offset.offset,
                        stats.offset.multiplier,
                        stats.strides.batch,
                        stats.strides.head,
                        stats.strides.seq,
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

/// The dK/dV kernel's arguments (`_PARAMS["dkdv"]`): 61 arguments, the offset arguments after
/// `seqinfo_k1`. With @p U2, the V-head variants' (`_PARAMS["dkdv_u2"]`, the dK-only and dV-only
/// objects alike): 62, `num_head_v` right after `num_head_k`.
template <bool U2>
struct DkdvKernargsT
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
    int64_t mulQ = 0;
    int64_t mulK = 0;
    native::OffsetArgs offV;
    native::OffsetArgs offDO;
    native::OffsetArgs offDK;
    native::OffsetArgs offDV;
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
    /// Read only when @p U2.
    int32_t numHeadV = 0;
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
        const auto leading = std::tie(q,
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
                                      mulQ,
                                      mulK,
                                      offV.offset,
                                      offV.multiplier,
                                      offDO.offset,
                                      offDO.multiplier,
                                      offDK.offset,
                                      offDK.multiplier,
                                      offDV.offset,
                                      offDV.multiplier,
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
                                      numHeadK);
        const auto trailing = std::tie(hdimQk,
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
        if constexpr(U2)
        {
            return std::tuple_cat(leading, std::tie(numHeadV), trailing);
        }
        else
        {
            return std::tuple_cat(leading, trailing);
        }
    }
};

/// The dQ kernel's arguments (`_PARAMS["dq"]`): 59 arguments, the offset arguments after
/// `seqinfo_k1`. With @p U2, the V-head variant's (`_PARAMS["dq_u2"]`): 60, `num_head_v` right
/// after `num_head_k`.
template <bool U2>
struct DqKernargsT
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
    int64_t mulQ = 0;
    int64_t mulK = 0;
    native::OffsetArgs offV;
    native::OffsetArgs offDO;
    native::OffsetArgs offDQ;
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
    /// Read only when @p U2.
    int32_t numHeadV = 0;
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
        const auto leading = std::tie(q,
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
                                      mulQ,
                                      mulK,
                                      offV.offset,
                                      offV.multiplier,
                                      offDO.offset,
                                      offDO.multiplier,
                                      offDQ.offset,
                                      offDQ.multiplier,
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
                                      numHeadK);
        const auto trailing = std::tie(hdimQk,
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
        if constexpr(U2)
        {
            return std::tuple_cat(leading, std::tie(numHeadV), trailing);
        }
        else
        {
            return std::tuple_cat(leading, trailing);
        }
    }
};

using DkdvKernargs = DkdvKernargsT<false>;
using DkdvU2Kernargs = DkdvKernargsT<true>;
using DqKernargs = DqKernargsT<false>;
using DqU2Kernargs = DqKernargsT<true>;

/// The argument list of @p slot's kernel, the V-head variant's when @p u2.
const std::vector<KernelArgument>& signatureOfSlot(Slot slot, bool u2)
{
    static const auto s_prep = native::signatureOf<PrepKernargs>();
    static const auto s_dkdv = native::signatureOf<DkdvKernargs>();
    static const auto s_dkdvU2 = native::signatureOf<DkdvU2Kernargs>();
    static const auto s_dq = native::signatureOf<DqKernargs>();
    static const auto s_dqU2 = native::signatureOf<DqU2Kernargs>();
    if(slot == Slot::PREP)
    {
        return s_prep;
    }
    if(slot == Slot::DKDV)
    {
        return u2 ? s_dkdvU2 : s_dkdv;
    }
    return u2 ? s_dqU2 : s_dq;
}

// ---------------------------------------------------------------------------
// Launch arguments
// ---------------------------------------------------------------------------

struct LaunchArguments
{
    std::variant<PrepKernargs, DkdvKernargs, DkdvU2Kernargs, DqKernargs, DqU2Kernargs> kernel;
    std::optional<native::ReduceKernargs> reduce;
    /// Set on the prep launch only: it writes the seed region the other slots read.
    std::optional<native::SeedUpload> seedUpload;
    /// Set on the dq launch when the graph needs the tail-zero launch after the kernel and
    /// the reduce.
    std::optional<native::TailzeroKernargs> tailzero;
    /// The same launch of a graph whose V heads differ from K's: the V-head object's arguments.
    std::optional<native::TailzeroHvKernargs> tailzeroHv;
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
        // `seq_len_q` / `seq_len_kv` tensors, and nothing is packed (num_seqlens stays 0). A
        // dense tensor passes a null offset array and multiplier 0.
        prep.seqinfoQ0 = problem.paddingMask ? buffer(problem.seqLenQUid) : unused;
        prep.seqinfoQ1 = native::offsetArgsOf(problem.ragged.q, buffer).offset;
        prep.offO = native::offsetArgsOf(problem.ragged.o, buffer);
        prep.offDO = native::offsetArgsOf(problem.ragged.dO, buffer);
        prep.stats = native::statsArgsOf(problem, buffer);
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
        const auto dkdvArguments = [&](auto d) {
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
            d.seqinfoQ1 = native::offsetArgsOf(problem.ragged.q, buffer).offset;
            d.seqinfoK1 = native::offsetArgsOf(problem.ragged.k, buffer).offset;
            d.mulQ = problem.ragged.q.multiplier;
            d.mulK = problem.ragged.k.multiplier;
            d.offV = native::offsetArgsOf(problem.ragged.v, buffer);
            d.offDO = native::offsetArgsOf(problem.ragged.dO, buffer);
            d.offDK = native::offsetArgsOf(problem.ragged.dK, buffer);
            d.offDV = native::offsetArgsOf(problem.ragged.dV, buffer);
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
            return d;
        };
        if(plan.u2)
        {
            auto d = dkdvArguments(DkdvU2Kernargs{});
            d.numHeadV = toInt32(problem.numVHeads, "num_head_v");
            arguments.kernel = d;
        }
        else
        {
            arguments.kernel = dkdvArguments(DkdvKernargs{});
        }
        return arguments;
    }

    const auto dqArguments = [&](auto d) {
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
        d.seqinfoQ1 = native::offsetArgsOf(problem.ragged.q, buffer).offset;
        d.seqinfoK1 = native::offsetArgsOf(problem.ragged.k, buffer).offset;
        d.mulQ = problem.ragged.q.multiplier;
        d.mulK = problem.ragged.k.multiplier;
        d.offV = native::offsetArgsOf(problem.ragged.v, buffer);
        d.offDO = native::offsetArgsOf(problem.ragged.dO, buffer);
        d.offDQ = native::offsetArgsOf(problem.ragged.dQ, buffer);
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
        if(plan.tailzeroMask != 0)
        {
            const auto tailzeroOf = [&](auto hv) {
                return native::tailzeroKernargsOf<decltype(hv)::value>(problem,
                                                                       buffer(plan.uids.dQ),
                                                                       buffer(plan.uids.dK),
                                                                       buffer(plan.uids.dV),
                                                                       buffer(problem.seqLenQUid),
                                                                       buffer(problem.seqLenKvUid));
            };
            if(plan.u2)
            {
                arguments.tailzeroHv = tailzeroOf(std::true_type{});
            }
            else
            {
                arguments.tailzero = tailzeroOf(std::false_type{});
            }
        }
        return d;
    };
    if(plan.u2)
    {
        auto d = dqArguments(DqU2Kernargs{});
        d.numHeadV = toInt32(problem.numVHeads, "num_head_v");
        arguments.kernel = d;
    }
    else
    {
        arguments.kernel = dqArguments(DqKernargs{});
    }
    return arguments;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The slot's kernel, the dV-only kernel of a `dkdv_u2` row, the reduce kernel when the graph
/// needs one, and the plan.
class PreparedFlydslSdpaBwdSplit : public PreparedDispatch
{
public:
    PreparedFlydslSdpaBwdSplit(SplitPlan plan,
                               IngestorKernelCode code,
                               std::optional<IngestorKernelCode> dvonlyCode,
                               std::optional<IngestorKernelCode> reduceCode,
                               std::optional<IngestorKernelCode> tailzeroCode)
        : _plan(std::move(plan))
        , _code(std::move(code))
        , _dvonly(std::move(dvonlyCode))
        , _reduce(std::move(reduceCode))
        , _tailzero(std::move(tailzeroCode))
    {
    }

    bool hasDvonly() const
    {
        return _dvonly.has_value();
    }

    compilation::IRunnableKernel& dvonlyForStream(hipStream_t stream) const
    {
        return _dvonly->kernelForStream(stream);
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

    compilation::IRunnableKernel& tailzeroForStream(hipStream_t stream) const
    {
        return _tailzero->kernelForStream(stream);
    }

private:
    SplitPlan _plan;
    IngestorKernelCode _code;
    std::optional<IngestorKernelCode> _dvonly;
    std::optional<IngestorKernelCode> _reduce;
    std::optional<IngestorKernelCode> _tailzero;
};

/**
 * @brief The native dispatch behind one slot's UDD: reports the slot's scratch, and launches
 *        the slot's kernel; prep first uploads the seed, dq follows its kernel with the reduce
 *        kernel when the graph's dBias is broadcast.
 *
 * With @p SOURCES the dispatch is multi-launch. On dq it is the `dq_reduce` dispatch: every row
 * carries the dQ kernel and the dBias reduce kernel as the launches `dq` and `dbias_reduce` of
 * its `sources`, and each is built from its own entry; the plain `dq` dispatch serves the rows
 * that name no reduce kernel and refuses one that does. On dkdv @p SOURCES is the `dkdv_u2`
 * dispatch (dK-only and dV-only launches).
 *
 * With @p DQ_U2 (dq only) the dispatch serves the rows of `u2_role` `dq`, whose launches are
 * the V-head dQ kernel and the V-head tail-zero kernel; the standard dq dispatches refuse
 * such a row and `dq_u2` refuses any other.
 *
 * Split handlers do not save plans: supportsSaving() is the SDK default, false.
 */
template <Slot SLOT, bool SOURCES = false, bool DQ_U2 = false>
class FlydslSdpaBwdSplitDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    FlydslSdpaBwdSplitDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                      const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
        static_assert(!SOURCES || SLOT != Slot::PREP, "prep is one launch");
        static_assert(!DQ_U2 || (SLOT == Slot::DQ && !SOURCES), "dq_u2 is a dq dispatch");
    }

    std::set<std::string> launchNames() const override
    {
        if constexpr(SLOT == Slot::DKDV && SOURCES)
        {
            return {std::string(DKDV_LAUNCH), std::string(DVONLY_LAUNCH)};
        }
        if constexpr(SLOT == Slot::DQ)
        {
            std::set<std::string> names{std::string(DQ_LAUNCH), std::string(TAILZERO_LAUNCH)};
            if constexpr(SOURCES)
            {
                names.insert(std::string(REDUCE_LAUNCH));
            }
            return names;
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

        // Every launch's definition is resolved, and the reduce and tail-zero launches'
        // argument lists checked, before any code object loads, so a row that lacks a launch
        // or disagrees on an ABI fails on the host whatever the state of the archive.
        [[maybe_unused]] const bool reduces = !plan.reduceSymbol.empty();
        [[maybe_unused]] const bool zeroesTails = plan.tailzeroMask != 0;
        const auto& slotSignature = signatureOfSlot(SLOT, plan.u2);
        std::optional<KernelDefinition> dqLaunch;
        std::optional<KernelDefinition> dkLaunch;
        std::optional<KernelDefinition> dvonlyKernel;
        std::optional<KernelDefinition> reduceKernel;
        std::optional<KernelDefinition> tailzeroKernel;
        if constexpr(SLOT == Slot::DKDV)
        {
            // A row serves V heads of its own exactly when its pack's dispatch launches both
            // objects.
            if(plan.u2 != SOURCES)
            {
                native::failBadParam(ERROR_SCOPE,
                                     "kernel '" + kernel.name + "' has u2_role "
                                         + (plan.u2 ? "dk" : "none") + " but is dispatched by the "
                                         + (SOURCES ? "dkdv_u2" : "dkdv") + " handler");
            }
            if constexpr(SOURCES)
            {
                dkLaunch = launchKernelDefinition(kernel, std::string(DKDV_LAUNCH));
                dvonlyKernel = launchKernelDefinition(kernel, std::string(DVONLY_LAUNCH));
                for(const auto* launch : {&*dkLaunch, &*dvonlyKernel})
                {
                    requireSignatureMatch(
                        launch->source.signature,
                        slotSignature,
                        launch->source.symbol,
                        describeDescriptor("kernel", kernel.name, kernel.kernelId));
                }
            }
        }
        if constexpr(SLOT == Slot::DQ)
        {
            if(plan.u2 != DQ_U2)
            {
                native::failBadParam(ERROR_SCOPE,
                                     "kernel '" + kernel.name + "' has u2_role "
                                         + (plan.u2 ? "dq" : "none") + " but is dispatched by the "
                                         + (DQ_U2 ? "dq_u2" : "dq") + " handler");
            }
            dqLaunch = launchKernelDefinition(kernel, std::string(DQ_LAUNCH));
            if constexpr(SOURCES)
            {
                if(reduces)
                {
                    reduceKernel = launchKernelDefinition(kernel, std::string(REDUCE_LAUNCH));
                    requireSignatureMatch(
                        reduceKernel->source.signature,
                        native::reduceSignature(),
                        reduceKernel->source.symbol,
                        describeDescriptor("kernel", kernel.name, kernel.kernelId));
                }
            }
            else if(reduces)
            {
                native::failBadParam(ERROR_SCOPE,
                                     "kernel '" + kernel.name + "' names the reduce kernel '"
                                         + plan.reduceSymbol
                                         + "' but is dispatched by the dq handler, which "
                                         + "launches no reduce: the dq_reduce handler does");
            }
            if(zeroesTails)
            {
                tailzeroKernel = launchKernelDefinition(kernel, std::string(TAILZERO_LAUNCH));
                requireSignatureMatch(tailzeroKernel->source.signature,
                                      plan.u2 ? native::tailzeroHvSignature()
                                              : native::tailzeroSignature(),
                                      tailzeroKernel->source.symbol,
                                      describeDescriptor("kernel", kernel.name, kernel.kernelId));
            }
        }
        const KernelDefinition* slotSource = &kernel;
        if(dqLaunch.has_value())
        {
            slotSource = &*dqLaunch;
        }
        else if(dkLaunch.has_value())
        {
            slotSource = &*dkLaunch;
        }
        const KernelDefinition& slotKernel = *slotSource;

        // Built from the arch alone; see FlydslRmsNormNative.cpp's prepare().
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, slotKernel, options, slotSignature);
        code.setBlockSize(plan.blockThreads, 1, 1);
        code.setGridSize(plan.grid.x, plan.grid.y, plan.grid.z);

        // The dV-only object runs at the dK-only object's block over the V heads.
        std::optional<IngestorKernelCode> dvonlyCode;
        if(dvonlyKernel.has_value())
        {
            dvonlyCode.emplace(buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, *dvonlyKernel, options, slotSignature));
            dvonlyCode->setBlockSize(plan.blockThreads, 1, 1);
            dvonlyCode->setGridSize(plan.dvonlyGrid.x, plan.dvonlyGrid.y, plan.dvonlyGrid.z);
        }

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
        std::optional<IngestorKernelCode> tailzeroCode;
        if(tailzeroKernel.has_value())
        {
            tailzeroCode.emplace(buildIngestorKernelCode(_kernelCompiler,
                                                         _kpackLoader,
                                                         context,
                                                         *tailzeroKernel,
                                                         options,
                                                         plan.u2 ? native::tailzeroHvSignature()
                                                                 : native::tailzeroSignature()));
            tailzeroCode->setBlockSize(native::TAILZERO_THREADS, 1, 1);
            tailzeroCode->setGridSize(plan.tailzeroGrid.x, plan.tailzeroGrid.y, 1);
        }
        return std::make_unique<PreparedFlydslSdpaBwdSplit>(std::move(plan),
                                                            std::move(code),
                                                            std::move(dvonlyCode),
                                                            std::move(reduceCode),
                                                            std::move(tailzeroCode));
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
        const auto launchKernel = [&](compilation::IRunnableKernel& kernel) {
            std::visit(
                [&](const auto& kernargs) {
                    std::apply([&](const auto&... argument) { kernel.launch(stream, argument...); },
                               kernargs.tie());
                },
                arguments.kernel);
        };
        launchKernel(preparedSplit.kernelForStream(stream));
        // A `dkdv_u2` row: the dV-only launch follows the dK-only one with the same arguments.
        if(preparedSplit.hasDvonly())
        {
            launchKernel(preparedSplit.dvonlyForStream(stream));
        }
        if(arguments.reduce.has_value())
        {
            std::apply(
                [&](const auto&... argument) {
                    preparedSplit.reduceForStream(stream).launch(stream, argument...);
                },
                arguments.reduce->tie());
        }
        if(arguments.tailzero.has_value())
        {
            std::apply(
                [&](const auto&... argument) {
                    preparedSplit.tailzeroForStream(stream).launch(stream, argument...);
                },
                arguments.tailzero->tie());
        }
        if(arguments.tailzeroHv.has_value())
        {
            std::apply(
                [&](const auto&... argument) {
                    preparedSplit.tailzeroForStream(stream).launch(stream, argument...);
                },
                arguments.tailzeroHv->tie());
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

/// Process-lifetime: the registry holds non-owning pointers to the handlers, which share
/// one compiler, one loader and the pack's module cache.
template <Slot SLOT, bool SOURCES = false, bool DQ_U2 = false>
const FlydslSdpaBwdSplitDispatchHandler<SLOT, SOURCES, DQ_U2>& flydslSdpaBwdSplitDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslSdpaBwdSplitKpackModuleCache());
    static const FlydslSdpaBwdSplitDispatchHandler<SLOT, SOURCES, DQ_U2> s_dispatchHandler(
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
    scope.add(std::string(DKDV_U2_DISPATCH_SYMBOL),
              &flydslSdpaBwdSplitDispatchHandler<Slot::DKDV, true>());
    scope.add(std::string(DQ_U2_DISPATCH_SYMBOL),
              &flydslSdpaBwdSplitDispatchHandler<Slot::DQ, false, true>());
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
    if(slot == Slot::DKDV && plan.u2)
    {
        dump.dvonly = true;
        dump.dvonlyBlock = plan.blockThreads;
        dump.dvonlyGrid = {plan.dvonlyGrid.x, plan.dvonlyGrid.y, plan.dvonlyGrid.z};
    }
    if(arguments.reduce.has_value())
    {
        dump.reduces = true;
        dump.reduceSymbol = plan.reduceSymbol;
        dump.reduceKernarg = native::kernargImage(*arguments.reduce);
        dump.reduceBlock = native::REDUCE_THREADS;
        dump.reduceGrid = {plan.reduceGrid.x, plan.reduceGrid.y, 1};
    }
    if(arguments.tailzero.has_value())
    {
        dump.tailzero = true;
        dump.tailzeroKernarg = native::kernargImage(*arguments.tailzero);
        dump.tailzeroBlock = native::TAILZERO_THREADS;
        dump.tailzeroGrid = {plan.tailzeroGrid.x, plan.tailzeroGrid.y, 1};
    }
    if(arguments.tailzeroHv.has_value())
    {
        dump.tailzero = true;
        dump.tailzeroKernarg = native::kernargImage(*arguments.tailzeroHv);
        dump.tailzeroBlock = native::TAILZERO_THREADS;
        dump.tailzeroGrid = {plan.tailzeroGrid.x, plan.tailzeroGrid.y, 1};
    }
    if(arguments.seedUpload.has_value())
    {
        dump.seedUploaded = true;
        dump.seedWords.assign(arguments.seedUpload->words.begin(),
                              arguments.seedUpload->words.end());
    }
    return dump;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
