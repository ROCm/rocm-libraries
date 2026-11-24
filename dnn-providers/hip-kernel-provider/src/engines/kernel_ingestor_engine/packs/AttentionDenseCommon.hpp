// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "engines/kernel_ingestor_engine/packs/AttentionDensePersistent.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionDenseStrides.hpp"
#include "engines/kernel_ingestor_engine/packs/SdpaGraphMatch.hpp"

/**
 * @file AttentionDenseCommon.hpp
 * @brief What the gfx942 and gfx950 dense-attention ingestor packs share: the launch state
 *        and its saved form, and the matching and dispatch both engines run, parameterised
 *        by an AttentionDenseArch descriptor.
 *
 * The two kernels are `rocke/library/kernels/gfx942/attention_dense.py` and
 * `rocke/library/kernels/gfx950/attention_dense.py`. They take the same ABI, launch on the
 * same grid formula and are bound from the same graph contract, so one copy of that code
 * serves both engines. What differs per arch reaches it through the descriptor: the tile
 * rules and launch geometry, whether a causal sliding window is served, whether the
 * candidate's head counts are compared, the launch grids its catalog carries
 * (AttentionDensePersistent.hpp), and the launch contracts. Scoring and
 * every read of a KMD field only one engine declares stay in that engine's own
 * `Gfx9xxAttentionDenseNative.cpp`, which also owns the module cache and registration.
 */

// The provider's plugin handle (core/Handle.hpp), named here only as the handler's handle type.
struct Handle;

namespace hip_kernel_provider::compilation
{
class KpackKernelLoader;
}

namespace hip_kernel_provider::kernel_ingestor_engine
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The stats uid of a graph that binds no stats (LSE) output.
inline constexpr int64_t ATTENTION_DENSE_NO_STATS = -1;

/// An additive bias the graph carries (its attn_mask): its uid and the element strides of
/// its batch, query-head and query-row axes. The key axis is unit stride, and an axis of
/// extent 1 broadcasts through stride 0.
struct AttentionDenseBias
{
    int64_t uid = 0;
    std::array<int32_t, 3> strides{0, 0, 0};
};

/// The tensor uids and derived scalars a matched dense-attention graph binds.
struct AttentionDenseBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t causal = 0;
    int64_t slidingWindow = 0;
    float scale = 0.0F;
    /// The stats output the kernel writes the LSE to, or ATTENTION_DENSE_NO_STATS. Only an
    /// arch that serves stats (AttentionDenseArch::servesStats) ever binds one.
    int64_t stats = ATTENTION_DENSE_NO_STATS;
    /// The element strides the kernel addresses Q, K, V and O with, in
    /// attention_dense::DENSE_STRIDE_NAMES order, in canonical form
    /// (attention_dense::canonicalDenseStrides: an extent-1 axis carries 0, or the head size
    /// for the token axis, whatever the graph declared). prepare() takes them from the
    /// predicate that admitted the operands (attention_dense::denseStridesAdmit), so for
    /// packed BSHD operands they equal rocKE's `packed_launch_strides`; the saved-plan
    /// reader re-runs that predicate on them. Passed only under a contract whose kernels
    /// take them (attention_dense::takesStrides).
    AttentionDenseStrides strides{};
    /// The planned kernel's own block_n, saved by a contract that takes strides so the
    /// restore's slab check uses the tile the kernel was built with (prepare() sets it from
    /// the candidate; zero in a binding read from a legacy contract, which does not save it).
    int64_t blockN = 0;
    /// Whether the planned kernel's body reads the strides (AttentionDenseArch::readsStrides
    /// at prepare() time), saved by a contract that takes strides. A kernel whose body does not
    /// read them addresses packed storage, so its plan holds packed strides only.
    bool readsStrides = false;
    /// The additive bias the kernel adds to the scaled scores. Only an arch that serves a
    /// bias (AttentionDenseArch::servesBias) ever binds one.
    std::optional<AttentionDenseBias> bias = std::nullopt;
};

/// The graph facts the matcher and prepare() both need, derived once from the tensors by
/// sdpa_graph::problemFor().
using AttentionDenseProblem = sdpa_graph::SdpaShape;

/// Everything a dense-attention launch is computed from.
struct AttentionDenseLaunchInputs
{
    AttentionDenseBinding binding;
    AttentionDenseProblem problem;
    int64_t blockM = 0;
    /// The candidate's launch grid: the qb_minor grid unless its arch reads another
    /// (AttentionDenseArch::candidateGrid).
    attention_dense::AttentionDenseGrid grid;
};

namespace attention_dense
{

/// The launch values both engines' contracts store. `scale` is stored as a double, which
/// holds every float exactly. `data_type` is the DataType enumerator name.
inline constexpr std::string_view Q_UID_VALUE = "q.uid";
inline constexpr std::string_view K_UID_VALUE = "k.uid";
inline constexpr std::string_view V_UID_VALUE = "v.uid";
inline constexpr std::string_view O_UID_VALUE = "o.uid";
inline constexpr std::string_view CAUSAL_VALUE = "causal";
inline constexpr std::string_view SLIDING_WINDOW_VALUE = "sliding_window";
inline constexpr std::string_view BATCH_VALUE = "batch";
inline constexpr std::string_view SEQLEN_Q_VALUE = "seqlen_q";
inline constexpr std::string_view SEQLEN_KV_VALUE = "seqlen_kv";
inline constexpr std::string_view NUM_QUERY_HEADS_VALUE = "num_query_heads";
inline constexpr std::string_view NUM_KV_HEADS_VALUE = "num_kv_heads";
inline constexpr std::string_view HEAD_SIZE_VALUE = "head_size";
inline constexpr std::string_view BLOCK_M_VALUE = "block_m";
inline constexpr std::string_view SCALE_VALUE = "scale";
inline constexpr std::string_view DATA_TYPE_VALUE = "data_type";
inline constexpr std::string_view STRIDE_LAYOUT_VALUE = "stride_layout";
/// Saved beside the twelve strides by a contract that takes them: the kernel's block_n, and
/// whether its body reads the strides (0 or 1).
inline constexpr std::string_view BLOCK_N_VALUE = "block_n";
inline constexpr std::string_view READS_STRIDES_VALUE = "reads_strides";
/// The stats output's uid, or ATTENTION_DENSE_NO_STATS. Saved only by an arch that serves
/// stats; it also selects the kernel arguments a restored plan launches.
inline constexpr std::string_view STATS_UID_VALUE = "stats.uid";
/// The launch grid, saved only by an arch whose catalog carries more than the qb_minor
/// grid: `grid_order` (qb_minor, bt_hkv_minor, hq_minor_swz or persistent), and on the
/// persistent grid its `persist_decode` and `num_persistent` (empty and 0 off it). They
/// also select the kernel arguments a restored plan launches. The fast-division magic
/// numbers and the grid are recomputed, never saved.
inline constexpr std::string_view GRID_ORDER_VALUE = "grid_order";
inline constexpr std::string_view PERSIST_DECODE_VALUE = "persist_decode";
inline constexpr std::string_view NUM_PERSISTENT_VALUE = "num_persistent";
/// The additive bias's uid and its batch, query-head and query-row element strides, saved
/// only by an arch that serves a bias and only for a plan that has one (its bias contract).
inline constexpr std::string_view BIAS_UID_VALUE = "bias.uid";
inline constexpr std::string_view BIAS_STRIDE_BATCH_VALUE = "bias.stride_batch";
inline constexpr std::string_view BIAS_STRIDE_HEAD_VALUE = "bias.stride_head";
inline constexpr std::string_view BIAS_STRIDE_QUERY_VALUE = "bias.stride_query";

/// The one stride layout the kernels support. prepare() refuses every other layout.
inline constexpr std::string_view BSHD_LAYOUT = "bshd";

/// The KMD field both manifests declare: 1 for a ragged build, 0 for a shape-generic one.
/// kernelMatches serves only the latter, and each arch's stride rule (candidateReadsStrides)
/// reads it as rocKE's `_reads_dense_strides` does.
inline constexpr std::string_view RAGGED_FIELD = "ragged";

/// Whether rocKE's additive-bias bodies read the twelve strides. They do not, on either arch:
/// they address packed Q, K, V and O (`_reads_dense_strides`). Three rules read this one value:
/// the kernel match's (AttentionDenseArch::readsStrides), the graph match's decline of a bias
/// over operands that are not packed BSHD, and the saved-plan reader's refusal of a bias plan
/// saved with `reads_strides` 1.
///
/// One value for both arches and every bias body, so it is a fact, not a switch: it holds only
/// while no bias body of either arch reads the strides. rocKE keeps the rule per arch and per
/// body, and a port of the strides to the bias bodies may reach one arch, or some bodies, first.
/// Setting this to true for such a port would serve strided graphs to the bias bodies of the
/// other arch too. A partial port instead moves the bias exclusion back into each arch's
/// candidateReadsStrides (which then answers for its bias bodies), drops this constant and its
/// early-out in readsStrides, and removes the graph-match decline and the bias-contract
/// `reads_strides` refusal, which the per-candidate rule then covers.
inline constexpr bool BIAS_BODIES_READ_STRIDES = false;

/// The predicate's view of @p problem for a candidate whose tile has @p blockN key rows.
DenseStrideProblem denseStrideProblem(const AttentionDenseProblem& problem, int64_t blockN);

/**
 * @brief The packed BSHD launch strides of @p problem in canonical form, in
 *        DENSE_STRIDE_NAMES order, or nullopt when one is not a non-negative i32.
 *
 * rocKE's `packed_launch_strides` (attention_dense::packedLaunchStrides of the predicate's
 * header, which has no i32 bound) restricted to values a kernel argument holds; named apart
 * from it so a caller that needs kernel arguments cannot pick the unbounded one by its argument
 * type. For a tensor of H heads, S tokens and head size D stored token-major with the head
 * fastest: batch stride S*H*D, head stride D, token stride H*D; an extent-1 axis then carries
 * 0, or D for the token axis.
 */
std::optional<AttentionDenseStrides> packedI32LaunchStrides(const AttentionDenseProblem& problem);

/// The kernel arguments a launch contract passes.
enum class AttentionDenseAbi
{
    /// q, k, v, o, scale, batch, seqlen_q, seqlen_kv: kernels that bake the head counts.
    BAKED_HEAD_COUNTS,
    /// The same, then num_query_heads and num_kv_heads: kernels that read them at runtime.
    RUNTIME_HEAD_COUNTS,
    /// RUNTIME_HEAD_COUNTS, then additive_bias_ptr and the bias's batch, query-head and
    /// query-row element strides (last, after any lse_ptr): kernels that add a bias and
    /// address packed Q/K/V/O. The form of the bias contracts that predate the strides.
    RUNTIME_HEAD_COUNTS_WITH_BIAS,
    /// RUNTIME_HEAD_COUNTS's arguments, then the twelve DENSE_STRIDE_NAMES strides last:
    /// kernels that take the twelve strides (a body that does not read them ignores them).
    RUNTIME_STRIDES,
    /// RUNTIME_STRIDES's arguments with the bias pointer and its three strides before the
    /// twelve strides: kernels that add a bias and take the strides (a body that does not
    /// read them ignores them).
    RUNTIME_STRIDES_WITH_BIAS
};

/// Whether a kernel of @p abi reads num_query_heads and num_kv_heads at runtime.
constexpr bool takesHeadCounts(AttentionDenseAbi abi)
{
    return abi != AttentionDenseAbi::BAKED_HEAD_COUNTS;
}

/// Whether a kernel of @p abi takes the twelve strides, and its contract saves them.
constexpr bool takesStrides(AttentionDenseAbi abi)
{
    return abi == AttentionDenseAbi::RUNTIME_STRIDES
           || abi == AttentionDenseAbi::RUNTIME_STRIDES_WITH_BIAS;
}

/// Whether a kernel of @p abi takes the bias pointer and its three strides, and its contract
/// saves the bias values.
constexpr bool addsBias(AttentionDenseAbi abi)
{
    return abi == AttentionDenseAbi::RUNTIME_HEAD_COUNTS_WITH_BIAS
           || abi == AttentionDenseAbi::RUNTIME_STRIDES_WITH_BIAS;
}

/// A versioned dispatch name, under which a plan's launch values are saved, and the kernel
/// arguments a launch under it passes.
struct AttentionDenseContract
{
    std::string_view symbol;
    AttentionDenseAbi abi = AttentionDenseAbi::RUNTIME_HEAD_COUNTS;
};

/// How many superseded contracts an arch can keep restoring (AttentionDenseArch). A change of the
/// kernel arguments supersedes two contracts at once, the current one and the bias one. gfx950
/// keeps three today (v1, v2, v3), so 8 holds the next two such changes (3 + 2 + 2 = 7) with a
/// slot to spare. A larger cap costs only empty slots in each descriptor.
inline constexpr std::size_t ATTENTION_DENSE_MAX_LEGACY_CONTRACTS = 8;

/// The part a contract plays in its arch's contract table (AttentionDenseArch::setContracts).
enum class AttentionDenseContractRole
{
    /// The contract the catalog dispatches through, and plans without a bias are saved under.
    CURRENT,
    /// The contract plans with an additive bias are saved under. The current handler serves it.
    BIAS,
    /// A superseded contract whose plans still restore, through a handler of its own.
    LEGACY
};

/// One row of an arch's contract table: a contract and the part it plays.
struct AttentionDenseContractRow
{
    AttentionDenseContract contract;
    AttentionDenseContractRole role = AttentionDenseContractRole::LEGACY;
};

/// Whether @p table can describe one arch: exactly one current row, at most one bias row and
/// that one after the current row, and no more legacy rows than
/// ATTENTION_DENSE_MAX_LEGACY_CONTRACTS. Each engine static_asserts it on its table.
///
/// Rows are oldest first, and the bias contract is versioned with the current one
/// (AttentionDenseArch::biasContract): a new current version needs a new bias version after it.
/// So a bump that adds a current row but leaves the old bias row in place puts the bias row
/// before the current one, and fails here instead of saving new bias plans under the old name.
/// The rule is one way: a new bias version alone (current, then legacy, then bias) passes.
template <std::size_t N>
constexpr bool isContractTable(const std::array<AttentionDenseContractRow, N>& table)
{
    std::size_t current = 0;
    std::size_t bias = 0;
    std::size_t legacy = 0;
    bool biasBeforeCurrent = false;
    for(const auto& row : table)
    {
        biasBeforeCurrent
            = biasBeforeCurrent || (row.role == AttentionDenseContractRole::BIAS && current == 0);
        current += row.role == AttentionDenseContractRole::CURRENT ? 1 : 0;
        bias += row.role == AttentionDenseContractRole::BIAS ? 1 : 0;
        legacy += row.role == AttentionDenseContractRole::LEGACY ? 1 : 0;
    }
    return current == 1 && bias <= 1 && !biasBeforeCurrent
           && legacy <= ATTENTION_DENSE_MAX_LEGACY_CONTRACTS;
}

/// The contract of @p table's first row in @p role, or an empty one when no row has it.
template <std::size_t N>
constexpr AttentionDenseContract
    contractInRole(const std::array<AttentionDenseContractRow, N>& table,
                   AttentionDenseContractRole role)
{
    for(const auto& row : table)
    {
        if(row.role == role)
        {
            return row.contract;
        }
    }
    return {};
}

/// The tokens an engine's graph match binds, spelled per engine.
struct AttentionDenseTokens
{
    std::string_view q;
    std::string_view k;
    std::string_view v;
    std::string_view o;
    /// The mask type graphMatches derived, so kernelMatches does not re-derive it.
    std::string_view causal;
    /// The sliding window the candidate must have been built with: the number of keys a
    /// query attends, counting the diagonal (the graph's left_bound), or 0 for none.
    std::string_view slidingWindow;
    /// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
    std::string_view scaleBits;
    /// The stats output's uid, or ATTENTION_DENSE_NO_STATS. Bound only by an arch that
    /// serves stats; empty for one that does not.
    std::string_view stats;
    /// 1 when the graph carries an additive bias (attn_mask), else 0. Bound only by an arch
    /// that serves a bias; empty for one that does not. The four tokens below are bound only
    /// when it is 1.
    std::string_view bias;
    std::string_view biasUid;
    std::string_view biasStrideBatch;
    std::string_view biasStrideHead;
    std::string_view biasStrideQuery;
};

/// The grid and block a variant must launch with.
struct AttentionDenseGeometry
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;
};

/**
 * @brief Everything the shared matching and dispatch ask of one arch.
 *
 * Each engine defines one at namespace scope in its own native file. The function pointers
 * are that arch's restatement of its own Python (its Gfx9xxAttentionDenseGeometry.hpp),
 * except the keyless-window rule (windowLeavesRowsWithoutKeys), which is this engine's
 * own policy.
 */
struct AttentionDenseArch
{
    /// "gfx942" or "gfx950": the head of every exception this module throws for the arch.
    std::string_view label;
    /// The engine name its decline lines carry.
    std::string_view engineName;
    AttentionDenseTokens tokens;
    /// Whether (blockM, blockN) is a tile the kernel is built with at headSize. Total over
    /// every int64 triple.
    bool (*isSupportedTile)(int64_t headSize, int64_t blockM, int64_t blockN) = nullptr;
    /// The launch geometry of a candidate's own block_m over a graph's shape. Throws
    /// HipdnnPluginException with HIPDNN_PLUGIN_STATUS_BAD_PARAM for a block_m the arch
    /// does not build or a non-positive extent.
    AttentionDenseGeometry (*geometry)(int64_t blockM,
                                       int64_t seqLenQ,
                                       int64_t numQueryHeads,
                                       int64_t batch,
                                       const std::string& kernelName)
        = nullptr;
    /// Whether this arch's graph match serves a sliding-window mask. An arch that does not
    /// declines every windowed mask.
    bool servesWindows = false;
    /// Whether a candidate writes the softmax statistics (LSE) through a trailing lse_ptr,
    /// read from a KMD field only that arch's manifest declares; nullopt for a record
    /// without a valid value. nullptr for an arch whose catalog has no LSE kernel: its
    /// graph match declines every stats output under the one cause 'stats', binds no stats
    /// token, and its launch values carry no stats uid.
    std::optional<bool> (*candidateWritesLse)(
        const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
        = nullptr;
    /// The launch grid a candidate is built for, read from KMD fields only that arch's
    /// manifest declares; nullopt for a record without a valid one, which kernelMatches
    /// declines. nullptr for an arch whose catalog has only the qb_minor grid: every
    /// candidate launches it, and its launch values carry no grid.
    std::optional<AttentionDenseGrid> (*candidateGrid)(
        const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
        = nullptr;
    /// Whether a candidate's body, taken as one without an additive bias, addresses Q, K, V
    /// and O through the twelve stride arguments, read from KMD knob fields only that arch's
    /// manifest declares. It does not look at the bias field: for a bias record it answers as
    /// for the same body without the bias, which may be true although the bias body reads
    /// packed storage only. Asked only through readsStrides, which answers false for the bias
    /// bodies first (BIAS_BODIES_READ_STRIDES). The kernels of a catalog all take the
    /// arguments; one whose body does not read them addresses packed BSHD storage, so a graph
    /// whose operands are not packed is served only by candidates for which readsStrides is
    /// true. Required (isConsistent): the graph match admits strided layouts on every arch, so
    /// an arch none of whose kernels reads the strides sets a hook that answers false.
    bool (*candidateReadsStrides)(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
        = nullptr;
    /// Whether a candidate adds the additive bias, which also appends four kernel arguments
    /// (see AttentionDenseAbi::RUNTIME_HEAD_COUNTS_WITH_BIAS), read from a KMD field only
    /// that arch's manifest declares; nullopt for a record without a valid value, which
    /// kernelMatches declines. nullptr for an arch whose catalog has no bias kernel: its
    /// graph match declines an attn_mask under the cause 'attn_mask', binds no bias token,
    /// and its launch values carry no bias.
    std::optional<bool> (*candidateAddsBias)(
        const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
        = nullptr;
    // The three contract roles below are filled from the arch's one contract table
    // (setContracts). A contract's value set follows its ABI (launchValues): the strides with
    // block_n and reads_strides where it takes them, `stride_layout` where it does not, and the
    // bias values where it adds a bias.

    /// The contract the engine's catalog dispatches through and new plans are saved under.
    AttentionDenseContract currentContract;
    /// The superseded contracts whose plans still restore, each launching its own older
    /// kernel's arguments through a handler of its own; an empty symbol marks an unused slot.
    std::array<AttentionDenseContract, ATTENTION_DENSE_MAX_LEGACY_CONTRACTS> legacyContracts{};
    /// The contract a plan with an additive bias is saved under, for an arch that serves one
    /// (an empty symbol for one that does not). Its values are currentContract's plus the
    /// bias values, and it launches AttentionDenseAbi::RUNTIME_STRIDES_WITH_BIAS. The current
    /// handler serves both contracts, since a plan may only be saved under a name registered
    /// to the handler that saves it: a graph without a bias saves under currentContract, one
    /// with a bias under this one. A legacy handler has no bias form. The bias contract is
    /// versioned with currentContract: a change that needs a new current version needs a new
    /// bias version too, and both superseded ones become legacy rows.
    AttentionDenseContract biasContract;

    /// Fills currentContract, biasContract and legacyContracts from @p table, the arch's one
    /// list of its contracts (isContractTable). Legacy rows keep their table order.
    template <std::size_t N>
    constexpr void setContracts(const std::array<AttentionDenseContractRow, N>& table)
    {
        std::size_t slot = 0;
        for(const auto& row : table)
        {
            if(row.role == AttentionDenseContractRole::CURRENT)
            {
                currentContract = row.contract;
            }
            else if(row.role == AttentionDenseContractRole::BIAS)
            {
                biasContract = row.contract;
            }
            else
            {
                legacyContracts.at(slot++) = row.contract;
            }
        }
    }

    /// The contract plans saved under @p symbol restore through: the current one, the bias
    /// one (where served) or a legacy one; nullptr for any other name, the unversioned one
    /// included.
    constexpr const AttentionDenseContract* contractNamed(std::string_view symbol) const
    {
        if(symbol == currentContract.symbol)
        {
            return &currentContract;
        }
        if(servesBias() && symbol == biasContract.symbol)
        {
            return &biasContract;
        }
        for(const auto& legacy : legacyContracts)
        {
            if(!legacy.symbol.empty() && symbol == legacy.symbol)
            {
                return &legacy;
            }
        }
        return nullptr;
    }

    constexpr bool servesSlidingWindow() const
    {
        return servesWindows;
    }

    constexpr bool servesBias() const
    {
        return candidateAddsBias != nullptr;
    }

    constexpr bool servesStats() const
    {
        return candidateWritesLse != nullptr;
    }

    constexpr bool servesGrids() const
    {
        return candidateGrid != nullptr;
    }

    /// Whether @p kernel's body reads the twelve strides: false for a body that adds a bias
    /// (BIAS_BODIES_READ_STRIDES) or whose bias field is malformed, else the arch's own rule
    /// (candidateReadsStrides). The kernel match, prepare() and the served-set census all ask
    /// this, so the bias exclusion is stated once for both arches.
    bool readsStrides(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel) const;

    /// Whether the descriptor is coherent. Every arch has its stride rule
    /// (candidateReadsStrides), and a named current contract whose kernels take the strides
    /// and no bias: the graph match admits strided layouts on every arch, so a current
    /// contract without the strides would launch a strided plan as a packed one. No two
    /// contracts share a name, so a saved name selects exactly one. A served stats output has
    /// its token spelled. A bias hook and a bias contract come together, with all five bias
    /// tokens spelled and a bias contract whose kernels take the strides and the bias. A hook
    /// without its contract would otherwise prepare under an empty name and the default
    /// RUNTIME_HEAD_COUNTS (10 arguments); a contract without its hook would never be
    /// reached. Each engine static_asserts it on its descriptor.
    constexpr bool isConsistent() const
    {
        if(candidateReadsStrides == nullptr || currentContract.symbol.empty()
           || !takesStrides(currentContract.abi) || addsBias(currentContract.abi))
        {
            return false;
        }
        if(servesStats() && tokens.stats.empty())
        {
            return false;
        }
        if(servesBias() != !biasContract.symbol.empty())
        {
            return false;
        }
        if(servesBias()
           && (tokens.bias.empty() || tokens.biasUid.empty() || tokens.biasStrideBatch.empty()
               || tokens.biasStrideHead.empty() || tokens.biasStrideQuery.empty()
               || !takesStrides(biasContract.abi) || !addsBias(biasContract.abi)))
        {
            return false;
        }
        return contractNamesAreDistinct();
    }

private:
    /// Whether the named contracts (current, bias where served, and each used legacy slot)
    /// have pairwise distinct names.
    constexpr bool contractNamesAreDistinct() const
    {
        std::array<std::string_view, ATTENTION_DENSE_MAX_LEGACY_CONTRACTS + 2> names{};
        std::size_t count = 0;
        names.at(count++) = currentContract.symbol;
        if(servesBias())
        {
            names.at(count++) = biasContract.symbol;
        }
        for(const auto& legacy : legacyContracts)
        {
            if(!legacy.symbol.empty())
            {
                names.at(count++) = legacy.symbol;
            }
        }
        for(std::size_t first = 0; first < count; ++first)
        {
            for(std::size_t second = first + 1; second < count; ++second)
            {
                if(names.at(first) == names.at(second))
                {
                    return false;
                }
            }
        }
        return true;
    }
};

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * Each decline of a graph with an SDPA-forward node logs its cause at INFO; any other
 * graph declines silently.
 *
 * @warning Returning std::nullopt empties the engine's WHOLE catalog and skips EVERY
 *          remaining pack, not just this one.
 */
std::optional<hipdnn_plugin_sdk::ingestor::BoundTokens>
    graphMatches(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                 const AttentionDenseArch& arch);

/// Kernel-scoped applicability: does THIS candidate's baked metadata fit the graph?
bool kernelMatches(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                   const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
                   const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                   const AttentionDenseArch& arch);

/// A candidate's integer metadata field, or nullopt when it is absent or holds another
/// type. Total: never throws, so a malformed record declines instead of faulting a match.
std::optional<int64_t> integerMetadata(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                                       std::string_view field);

/// A candidate's string metadata field, or nullopt when it is absent or holds another type.
/// Total, as integerMetadata.
std::optional<std::string>
    stringMetadata(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                   std::string_view field);

/// Whether the candidate adds a bias (which also appends four kernel arguments), or nullopt
/// for a malformed record: the descriptor hook of an arch that serves a bias
/// (AttentionDenseArch::candidateAddsBias). An absent use_additive_bias is the KMD default
/// 0; a field holding another type or a value other than 0/1 is malformed and declines,
/// rather than reading as a plain kernel. Total: never throws.
std::optional<bool>
    candidateAdditiveBias(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/**
 * @brief Refuses a binding and a launch contract that disagree on the additive bias.
 *
 * A binding that carries a bias must be prepared under the contract whose kernel takes the
 * bias pointer and strides (RUNTIME_HEAD_COUNTS_WITH_BIAS), and only then. The other
 * contracts pass no bias, so a bias binding under one would launch a kernel that silently
 * ignores it (a wrong result, not a crash), and a bias contract without a binding has no
 * pointer to pass. prepare() and restoreLaunch() derive each of the two from the other, so
 * this holds today; it is checked once before the geometry is applied, so a change to either
 * derivation fails the plan instead of the launch.
 *
 * @param archLabel  The arch's label, for the message.
 * @param kernelName The kernel's name, for the message.
 * @throws HipdnnPluginException with INTERNAL_ERROR (a defect of the caller, not a property
 *         of the graph) naming which side carries the bias.
 */
void requireBiasBindingMatchesContract(const AttentionDenseBinding& binding,
                                       const AttentionDenseContract& contract,
                                       std::string_view archLabel,
                                       std::string_view kernelName);

/**
 * @brief Refuses a launch whose output shares bytes with its Q, K or V view.
 *
 * Asked by AttentionDenseKernelArguments' constructor, which every launch builds its kernel
 * arguments with: the one point where the four device pointers and the strides the kernel is
 * launched with are both known (attention_dense::denseOutputOverlapsInput says why the kernels
 * cannot serve such an output). A null pointer is skipped: it names no buffer to share.
 *
 * @param operands   The device pointers of Q, K, V and O, in that order.
 * @param archLabel  The arch's label, for the message.
 * @throws HipdnnPluginException with BAD_PARAM naming the input the output overlaps.
 */
void requireOutputDisjointFromInputs(const AttentionDenseLaunchInputs& launch,
                                     const std::array<const void*, 4>& operands,
                                     std::string_view archLabel);

/**
 * @brief Refuses a launch whose stats (LSE) output shares bytes with its Q, K, V or O view, or
 *        with the additive bias its kernel reads.
 *
 * Asked by AttentionDenseKernelArguments' constructor after requireOutputDisjointFromInputs, for
 * a launch that writes LSE (its binding has a stats output); a launch that writes none passes
 * the kernel no lse_ptr, so its LSE pointer is not looked at. The stats output is the packed
 * FP32 [B, Hq, Sq, 1] tensor the graph match admits, and the kernel writes it while other
 * workgroups still read the inputs and write O (attention_dense::denseStatsOverlapsOperand gives
 * the spans compared). A null pointer is skipped: it names no buffer to share.
 *
 * @param operands   The device pointers of Q, K, V and O, in that order.
 * @param lse        The device pointer of the stats output.
 * @param bias       The device pointer of the additive bias, or null for a launch whose contract
 *                   adds no bias (its kernel reads none). Its span comes from the binding's bias
 *                   strides; a binding without a bias compares none.
 * @param archLabel  The arch's label, for the message.
 * @throws HipdnnPluginException with BAD_PARAM naming the operand the stats output overlaps.
 */
void requireStatsDisjointFromOperands(const AttentionDenseLaunchInputs& launch,
                                      const std::array<const void*, 4>& operands,
                                      const void* lse,
                                      const void* bias,
                                      std::string_view archLabel);

/// The launch shape of @p launch, for AttentionDensePersistent.hpp's grid functions.
AttentionDenseGridShape gridShape(const AttentionDenseLaunchInputs& launch);

/// A candidate's (block_m, block_n) tile, read from its completed metadata.
struct AttentionDenseTile
{
    int64_t blockM = 0;
    int64_t blockN = 0;
};

/**
 * @brief The candidate's tile, or nullopt when it has none the arch can launch.
 *
 * Validated against the arch's isSupportedTile. A record omitting block_m or block_n
 * reaches here completed to 256/64 through the KMD defaults, so an absent field means an
 * uncompleted or malformed record: it is declined rather than given a default here,
 * because a substituted tile launches a binary with another binary's grid.
 */
std::optional<AttentionDenseTile>
    candidateTile(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                  const AttentionDenseArch& arch);

/**
 * @brief The native dispatch behind an engine's UDD: one handler per handler-owning contract
 *        of the arch, and their registration.
 *
 * The current handler serves the current contract and the bias one; each legacy contract has
 * a handler of its own, which launches its own older kernel's arguments. A handler saves new
 * plans under its contract's name and restores only plans saved under a name it serves. Built
 * from the descriptor, whose contracts come from the arch's one contract table, so a row added
 * there is dispatched and registered with no other edit. Process-lifetime: the caller keeps it
 * alive for as long as its handlers are registered, and @p kpackLoader and @p arch for as long
 * as it lives.
 */
class AttentionDenseDispatchHandlers
{
public:
    AttentionDenseDispatchHandlers(const compilation::KpackKernelLoader& kpackLoader,
                                   const AttentionDenseArch& arch);

    /// Registers the current handler under @p unversionedSymbol (the name the catalog's
    /// descriptors carry) and under the current and bias contracts' names, and each legacy
    /// handler under its own contract's name.
    void registerWith(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope,
                      std::string_view unversionedSymbol) const;

private:
    const AttentionDenseArch& _arch;
    std::unique_ptr<hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>> _current;
    /// One per used slot of the arch's legacyContracts, in the same slot; null elsewhere.
    std::array<std::unique_ptr<hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>>,
               ATTENTION_DENSE_MAX_LEGACY_CONTRACTS>
        _legacy;
};

/**
 * @brief The launch state prepare() builds a plan from, for @p kernel matched to the graph
 *        @p bound was bound from: the bindings, the shape, the candidate's tile and grid.
 *
 * Everything prepare() decides before it touches the code object. It re-checks what
 * graphMatches and kernelMatches established, as defence in depth: all four operands go
 * through the stride predicate (denseStridesAdmit) with the candidate's own block_n, and a
 * layout other than packed BSHD needs a candidate whose body reads the strides
 * (AttentionDenseArch::readsStrides). The binding's strides are the predicate's
 * canonical ones, the values the launch passes. Separate from prepare() so the state is
 * observable without a code archive.
 *
 * @throws HipdnnPluginException with BAD_PARAM when a token is missing, an operand is not
 *         what the kernel can address, or the candidate has no tile or grid the arch launches.
 */
AttentionDenseLaunchInputs
    resolveLaunch(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                  const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
                  const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                  const AttentionDenseArch& arch);

/**
 * @brief The kernel signature @p abi passes, for a kernel on @p grid that writes LSE
 *        (@p writesLse) or not.
 *
 * q_ptr, k_ptr, v_ptr, o_ptr, scale, batch, seqlen_q, seqlen_kv; then num_query_heads and
 * num_kv_heads unless the ABI bakes them; on the persistent grid (gfx942 only) a
 * `<d>_magic`, `<d>_shift` i32 pair per fast-division divisor of its decode; lse_ptr for a
 * kernel that writes LSE (gfx942 only); under a bias ABI the bias pointer and its three i32
 * strides; and under RUNTIME_STRIDES and RUNTIME_STRIDES_WITH_BIAS the twelve
 * DENSE_STRIDE_NAMES i32 strides last. The names are the rocKE builder's own (see the
 * definition for why they are load-bearing); the offsets mirror the packed kernarg layout.
 */
std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>
    kernelSignature(AttentionDenseAbi abi, const AttentionDenseGrid& grid, bool writesLse);

/**
 * @brief The kernel arguments one launch passes, in kernelSignature's order, and the
 *        storage they point into.
 *
 * Built from a launch's inputs and the ABI of the contract its code object was checked
 * against: the four operand pointers (@p operands, in the order q, k, v, o) and scale; batch,
 * seqlen_q and seqlen_kv; the head counts unless the ABI bakes them; @p fastDiv's pairs (empty
 * off the persistent grid); @p lse when the binding has a stats output; under a bias ABI
 * @p bias and the binding's three bias strides; and under a strides ABI the binding's twelve
 * strides. A plan restored under a legacy contract therefore passes exactly its own older
 * kernel's arguments; one without strides passes none.
 *
 * The constructor first refuses an output that shares bytes with an input
 * (requireOutputDisjointFromInputs) and, for a launch that writes LSE, a stats output that shares
 * bytes with Q, K, V, O or the bias the contract passes (requireStatsDisjointFromOperands), so no
 * launch can pass the kernel such a pointer set: the checks sit where the pointers become kernel
 * arguments, not in a caller that could skip them.
 *
 * Not copyable or movable, because params() points into the object.
 *
 * @param archLabel The arch's label, for the refusal's message.
 * @throws HipdnnPluginException with BAD_PARAM when the output overlaps Q, K or V, or the stats
 *         output overlaps Q, K, V, O or the bias.
 */
class AttentionDenseKernelArguments
{
public:
    AttentionDenseKernelArguments(const AttentionDenseLaunchInputs& launch,
                                  AttentionDenseAbi abi,
                                  const std::vector<AttentionDenseFastDivPair>& fastDiv,
                                  const std::array<void*, 4>& operands,
                                  void* lse,
                                  void* bias,
                                  std::string_view archLabel);
    AttentionDenseKernelArguments(const AttentionDenseKernelArguments&) = delete;
    AttentionDenseKernelArguments& operator=(const AttentionDenseKernelArguments&) = delete;
    AttentionDenseKernelArguments(AttentionDenseKernelArguments&&) = delete;
    AttentionDenseKernelArguments& operator=(AttentionDenseKernelArguments&&) = delete;
    ~AttentionDenseKernelArguments() = default;

    /// One pointer per kernel argument, to its value, in kernarg order.
    const std::vector<void*>& params() const
    {
        return _params;
    }

    /// The same, as the launcher takes it.
    void** data()
    {
        return _params.data();
    }

private:
    std::array<void*, 4> _operands;
    float _scale;
    std::array<int32_t, 5> _shape;
    std::vector<AttentionDenseFastDivPair> _fastDiv;
    void* _lse;
    void* _bias;
    std::array<int32_t, 3> _biasStrides{};
    std::array<int32_t, ATTENTION_DENSE_STRIDE_COUNT> _strides{};
    std::vector<void*> _params;
};

/// The launch values a plan under @p contract saves. Every contract saves the fifteen both
/// engines share (the uids, mask, shape, `block_m`, `scale` and `data_type` values above). A
/// contract that takes strides (takesStrides) then saves the binding's twelve strides under
/// their DENSE_STRIDE_NAMES, then `block_n` and `reads_strides`; every other contract saves
/// `stride_layout` (`bshd`) instead. One that adds a bias (addsBias) also saves the binding's
/// `bias.*` values. `stats.uid` follows where the arch serves stats, and `grid_order`,
/// `persist_decode` and `num_persistent` where it serves grids past qb_minor.
hipdnn_plugin_sdk::ingestor::MetadataValues launchValues(const AttentionDenseBinding& binding,
                                                         const AttentionDenseProblem& problem,
                                                         int64_t blockM,
                                                         const AttentionDenseArch& arch,
                                                         const AttentionDenseContract& contract,
                                                         const AttentionDenseGrid& grid = {});

/**
 * @brief Reads @p arch's launch values back into the launch state.
 *
 * The dispatch name must be one of the arch's contracts (the current one, the bias one where
 * served, or a legacy one); anything else, including the unversioned name, is refused against
 * the current one. The value set is that contract's (launchValues). `batch`, `seqlen_q`,
 * `seqlen_kv`, the head counts and `head_size` must be in [1, INT32_MAX]; the kernel takes the
 * first three as 32-bit integers. `block_m` must be one the arch's geometry accepts; a contract
 * without strides does not save `block_n`, so its plans do not re-run the arch's full tile rule
 * (the strided contracts do, below). `scale` must convert to a float and back without change,
 * within [2^-64, 2^4]. `data_type` must be a type the kernel is built for. A contract without
 * strides saves `stride_layout`, which must be `bshd`; one with strides saves the twelve, each
 * an i32 in [0, INT32_MAX], then the kernel's `block_n` (a tile the arch builds with `block_m`)
 * and `reads_strides` (0 or 1), and the restore re-runs the graph match's gate on them: the
 * saved dims and strides, at the default 16-byte alignment and the saved block_n, must pass
 * attention_dense::denseStridesAdmit and be in its canonical form. The strides must be the
 * packed ones when `reads_strides` is 0 (the kernel addresses packed storage), and under the bias
 * contract `reads_strides` must be 0 (BIAS_BODIES_READ_STRIDES). A legacy read fills the
 * binding's strides with the packed values; its launch does not pass them.
 *
 * Beyond those rules the saved `reads_strides` and `block_n` are trusted, as `block_m` is. A
 * saved plan's code object carries no KMD knobs (SavedKernelCode), so the restore cannot ask
 * which body or tile it is: a plan edited to say `reads_strides` 1 for a body that does not read
 * the strides restores. The stride checks above hold the saved layout to what the saved flag and
 * tile allow; they are not a second gate on the kernel. The graph and kernel match, run when
 * the plan was built, are that gate.
 *
 * As both arches' graph and kernel match require, `causal` must be 0 or 1; `sliding_window`
 * non-negative, 0 unless `causal` is 1, and not so narrow that the last query rows see no
 * key (seqlen_q - seqlen_kv >= sliding_window), except as the next sentence says;
 * `num_query_heads` a multiple of `num_kv_heads`; `head_size` 64 or 128; K/V below 2^31
 * bytes and Q/O below 2^31 elements; and `seqlen_q` a multiple of `block_m`, since the
 * kernel addresses Q and O without a bounds check. An arch that serves windows (gfx942)
 * restores a window that leaves the last query rows no key for a plan without a stats
 * output, and refuses it for a plan with one. One exception: gfx950's graph match declines
 * every window, but its reader still takes a causal one within these rules: its frozen v1/v2
 * samples hold one.
 *
 * An arch that serves stats also reads `stats.uid`, which must be ATTENTION_DENSE_NO_STATS
 * or a uid (non-negative). An arch that serves grids also reads `grid_order`, which must
 * name one; on the persistent grid `persist_decode` must name a decode the kernel reads the
 * shape for at run time (qb_major, hkv_major, bt_hkv_minor or hq_minor_swz; never `auto`
 * or gqa_pair*) and `num_persistent` be in [1, INT32_MAX], and off it they must be empty
 * and 0. The grid must be legal at the saved shape (AttentionDensePersistent.hpp
 * gridIllegality). The restore selects the kernel's argument list from the saved stats uid
 * and grid before this reader runs, so its signature check, which this reader does not
 * see, refuses a kernel whose LSE output or grid disagrees with the plan.
 *
 * An arch that serves a bias also reads, under its bias contract (and only there), the
 * bias's `bias.uid` and its three strides, each in [0, MAX_BIAS_INDEX]; the largest bias
 * element the saved shape reaches must be at most MAX_BIAS_INDEX, the same bounds the graph
 * match puts on a graph's attn_mask.
 *
 * @throws HipdnnPluginException with a load refusal when the dispatch name is neither
 *         contract, or when a value is missing, extra, of another type or out of range.
 */
AttentionDenseLaunchInputs
    readLaunchInputs(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                     const AttentionDenseArch& arch);

} // namespace attention_dense
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
