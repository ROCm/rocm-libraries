// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/packs/AttentionDenseCommon.hpp"

/**
 * @file Gfx942AttentionDenseLaunchValues.hpp
 * @brief The saved form of the gfx942 dense-attention ingestor pack's launch state: a
 *        versioned dispatch name and the named, typed values the launch is computed from.
 *        The pack also reads its saved form back into its launch inputs.
 *
 * The launch state (AttentionDenseBinding, AttentionDenseProblem) and the value names are
 * shared with the gfx950 pack and live in AttentionDenseCommon.hpp.
 *
 * The version in a dispatch name identifies the contract between the saved values and the
 * launch. A saved plan pairs an older code object with this launch. Use a new version for
 * a change to any of these:
 *   - the names, types or meanings of the values.
 *   - the kernel arguments the launch passes: their order, count, kinds, sizes or meanings.
 *   - how the launch derives grid, block or dynamic LDS from the values.
 * Keep the old version as a legacy row of the engine's contract table
 * (GFX942_ATTENTION_DENSE_CONTRACTS below) while plans saved under it must load: that row
 * registers its handler.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Contract v1: the launch passes 10 kernel arguments (q, k, v, o, scale, batch, seqlen_q,
/// seqlen_kv, num_query_heads, num_kv_heads); to a kernel on the persistent grid then a
/// `<d>_magic`, `<d>_shift` i32 pair per fast-division divisor of its decode (2 or 3
/// pairs); and to a kernel that writes LSE last lse_ptr, the stats output: 10, 11, 14, 15,
/// 16 or 17 arguments. The grid is (nqb, Hq, B), (B, Hq, nqb) or (8 nqb, Hq / 8, B) at the
/// qb_minor, bt_hkv_minor or hq_minor_swz order, or (num_persistent, 1, 1). Versions are per
/// dispatch name, so this is gfx942's first contract, not gfx950's v1. Its handler stays
/// registered so plans saved under it still load.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1
    = "hipkernel.gfx942_attention_dense.dispatch.v1";

/// Contract v2: the v1 values plus the bias uid and its batch, query-head and query-row
/// element strides (attention_dense::BIAS_*_VALUE); the launch passes the v1 arguments, then
/// additive_bias_ptr and the three strides, last after any lse_ptr: v1's 10, 11, 14, 15, 16
/// or 17 arguments plus four. Plans with an additive bias were saved under it before the
/// strides became runtime arguments; its handler stays registered so they still load.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V2
    = "hipkernel.gfx942_attention_dense.dispatch.v2";

/// Contract v3, the strided contract: v1's values with `stride_layout` replaced by the twelve
/// element strides (attention_dense::DENSE_STRIDE_NAMES), and the launch passes v1's
/// arguments and then those strides last (22, 23, 26, 27, 28 or 29 kernel arguments), for
/// kernels that take the twelve strides (a body that does not read them ignores them). It also
/// saves the kernel's `block_n` and `reads_strides` (whether its body reads them: 0 for the KDB
/// and VPF pipelines and the bias kernels, whose plans hold packed strides only). Its role
/// (current today) is the contract table's.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V3
    = "hipkernel.gfx942_attention_dense.dispatch.v3";

/// Contract v4, the strided bias contract: v3's values plus the bias uid and its batch,
/// query-head and query-row element strides (attention_dense::BIAS_*_VALUE); the launch passes
/// v3's arguments with additive_bias_ptr and the three bias strides before the twelve strides
/// (v3's 22, 23, 26, 27, 28 or 29 arguments plus four). Its role (the bias contract today) is
/// the contract table's.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V4
    = "hipkernel.gfx942_attention_dense.dispatch.v4";

/**
 * Every launch contract of the gfx942 engine, oldest first: the dispatch name its plans are
 * saved under, the kernel arguments a launch under it passes (which also decide what a plan under
 * it saves, attention_dense::launchValues), and its role. The engine's descriptor, its dispatch
 * handlers and their registration, the restore's name lookup and the two names below all derive
 * from this one table. The bias contract is versioned with the current one, so a new current
 * version is two names above, two new rows (current, then bias) and two changed rows (the
 * superseded current and bias rows turn LEGACY); isContractTable refuses a bias row left before
 * the current one. A new bias version alone is one name, one new row and one changed row. No
 * gfx942 kernel ever baked the head counts, so every row takes them at runtime.
 */
inline constexpr std::array<attention_dense::AttentionDenseContractRow, 4>
    GFX942_ATTENTION_DENSE_CONTRACTS{{
        // v1: the ten arguments, then any fast-division pairs and lse_ptr; plans save
        // `stride_layout`, the stats uid and the grid.
        {{GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1,
          attention_dense::AttentionDenseAbi::RUNTIME_HEAD_COUNTS},
         attention_dense::AttentionDenseContractRole::LEGACY},
        // v2: v1's arguments and the bias pointer and strides, last; v1's values and the bias's.
        {{GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V2,
          attention_dense::AttentionDenseAbi::RUNTIME_HEAD_COUNTS_WITH_BIAS},
         attention_dense::AttentionDenseContractRole::LEGACY},
        // v3: v1's arguments and the twelve strides, last; v1's values with the strides, block_n
        // and reads_strides in place of `stride_layout`.
        {{GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V3,
          attention_dense::AttentionDenseAbi::RUNTIME_STRIDES},
         attention_dense::AttentionDenseContractRole::CURRENT},
        // v4: v3's with the bias's four before the strides; v3's values and the bias's.
        {{GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V4,
          attention_dense::AttentionDenseAbi::RUNTIME_STRIDES_WITH_BIAS},
         attention_dense::AttentionDenseContractRole::BIAS},
    }};
static_assert(
    attention_dense::isContractTable(GFX942_ATTENTION_DENSE_CONTRACTS),
    "GFX942_ATTENTION_DENSE_CONTRACTS needs one current row, at most one bias row and that one "
    "after it (a new current version needs a new bias version), and at most "
    "ATTENTION_DENSE_MAX_LEGACY_CONTRACTS legacy rows (raise the cap to keep more)");

/// The contract new plans without an additive bias are saved under: the table's current row.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED
    = attention_dense::contractInRole(GFX942_ATTENTION_DENSE_CONTRACTS,
                                      attention_dense::AttentionDenseContractRole::CURRENT)
          .symbol;

/// The contract new plans with an additive bias are saved under: the table's bias row.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED_BIAS
    = attention_dense::contractInRole(GFX942_ATTENTION_DENSE_CONTRACTS,
                                      attention_dense::AttentionDenseContractRole::BIAS)
          .symbol;

/// Everything a gfx942 dense-attention launch is computed from.
using Gfx942AttentionDenseLaunchInputs = AttentionDenseLaunchInputs;

/// The engine's arch descriptor: its current, bias and legacy contracts, each with the kernel
/// arguments it launches (attention_dense::kernelSignature), and the per-candidate rules the
/// shared matching and dispatch apply. Process-lifetime.
const attention_dense::AttentionDenseArch& gfx942AttentionDenseArch();

/// Reads the values of the strided contract, the strided bias contract, or of the
/// `hipkernel.gfx942_attention_dense.dispatch.v1` or `.v2` contract (the legacy ones differ
/// in the bias values v2 adds and the kernel arguments the launch passes). `block_m` must be
/// one gfx942 builds: a positive multiple of 32 within a 1024-thread workgroup. For every
/// other rule, including those for the strides, the stats, the grid and the bias values, see
/// attention_dense::readLaunchInputs.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is none of
///         those contracts, or when a value is missing, extra, of another type or out of range.
Gfx942AttentionDenseLaunchInputs readGfx942AttentionDenseLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
