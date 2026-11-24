// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "engines/kernel_ingestor_engine/packs/AttentionDenseCommon.hpp"

/**
 * @file Gfx950AttentionDenseLaunchValues.hpp
 * @brief The saved form of the gfx950 dense-attention ingestor pack's launch state: a
 *        versioned dispatch name and the named, typed values the launch is computed from.
 *        The pack also reads its saved form back into its launch inputs.
 *
 * The launch state (AttentionDenseBinding, AttentionDenseProblem) and the value names are
 * shared with the gfx942 pack and live in AttentionDenseCommon.hpp.
 *
 * The version in a dispatch name identifies the contract between the saved values and the
 * launch. A saved plan pairs an older code object with this launch. Use a new version for
 * a change to any of these:
 *   - the names, types or meanings of the values.
 *   - the kernel arguments the launch passes: their order, count, kinds, sizes or meanings.
 *   - how the launch derives grid, block or dynamic LDS from the values.
 * Keep the old version as a legacy row of the engine's contract table
 * (GFX950_ATTENTION_DENSE_CONTRACTS below) while plans saved under it must load: that row
 * registers its handler.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Contract v1: the launch passes 8 kernel arguments (q, k, v, o, scale, batch, seqlen_q,
/// seqlen_kv), for kernels that bake the head counts. Its handler stays registered so plans
/// saved under it still load.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1
    = "hipkernel.gfx950_attention_dense.dispatch.v1";

/// Contract v2: the same values, and the launch also passes num_query_heads and
/// num_kv_heads (10 kernel arguments), for kernels that read the head counts at runtime.
/// Its handler stays registered so plans saved under it still load.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2
    = "hipkernel.gfx950_attention_dense.dispatch.v2";

/// Contract v3: the v2 values plus the bias uid and its batch, query-head and query-row
/// element strides (attention_dense::BIAS_*_VALUE); the launch passes the v2 arguments, then
/// additive_bias_ptr and the three strides (14 kernel arguments). Plans with an additive bias
/// were saved under it before the strides became runtime arguments; its handler stays
/// registered so they still load.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V3
    = "hipkernel.gfx950_attention_dense.dispatch.v3";

/// Contract v4, the strided contract: v2's values with `stride_layout` replaced by the twelve
/// element strides (attention_dense::DENSE_STRIDE_NAMES), and the launch passes v2's ten
/// arguments and then those strides (22 kernel arguments), for kernels that take the twelve
/// strides (a body that does not read them ignores them). It also saves the kernel's `block_n`
/// and `reads_strides` (whether its body reads them, see candidateReadsStrides: 1 for the grid
/// body and the persistent work orders that decode through the shape arguments, 0 for the
/// ragged body, the gqa_pair work orders and the bias bodies, whose plans hold packed strides
/// only). Its role (current today) is the contract table's.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V4
    = "hipkernel.gfx950_attention_dense.dispatch.v4";

/// Contract v5, the strided bias contract: v4's values plus the bias uid and its batch,
/// query-head and query-row element strides (attention_dense::BIAS_*_VALUE); the launch passes
/// v2's ten arguments, additive_bias_ptr and the three bias strides, and then the twelve
/// strides (26 kernel arguments). Its role (the bias contract today) is the contract table's.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V5
    = "hipkernel.gfx950_attention_dense.dispatch.v5";

/**
 * Every launch contract of the gfx950 engine, oldest first: the dispatch name its plans are
 * saved under, the kernel arguments a launch under it passes (which also decide what a plan under
 * it saves, attention_dense::launchValues), and its role. The engine's descriptor, its dispatch
 * handlers and their registration, the restore's name lookup and the two names below all derive
 * from this one table. The bias contract is versioned with the current one, so a new current
 * version is two names above, two new rows (current, then bias) and two changed rows (the
 * superseded current and bias rows turn LEGACY); isContractTable refuses a bias row left before
 * the current one. A new bias version alone is one name, one new row and one changed row.
 */
inline constexpr std::array<attention_dense::AttentionDenseContractRow, 5>
    GFX950_ATTENTION_DENSE_CONTRACTS{{
        // v1: the eight arguments of kernels that bake the head counts; plans save
        // `stride_layout`.
        {{GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1,
          attention_dense::AttentionDenseAbi::BAKED_HEAD_COUNTS},
         attention_dense::AttentionDenseContractRole::LEGACY},
        // v2: v1's arguments and the two head counts (10); v1's values.
        {{GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2,
          attention_dense::AttentionDenseAbi::RUNTIME_HEAD_COUNTS},
         attention_dense::AttentionDenseContractRole::LEGACY},
        // v3: v2's arguments and the bias pointer and strides (14); v2's values and the bias's.
        {{GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V3,
          attention_dense::AttentionDenseAbi::RUNTIME_HEAD_COUNTS_WITH_BIAS},
         attention_dense::AttentionDenseContractRole::LEGACY},
        // v4: v2's arguments and the twelve strides (22); the strides, block_n and reads_strides.
        {{GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V4,
          attention_dense::AttentionDenseAbi::RUNTIME_STRIDES},
         attention_dense::AttentionDenseContractRole::CURRENT},
        // v5: v4's with the bias's four before the strides (26); v4's values and the bias's.
        {{GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V5,
          attention_dense::AttentionDenseAbi::RUNTIME_STRIDES_WITH_BIAS},
         attention_dense::AttentionDenseContractRole::BIAS},
    }};
static_assert(
    attention_dense::isContractTable(GFX950_ATTENTION_DENSE_CONTRACTS),
    "GFX950_ATTENTION_DENSE_CONTRACTS needs one current row, at most one bias row and that one "
    "after it (a new current version needs a new bias version), and at most "
    "ATTENTION_DENSE_MAX_LEGACY_CONTRACTS legacy rows (raise the cap to keep more)");

/// The contract new plans without an additive bias are saved under: the table's current row.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED
    = attention_dense::contractInRole(GFX950_ATTENTION_DENSE_CONTRACTS,
                                      attention_dense::AttentionDenseContractRole::CURRENT)
          .symbol;

/// The contract new plans with an additive bias are saved under: the table's bias row.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED_BIAS
    = attention_dense::contractInRole(GFX950_ATTENTION_DENSE_CONTRACTS,
                                      attention_dense::AttentionDenseContractRole::BIAS)
          .symbol;

/// Everything a gfx950 dense-attention launch is computed from.
using Gfx950AttentionDenseLaunchInputs = AttentionDenseLaunchInputs;

/// The engine's arch descriptor: its current, bias and legacy contracts, each with the kernel
/// arguments it launches (attention_dense::kernelSignature), and the per-candidate rules the
/// shared matching and dispatch apply. Process-lifetime.
const attention_dense::AttentionDenseArch& gfx950AttentionDenseArch();

/// Reads the values of the strided contract, the strided bias contract, or of the
/// `hipkernel.gfx950_attention_dense.dispatch.v1`, `.v2` or `.v3` contract (v1 and v2 share
/// one value set and differ only in the kernel arguments the launch passes; v3 adds the bias
/// values). `block_m` must be 128 or 256, the two gfx950 builds. For every other rule, the
/// strides and the bias values included, and the one gfx950 exception to them, see
/// attention_dense::readLaunchInputs.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is none of
///         those contracts, or when a value is missing, extra, of another type or out of range.
Gfx950AttentionDenseLaunchInputs readGfx950AttentionDenseLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
