// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
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
 * Keep the handler of the old version registered while plans saved under it must load.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Contract v1: the launch passes 10 kernel arguments (q, k, v, o, scale, batch, seqlen_q,
/// seqlen_kv, num_query_heads, num_kv_heads); to a kernel on the persistent grid then a
/// `<d>_magic`, `<d>_shift` i32 pair per fast-division divisor of its decode (2 or 3
/// pairs); and to a kernel that writes LSE last lse_ptr, the stats output: 10, 11, 14, 15,
/// 16 or 17 arguments. The grid is (nqb, Hq, B), (B, Hq, nqb) or (8 nqb, Hq / 8, B) at the
/// qb_minor, bt_hkv_minor or hq_minor_swz order, or (num_persistent, 1, 1). New plans are
/// saved under it for a graph without an additive bias. Versions are per dispatch name, so
/// this is gfx942's first contract, not gfx950's v1.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1
    = "hipkernel.gfx942_attention_dense.dispatch.v1";

/// Contract v2: the v1 values plus the bias uid and its batch, query-head and query-row
/// element strides (attention_dense::BIAS_*_VALUE); the launch passes the v1 arguments, then
/// additive_bias_ptr and the three strides, last after any lse_ptr: v1's 10, 11, 14, 15, 16
/// or 17 arguments plus four. New plans with an additive bias are saved under it. The
/// handler of v1 also serves it, so v1 stays the contract of every plan without a bias and
/// those plans are unchanged.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V2
    = "hipkernel.gfx942_attention_dense.dispatch.v2";

/// The ALiBi contract: v1's values, and the launch also passes a null alibi_slopes_ptr
/// after the head counts and any fast-division pairs, before lse_ptr, for use_alibi
/// kernels: 11, 12, 15, 16, 17 or 18 arguments. Named rather than numbered, so it does not
/// take the next version of the plain contract.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_ALIBI_V1
    = "hipkernel.gfx942_attention_dense.dispatch.alibi.v1";

/// The values of the `hipkernel.gfx942_attention_dense.dispatch.v1`, `.v2` and `.alibi.v1`
/// contracts (attention_dense::launchValues): the sixteen both arches save, and, as gfx942
/// serves stats and grids, `stats.uid`, `grid_order`, `persist_decode` and `num_persistent`.
/// v2 adds the four bias values, which this writes exactly when the binding has a bias.
hipdnn_plugin_sdk::ingestor::MetadataValues
    gfx942AttentionDenseLaunchValues(const AttentionDenseBinding& binding,
                                     const AttentionDenseProblem& problem,
                                     int64_t blockM,
                                     const attention_dense::AttentionDenseGrid& grid = {});

/// Everything a gfx942 dense-attention launch is computed from.
using Gfx942AttentionDenseLaunchInputs = AttentionDenseLaunchInputs;

/// Reads the values of the `hipkernel.gfx942_attention_dense.dispatch.v1` or `.v2` contract
/// (they differ in the bias values v2 adds and the kernel arguments the launch passes).
/// `block_m` must be one gfx942 builds: a positive multiple of 32 within a 1024-thread CTA.
/// For every other rule, including those for the stats, grid and bias values, see
/// attention_dense::readLaunchInputs.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is neither
///         contract, or when a value is missing, extra, of another type or out of range.
Gfx942AttentionDenseLaunchInputs readGfx942AttentionDenseLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
