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

/// The strided contract, v3: v1's values with `stride_layout` replaced by the twelve
/// element strides (attention_dense::DENSE_STRIDE_NAMES), and the launch passes v1's
/// arguments and then those strides last (22, 23, 26, 27, 28 or 29 kernel arguments), for
/// kernels that read every stride at runtime. It also saves the kernel's `block_n` and
/// `reads_strides` (whether its body reads them: 0 for the KDB and VPF pipelines and the bias
/// kernels, whose plans hold packed strides only). New plans without an additive bias are
/// saved under it. Its version is spelled only here: the next free number of this dispatch
/// name.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED
    = "hipkernel.gfx942_attention_dense.dispatch.v3";

/// The strided bias contract, v4: v3's values plus the bias uid and its batch, query-head and
/// query-row element strides (attention_dense::BIAS_*_VALUE); the launch passes v3's
/// arguments with additive_bias_ptr and the three bias strides before the twelve strides
/// (v3's 22, 23, 26, 27, 28 or 29 arguments plus four). New plans with an additive bias are
/// saved under it. The strided handler also serves it.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED_BIAS
    = "hipkernel.gfx942_attention_dense.dispatch.v4";

/// The values of the `hipkernel.gfx942_attention_dense.dispatch.v1` and `.v2` contracts
/// (attention_dense::launchValues): the sixteen both arches save without strides,
/// `stride_layout` among them, and, as gfx942 serves stats and grids, `stats.uid`,
/// `grid_order`, `persist_decode` and `num_persistent`. v2 adds the four bias values, which
/// this writes exactly when the binding has a bias.
hipdnn_plugin_sdk::ingestor::MetadataValues
    gfx942AttentionDenseLaunchValues(const AttentionDenseBinding& binding,
                                     const AttentionDenseProblem& problem,
                                     int64_t blockM,
                                     const attention_dense::AttentionDenseGrid& grid = {});

/// The values of the strided contract (GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED), which
/// new plans are saved under: v1's with `stride_layout` replaced by the binding's twelve
/// strides, its block_n and its reads-strides flag; for a binding with an additive bias,
/// those of the strided bias contract (GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED_BIAS),
/// which add the four bias values.
hipdnn_plugin_sdk::ingestor::MetadataValues
    gfx942AttentionDenseStridedLaunchValues(const AttentionDenseBinding& binding,
                                            const AttentionDenseProblem& problem,
                                            int64_t blockM,
                                            const attention_dense::AttentionDenseGrid& grid = {});

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
