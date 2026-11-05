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
/// saved under it. Versions are per dispatch name, so this is gfx942's first contract, not
/// gfx950's v1.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1
    = "hipkernel.gfx942_attention_dense.dispatch.v1";

/// The values of the `hipkernel.gfx942_attention_dense.dispatch.v1` contract
/// (attention_dense::launchValues): the sixteen gfx950's v2 saves; `stats.uid`, the stats
/// output's uid or ATTENTION_DENSE_NO_STATS; and the launch grid, `grid_order` (qb_minor,
/// bt_hkv_minor, hq_minor_swz or persistent) with, on the persistent grid, its
/// `persist_decode` and `num_persistent` (empty and 0 off it). `scale` is stored as a
/// double, which holds every float exactly. `data_type` is the DataType enumerator name.
/// The fast-division magic numbers and the grid are recomputed on restore, never saved.
hipdnn_plugin_sdk::ingestor::MetadataValues
    gfx942AttentionDenseLaunchValues(const AttentionDenseBinding& binding,
                                     const AttentionDenseProblem& problem,
                                     int64_t blockM,
                                     const attention_dense::AttentionDenseGrid& grid = {});

/// Everything a gfx942 dense-attention launch is computed from.
using Gfx942AttentionDenseLaunchInputs = AttentionDenseLaunchInputs;

/// Reads the values of the `hipkernel.gfx942_attention_dense.dispatch.v1` contract.
/// `batch`, `seqlen_q` and `seqlen_kv` must be in [1, INT32_MAX], because the kernel takes
/// them as 32-bit integers. The head counts and `head_size` must be in [1, INT32_MAX].
/// `block_m` must be one gfx942 builds: a positive multiple of 32 within a 1024-thread CTA
/// (block_n is not saved, so the full tile rule cannot be re-run). `scale` must convert to
/// a float and back without change, within [2^-64, 2^4]. `data_type` must be a type the
/// kernel is built for, and `stride_layout` must be `bshd`. `causal` must be 0 or 1,
/// and `sliding_window` non-negative, 0 unless `causal` is 1, and not so narrow that the
/// last query rows see no key (seqlen_q - seqlen_kv >= sliding_window). As the graph and
/// kernel match require, `num_query_heads` must be a multiple of `num_kv_heads`,
/// `head_size` 64 or 128, K/V below 2^31 bytes, Q/O below 2^31 elements, and `seqlen_q` a
/// multiple of `block_m`: the kernel addresses Q and O without a bounds check.
/// `stats.uid` must be ATTENTION_DENSE_NO_STATS (-1) or a non-negative uid. `grid_order`
/// must name a grid; on the persistent grid `persist_decode` must be qb_major, hkv_major,
/// bt_hkv_minor or hq_minor_swz (never `auto` or gqa_pair*) and `num_persistent` in
/// [1, INT32_MAX], and off it they must be empty and 0. The grid must be legal at the
/// saved shape: hq_minor_swz needs Hq (and on the persistent grid num_persistent) a
/// multiple of 8 and an even query-block count under causal balancing, and a persistent
/// decode's fast-division dividends must stay below 2^30. Restore selects the argument
/// list from `stats.uid`, `grid_order` and `persist_decode` before this reader runs, so its
/// signature check refuses a kernel whose LSE output or grid disagrees with the plan.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not the
///         contract, or when a value is missing, extra, of another type or out of range.
Gfx942AttentionDenseLaunchInputs readGfx942AttentionDenseLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
