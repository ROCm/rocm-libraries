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
/// (attention_dense::launchValues): the sixteen both arches save, and, as gfx942 serves
/// stats and grids, `stats.uid`, `grid_order`, `persist_decode` and `num_persistent`.
hipdnn_plugin_sdk::ingestor::MetadataValues
    gfx942AttentionDenseLaunchValues(const AttentionDenseBinding& binding,
                                     const AttentionDenseProblem& problem,
                                     int64_t blockM,
                                     const attention_dense::AttentionDenseGrid& grid = {});

/// Everything a gfx942 dense-attention launch is computed from.
using Gfx942AttentionDenseLaunchInputs = AttentionDenseLaunchInputs;

/// Reads the values of the `hipkernel.gfx942_attention_dense.dispatch.v1` contract.
/// `block_m` must be one gfx942 builds: a positive multiple of 32 within a 1024-thread CTA.
/// For every other rule, including those for the stats and grid values, see
/// attention_dense::readLaunchInputs.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not the
///         contract, or when a value is missing, extra, of another type or out of range.
Gfx942AttentionDenseLaunchInputs readGfx942AttentionDenseLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
