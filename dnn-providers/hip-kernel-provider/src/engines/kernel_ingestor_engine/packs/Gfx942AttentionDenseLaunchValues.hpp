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
/// seqlen_kv, num_query_heads, num_kv_heads). New plans are saved under it. Versions are
/// per dispatch name, so this is gfx942's first contract, not gfx950's v1.
inline constexpr std::string_view GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1
    = "hipkernel.gfx942_attention_dense.dispatch.v1";

/// The values of the `hipkernel.gfx942_attention_dense.dispatch.v1` contract
/// (attention_dense::launchValues). `scale` is stored as a double, which holds every float
/// exactly. `data_type` is the DataType enumerator name.
hipdnn_plugin_sdk::ingestor::MetadataValues gfx942AttentionDenseLaunchValues(
    const AttentionDenseBinding& binding, const AttentionDenseProblem& problem, int64_t blockM);

/// Everything a gfx942 dense-attention launch is computed from.
using Gfx942AttentionDenseLaunchInputs = AttentionDenseLaunchInputs;

/// Reads the values of the `hipkernel.gfx942_attention_dense.dispatch.v1` contract.
/// `batch`, `seqlen_q` and `seqlen_kv` must be in [1, INT32_MAX], because the kernel takes
/// them as 32-bit integers. The head counts and `head_size` must be in [1, INT32_MAX].
/// `block_m` must be one gfx942 builds: a positive multiple of 32 within a 1024-thread CTA
/// (block_n is not saved, so the full tile rule cannot be re-run). `scale` must convert to
/// a float and back without change; any finite scale is served. `data_type` must be a type
/// the kernel is built for, and `stride_layout` must be `bshd`. `causal` must be 0 or 1,
/// and `sliding_window` non-negative, 0 unless `causal` is 1, and not so narrow that the
/// last query rows see no key (seqlen_q - seqlen_kv >= sliding_window). As the graph and
/// kernel match require, `num_query_heads` must be a multiple of `num_kv_heads`,
/// `head_size` 64 or 128, K/V below 2^31 bytes, Q/O below 2^31 elements, and `seqlen_q` a
/// multiple of `block_m`: the kernel addresses Q and O without a bounds check.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is not the
///         contract, or when a value is missing, extra, of another type or out of range.
Gfx942AttentionDenseLaunchInputs readGfx942AttentionDenseLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
