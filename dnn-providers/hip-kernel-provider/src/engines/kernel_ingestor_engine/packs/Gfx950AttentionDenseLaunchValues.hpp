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
 * Keep the handler of the old version registered while plans saved under it must load.
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

/// The strided contract, v4: v2's values with `stride_layout` replaced by the twelve
/// element strides (attention_dense::DENSE_STRIDE_NAMES), and the launch passes v2's ten
/// arguments and then those strides (22 kernel arguments), for kernels that read every
/// stride at runtime. It also saves the kernel's `block_n` and `reads_strides` (whether its
/// body reads them, see candidateReadsStrides: 1 for the grid body and the persistent work
/// orders that decode through the shape arguments, 0 for the ragged body, the gqa_pair work
/// orders and the bias bodies, whose plans hold packed strides only). New plans without an
/// additive bias are saved under it. Its version is spelled only here: the next free number
/// of this dispatch name.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED
    = "hipkernel.gfx950_attention_dense.dispatch.v4";

/// The strided bias contract, v5: v4's values plus the bias uid and its batch, query-head and
/// query-row element strides (attention_dense::BIAS_*_VALUE); the launch passes v2's ten
/// arguments, additive_bias_ptr and the three bias strides, and then the twelve strides (26
/// kernel arguments). New plans with an additive bias are saved under it. The strided handler
/// also serves it.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED_BIAS
    = "hipkernel.gfx950_attention_dense.dispatch.v5";

/// The values of the `hipkernel.gfx950_attention_dense.dispatch.v1`, `.v2` and `.v3`
/// contracts. v1 and v2 share one value set (attention_dense::launchValues): the sixteen both
/// arches save without strides, `stride_layout` among them. v3 adds the four bias values,
/// which this writes exactly when the binding has a bias.
hipdnn_plugin_sdk::ingestor::MetadataValues gfx950AttentionDenseLaunchValues(
    const AttentionDenseBinding& binding, const AttentionDenseProblem& problem, int64_t blockM);

/// The values of the strided contract (GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED), which
/// new plans are saved under: the fifteen both arches share, the binding's twelve strides,
/// its block_n and its reads-strides flag.
/// For a binding with an additive bias, the values of the strided bias contract
/// (GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_STRIDED_BIAS) instead, which add the four bias
/// values.
hipdnn_plugin_sdk::ingestor::MetadataValues gfx950AttentionDenseStridedLaunchValues(
    const AttentionDenseBinding& binding, const AttentionDenseProblem& problem, int64_t blockM);

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
/// @throws HipdnnPluginException with a load refusal when the dispatch name is none of the
///         three contracts, or when a value is missing, extra, of another type or out of
///         range.
Gfx950AttentionDenseLaunchInputs readGfx950AttentionDenseLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
