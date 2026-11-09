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
/// New plans are saved under it.
inline constexpr std::string_view GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2
    = "hipkernel.gfx950_attention_dense.dispatch.v2";

/// The values of the `hipkernel.gfx950_attention_dense.dispatch.v1` and `.v2` contracts,
/// which share one value set (attention_dense::launchValues): the sixteen both arches save.
hipdnn_plugin_sdk::ingestor::MetadataValues gfx950AttentionDenseLaunchValues(
    const AttentionDenseBinding& binding, const AttentionDenseProblem& problem, int64_t blockM);

/// Everything a gfx950 dense-attention launch is computed from.
using Gfx950AttentionDenseLaunchInputs = AttentionDenseLaunchInputs;

/// Reads the values of the `hipkernel.gfx950_attention_dense.dispatch.v1` or `.v2`
/// contract (one value set; the versions differ only in the kernel arguments the launch
/// passes). `block_m` must be 128 or 256, the two gfx950 builds. For every other rule, and
/// the one gfx950 exception to them, see attention_dense::readLaunchInputs.
///
/// @throws HipdnnPluginException with a load refusal when the dispatch name is neither
///         contract, or when a value is missing, extra, of another type or out of range.
Gfx950AttentionDenseLaunchInputs readGfx950AttentionDenseLaunchInputs(
    const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs);

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
