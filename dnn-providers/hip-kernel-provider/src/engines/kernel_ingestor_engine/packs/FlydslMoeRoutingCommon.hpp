// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstddef>
#include <cstdint>

/**
 * @file FlydslMoeRoutingCommon.hpp
 * @brief The flyDSL MoE routing pack's launch geometry: the kernels' block sizes, the
 *        bounds the matcher applies, and how the pairs are split into blocks. Shared with
 *        the tests, which size their own workspaces by it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_moe_routing
{

// The builder's launch geometry, asserted against the kernel source by
// `generators/gen_moe_routing.py` before any object is written.
/// The histogram and scatter kernels' block, and the pairs one tile holds.
constexpr unsigned int TILE_THREADS = 256;
/// The prefix kernel's single block.
constexpr unsigned int PREFIX_THREADS = 1024;
/// The prefix kernel reads the workspace in 16-byte groups; it is padded to a whole one.
constexpr int64_t PREFIX_ITEMS = 4;

/// The pair count's bound: every pair's byte offset is a 32-bit buffer offset, and the
/// kernels' 0x7fffffff "no element" offset must lie past every real one.
constexpr int64_t MAX_PAIRS = (int64_t{1} << 29) - 1;
/// num_experts within the kernels' int32 arguments; the objects bound it further (e_max).
constexpr int64_t MAX_EXPERTS = 0x7fffffff;

/// Used when the device reports no CU count.
constexpr int64_t FALLBACK_COMPUTE_UNITS = 64;
/// The workspace entries (experts x blocks) the split aims to stay within: the prefix
/// kernel scans them in a single block, so its time grows with them.
constexpr int64_t PREFIX_ENTRY_BUDGET = 65536;

/// How the pairs are split: `blocks` blocks of `span` pairs (a multiple of the tile), the
/// last possibly short.
struct RoutingSplit
{
    int64_t span = 0;
    int64_t blocks = 0;
};

/// At most one block per compute unit, one per tile of pairs, and, for many experts,
/// fewer still, so the prefix kernel's single block scans at most about
/// PREFIX_ENTRY_BUDGET entries (never fewer than one block).
inline RoutingSplit routingSplit(int64_t pairs, int64_t experts, int computeUnits)
{
    const int64_t tile = TILE_THREADS;
    const int64_t tiles = std::max<int64_t>(1, (pairs + tile - 1) / tile);
    const int64_t units = computeUnits > 0 ? computeUnits : FALLBACK_COMPUTE_UNITS;
    const int64_t byBudget
        = std::max<int64_t>(1, PREFIX_ENTRY_BUDGET / std::max<int64_t>(1, experts));
    const int64_t wanted = std::max<int64_t>(1, std::min({tiles, units, byBudget}));
    const int64_t span = ((tiles + wanted - 1) / wanted) * tile;
    return {span, std::max<int64_t>(1, (pairs + span - 1) / span)};
}

/// The counts workspace: one int32 per (expert, block), padded to a whole 16-byte group.
inline size_t workspaceBytesFor(int64_t experts, int64_t blocks)
{
    const int64_t entries = experts * blocks;
    const int64_t padded = (entries + PREFIX_ITEMS - 1) / PREFIX_ITEMS * PREFIX_ITEMS;
    return static_cast<size_t>(padded) * sizeof(int32_t);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_moe_routing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
