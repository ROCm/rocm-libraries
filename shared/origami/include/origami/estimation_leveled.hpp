// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <vector>

#include "origami/hardware.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

/**
 * @brief Leveled coarse-to-fine estimation over a candidate set.
 *
 * The GEMM analytical model's internal scoring cascade (implemented in
 * estimation_leveled.cpp on top of the gemm:: primitives in gemm_common.cpp):
 *   - levels 0+1 (fused, context-free): fast-reject/feasibility filter + a
 *     compute-bound proxy (per-tile compute x a cheap num_output_tiles/N_CU
 *     timestep estimate). No context is built, so pruned configs never pay for
 *     context construction (streamk launch-parameter selection).
 *   - level 2: build the context; memory-aware roofline (flat ~0.5 cache proxy).
 *   - level 3: full analytical latency (estimation_latency_from_context).
 * The working set is pruned between the coarse levels (funnel), and a per-config
 * context is carried so finer levels reuse coarser work.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param configs Full candidate array (indexed by @p survivors).
 * @param survivors Indices into @p configs to score.
 * @return scored_configs_t (cost, original index) pairs for the full-detail
 *         survivors, sorted ascending by cost.
 */
ORIGAMI_EXPORT scored_configs_t score_estimation_leveled(
    const problem_t& problem,
    const hardware_t& hardware,
    const std::vector<config_t>& configs,
    const std::vector<std::size_t>& survivors);

// TEMPORARY (diagnostic): if non-null, score_estimation_leveled records, per config
// index, the level at which that config left the cascade. The caller sizes it to
// configs.size() (init to -1) before calling and reads it after. Codes:
//   0 = infeasible (L0), 1 = fast_reject (L0), 2 = compute-proxy reject (L1),
//   11 = pruned by L1 keep-fraction, 12 = pruned by L2 keep-fraction,
//   100 + rank = survived to the final level (rank 0 == best).
ORIGAMI_EXPORT extern thread_local std::vector<int>* g_prune_trace;

}  // namespace gemm
}  // namespace origami
