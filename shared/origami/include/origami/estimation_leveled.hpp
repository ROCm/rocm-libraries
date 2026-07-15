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
 *   - levels 0+1 (fused, context-free): a fast-reject/feasibility filter plus a
 *     compute-vs-memory proxy (per-tile compute vs A/B read bandwidth, scaled by
 *     the K-iteration count, over a cheap num_output_tiles/N_CU timestep). No
 *     context is built, so pruned configs never pay for context construction
 *     (streamk launch-parameter selection).
 *   - prune: keep the cheapest survivors (funnel) before the full level. Split-K-
 *     prone problems skip this prune -- their winner turns on cross-CU K-reduction
 *     reuse the coarse proxy cannot see -- so every candidate reaches the full level.
 *   - full level: build the context per survivor and score at full analytical
 *     latency (estimation_latency_from_context).
 * The result is sorted ascending by the full-level latency.
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

}  // namespace gemm
}  // namespace origami
