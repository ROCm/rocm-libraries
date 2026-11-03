// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Internal header: the leveled estimation cascade used by rank_configs() when
// ORIGAMI_LEVELED_ESTIMATION is set. Not installed and not exported.

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "origami/hardware.hpp"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

/// Output of rank_leveled_estimation().
struct leveled_estimation_result_t {
  /// (latency, index into the configs) for the configs scored at the full level and not
  /// rejected there, in ascending index order.
  std::vector<std::pair<double, std::size_t>> scored;
  /// Indices of the configs that passed level 0 but were pruned at level 1, in proxy
  /// order (cheapest first, index order on equal proxy cost). They were not scored with
  /// the full model.
  std::vector<std::size_t> pruned;
};

/**
 * @brief Whether the leveled cascade prunes any config for this problem.
 *
 * False for GEMV shapes (M == 1 or N == 1) and for split-K-prone shapes: a reference
 * 256x256 tiling underfills the problem's CU budget and K has at least 64 iterations of a
 * 64-deep k-tile. For those shapes the cascade would score every config that passes
 * level 0 with the full model, which is the single-pass ranking, so rank_configs() checks
 * this first and takes the single pass directly. It depends on the problem and the CU
 * budget only, not on the configs.
 */
bool leveled_estimation_prunes(const problem_t& problem, const hardware_t& hardware);

/**
 * @brief Score configs with the leveled (coarse-to-fine) estimation cascade.
 *
 * 1. Level 0: drop configs that the single-pass ranking rejects without running the
 *    model (LDS capacity, then fast_reject()).
 * 2. Level 1: score the rest with a cheap context-free proxy and keep the best
 *    max(kLeveledMinKeep, n * kLeveledKeepFraction). Configs whose proxy cost equals
 *    that of the last kept config are kept too, so the cut never splits a group of
 *    equal proxy cost. The prune is skipped when leveled_estimation_prunes() is false.
 * 3. Full level: build the context for each survivor and score it with
 *    estimation_latency_from_context(), the same latency the single-pass ranking computes.
 *
 * All configs must use the estimation prediction mode. Every config that passes level 0
 * appears in exactly one of the two output lists.
 *
 * @return The scored survivors and the pruned configs. Both are empty if every config is
 *         rejected at level 0.
 */
leveled_estimation_result_t rank_leveled_estimation(const problem_t& problem,
                                                    const hardware_t& hardware,
                                                    const std::vector<config_t>& configs);

}  // namespace gemm
}  // namespace origami
